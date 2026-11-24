"""Grounded dialogue stage inside the native data-generation execution."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from llm_platform_core.dense_dialogue import KINDS, node_quota, prompt, schema, validate

from .bounded_async import bounded_map_unordered


def imported_dialogue_state(stream, targets):
    """Count verified batches already imported into this attempt's partial output."""
    path = Path(stream.name)
    counts = Counter()
    completed_nodes = 0
    completed_ordinals = set()
    if path.exists():
        with path.open(encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                batch = json.loads(line)
                completed_nodes += 1
                ordinal = batch.get("metadata", {}).get("source_ordinal")
                if ordinal is not None:
                    completed_ordinals.add(int(ordinal))
                for row in batch.get("dialogue_records", []):
                    kind = row.get("dialogue_kind")
                    if kind in KINDS:
                        counts[kind] += 1
    return completed_nodes, counts, completed_ordinals


async def generate_dialogues(source, config, request, stream, provenance, report, emit, attempt):
    items = list(source)
    targets = {
        "single_qa": config.dialogue_single_target,
        "multi_qa": config.dialogue_multi_target,
        "natural_chat": config.dialogue_natural_target,
    }
    resumed_nodes, counts, completed_ordinals = imported_dialogue_state(stream, targets)
    all_ordinals = sorted(completed_ordinals | {int(item["source_ordinal"]) for item in items})
    ordinal_positions = {ordinal: index for index, ordinal in enumerate(all_ordinals)}
    effective_targets = Counter(counts)
    for item in items:
        quota = node_quota(ordinal_positions[int(item["source_ordinal"])], len(all_ordinals), targets)
        effective_targets.update(quota)
    audit_path = attempt / "generation-validation.jsonl"
    concurrency = int(config.batch_size)

    async def one_node(pair):
        index, item = pair
        quota = node_quota(
            ordinal_positions[int(item["source_ordinal"])], len(all_ordinals), targets
        )
        source_node = item["qa_sources"][0]
        bounded = {**item, "prompt": prompt(source_node, quota), "qa_json_schema": schema(source_node, quota)}
        last_raw = ("", 0, False)
        last_error = ""
        for retry in range(2):
            try:
                _, raw, _, _ = await request(bounded, full_budget=True, validation_retry=retry)
                last_raw = raw
                if raw[2]:
                    raise ValueError("dialogue generation reached output token limit")
                rows = validate(json.loads(raw[0]), source_node, quota)
                return item, quota, rows, raw, ""
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        return item, quota, [], last_raw, last_error

    with audit_path.open("a", encoding="utf-8") as audit:
        completed = 0
        async for item, quota, rows, raw, error in bounded_map_unordered(enumerate(items), one_node, concurrency):
            source_node = item["qa_sources"][0]
            output_rows = []
            for local_index, row in enumerate(rows):
                counts[row["kind"]] += 1
                output_rows.append({
                    "dialogue_id": "dlg_" + hashlib.sha256(
                        f"{provenance['input_dataset_version_id']}:{source_node['id']}:{row['kind']}:{local_index}".encode()
                    ).hexdigest()[:24],
                    "dialogue_kind": row["kind"],
                    "primary_node_id": source_node["id"],
                    "primary_atlas_order": source_node["atlas_order"],
                    "source_nodes": [{key: source_node[key] for key in ("id", "atlas_order", "concept")}],
                    "planning_evidence": [{"node_id": source_node["id"], "quote": quote} for quote in row["evidence"]],
                    "grounding_review": {"status": "program_evidence_validated"},
                    "review_status": "pending_independent_review",
                    "messages": row["messages"],
                })
            metadata = {
                **provenance,
                **{key: value for key, value in item.items() if key not in {"prompt", "qa_sources", "qa_json_schema"}},
                "generation_attempt_id": attempt.name,
                "dialogue_runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            stream.write(json.dumps({"text": "", "metadata": metadata, "dialogue_records": output_rows}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            audit.write(json.dumps({"primary_node_id": source_node["id"], "quota": quota,
                                    "response": raw[0], "tokens": raw[1], "error": error}, ensure_ascii=False) + "\n")
            audit.flush()
            completed += 1
            report["raw_completed_requests"] = report.get("raw_completed_requests", 0) + 1
            report["raw_generated_tokens"] = report.get("raw_generated_tokens", 0) + raw[1]
            if error:
                report["dialogue_failed_nodes"] = report.get("dialogue_failed_nodes", 0) + 1
            report.update(dialogue_count=sum(counts.values()), dialogue_distribution=dict(counts),
                          output_records=sum(counts.values()), completed_nodes=resumed_nodes + completed,
                          total_nodes=resumed_nodes + len(items))
            emit("progress", phase="dialogue_generated", completed_nodes=resumed_nodes + completed,
                 total_nodes=resumed_nodes + len(items),
                 output_records=sum(counts.values()), dialogue_distribution=dict(counts),
                 primary_concept=source_node["concept"])
    report.update(dialogue_targets=targets, dialogue_effective_targets=dict(effective_targets),
                  dialogue_target_reached=counts == effective_targets,
                  quality_status="candidates_pending_independent_review")


def finalize_dialogue_batches(temporary, destination, report):
    counts = Counter()
    total = 0
    with temporary.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8", newline="\n") as out:
        for line in source:
            batch = json.loads(line)
            for row in batch.get("dialogue_records", []):
                row["text"] = "\n".join(("用户：" if m["role"] == "user" else "助手：") + m["content"] for m in row["messages"])
                row["metadata"] = batch["metadata"]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[row["dialogue_kind"]] += 1
                total += 1
        out.flush()
        os.fsync(out.fileno())
    report.update(output_records=total, dialogue_count=total, dialogue_distribution=dict(counts))
    temporary.unlink()
