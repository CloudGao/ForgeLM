"""Grounded relational knowledge stage inside native data generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from llm_platform_core.dense_relations import prompt, schema, validate

from .bounded_async import bounded_map_unordered


def imported_relation_state(stream):
    path = Path(stream.name)
    completed_nodes, completed_ordinals, count = 0, set(), 0
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
                count += len(batch.get("relation_records", []))
    return completed_nodes, count, completed_ordinals


async def generate_relations(source, config, request, stream, provenance, report, emit, attempt):
    items = list(source)
    resumed_nodes, relation_count, _ = imported_relation_state(stream)
    audit_path = attempt / "generation-validation.jsonl"
    concurrency = int(config.batch_size)

    async def one_node(pair):
        _, item = pair
        source_node = item["qa_sources"][0]
        bounded = {
            **item,
            "prompt": prompt(source_node, config.relation_per_node_limit),
            "qa_json_schema": schema(source_node, config.relation_per_node_limit),
        }
        last_raw, last_error = ("", 0, False), ""
        for retry in range(2):
            try:
                _, raw, _, _ = await request(bounded, full_budget=True, validation_retry=retry)
                last_raw = raw
                if raw[2]:
                    raise ValueError("relation generation reached output token limit")
                rows = validate(json.loads(raw[0]), source_node, config.relation_per_node_limit)
                return item, rows, raw, ""
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        return item, [], last_raw, last_error

    distribution = Counter()
    with audit_path.open("a", encoding="utf-8") as audit:
        completed = 0
        async for item, rows, raw, error in bounded_map_unordered(enumerate(items), one_node, concurrency):
            source_node = item["qa_sources"][0]
            output_rows = []
            for local_index, row in enumerate(rows):
                distribution[row["relation_type"]] += 1
                output_rows.append({
                    "relation_id": "rel_" + hashlib.sha256(
                        f"{provenance['input_dataset_version_id']}:{source_node['id']}:{local_index}:{row['statement']}".encode()
                    ).hexdigest()[:24],
                    "relation_type": row["relation_type"],
                    "statement": row["statement"],
                    "primary_node_id": source_node["id"],
                    "primary_atlas_order": source_node["atlas_order"],
                    "source_nodes": [{key: source_node[key] for key in ("id", "atlas_order", "concept")}],
                    "supporting_evidence": [{"node_id": source_node["id"], "quote": quote} for quote in row["evidence"]],
                    "grounding_review": {"status": "program_evidence_validated"},
                    "review_status": "pending_independent_review",
                })
            metadata = {
                **provenance,
                **{key: value for key, value in item.items() if key not in {"prompt", "qa_sources", "qa_json_schema"}},
                "generation_attempt_id": attempt.name,
                "relation_runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            stream.write(json.dumps({"text": "", "metadata": metadata, "relation_records": output_rows}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            audit.write(json.dumps({"primary_node_id": source_node["id"], "response": raw[0],
                                    "tokens": raw[1], "accepted": len(rows), "error": error}, ensure_ascii=False) + "\n")
            audit.flush()
            completed += 1
            relation_count += len(rows)
            report["raw_completed_requests"] = report.get("raw_completed_requests", 0) + 1
            report["raw_generated_tokens"] = report.get("raw_generated_tokens", 0) + raw[1]
            if error:
                report["relation_failed_nodes"] = report.get("relation_failed_nodes", 0) + 1
            report.update(output_records=relation_count, completed_nodes=resumed_nodes + completed,
                          total_nodes=resumed_nodes + len(items), relation_distribution=dict(distribution))
            emit("progress", phase="relations_generated", completed_nodes=resumed_nodes + completed,
                 total_nodes=resumed_nodes + len(items), output_records=relation_count,
                 primary_concept=source_node["concept"])
    report.update(relation_per_node_limit=config.relation_per_node_limit,
                  quality_status="candidates_pending_independent_review")


def finalize_relation_batches(temporary, destination, report):
    distribution, total = Counter(), 0
    with temporary.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8", newline="\n") as out:
        for line in source:
            batch = json.loads(line)
            for row in batch.get("relation_records", []):
                row["text"] = row["statement"]
                row["metadata"] = batch["metadata"]
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                distribution[row["relation_type"]] += 1
                total += 1
        out.flush()
        os.fsync(out.fileno())
    report.update(output_records=total, relation_distribution=dict(distribution))
    temporary.unlink()
