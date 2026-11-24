"""QA stages inside the existing frozen data-generation Worker execution."""
import asyncio
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from .bounded_async import bounded_map_unordered

from llm_platform_core.dense_qa import (
    corpus_stats,
    group_review_prompt,
    group_duplicate_indices,
    digest,
    duplicate_kind,
    generation_prompt,
    planner_prompt,
    planner_prompt_v1,
    planner_prompt_v3,
    generation_prompt_v3,
    grouped_qa_prompt_v3,
    review_prompt_v3,
    validate_evidence_plans,
    validate_grouped_qa,
    review_prompt,
    stage_schema,
    validate_plans,
    validate_qa,
    validate_review,
)


class _CorpusStatsAccumulator:
    """Maintain live QA statistics without rescanning the growing corpus."""

    def __init__(self, rows=()):
        self.qa_count = 0
        self.node_counts = Counter()
        self.angles = Counter()
        self.difficulties = Counter()
        self.answer_lengths = Counter()
        self.node_angles = defaultdict(Counter)
        self.cross_node_count = 0
        self.single_node_count = 0
        self.annotate_only_count = 0
        self.add(rows)

    def add(self, rows):
        for row in rows:
            node_id = row["primary_node_id"]
            self.qa_count += 1
            self.node_counts[node_id] += 1
            self.angles[row["angle"]] += 1
            self.difficulties[row["difficulty"]] += 1
            self.answer_lengths[row["answer_length"]] += 1
            self.node_angles[node_id][row["angle"]] += 1
            source_count = len(row["source_nodes"])
            self.cross_node_count += source_count > 1
            self.single_node_count += source_count == 1
            self.annotate_only_count += row.get("review_policy") == "annotate_only"

    def snapshot(self):
        node_total = len(self.node_counts)
        return {
            "qa_count": self.qa_count,
            "primary_node_count": node_total,
            "qa_per_node": dict(self.node_counts),
            "qa_per_node_average": self.qa_count / max(1, node_total),
            "angle_distribution": dict(self.angles),
            "difficulty_distribution": dict(self.difficulties),
            "answer_length_distribution": dict(self.answer_lengths),
            "cross_node_count": self.cross_node_count,
            "single_node_count": self.single_node_count,
            "angle_coverage": {key: dict(value) for key, value in self.node_angles.items()},
            "quality_status": ("candidates_pending_independent_review" if self.annotate_only_count
                               else "same_model_review_passed_human_review_required"),
        }


async def generate_qa(source, config, request, stream, provenance, report, emit, attempt,
                      *, select_window=None, reset_cache=None, prepare_capacity=None):
    evidence_first = config.prompt_version == "dense-qa-v3"
    accepted = []
    if Path(stream.name).exists():
        with Path(stream.name).open(encoding="utf-8") as existing:
            for line in existing:
                accepted.extend(json.loads(line).get("qa_records", []))
    live_stats = _CorpusStatsAccumulator(accepted)
    report.update(qa_count=len(accepted), qa_rejected=0, qa_duplicate_removed=0,
                  qa_semantic_duplicate_removed=0, qa_grounding_failed=0, qa_planner_failed=0)
    node_target = config.qa_per_node_target
    one_shot = config.qa_planning_mode == "one_shot"
    grouped = config.qa_planning_mode == "grouped"
    plan_limit = node_target if one_shot or grouped else 8
    report.update(qa_planning_mode=config.qa_planning_mode, qa_group_duplicate_removed=0, qa_plans_proposed=0, qa_plans_validated=0)
    report.update(qa_per_node_target=node_target, qa_node_shortfalls=[],
                  qa_review_policy="annotate_only" if one_shot or grouped else "filter",
                  qa_duplicate_flagged=0, qa_review_flagged=0, qa_group_duplicate_flagged=0, qa_group_review_failed=0)
    automatic = config.batch_size == "auto"
    gate = asyncio.Semaphore(min(config.max_batch_size, 64) if automatic else int(config.batch_size))
    tuned = not automatic
    probing = False
    audit_path = attempt / "generation-validation.jsonl"
    with audit_path.open("a", encoding="utf-8") as audit:
        async def ask(item, prompt, stage, *, group_count=None):
            bounded = {**item, "prompt": prompt, "qa_json_schema": stage_schema(stage, item["qa_sources"], evidence_first=evidence_first, plan_limit=plan_limit, group_count=group_count)}
            async with gate:
                _, result, _, _ = await request(bounded, full_budget=True)
            report["raw_generated_tokens"] = report.get("raw_generated_tokens", 0) + result[1]
            report["raw_completed_requests"] = report.get("raw_completed_requests", 0) + 1
            audit.write(json.dumps({"stage": stage, "autotune_probe": probing, "primary_node_id": item["qa_sources"][0]["id"],
                                    "source_sha256": item["source_sha256"], "request": prompt,
                                    "response": result[0], "tokens": result[1], "truncated": result[2]}, ensure_ascii=False) + "\n")
            audit.flush()
            if result[2]:
                raise ValueError("QA stage reached output token limit")
            value = json.loads(result[0])
            if not isinstance(value, dict):
                raise ValueError("QA stage requires JSON object")
            return value

        if grouped:
            items = list(source)
            report.update(qa_grouped_requests=0, qa_grouped_invalid=0, qa_group_duplicate_flagged=0,
                          qa_review_policy="program_validated_pending_independent_review")

            async def grouped_node(item):
                sources = item["qa_sources"]
                emit("progress", phase="qa_grouped_generation", qa_count=len(accepted), qa_limit=config.qa_limit,
                     primary_concept=sources[0]["concept"], node_qa_count=0)
                try:
                    raw = await ask(item, grouped_qa_prompt_v3(sources, node_target), "group_generate")
                    report["qa_grouped_requests"] += 1
                    proposed = len(raw.get("items", [])) if isinstance(raw.get("items"), list) else 0
                    pairs = validate_grouped_qa(raw, sources, node_target)
                    report["qa_plans_proposed"] += proposed
                    report["qa_plans_validated"] += len(pairs)
                    report["qa_grouped_invalid"] += max(0, proposed - len(pairs))
                except ValueError as exc:
                    report["qa_planner_failed"] += 1
                    audit.write(json.dumps({"stage": "group_generate_rejected", "primary_node_id": sources[0]["id"],
                                            "reason": str(exc)}, ensure_ascii=False) + "\n")
                    audit.flush()
                    return item, []
                result = []
                for index, (plan, qa) in enumerate(pairs):
                    required = [n for n in sources if n["id"] in plan["source_ids"]]
                    kind = duplicate_kind(qa, result)
                    flags = []
                    if kind:
                        flags.append({"type": "duplicate", "method": kind})
                        report["qa_duplicate_flagged"] += 1
                    signature = digest({"sources": sorted(plan["source_ids"]), "intent": plan["intent"]})
                    result.append({"qa_id": "qa_" + digest({"version": provenance["input_dataset_version_id"],
                                  "primary": sources[0]["id"], "question": qa["question"], "grouped_index": index})[:24],
                        "primary_node_id": sources[0]["id"], "primary_atlas_order": sources[0]["atlas_order"],
                        "source_nodes": [{k: n[k] for k in ("id", "atlas_order", "concept")} for n in required],
                        "angle": plan["angle"], "difficulty": plan["difficulty"], **qa,
                        "answer_length": "short" if len(qa["answer"]) < 150 else "medium" if len(qa["answer"]) <= 400 else "long",
                        "generation_batch": "0001", "question_signature": signature, "node_generation_round": 1,
                        "canonical_source_ids": sorted(plan["source_ids"]),
                        "grounding_review": {"status": "program_evidence_validated", "evidence": plan["evidence"]},
                        "review_status": "pending_independent_review", "review_flags": flags,
                        "review_policy": "annotate_only", "planning_evidence": plan["evidence"],
                        "messages": [{"role": "user", "content": qa["question"]}, {"role": "assistant", "content": qa["answer"]}]})
                return item, result

            if automatic and items:
                if select_window is None:
                    raise RuntimeError("Grouped QA auto mode requires the platform throughput selector")
                probing = True
                sample = items[:min(len(items), config.max_batch_size, 8)]
                cap = len(sample)

                async def trial(window):
                    nonlocal gate
                    if prepare_capacity is not None:
                        await prepare_capacity(window)
                    if reset_cache is not None and not await reset_cache():
                        raise RuntimeError("vLLM could not reset idle prefix cache for grouped QA autotuning")
                    gate = asyncio.Semaphore(window)
                    tokens_before = report.get("raw_generated_tokens", 0)
                    counters = {key: report.get(key, 0) for key in (
                        "qa_grouped_requests", "qa_grouped_invalid", "qa_plans_proposed",
                        "qa_plans_validated", "qa_planner_failed", "qa_duplicate_flagged")}
                    began = time.perf_counter()
                    async with asyncio.timeout(600):
                        values = await asyncio.gather(*(grouped_node(item) for item in sample))
                    elapsed = max(time.perf_counter() - began, 1e-9)
                    tokens = report.get("raw_generated_tokens", 0) - tokens_before
                    valid = sum(len(rows) for _, rows in values)
                    report.update(counters)
                    return dict(seconds=elapsed, input_records=len(sample), requests=len(sample), raw_tokens=tokens,
                                accepted_records=valid, raw_tokens_per_second=tokens / elapsed,
                                accepted_tokens_per_second=0)

                report["qa_probe_stage"] = "group_generate"
                report["qa_probe_plan_count"] = len(sample)
                selected = await select_window(cap, trial, report)
                if prepare_capacity is not None:
                    await prepare_capacity(selected)
                probing = False
                gate = asyncio.Semaphore(selected)
                tuned = True
                emit("generation.batch_selected", **report)

            completed = 0
            worker_count = min(len(items), selected if automatic else int(config.batch_size))
            async for item, node_rows in bounded_map_unordered(items, grouped_node, worker_count):
                remaining = config.qa_limit - len(accepted)
                node_rows = node_rows[:max(0, remaining)]
                accepted.extend(node_rows)
                sources = item["qa_sources"]
                if len(node_rows) < node_target:
                    report["qa_node_shortfalls"].append({"node_id": sources[0]["id"], "concept": sources[0]["concept"],
                        "target": node_target, "accepted": len(node_rows), "rounds": 1})
                metadata = {**provenance, **{k: v for k, v in item.items() if k not in {"prompt", "qa_sources"}},
                            "generation_tokens": 0, "generation_attempt_id": attempt.name,
                            "qa_runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
                stream.write(json.dumps({"text": "", "metadata": metadata, "qa_records": node_rows}, ensure_ascii=False) + "\n")
                stream.flush(); os.fsync(stream.fileno())
                completed += 1
                live_stats.add(node_rows)
                report.update(live_stats.snapshot())
                emit("progress", phase="qa_grouped_reviewed", qa_count=len(accepted), qa_limit=config.qa_limit,
                     output_records=len(accepted), completed_nodes=completed, total_nodes=len(items),
                     primary_concept=sources[0]["concept"], raw_generated_tokens=report.get("raw_generated_tokens", 0))
            report.update(output_records=len(accepted), qa_target_reached=len(accepted) == config.qa_limit)
            return

        for item in source:
            if len(accepted) >= config.qa_limit:
                break
            sources = item["qa_sources"]
            node_rows = []
            for node_round in range(config.qa_node_max_rounds if node_target else 1):
                if len(accepted) >= config.qa_limit or (node_target and len(node_rows) >= node_target):
                    break
                emit("progress", phase="qa_planning", qa_count=len(accepted), qa_limit=config.qa_limit,
                     primary_concept=sources[0]["concept"], node_round=node_round + 1, node_qa_count=len(node_rows))
                try:
                    plan_prompt = planner_prompt_v3(sources, remaining=(node_target - len(node_rows)) if node_target else None, previous=[r["question"] for r in node_rows], one_shot_target=node_target if one_shot else None) if evidence_first else (planner_prompt_v1(sources) if config.prompt_version == "dense-qa-v1" else planner_prompt(sources))
                    raw_plans = await ask(item, plan_prompt, "planner")
                    plans = validate_evidence_plans(raw_plans, sources, plan_limit=plan_limit) if evidence_first else validate_plans(raw_plans, sources)
                    report["qa_plans_proposed"] += len(raw_plans.get("plans", []))
                    report["qa_plans_validated"] += len(plans)
                except ValueError as exc:
                    plans = []
                    report["qa_planner_failed"] += 1
                    audit.write(json.dumps({"stage": "planner_rejected", "reason": str(exc)}, ensure_ascii=False) + "\n")

                async def draft(plan, sources=sources, item=item):
                    required = [n for n in sources if n["id"] in plan["source_ids"]]
                    try:
                        qa = validate_qa(await ask(item, (generation_prompt_v3 if evidence_first else generation_prompt)(required, plan), "generate"))
                        return plan, required, qa
                    except ValueError as exc:
                        report["qa_rejected"] += 1
                        audit.write(json.dumps({"stage": "draft_rejected", "reason": str(exc)}, ensure_ascii=False) + "\n")
                        return None

                if not tuned and plans:
                    if select_window is None:
                        raise RuntimeError("QA auto mode requires the platform throughput selector")
                    # Probe the same real, grounded draft requests for each window.
                    # Probe candidates never enter the output/checkpoint corpus.
                    probing = True
                    rejected_before = report["qa_rejected"]
                    await draft(plans[0])  # exclude lazy model/schema setup
                    cap = min(config.max_batch_size, 64, len(plans))

                    async def trial(window):
                        nonlocal gate
                        if prepare_capacity is not None:
                            await prepare_capacity(min(window, cap))
                            await draft(plans[0])
                        if reset_cache is not None and not await reset_cache():
                            raise RuntimeError("vLLM could not reset idle prefix cache for QA autotuning")
                        gate = asyncio.Semaphore(window)
                        tokens_before = report.get("raw_generated_tokens", 0)
                        began = time.perf_counter()
                        async with asyncio.timeout(240):
                            values = await asyncio.gather(*(draft(p) for p in plans))
                        elapsed = max(time.perf_counter() - began, 1e-9)
                        tokens = report.get("raw_generated_tokens", 0) - tokens_before
                        valid = sum(v is not None for v in values)
                        # Draft validation is structural, not factual certification.
                        return dict(seconds=elapsed, input_records=len(plans), requests=len(plans),
                                    raw_tokens=tokens, accepted_records=valid,
                                    raw_tokens_per_second=tokens / elapsed,
                                    accepted_tokens_per_second=0)

                    report["qa_probe_stage"] = "generate"
                    report["qa_probe_plan_count"] = len(plans)
                    selected = await select_window(cap, trial, report)
                    if prepare_capacity is not None:
                        await prepare_capacity(min(selected, cap))
                    report["qa_probe_rejected"] = report["qa_rejected"] - rejected_before
                    report["qa_rejected"] = rejected_before
                    probing = False
                    gate = asyncio.Semaphore(selected)
                    tuned = True
                    emit("generation.batch_selected", **report)
                    (attempt / "generation-runtime.json").write_text(
                        json.dumps(report, ensure_ascii=False), encoding="utf-8")

                candidates = [r for r in await asyncio.gather(*(draft(p) for p in plans)) if r]
                # Only overlapping sources enter semantic comparison; unrelated topics cannot support facts.
                prior = [r for r in accepted if set(n["id"] for n in r["source_nodes"]) & set(n["id"] for n in sources)]

                async def review(index, candidate, prior=prior, candidates=candidates, item=item, sources=sources):
                    plan, required, qa = candidate
                    previous = prior[-30:] + [c[2] for c in candidates[:index]]
                    kind = duplicate_kind(qa, accepted + [c[2] for c in candidates[:index]])
                    if kind and not one_shot:
                        report["qa_duplicate_removed"] += 1
                        return None
                    flags = []
                    judgment = {"status": "unavailable"}
                    if kind:
                        flags.append({"type": "duplicate", "method": kind})
                        report["qa_duplicate_flagged"] += 1
                    try:
                        judgment = await ask(item, (review_prompt_v3 if evidence_first else review_prompt)(required, plan, qa,
                            [{"question": r["question"]} for r in previous]), "review")
                        if judgment.get("semantic_duplicate") is True and not one_shot:
                            report["qa_semantic_duplicate_removed"] += 1
                        validate_review(judgment, required)
                    except ValueError as exc:
                        report["qa_grounding_failed"] += 1
                        if not one_shot:
                            audit.write(json.dumps({"stage": "qa_rejected", "question": qa["question"], "reason": str(exc)}, ensure_ascii=False) + "\n")
                            return None
                        flags.append({"type": "review_failed", "reason": str(exc)})
                        report["qa_review_flagged"] += 1
                    signature = digest({"sources": sorted(plan["source_ids"]), "intent": plan["intent"]})
                    return {"qa_id": "qa_" + digest({"version": provenance["input_dataset_version_id"],
                                "primary": sources[0]["id"], "question": qa["question"],
                                **({"plan": plan, "candidate_index": index} if one_shot else {})})[:24],
                            "primary_node_id": sources[0]["id"], "primary_atlas_order": sources[0]["atlas_order"],
                            "source_nodes": [{k: n[k] for k in ("id", "atlas_order", "concept")} for n in required],
                            "angle": plan["angle"], "difficulty": plan["difficulty"], **qa,
                            "answer_length": "short" if len(qa["answer"]) < 150 else "medium" if len(qa["answer"]) <= 400 else "long",
                            "generation_batch": "0001", "question_signature": signature,
                            **({"node_generation_round": node_round + 1} if node_target else {}),
                            "canonical_source_ids": sorted(plan["source_ids"]),
                            "grounding_review": judgment,
                            **({"review_status": "pending_independent_review", "review_flags": flags,
                                "review_policy": "annotate_only"} if one_shot else {}),
                            **({"planning_evidence": plan["evidence"]} if evidence_first else {}),
                            "messages": [{"role": "user", "content": qa["question"]}, {"role": "assistant", "content": qa["answer"]}]}

                reviewed = await asyncio.gather(*(review(i, c) for i, c in enumerate(candidates)))
                remaining = min(config.qa_limit - len(accepted), node_target - len(node_rows)) if node_target else config.qa_limit - len(accepted)
                rows = [r for r in reviewed if r][:remaining]
                if one_shot and len(rows) > 1:
                    emit("progress", phase="qa_group_review", primary_concept=sources[0]["concept"], qa_count=len(accepted), group_size=len(rows))
                    try:
                        group_verdict = await ask(item, group_review_prompt(sources, rows), "group_review", group_count=len(rows))
                        group_duplicate_indices(group_verdict, len(rows))  # Validate references; never use verdict to delete.
                        flagged = set()
                        for group in group_verdict["duplicate_groups"]:
                            for i in group["indices"]:
                                flagged.add(i)
                                rows[i]["review_flags"].append({"type": "group_duplicate_suspected",
                                    "candidate_ids": [rows[j]["qa_id"] for j in group["indices"]],
                                    "reason": group["reason"]})
                        report["qa_group_duplicate_flagged"] += len(flagged)
                    except ValueError as exc:
                        report["qa_group_review_failed"] += 1
                        for row in rows:
                            row["review_flags"].append({"type": "group_review_failed", "reason": str(exc)})
                        audit.write(json.dumps({"stage": "group_review_flagged", "reason": str(exc)}, ensure_ascii=False) + "\n")
                node_rows.extend(rows)
                accepted.extend(rows)
                live_stats.add(rows)
            if node_target and len(node_rows) < node_target:
                report["qa_node_shortfalls"].append({"node_id": sources[0]["id"], "concept": sources[0]["concept"],
                    "target": node_target, "accepted": len(node_rows), "rounds": node_round + 1})
            metadata = {**provenance, **{k: v for k, v in item.items() if k not in {"prompt", "qa_sources"}},
                        "generation_tokens": 0, "generation_attempt_id": attempt.name,
                        "qa_runtime_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            # One atomic JSONL checkpoint record per mother node; final artifact is one row per QA.
            stream.write(json.dumps({"text": "", "metadata": metadata, "qa_records": node_rows}, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            report.update(live_stats.snapshot())
            emit("progress", phase="qa_reviewed", qa_count=len(accepted), qa_limit=config.qa_limit,
                 output_records=len(accepted), primary_concept=sources[0]["concept"],
                 raw_generated_tokens=report.get("raw_generated_tokens", 0))
        report.update(output_records=len(accepted), qa_target_reached=len(accepted) == config.qa_limit)


def finalize_qa_batches(temporary, destination, report):
    rows = []
    with temporary.open(encoding="utf-8") as source, destination.open("w", encoding="utf-8", newline="\n") as out:
        for line in source:
            batch = json.loads(line)
            for qa in batch.get("qa_records", []):
                row = {**qa, "text": "问：" + qa["question"] + "\n答：" + qa["answer"], "metadata": batch["metadata"]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
        out.flush()
        os.fsync(out.fileno())
    report.update(corpus_stats(rows), output_records=len(rows))
    temporary.unlink()
