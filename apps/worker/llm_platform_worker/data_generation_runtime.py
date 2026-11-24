"""Bounded local GPU batch generation. Invoked only with a frozen Worker task."""

import asyncio
import csv
import gc
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import deque
from datetime import UTC, datetime
from itertools import chain, islice
from pathlib import Path

import llm_platform_core.data_generation as generation_contract
from llm_platform_core.data_generation import (
    PROMPT_VERSIONS,
    DataGenerationCreate,
    FactTranslationCreate,
    factual_surface_anchors,
    prompt_for,
    validate_factual_surface,
)
from llm_platform_core.fact_translation import (
    finish_translation,
    translation_plan,
    translation_prompt,
    translation_schema,
)

from .adapters.data_import import DataImportAdapter
from .file_cache import advise_sequential, release_file_cache, release_file_cache_descriptor
from .project_storage import project_data_asset_root, scoped_runtime_path


def emit(event, **payload):
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


GENERATION_RUNTIME_POLICY = "zh-surface-guard-feedback-v1"
GENERATION_INPUT_POLICY = "zh-fact-input-guards-v6"
NATURAL_RUNTIME_POLICY = "natural-source-names-v8"


def render_generation_request(tokenizer, item, config, previous_text="", validation_error=""):
    prompt = item["prompt"]
    direction = (item.get("quality_anchors") or {}).get("temporal_direction")
    if direction:
        prompt += "\n本卡先后关系请按主体、客体原顺序叙述，使用“" + direction + "”，不要交换主客体改成反向说法。"
    messages = [{"role": "user", "content": prompt}]
    feedback = ""
    if validation_error:
        issues = []
        if "number_mismatch" in validation_error:
            numbers = (item.get("quality_anchors") or {}).get("numbers", [])
            issues.append("数字必须与原始事实一致，不新增或遗漏数字；世代、届次可用等值中文序数，其余保留阿拉伯数字。原始数字：" + json.dumps(numbers))
        if "timezone_mismatch" in validation_error:
            offsets = (item.get("quality_anchors") or {}).get("utc_offsets", [])
            issues.append("完整保留时区正负号、小时和分钟，写成协调世界时加原始偏移，不推断地域或夏令时。原始偏移：" + json.dumps(offsets))
        if "untranslated_words" in validation_error or "Chinese" in validation_error or "non_chinese" in validation_error:
            issues.append(
                "仅允许列表中的来源专名在无可靠译名时保留，其余外文词和句子译成中文；不以删除事实避开翻译。"
                if config.proper_name_policy == "source_exact" else
                "将残留外文译成中文；不附括号原名、英文缩写或外文标题，不以删除事实来避开翻译。"
            )
        if "obvious_relation_contradiction" in validation_error:
            expected = [check["expected"] for check in (item.get("quality_anchors") or {}).get("obvious_relation_checks", [])]
            issues.append("关系方向写反或把类型误作机构隶属。严格按以下主体—关系—客体重写："
                          + json.dumps(expected, ensure_ascii=False) + "。")
        if "source_boilerplate_as_content" in validation_error:
            issues.append("来源快照、知识库记录等是来源说明，不能充当人物、地点或其他事实，也不要复述来源免责声明。")
        if "association_promoted_to_definition" in validation_error:
            issues.append("输入只说明词语关联或使用语境，不能写成指代、定义、等同或必然因果；仅陈述有关联。")
        if "duplicated_entity_name" in validation_error:
            issues.append("实体姓名出现连续重复，删去重复片段并依据原始事实重写。")
        if "temporal_direction_word_mismatch" in validation_error:
            issues.append("先后关系的方向词错误，按原主体和客体顺序使用“" + str(direction) + "”，不要倒置。")
        if not issues:
            issues.append("只输出完整中文陈述正文，去掉思考标签、问答标记及任务说明。")
        feedback = "上一版未通过校验。" + "".join(issues) + "依据最初事实修正，保持主体、关系方向和限定；只输出修正后的正文。"
        messages += [{"role": "assistant", "content": previous_text}, {"role": "user", "content": feedback}]
    while True:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        length = len(tokenizer.encode(rendered, add_special_tokens=False))
        if length <= config.max_input_tokens:
            return rendered, length
        # Preserve the complete frozen source. Omit optional correction context
        # instead of truncating facts or rejecting an otherwise valid request.
        if len(messages) > 1:
            messages = [{"role": "user", "content": prompt + "\n\n" + feedback}]
        elif feedback:
            messages = [{"role": "user", "content": prompt}]
            feedback = ""
        else:
            raise ValueError("Input exceeds max_input_tokens; reduce chunk_characters (no silent truncation)")


def chinese_sampling_constraint(item, enabled=True):
    if not enabled or not (item.get("quality_anchors") or {}).get("chinese_only"):
        return {}
    if item["quality_anchors"].get("requires_latin_spelling"):
        raise ValueError("前置纯中文约束不能用于英文词形/词源卡：请先明确保留外文或筛选此类输入，不能硬译为中文词间关系")
    from vllm.sampling_params import StructuredOutputsParams

    return {"structured_outputs": StructuredOutputsParams(regex=r"[^A-Za-z]*")}


class GenerationRateWindow:
    def __init__(self, now, tokens=0, records=0, seconds=120):
        self.samples = deque([(now, tokens, records)])
        self.seconds = seconds

    def update(self, now, tokens, records):
        self.samples.append((now, tokens, records))
        while len(self.samples) > 2 and self.samples[1][0] <= now - self.seconds:
            self.samples.popleft()
        start, previous_tokens, previous_records = self.samples[0]
        elapsed = max(now - start, 1e-9)
        return {
            "recent_tokens_per_second": (tokens - previous_tokens) / elapsed,
            "recent_documents_per_second": (records - previous_records) / elapsed,
            "rate_window_seconds": elapsed,
        }


class GenerationDiagnostics:
    """Sample per-record diagnostics; durable skip journals retain rejected rows."""

    def __init__(self, emit_event, interval=30):
        self.emit_event, self.interval = emit_event, interval
        self.counts, self.last_emitted = {}, {}

    def emit(self, event, **payload):
        self.counts[event] = self.counts.get(event, 0) + 1
        now = time.perf_counter()
        if now - self.last_emitted.get(event, float("-inf")) >= self.interval:
            self.emit_event(event, **payload, total_occurrences=self.counts[event],
                            diagnostic_sample=True)
            self.last_emitted[event] = now


async def log_vllm_stats(engine, interval=30):
    # Keep observing stalled/slow requests too, independently of completions.
    while True:
        await asyncio.sleep(interval)
        try:
            await engine.do_log_stats()
        except Exception as exc:
            emit("log", message=f"vLLM statistics unavailable: {type(exc).__name__}")


def write_generation_snapshot(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


async def select_continuous_window(cap, run_trial, report, *, start_small=False):
    """Measure natural, validated completions with continuous refill, including retries.

    The scheduler ceiling stays fixed. Admission below that ceiling limits active
    requests; a larger admission window measures whether queueing hides refill gaps.
    Compare the baseline twice so a tiny/noisy gain does not change the setting.
    """
    if start_small:
        candidates = sorted({min(cap, n) for n in (1, 2, 4, 8, 16, 32, 64)} | {cap})
        probes = report.setdefault("probes", [])
        safe = []
        async def measure(window, direction):
            emit("progress", phase="autotuning", submission_window=window, direction=direction)
            try:
                result = await run_trial(window)
            except (RuntimeError, ValueError, TimeoutError) as exc:
                row = dict(submission_window=window, direction=direction, status="failed",
                           error_type=type(exc).__name__, error=str(exc)[:1500])
                probes.append(row)
                emit("generation.probe", **row)
                return None
            row = dict(submission_window=window, direction=direction, status="succeeded", **result)
            probes.append(row)
            emit("generation.probe", **row)
            return row

        for window in candidates:
            if await measure(window, "ascending") is None:
                break
            safe.append(window)
        if not safe:
            raise RuntimeError("QA capacity search failed at batch 1; see recorded probe error")
        report["highest_verified_capacity"] = safe[-1]
        descending = []
        for window in reversed(safe):
            row = await measure(window, "descending")
            if row is not None:
                descending.append(row)
        if not descending:
            raise RuntimeError("QA capacity recovery failed at all previously verified capacities")
        metric = "accepted_tokens_per_second" if any(r.get("accepted_tokens_per_second", 0) > 0 for r in descending) else "raw_tokens_per_second"
        winner = max(descending, key=lambda r: (r[metric], -r["submission_window"]))
        selected = winner["submission_window"]
        report.update(selection="vllm_ascending_capacity_descending_throughput", selection_metric=metric,
                      selected_batch_size=selected, submission_window=selected, search_cap=cap)
        return selected

    baseline = 1 if start_small else max(128, cap)
    values = (1, 2, 4, 8, 16, 32, 64) if start_small else (8, 16, 32, 64)
    candidates = sorted({min(cap, value) for value in values} | {cap})
    candidates = [baseline, *(v for v in candidates if v != baseline), baseline]
    probes = report.setdefault("probes", [])
    for window in candidates:
        emit("progress", phase="autotuning", submission_window=window,
             completed_probes=len(probes), total_probes=len(candidates),
             sampled_at=datetime.now(UTC).isoformat())
        # Any engine failure is fatal: a dead vLLM engine cannot safely service
        # another candidate. Do not disguise it as a successful fallback.
        try:
            result = await run_trial(window)
        except TimeoutError:
            probes.append({"submission_window": window, "status": "timeout"})
            report.update(selection="vllm_probe_timeout_baseline", submission_window=baseline,
                          selected_batch_size=cap, search_cap=cap)
            return baseline
        row = {"submission_window": window, **result}
        probes.append(row)
        emit("generation.probe", **row)
    metric = "accepted_tokens_per_second"
    if not any(row[metric] > 0 for row in probes):
        metric = "raw_tokens_per_second"
    baseline_rows = [row for row in probes if row["submission_window"] == baseline]
    baseline_score = sum(row[metric] for row in baseline_rows) / len(baseline_rows)
    winner = max(probes, key=lambda row: (row[metric], -row["submission_window"]))
    selected = winner["submission_window"] if winner[metric] > baseline_score * 1.05 else baseline
    if selected != baseline:
        try:
            confirmation = {"submission_window": selected, "confirmation": True, **await run_trial(selected)}
        except TimeoutError:
            confirmation = {"submission_window": selected, "confirmation": True, "status": "timeout"}
        probes.append(confirmation)
        emit("generation.probe", **confirmation)
        if confirmation.get(metric, 0) <= baseline_score * 1.05:
            selected = baseline
    report.update(selection="vllm_measured_continuous", selection_metric=metric,
                  baseline_score=baseline_score, submission_window=selected,
                  selected_batch_size=min(cap, selected), search_cap=cap)
    return selected


def records(path, field, *, with_source=False, start=0, stop=None, expected_total=None):
    """Strict input parsing: no malformed JSON becoming fabricated source prose."""
    def unpack(value):
        text = value if isinstance(value, str) else value[field]
        source_kind = (value.get("source") or (value.get("metadata") or {}).get("source_kind")) if isinstance(value, dict) else None
        if not isinstance(source_kind, str) or source_kind not in {"wikidata5m", "conceptnet", "atomic2020"}:
            source_kind = None
        return (text, source_kind) if with_source else text

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        if expected_total is not None and parquet.metadata.num_rows != expected_total:
            raise ValueError("Frozen source record count differs from Parquet row count")
        columns = [field]
        if with_source and "source" in parquet.schema_arrow.names and field != "source":
            columns.append("source")
        offset = 0
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if offset + count <= start:
                offset += count
                continue
            if stop is not None and offset >= stop:
                break
            for batch in parquet.iter_batches(batch_size=256, columns=columns, row_groups=[group]):
                end = offset + batch.num_rows
                lo, hi = max(0, start - offset), min(batch.num_rows, stop - offset if stop is not None else batch.num_rows)
                if hi > lo:
                    yield from (unpack(row) for row in batch.slice(lo, hi - lo).to_pylist())
                offset = end
                if stop is not None and offset >= stop:
                    break
            release_file_cache(path)
    elif suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as stream:
            lines = (line for line in stream if line.strip())
            for line in islice(lines, start, stop):
                yield unpack(json.loads(line))
            if expected_total is not None and stop == expected_total and next(lines, None) is not None:
                raise ValueError("Frozen source record count is smaller than JSONL row count")
    elif suffix in {".csv", ".tsv"}:
        with path.open(encoding="utf-8", newline="") as stream:
            for row in islice(csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ","), start, stop):
                yield unpack(row)
    elif suffix in {".txt", ".md"}:
        with path.open(encoding="utf-8") as stream:
            yield from (unpack(line) for line in islice(stream, start, stop))
    else:
        raise ValueError(f"Unsupported source format: {path.name}; convert to JSONL/Parquet first")


def generation_content(chunk, mode, style, config, source_kind):
    if mode == "fact_translation":
        plan = translation_plan(chunk, source_kind)
        return {"prompt": translation_prompt(plan), "translation_plan": plan,
                "quality_anchors": factual_surface_anchors(chunk, "facts", source_kind,
                                                            proper_name_policy="source_exact")}
    anchors = (factual_surface_anchors(chunk, style, source_kind, proper_name_policy=config.proper_name_policy)
               if config.prompt_version in {"cpt-generation-v5", "cpt-generation-v6", "cpt-generation-v7", "cpt-generation-v8"}
               and mode == "knowledge" else None)
    names = anchors.get("source_proper_names", []) if anchors is not None and config.proper_name_policy == "source_exact" else None
    return {
        "prompt": prompt_for(chunk, mode, style, config.prompt_version, proper_names=names),
        **({"quality_anchors": anchors} if anchors is not None else {}),
    }


def jobs(binding, config, mode, report=None):
    if config.prompt_version in {"dense-qa-v1", "dense-qa-v2", "dense-qa-v3", "dense-dialogue-v1", "dense-relations-v1"}:
        from llm_platform_core.dense_qa import source_jobs

        yield from source_jobs(binding, config, report if report is not None else {})
        return
    if config.style_order == "style_major" or mode in {"translation", "fact_translation"}:
        yield from _style_major_jobs(binding, config, mode, report)
        return
    # A single streaming pass; keep source/style identities independent of order.
    single = config.model_copy(update={"styles": config.styles[:1]})
    for item in _style_major_jobs(binding, single, mode, report, include_text=True):
        chunk = item.pop("_source_chunk")
        for style in config.styles:
            yield {
                **item,
                "style": style,
                **generation_content(chunk, mode, style, config, item.get("source_kind")),
            }


def _style_major_jobs(binding, config, mode, report=None, *, include_text=False):
    stats = report if report is not None else {}
    stats.update(
        source_records=0, accepted_source_records=0, skipped_empty_records=0, skipped_long_records=0
    )
    root = Path(binding["runtime_path"]).resolve()
    base = root.parent if root.is_file() else root
    names = binding.get("logical_files") or ([root.name] if root.is_file() else [])
    if not names:
        raise ValueError("Input snapshot requires a frozen file manifest")
    selection = config.record_range
    if selection:
        if selection.logical_file not in names:
            raise ValueError("Record range file is not in the frozen input manifest")
        names = [selection.logical_file]
    styles = [mode] if mode in {"translation", "fact_translation"} else config.styles
    # Read the frozen source once per style so every request sharing an
    # instruction prefix stays contiguous across the complete run.  The input
    # is streamed on every pass, so this does not retain the corpus in RAM.
    for style_index, style in enumerate(styles):
        for name in names:
            path = (base / name).resolve()
            path.relative_to(base)
            # Metadata in an imported HF snapshot is not knowledge text.
            if path.suffix.lower() not in {
                ".jsonl",
                ".ndjson",
                ".parquet",
                ".csv",
                ".tsv",
                ".txt",
                ".md",
            }:
                continue
            first, stop = (selection.start, selection.stop) if selection else (0, None)
            for ordinal, (text, source_kind) in enumerate(
                records(path, config.text_field, with_source=True, start=first, stop=stop,
                        expected_total=selection.total if selection else None), start=first
            ):
                if style_index == 0:
                    stats["source_records"] += 1
                if not isinstance(text, str):
                    raise ValueError(f"Text field is not a string: {name}:{ordinal}")
                if not text.strip():
                    if style_index == 0:
                        stats["skipped_empty_records"] += 1
                    continue
                if (
                    config.max_source_characters is not None
                    and len(text) > config.max_source_characters
                ):
                    if style_index == 0:
                        stats["skipped_long_records"] += 1
                    continue
                if style_index == 0:
                    stats["accepted_source_records"] += 1
                digest = hashlib.sha256(text.encode()).hexdigest()
                width = len(text) if config.document_mode == "whole" else config.chunk_characters
                for start in range(0, len(text), width):
                    chunk = text[start : start + width]
                    yield {
                        **({"_source_chunk": chunk} if include_text else {}),
                        **({"source_kind": source_kind} if source_kind else {}),
                        **generation_content(chunk, mode, style, config, source_kind),
                        "source_file": name,
                        "source_ordinal": ordinal,
                        "source_sha256": digest,
                        "character_start": start,
                        "character_end": start + len(chunk),
                        "style": style,
                    }
            if selection and stats["source_records"] != selection.stop - selection.start:
                raise ValueError("Frozen record range exceeds the source file")


def validate_output(text, truncated=False, *, forbid_qa=False):
    if truncated:
        raise ValueError(
            "Generation reached max_new_tokens; increase output budget or reduce chunk size"
        )
    # Empty no-thinking wrappers are harmless; nonempty reasoning is never CPT text.
    text = re.sub(r"^\s*<think>\s*</think>\s*", "", text).strip()
    if "【无法可靠生成】" in text:
        raise ValueError("unresolved_source")
    if not text or re.search(r"</?think\b|</?analysis\b", text, re.I):
        raise ValueError("Empty output or unexpected thinking content")
    if not re.search(r"[\u3400-\u9fff]", text):
        raise ValueError("Expected Chinese CPT output")
    if forbid_qa and re.search(
        r"(?im)^\s*(?:[#*>-]+\s*)?(?:\*\*)?(?:问|答|问题|回答|Q|A)\s*[:：]", text
    ):
        raise ValueError("QA output is forbidden by the frozen prose-only protocol")
    return text


def generation_job_key(item):
    """Stable identity of one source span and expression in a frozen input."""
    return (
        str(item["source_file"]),
        int(item["source_ordinal"]),
        str(item["source_sha256"]),
        int(item["character_start"]),
        int(item["character_end"]),
        str(item["style"]),
    )


def generation_checkpoint_identity(task):
    payload = task["payload"]
    frozen = {
        "task_id": task.get("id"),
        "output_dataset_version_id": payload["output_dataset_version_id"],
        "generation": payload["generation"],
        "input_artifact_hash": payload["generation_input"]["artifact_hash"],
        "model_identity_hash": payload["generation_model"]["identity_hash"],
    }
    return hashlib.sha256(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _scan_checkpoint_rows(path, provenance, *, skipped=False):
    """Reject corrupt checkpoints; discard only an incomplete final JSONL line."""
    keys, rows, tokens = set(), 0, 0
    if not path.exists():
        return keys, rows, tokens
    valid_bytes = 0
    with path.open("rb+") as stream:
        advise_sequential(stream.fileno())
        while line := stream.readline():
            if not line.endswith(b"\n"):
                stream.truncate(valid_bytes)
                break
            row = json.loads(line)
            metadata = row["metadata"]
            if any(metadata.get(key) != value for key, value in provenance.items()):
                raise ValueError("Generation checkpoint provenance does not match frozen task")
            key = generation_job_key(metadata)
            if key in keys:
                raise ValueError("Generation checkpoint contains duplicate source/style rows")
            if skipped and not row.get("reason"):
                raise ValueError("Generation skip checkpoint has no reason")
            if not skipped and not isinstance(row.get("text"), str):
                raise ValueError("Generation checkpoint has no text")
            keys.add(key)
            rows += 1
            tokens += int(metadata.get("generation_tokens") or 0)
            valid_bytes = stream.tell()
        stream.flush()
        os.fsync(stream.fileno())
        release_file_cache_descriptor(stream.fileno())
    return keys, rows, tokens


def prepare_generation_checkpoint(task, root, output, provenance, report):
    """Copy a verified previous attempt into this attempt, then resume missing jobs."""
    signature = generation_checkpoint_identity(task)
    manifest = {
        "schema": "forgelm-generation-resume-v1",
        "task_id": task.get("id"),
        "output_dataset_version_id": task["payload"]["output_dataset_version_id"],
        "identity": signature,
    }
    temporary, skipped_path = output / "data.partial", output / "skipped.records"
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    if task.get("id"):
        for candidate in (candidate for resume_root in [root, *(Path(p) for p in task.get("storage_resume_roots", []))]
                          for candidate in resume_root.glob(f"{manifest['output_dataset_version_id']}-*")):
            if candidate == output or candidate.is_symlink() or not candidate.is_dir():
                continue
            previous = candidate / "resume-state.json"
            partial = candidate / "data.partial"
            if not previous.is_file() or not partial.is_file():
                continue
            if json.loads(previous.read_text(encoding="utf-8")) == manifest:
                candidates.append((partial.stat().st_mtime_ns, candidate))
    if candidates:
        from .checkpoint_copy import copy_checkpoint

        _, previous = max(candidates)
        report["resume_copy_mode"] = copy_checkpoint(previous / "data.partial", temporary)
        release_file_cache(previous / "data.partial")
        if (previous / "skipped.records").exists():
            copy_checkpoint(previous / "skipped.records", skipped_path)
            release_file_cache(previous / "skipped.records")
        report["resumed_from"] = previous.name
    completed, resumed, tokens = _scan_checkpoint_rows(temporary, provenance)
    skipped, skipped_count, _ = _scan_checkpoint_rows(skipped_path, provenance, skipped=True)
    if completed & skipped:
        raise ValueError("Generation checkpoint both completed and skipped a source/style")
    manifest_path = output / "resume-state.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    report.update(
        resumed_records=resumed,
        resumed_generated_tokens=tokens,
        resumed_skipped_records=skipped_count,
    )
    return completed | skipped, resumed, tokens, skipped_count, skipped_path


def vllm_memory_budget(requested, torch_module, system_release=None):
    """Respect configured utilization and observed free VRAM, without extra reserve."""
    release = platform.release() if system_release is None else system_release
    if "microsoft" not in release.lower():
        return requested
    free, total = torch_module.cuda.mem_get_info()
    reserve = 0
    budget = min(requested, free / total)
    if budget <= 0:
        raise RuntimeError("Insufficient free GPU memory")
    emit("generation.memory_budget", requested_fraction=requested,
         effective_fraction=budget, free_bytes=free, total_bytes=total,
         reserve_bytes=reserve, policy="wsl_available_memory_v2")
    return budget


def vllm_scheduler_limits(config):
    """Keep enough queued decode work to saturate a local throughput worker."""
    return {
        "max_num_seqs": min(config.max_batch_size, 64),
        "max_num_batched_tokens": min(max(config.max_input_tokens * 2, 2048), 8192),
        # Qwen3.5's GDN decode dispatch is dominated by eager kernel-launch
        # overhead at small/medium batches. Its vLLM CUDA graphs are required
        # for normal throughput on a single consumer GPU.
        "enforce_eager": False,
    }


def submission_window(config, selected_batch_size):
    """Keep a short ready queue without burying active decodes behind prefill."""
    if config.engine != "vllm":
        return selected_batch_size
    # On the 12 GiB Qwen3-4B-AWQ Wiki workload, 384 waiting/submitted
    # requests cut natural-output throughput by more than half versus 128.
    # The engine still has enough queued work to refill its 48/64 active slots.
    return max(selected_batch_size, 128)


class LocalEngine:
    def __init__(self, path, config):
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CPT generation requires the assigned CUDA GPU")
        self.torch, self.config = torch, config
        self.version = transformers.__version__
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        if not self.tokenizer.chat_template:
            raise ValueError("Selected model has no chat template")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        torch.manual_seed(config.seed)
        if config.engine == "vllm":
            import vllm

            self.version = vllm.__version__
            self.model = vllm.LLM(
                model=path,
                tokenizer=path,
                trust_remote_code=False,
                tensor_parallel_size=1,
                quantization=None if config.quantization == "none" else config.quantization,
                max_model_len=config.max_input_tokens + config.max_new_tokens,
                gpu_memory_utilization=vllm_memory_budget(config.gpu_memory_utilization, torch),
                **vllm_scheduler_limits(config),
                seed=config.seed,
                distributed_executor_backend="uni",
                enable_prefix_caching=config.enable_prefix_caching,
                language_model_only=True,
            )
        else:
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_utilization)
            model_config = AutoConfig.from_pretrained(
                path, local_files_only=True, trust_remote_code=False
            )
            # Qwen3.5 is a conditional multimodal model; only text is supplied here.
            loader = AutoModelForCausalLM
            if model_config.model_type in {"qwen3_5", "qwen3_5_moe"}:
                loader = transformers.AutoModelForImageTextToText
            self.model = (
                loader.from_pretrained(
                    path,
                    local_files_only=True,
                    trust_remote_code=False,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
                .to("cuda")
                .eval()
            )

    def render(self, prompts, *, return_lengths=False):
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        lengths = [len(self.tokenizer.encode(text, add_special_tokens=False)) for text in rendered]
        if max(lengths) > self.config.max_input_tokens:
            raise ValueError(
                "Input exceeds max_input_tokens; reduce chunk_characters (no silent truncation)"
            )
        return (rendered, lengths) if return_lengths else rendered

    def generate(self, prompts, *, probe=False, full_budget=False):
        cfg = self.config
        rendered, prompt_lengths = self.render(prompts, return_lengths=True)
        limit = min(cfg.probe_max_new_tokens, cfg.max_new_tokens) if probe else cfg.max_new_tokens
        if cfg.engine == "vllm":
            from vllm import SamplingParams

            limits = [limit] * len(rendered)
            if not probe and not full_budget:
                # Whole-document rewrites normally stay near the source length.
                # Bound each first pass so one failed EOS cannot hold an entire
                # continuous window at the global 4096-token ceiling.  A
                # truncated request is retried below with the full frozen budget.
                limits = [
                    min(cfg.max_new_tokens, 3072, max(512, prompt_length * 2))
                    for prompt_length in prompt_lengths
                ]
            params = [
                SamplingParams(
                    max_tokens=item_limit,
                    min_tokens=item_limit if probe else 0,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    repetition_penalty=cfg.repetition_penalty,
                    seed=cfg.seed,
                )
                for item_limit in limits
            ]
            responses = self.model.generate(
                rendered,
                params,
                use_tqdm=False,
            )
            return [
                (
                    r.outputs[0].text,
                    len(r.outputs[0].token_ids),
                    r.outputs[0].finish_reason == "length",
                )
                for r in responses
            ]
        inputs = self.tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        options = dict(
            max_new_tokens=limit,
            min_new_tokens=limit if probe else 0,
            do_sample=cfg.temperature > 0,
            repetition_penalty=cfg.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if cfg.temperature > 0:
            options.update(temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        generated = output[:, inputs.input_ids.shape[1] :].tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        result = []
        for tokens in generated:
            ended = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
            actual = tokens if ended is None else tokens[: ended + 1]
            result.append(
                (
                    self.tokenizer.decode(actual, skip_special_tokens=True),
                    len(actual),
                    ended is None and len(actual) >= limit,
                )
            )
        return result

    def cleanup_oom(self):
        gc.collect()
        self.torch.cuda.empty_cache()

    def memory(self, reset=False):
        if reset:
            self.torch.cuda.reset_peak_memory_stats()
        free, total = self.torch.cuda.mem_get_info()
        return {
            "gpu_free_bytes": free,
            "gpu_total_bytes": total,
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(),
        }


def is_oom(exc):
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def tune_batch(engine, samples, config, report):
    if config.batch_size != "auto":
        return config.batch_size
    if config.engine == "vllm":
        # The engine owns active-sequence admission and KV paging.  A synchronous
        # probe is itself a batch barrier and badly understates continuous-queue
        # throughput, so select the scheduler ceiling without generating throwaway
        # text.  The production window below remains OOM-adaptive.
        batch = vllm_scheduler_limits(config)["max_num_seqs"]
        report.update(
            selected_batch_size=batch,
            search_cap=batch,
            selection="vllm_scheduler_ceiling",
        )
        emit("generation.batch_selected", **report)
        return batch
    # Longest representative source in a bounded lookahead; later batches can downshift.
    prompt = max((item["prompt"] for item in samples), key=len)
    engine.generate([prompt], probe=True)  # warm kernels, exclude cold load from timing
    candidates = []
    batch, low, high = 1, 0, None
    while True:
        if hasattr(engine, "memory"):
            engine.memory(reset=True)
        started = time.perf_counter()
        try:
            values = engine.generate([prompt] * batch, probe=True)
            elapsed = time.perf_counter() - started
            candidate = {
                "batch_size": batch,
                "status": "ok",
                "tokens_per_second": sum(v[1] for v in values) / max(elapsed, 1e-9),
            }
            candidates.append(candidate)
            low = batch
        except RuntimeError as exc:
            if not is_oom(exc):
                raise
            candidate = {"batch_size": batch, "status": "oom"}
            high = batch
        if candidate["status"] == "oom":
            engine.cleanup_oom()
        if hasattr(engine, "memory"):
            candidate.update(engine.memory())
        report["probes"].append(candidate)
        emit("generation.batch_probe", **candidate)
        if high is not None:
            if high - low <= 1:
                break
            batch = (low + high) // 2
        elif batch >= config.max_batch_size:
            break
        else:
            batch = min(config.max_batch_size, batch * 2)
    if not candidates:
        raise RuntimeError("No batch fits GPU memory")
    selected = max(candidates, key=lambda row: (row["tokens_per_second"], -row["batch_size"]))
    report.update(
        selected_batch_size=selected["batch_size"],
        largest_tested_safe_batch=max(c["batch_size"] for c in candidates),
        probe_decode_tokens=min(config.probe_max_new_tokens, config.max_new_tokens),
        search_cap=config.max_batch_size,
        selection="measured_tokens_per_second",
    )
    emit("generation.batch_selected", **report)
    return selected["batch_size"]


def eligible_generation_jobs(source, provenance, report, skip_stream, completed_keys=None, *, config=None):
    """Journal incompatible whole cards before any model request, including probes."""
    report.setdefault("input_policy", GENERATION_INPUT_POLICY)
    selection = config.fact_input_selection if config else "standard"
    report.setdefault("fact_input_selection", selection)
    report.setdefault("preflight_skipped_records", 0)
    report.setdefault("preflight_skip_reasons", {})
    report.setdefault("skipped_generation_records", int(report.get("resumed_skipped_records") or 0))
    diagnostics = GenerationDiagnostics(emit)
    last_flush = time.perf_counter()
    for item in source:
        if completed_keys is not None and generation_job_key(item) in completed_keys:
            continue
        anchors = item.get("quality_anchors") or {}
        spelling = anchors.get("requires_latin_spelling")
        opaque = anchors.get("opaque_label_facts") if selection == "chinese_ready" or "translation_plan" in item else []
        taxon = anchors.get("untranslated_taxon_facts") if selection == "chinese_ready" else []
        translation = anchors.get("latin_translation_facts")
        entities = anchors.get("untranslated_entity_facts") if selection == "chinese_ready" else []
        associations = (selection == "chinese_ready" and anchors.get("associations_only")
                        and any(re.search(r"[A-Za-z]", label) for label in anchors.get("title_candidates", [])))
        lexical = selection == "chinese_ready" and anchors.get("foreign_lexical_only")
        if not (anchors.get("chinese_only") and (spelling or opaque or taxon or translation or entities or associations or lexical)):
            yield item
            continue
        if skip_stream is None:
            raise RuntimeError("Input rejection requires a durable skip journal")
        key = generation_job_key(item)
        reason = ("input_latin_spelling" if spelling else "input_opaque_label" if opaque
                  else "input_untranslated_taxon" if taxon else "input_latin_translation" if translation
                  else "input_untranslated_entity" if entities else "input_foreign_association_only" if associations
                  else "input_foreign_lexical_only")
        matches = (anchors.get("latin_spelling_facts", []) if spelling
                   else opaque or taxon or translation or entities or anchors.get("title_candidates", []))
        skip_stream.write(json.dumps({
            "metadata": {**provenance, **{k: v for k, v in item.items() if k not in {"prompt", "quality_anchors", "translation_plan"}},
                         "generation_input_policy": GENERATION_INPUT_POLICY, "generation_tokens": 0,
                         "fact_input_selection": selection},
            "reason": reason, "stage": "before_generation", "model_requests": 0,
            # Full facts remain in the immutable source; the journal needs the
            # resume key and bounded examples, not another copy of all anchors.
            "matched_facts": matches[:8], "matched_fact_count": len(matches),
        }, ensure_ascii=False) + "\n")
        if completed_keys is not None:
            completed_keys.add(key)
        report["preflight_skipped_records"] += 1
        report["preflight_skip_reasons"][reason] = report["preflight_skip_reasons"].get(reason, 0) + 1
        report["skipped_generation_records"] += 1
        now = time.perf_counter()
        if report["preflight_skipped_records"] % 64 == 0 or now - last_flush >= 5:
            skip_stream.flush()
            os.fsync(skip_stream.fileno())
            release_file_cache_descriptor(skip_stream.fileno())
            last_flush = now
        diagnostics.emit("generation.input_skipped", reason=reason,
                         source_ordinal=item["source_ordinal"], style=item["style"],
                         input_policy=GENERATION_INPUT_POLICY,
                         preflight_skipped_records=report["preflight_skipped_records"])


def generate_records(engine, source, config, stream, provenance, report, skip_stream=None):
    report.setdefault("length_retries", 0)
    source = eligible_generation_jobs(source, provenance, report, skip_stream, config=config)
    pending = list(islice(source, config.max_batch_size))
    if not pending:
        if report["preflight_skipped_records"]:
            report.update(output_records=0, generated_tokens=0, selection="no_eligible_input")
            return
        raise ValueError("No nonempty supported text records in input snapshot")
    batch_size = tune_batch(engine, pending, config, report)
    queue_size = submission_window(config, batch_size)
    report["submission_window"] = queue_size
    pending.extend(islice(source, queue_size - len(pending)))
    style_order = {style: index for index, style in enumerate(config.styles)}
    total = tokens = 0
    started = time.perf_counter()
    while pending:
        batch = pending[:queue_size]
        if config.engine == "vllm" and len(style_order) > 1 and config.style_order == "style_major":
            # Keep each style's identical instruction prefix adjacent. This is
            # useful to cache implementations with finer hybrid-model prefix
            # checkpoints and costs nothing when prefix caching is unavailable.
            batch = sorted(batch, key=lambda item: style_order.get(item.get("style"), 0))
        try:
            values = engine.generate([item["prompt"] for item in batch])
        except RuntimeError as exc:
            if not is_oom(exc) or len(batch) == 1:
                raise
            engine.cleanup_oom()
            queue_size = max(1, (len(batch) + 1) // 2)
            report["oom_backoffs"] += 1
            emit(
                "generation.batch_backoff",
                batch_size=batch_size,
                submission_window=queue_size,
                completed_records=total,
            )
            continue
        if len(values) != len(batch):
            raise ValueError("Engine returned a different number of records")
        truncated = [index for index, value in enumerate(values) if value[2]]
        if truncated and config.engine == "vllm":
            retry_values = engine.generate(
                [batch[index]["prompt"] for index in truncated], full_budget=True
            )
            if len(retry_values) != len(truncated):
                raise ValueError("Engine returned a different number of retry records")
            values = list(values)
            for index, value in zip(truncated, retry_values, strict=True):
                values[index] = value
            report["length_retries"] += len(truncated)
            emit(
                "generation.length_retry",
                retry_records=len(truncated),
                completed_records=total,
            )
        accepted, skipped = [], []
        for item, value in zip(batch, values, strict=True):
            if "【无法可靠生成】" in value[0]:
                skipped.append((item, "unresolved_source"))
                continue
            if value[2] and config.output_length_limit_policy == "skip":
                skipped.append((item, "output_length_limit"))
                continue
            text = validate_output(
                value[0], value[2], forbid_qa=config.prompt_version != "cpt-generation-v1"
            )
            if "quality_anchors" in item:
                text = validate_factual_surface(
                    text, item["quality_anchors"]
                )
            accepted.append((item, text, value[1]))
        for item, text, generated_tokens in accepted:
            metadata = {
                **provenance,
                **{k: v for k, v in item.items() if k not in {"prompt", "translation_plan"}},
                "generation_tokens": generated_tokens,
                "generation_input_policy": GENERATION_INPUT_POLICY,
            }
            stream.write(
                json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False) + "\n"
            )
        for item, reason in skipped:
            metadata = {**provenance, **{k: v for k, v in item.items() if k not in {"prompt", "translation_plan"}}}
            if skip_stream is not None:
                skip_stream.write(
                    json.dumps(
                        {"metadata": metadata, "reason": reason},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            emit(
                "generation.record_skipped",
                source_ordinal=item.get("source_ordinal"),
                style=item.get("style"),
                reason=reason,
            )
        report["skipped_generation_records"] = report.get("skipped_generation_records", 0) + len(skipped)
        total += len(accepted)
        tokens += sum(value[2] for value in accepted)
        del pending[: len(batch)]
        pending.extend(islice(source, queue_size - len(pending)))
        emit(
            "progress",
            phase="generating",
            document_count=total,
            generated_tokens=tokens,
            tokens_per_second=tokens / max(time.perf_counter() - started, 1e-9),
            batch_size=batch_size,
            submission_window=queue_size,
        )
    report.update(
        output_records=total,
        generated_tokens=tokens,
        final_batch_size=batch_size if config.engine == "vllm" else queue_size,
        final_submission_window=queue_size,
        elapsed_seconds=time.perf_counter() - started,
    )


async def wait_generation_requests(requests):
    """Observe every finished sibling before processing any result that may raise."""
    done, pending = await asyncio.wait(requests, return_when=asyncio.FIRST_COMPLETED)
    for completed in done:
        if not completed.cancelled():
            completed.exception()
    return done, pending


def shutdown_generation_engine(engine, torch=None):
    # A dead CUDA context can fail again during teardown. Preserve the request
    # failure and do not issue another CUDA synchronization after that failure.
    failing = sys.exc_info()[0] is not None
    try:
        engine.shutdown()
    except Exception as exc:
        if not failing:
            raise
        emit("generation.cleanup_failed", error_type=type(exc).__name__, error=str(exc)[:500])
    if torch is not None and not failing:
        torch.cuda.synchronize()


async def generate_records_vllm_async(
    model_path, source, config, stream, provenance, report, completed_keys=None, skip_stream=None,
    *, telemetry_dir=None,
):
    """Continuously refill vLLM as individual whole-document requests finish."""
    completed_keys = completed_keys if completed_keys is not None else set()
    source = eligible_generation_jobs(source, provenance, report, skip_stream, completed_keys, config=config)
    first = list(islice(source, 1))
    if not first:
        report.update(output_records=int(report.get("resumed_records") or 0),
                      generated_tokens=int(report.get("resumed_generated_tokens") or 0),
                      raw_completed_requests=0, raw_generated_tokens=0, selection="no_eligible_input")
        return
    source = iter(chain(first, source))
    emit("progress", phase="loading_model", sampled_at=datetime.now(UTC).isoformat())
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise ValueError("Selected model has no chat template")
    use_language_guard = config.chinese_output_constraint
    runtime_policy = GENERATION_RUNTIME_POLICY if use_language_guard else NATURAL_RUNTIME_POLICY
    limits = vllm_scheduler_limits(config)
    qa_mode = config.prompt_version in {"dense-qa-v1", "dense-qa-v2", "dense-qa-v3", "dense-dialogue-v1", "dense-relations-v1"}
    qa_auto = config.batch_size == "auto" and qa_mode
    if qa_mode and config.max_batch_size <= 8:
        limits.update(max_num_seqs=1, max_num_batched_tokens=256, enforce_eager=True)
    if config.batch_size != "auto":
        limits["max_num_seqs"] = config.batch_size
    elif qa_auto:
        limits.update(max_num_seqs=1, max_num_batched_tokens=512)
    def create_engine(capacity):
        return AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model_path,
                tokenizer=model_path,
                trust_remote_code=False,
                tensor_parallel_size=1,
                quantization=None if config.quantization == "none" else config.quantization,
                max_model_len=config.max_input_tokens + config.max_new_tokens,
                gpu_memory_utilization=vllm_memory_budget(config.gpu_memory_utilization, torch),
                max_num_seqs=capacity,
                max_num_batched_tokens=limits["max_num_batched_tokens"],
                enforce_eager=limits["enforce_eager"],
                seed=config.seed,
                distributed_executor_backend="uni",
                enable_prefix_caching=config.enable_prefix_caching,
                language_model_only=True,
                disable_log_stats=False,
            )
        )
    engine = create_engine(limits["max_num_seqs"])

    async def prepare_qa_capacity(capacity):
        nonlocal engine
        if engine is not None and capacity == limits["max_num_seqs"]:
            return
        emit("progress", phase="qa_engine_resizing", capacity=capacity)
        if engine is not None:
            try:
                shutdown_generation_engine(engine, torch)
            except Exception as exc:
                emit("generation.cleanup_failed", error_type=type(exc).__name__, error=str(exc)[:500])
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
        limits["max_num_seqs"] = 0
        engine = create_engine(capacity)
        limits["max_num_seqs"] = capacity
        report["scheduler_max_num_seqs"] = capacity

    async def reset_qa_cache():
        return await engine.reset_prefix_cache()

    async def select_qa_window(cap, trial, result):
        return await select_continuous_window(cap, trial, result, start_small=True)

    report.update(
        engine_version=vllm.__version__,
        runtime_policy=runtime_policy,
        runtime_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fact_rules_implementation_sha256=hashlib.sha256(Path(generation_contract.__file__).read_bytes()).hexdigest(),
        selected_batch_size=limits["max_num_seqs"],
        prefill_token_budget=limits["max_num_batched_tokens"],
        search_cap=limits["max_num_seqs"],
        selection="manual" if config.batch_size != "auto" else "pending_autotune",
        submission_window=(config.batch_size if config.batch_size != "auto"
                           else submission_window(config, limits["max_num_seqs"])),
        prefix_caching=config.enable_prefix_caching,
        validation_retry_limit=config.validation_retry_limit,
        proper_name_policy=config.proper_name_policy,
        **({"translation_protocol": config.translation_protocol,
            "translation_implementation_sha256": hashlib.sha256(
                Path(sys.modules[finish_translation.__module__].__file__).read_bytes()).hexdigest()}
           if isinstance(config, FactTranslationCreate) else {}),
    )
    request_number = 0

    async def one(item, *, full_budget=False, validation_retry=0, guard=use_language_guard,
                  previous_text="", validation_error=""):
        nonlocal request_number
        plan = item.get("translation_plan")
        if plan is not None and not plan["slots"]:
            return item, ("{}", 0, False), full_budget, validation_retry
        rendered, prompt_length = render_generation_request(
            tokenizer, item, config, previous_text, validation_error)
        output_limit = config.max_new_tokens
        if not full_budget:
            output_limit = min(output_limit, 3072, max(512, prompt_length * 2))
        constraint = chinese_sampling_constraint(item, guard)
        if plan is not None:
            from vllm.sampling_params import StructuredOutputsParams

            constraint = {"structured_outputs": StructuredOutputsParams(json=translation_schema(plan))}
        if "qa_json_schema" in item:
            from vllm.sampling_params import StructuredOutputsParams

            constraint = {"structured_outputs": StructuredOutputsParams(json=item["qa_json_schema"])}
        params = SamplingParams(
            max_tokens=output_limit,
            temperature=min(config.temperature, 0.1) if validation_retry else config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=(
                min(1.2, config.repetition_penalty + 0.1)
                if validation_retry
                else config.repetition_penalty
            ),
            seed=config.seed + validation_retry,
            output_kind=RequestOutputKind.FINAL_ONLY,
            **constraint,
        )
        request_number += 1
        final = None
        async for output in engine.generate(
            rendered, params, f"wiki-rewrite-{request_number}"
        ):
            final = output
        if final is None:
            raise RuntimeError("vLLM request ended without a final output")
        value = final.outputs[0]
        return item, (
            value.text,
            len(value.token_ids),
            value.finish_reason == "length",
        ), full_budget, validation_retry

    if config.prompt_version == "dense-dialogue-v1":
        from .dense_dialogue_runtime import generate_dialogues

        try:
            await generate_dialogues(source, config, one, stream, provenance, report, emit, telemetry_dir)
        finally:
            shutdown_generation_engine(engine, torch)
        return
    if config.prompt_version == "dense-relations-v1":
        from .dense_relations_runtime import generate_relations

        try:
            await generate_relations(source, config, one, stream, provenance, report, emit, telemetry_dir)
        finally:
            shutdown_generation_engine(engine, torch)
        return
    if config.prompt_version in {"dense-qa-v1", "dense-qa-v2", "dense-qa-v3"}:
        from .dense_qa_runtime import generate_qa

        try:
            await generate_qa(source, config, one, stream, provenance, report, emit, telemetry_dir,
                              select_window=select_qa_window,
                              prepare_capacity=prepare_qa_capacity if qa_auto else None,
                              reset_cache=reset_qa_cache if config.enable_prefix_caching else None)
        finally:
            shutdown_generation_engine(engine, torch)
        return

    def validate(item, value):
        if "translation_plan" in item:
            if value[2]:
                raise ValueError("fact_translation_output_length_limit")
            return finish_translation(value[0], item["translation_plan"])
        text = validate_output(value[0], value[2], forbid_qa=config.prompt_version != "cpt-generation-v1")
        if "quality_anchors" in item:
            text = validate_factual_surface(text, item["quality_anchors"])
        return text

    async def probe_one(item, *, guard=use_language_guard, audit=False):
        _, value, full_budget, validation_retry = await one(item, guard=guard)
        raw_tokens, requests = value[1], int(not ("translation_plan" in item and not item["translation_plan"]["slots"]))
        initial_text = value[0]
        initial_error = ""
        try:
            validate(item, value)
        except ValueError as exc:
            initial_error = str(exc)
        if value[2]:
            _, value, full_budget, validation_retry = await one(item, full_budget=True, guard=guard)
            raw_tokens += value[1]
            requests += 1
        try:
            validate(item, value)
        except ValueError as exc:
            if not value[2] and str(exc) != "unresolved_source" and config.validation_retry_limit:
                _, value, _, _ = await one(item, full_budget=True, validation_retry=1, guard=guard,
                                          previous_text=value[0], validation_error=str(exc))
                raw_tokens += value[1]
                requests += 1
        final_error = ""
        try:
            validate(item, value)
        except ValueError as exc:
            final_error = str(exc)
        result = dict(raw_tokens=raw_tokens, accepted_tokens=0 if final_error else value[1],
                      requests=requests, first_pass_accepted=int(not initial_error), accepted_records=int(not final_error))
        if audit and telemetry_dir is not None:
            with (telemetry_dir / "generation-validation.jsonl").open("a", encoding="utf-8") as audit_stream:
                audit_stream.write(json.dumps({
                    "policy": GENERATION_RUNTIME_POLICY if guard else NATURAL_RUNTIME_POLICY,
                    **item, "initial_text": initial_text, "initial_error": initial_error,
                    "final_text": value[0], "final_error": final_error, **result,
                }, ensure_ascii=False) + "\n")
        return result

    sample_limit = max(16, min(128, 131072 // config.max_new_tokens))
    probe_items = list(islice(source, sample_limit)) if config.batch_size == "auto" else []
    source = iter(chain(probe_items, source))
    if config.batch_size == "auto" and len(probe_items) < 16:
        report["selection"] = "insufficient_probe_input"
    compare_language_guard = use_language_guard and any((item.get("quality_anchors") or {}).get("chinese_only") for item in probe_items)

    async def trial(window, *, guard=use_language_guard, audit=False):
        # Flush cached full prompts before every candidate for comparable prefill.
        if config.enable_prefix_caching:
            if not await engine.reset_prefix_cache():
                raise RuntimeError("vLLM could not reset idle prefix cache for autotuning")
        iterator = iter(probe_items)
        pending = set()
        raw_tokens = accepted_tokens = requests = accepted_records = 0
        first_pass_accepted = 0
        audit = audit or (guard and compare_language_guard and "guarded" not in report.get("language_guard_comparison", {}))
        capture_sample = audit or not report.get("validation_sample_saved", False)
        began = time.perf_counter()
        try:
            async with asyncio.timeout(240):
                while True:
                    while len(pending) < window:
                        item = next(iterator, None)
                        if item is None:
                            break
                        pending.add(asyncio.create_task(probe_one(item, guard=guard, audit=capture_sample)))
                    if not pending:
                        break
                    done, pending = await wait_generation_requests(pending)
                    for completed in done:
                        row = completed.result()
                        raw_tokens += row["raw_tokens"]
                        accepted_tokens += row["accepted_tokens"]
                        requests += row["requests"]
                        first_pass_accepted += row["first_pass_accepted"]
                        accepted_records += row["accepted_records"]
        finally:
            for pending_task in pending:
                pending_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        elapsed = time.perf_counter() - began
        result = dict(seconds=elapsed, input_records=len(probe_items), requests=requests,
                      raw_tokens=raw_tokens, accepted_tokens=accepted_tokens,
                      first_pass_accepted=first_pass_accepted,
                      accepted_records=accepted_records, raw_tokens_per_second=raw_tokens / elapsed,
                      accepted_tokens_per_second=accepted_tokens / elapsed)
        if audit:
            report.setdefault("language_guard_comparison", {})["guarded" if guard else "legacy"] = result
        if capture_sample:
            report["validation_sample_saved"] = True
            report["validation_sample_records"] = len(probe_items)
        if telemetry_dir is not None:
            write_generation_snapshot(telemetry_dir / "generation-runtime.json",
                                      {**report, "current_probe": {"submission_window": window, **result}})
        return result

    try:
        if len(probe_items) >= 16:
            # Exclude lazy engine setup/compilation from the timed comparisons.
            await probe_one(probe_items[0])
            if compare_language_guard:
                emit("progress", phase="autotuning", submission_window=128, completed_probes=0,
                     total_probes=1, comparison="language_guard", sampled_at=datetime.now(UTC).isoformat())
                try:
                    await trial(128, guard=False, audit=True)
                except TimeoutError:
                    report["language_guard_comparison"] = {"legacy": {"status": "timeout"}}
            await select_continuous_window(limits["max_num_seqs"], trial, report)
        emit("generation.batch_selected", **report)
        if telemetry_dir is not None:
            write_generation_snapshot(telemetry_dir / "generation-runtime.json", report)
    except BaseException:
        shutdown_generation_engine(engine)
        raise

    current_style = None
    deferred = None

    def next_same_style():
        nonlocal current_style, deferred
        if deferred is not None:
            item, deferred = deferred, None
        else:
            try:
                item = next(source)
            except StopIteration:
                return None
        item_style = item.get("style")
        if current_style is None:
            current_style = item_style
        if item_style != current_style and config.style_order == "style_major":
            deferred = item
            return None
        return item

    active = set()
    queue_limit = report["submission_window"]
    total = int(report.get("resumed_records") or 0)
    tokens = int(report.get("resumed_generated_tokens") or 0)
    fresh_tokens = 0
    raw_tokens = raw_requests = retry_tokens = rejected_tokens = 0
    first_pass_requests = first_pass_accepted = 0
    last_reported_total = total
    report.setdefault("validation_retries", 0)
    started = last_progress = time.perf_counter()
    rate_window = GenerationRateWindow(started, records=total)
    raw_rate_window = GenerationRateWindow(started)
    diagnostics = GenerationDiagnostics(emit)

    def progress(now):
        nonlocal last_progress, last_reported_total
        stream.flush()
        os.fsync(stream.fileno())
        release_file_cache_descriptor(stream.fileno())
        if skip_stream is not None:
            skip_stream.flush()
            os.fsync(skip_stream.fileno())
            release_file_cache_descriptor(skip_stream.fileno())
        raw_rate = raw_rate_window.update(now, raw_tokens, raw_requests)
        values = dict(
            phase="generating", document_count=total, generated_tokens=tokens,
            accepted_tokens_this_attempt=fresh_tokens,
            tokens_per_second=fresh_tokens / max(now - started, 1e-9),
            raw_generated_tokens=raw_tokens, raw_completed_requests=raw_requests,
            retry_generated_tokens=retry_tokens, rejected_generated_tokens=rejected_tokens,
            validation_retries=report["validation_retries"], length_retries=report["length_retries"],
            diagnostic_counts=dict(diagnostics.counts),
            runtime_policy=runtime_policy,
            input_policy=GENERATION_INPUT_POLICY,
            preflight_skipped_records=report["preflight_skipped_records"],
            preflight_skip_reasons=dict(report["preflight_skip_reasons"]),
            first_pass_requests=first_pass_requests, first_pass_accepted=first_pass_accepted,
            first_pass_acceptance_rate=first_pass_accepted / max(first_pass_requests, 1),
            raw_tokens_per_second=raw_tokens / max(now - started, 1e-9),
            recent_raw_tokens_per_second=raw_rate["recent_tokens_per_second"],
            sampled_at=datetime.now(UTC).isoformat(),
            **rate_window.update(now, fresh_tokens, total),
            resumed_records=report.get("resumed_records", 0), skipped_records=report["skipped_generation_records"],
            batch_size=report["selected_batch_size"], scheduler_max_num_seqs=limits["max_num_seqs"],
            submission_window=queue_limit,
        )
        if telemetry_dir is not None:
            write_generation_snapshot(telemetry_dir / "generation-progress.json", values)
            with (telemetry_dir / "generation-metrics.jsonl").open("a", encoding="utf-8") as history:
                history.write(json.dumps(values, ensure_ascii=False) + "\n")
        emit("progress", **values)
        report.update(raw_generated_tokens=raw_tokens, raw_completed_requests=raw_requests,
                      retry_generated_tokens=retry_tokens, rejected_generated_tokens=rejected_tokens)
        last_progress, last_reported_total = now, total

    stats_task = asyncio.create_task(log_vllm_stats(engine))
    try:
        while True:
            while len(active) < queue_limit:
                item = next_same_style()
                if item is None:
                    break
                active.add(asyncio.create_task(one(item)))
            if not active:
                if deferred is None:
                    break
                current_style = None
                continue
            done, active = await wait_generation_requests(active)
            for completed in done:
                item, value, full_budget, validation_retry = completed.result()
                raw_tokens += value[1]
                raw_requests += int(not ("translation_plan" in item and not item["translation_plan"]["slots"]))
                if full_budget or validation_retry:
                    retry_tokens += value[1]
                else:
                    first_pass_requests += 1
                if value[2] and not full_budget:
                    rejected_tokens += value[1]
                    report["length_retries"] += 1
                    diagnostics.emit(
                        "generation.length_retry",
                        retry_records=1,
                        completed_records=total,
                    )
                    active.add(asyncio.create_task(one(item, full_budget=True)))
                    continue
                try:
                    text = validate(item, value)
                except ValueError as exc:
                    rejected_tokens += value[1]
                    if str(exc) == "unresolved_source":
                        # An explicit abstention is terminal, not an invitation to guess on retry.
                        validation_retry = 1
                    if value[2] and full_budget and config.output_length_limit_policy == "skip":
                        if skip_stream is None:
                            raise RuntimeError("Skip policy requires a durable skip journal") from exc
                        skip_stream.write(
                            json.dumps(
                                {
                                    "metadata": {
                                        **provenance,
                                        **{key: val for key, val in item.items() if key not in {"prompt", "translation_plan"}},
                                    },
                                    "reason": "output_length_limit",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        report["skipped_generation_records"] += 1
                        completed_keys.add(generation_job_key(item))
                        diagnostics.emit(
                            "generation.record_skipped",
                            source_ordinal=item["source_ordinal"],
                            style=item["style"],
                            reason="output_length_limit",
                            skipped_records=report["skipped_generation_records"],
                        )
                        continue
                    if value[2] and full_budget:
                        raise
                    if validation_retry < config.validation_retry_limit:
                        report["validation_retries"] += 1
                        diagnostics.emit(
                            "generation.validation_retry",
                            source_ordinal=item["source_ordinal"],
                            style=item["style"],
                            reason=str(exc),
                        )
                        active.add(
                            asyncio.create_task(one(item, full_budget=True, validation_retry=1,
                                                    previous_text=value[0], validation_error=str(exc)))
                        )
                        continue
                    if "quality_anchors" in item and not value[2]:
                        if skip_stream is None:
                            raise RuntimeError("Quality rejection requires a durable skip journal") from exc
                        skip_stream.write(json.dumps({
                            "metadata": {**provenance, **{k: v for k, v in item.items() if k not in {"prompt", "translation_plan"}}},
                            "reason": "fact_surface_validation", "detail": str(exc),
                            "rejected_text": value[0],
                        }, ensure_ascii=False) + "\n")
                        report["skipped_generation_records"] += 1
                        report["quality_rejected_records"] = report.get("quality_rejected_records", 0) + 1
                        completed_keys.add(generation_job_key(item))
                        diagnostics.emit("generation.record_skipped", source_ordinal=item["source_ordinal"],
                             style=item["style"], reason="fact_surface_validation", detail=str(exc),
                             skipped_records=report["skipped_generation_records"])
                        continue
                    raise
                metadata = {
                    **provenance,
                    **{key: val for key, val in item.items() if key not in {"prompt", "translation_plan"}},
                    "generation_tokens": value[1],
                    "generation_runtime_policy": runtime_policy,
                    "generation_input_policy": GENERATION_INPUT_POLICY,
                    "fact_input_selection": config.fact_input_selection,
                    "validation_retry_limit": config.validation_retry_limit,
                    "proper_name_policy": config.proper_name_policy,
                    "generation_attempt_id": telemetry_dir.name if telemetry_dir is not None else None,
                    **({"translation_protocol": config.translation_protocol,
                        "source_fact_count": len(item["translation_plan"]["facts"]),
                        "translation_model_requests": int(bool(item["translation_plan"]["slots"]))}
                       if "translation_plan" in item else {}),
                }
                stream.write(
                    json.dumps({"text": text, "metadata": metadata}, ensure_ascii=False)
                    + "\n"
                )
                total += 1
                if not full_budget and not validation_retry:
                    first_pass_accepted += 1
                tokens += value[1]
                fresh_tokens += value[1]
                completed_keys.add(generation_job_key(item))
            now = time.perf_counter()
            if total - last_reported_total >= 64 or now - last_progress >= 5:
                progress(now)
        progress(time.perf_counter())
    finally:
        stats_task.cancel()
        await asyncio.gather(stats_task, return_exceptions=True)
        for pending in active:
            pending.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        shutdown_generation_engine(engine, torch)
    report.update(
        output_records=total,
        generated_tokens=tokens,
        final_batch_size=report["selected_batch_size"],
        final_submission_window=queue_limit,
        elapsed_seconds=time.perf_counter() - started,
    )


def run(task):
    payload, context = task["payload"], task["runtime_context"]
    frozen = dict(payload["generation"])
    mode, prompt_version = frozen.pop("mode"), frozen.get("prompt_version")
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError("Unsupported frozen prompt version")
    config = (FactTranslationCreate if mode == "fact_translation" else DataGenerationCreate).model_validate(frozen)
    attempt = Path(context["attempt_dir"]).resolve()
    root = scoped_runtime_path(
        project_data_asset_root(context["project_workspace"]), "data-generation"
    )
    root.mkdir(parents=True, exist_ok=True)
    # Each attempt owns its directory; never overwrite an earlier snapshot.
    output = scoped_runtime_path(root, f"{payload['output_dataset_version_id']}-{attempt.name}")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "data.jsonl"
    temporary = output / "data.partial"
    report = {
        "probes": [],
        "oom_backoffs": 0,
        "length_retries": 0,
        "engine": config.engine,
        "enable_thinking": False,
    }
    model_path = payload["generation_model"]["runtime_path"]
    provenance = {
        "input_dataset_version_id": config.input_dataset_version_id,
        "input_artifact_hash": payload["generation_input"]["artifact_hash"],
        "model_weight_version_id": config.model_weight_version_id,
        "model_identity_hash": payload["generation_model"]["identity_hash"],
        "prompt_version": prompt_version,
        "mode": mode,
        "enable_thinking": False,
    }
    try:
        if config.engine == "vllm":
            from .generation_continuation import import_checkpoint_range, remaining_verified_jobs

            completed_keys, _, _, _, skipped_path = prepare_generation_checkpoint(
                task, root, output, provenance, report
            )
            imported = import_checkpoint_range(task, root, output, provenance, report, completed_keys)
            source = remaining_verified_jobs(
                jobs(payload["generation_input"], config, mode, report), completed_keys, imported
            )
            with (
                temporary.open("a", encoding="utf-8", newline="\n") as stream,
                skipped_path.open("a", encoding="utf-8", newline="\n") as skip_stream,
            ):
                asyncio.run(
                    generate_records_vllm_async(
                        model_path,
                        iter(source),
                        config,
                        stream,
                        provenance,
                        report,
                        completed_keys,
                        skip_stream,
                        telemetry_dir=attempt,
                    )
                )
                stream.flush()
                os.fsync(stream.fileno())
                skip_stream.flush()
                os.fsync(skip_stream.fileno())
        else:
            with (
                temporary.open("w", encoding="utf-8", newline="\n") as stream,
                (output / "skipped.records").open("w", encoding="utf-8", newline="\n") as skip_stream,
            ):
                source = iter(jobs(payload["generation_input"], config, mode, report))
                engine = LocalEngine(model_path, config)
                report["engine_version"] = engine.version
                generate_records(
                    engine,
                    source,
                    config,
                    stream,
                    provenance,
                    report,
                    skip_stream,
                )
                del engine
        if config.prompt_version == "dense-dialogue-v1":
            from .dense_dialogue_runtime import finalize_dialogue_batches

            finalize_dialogue_batches(temporary, destination, report)
        elif config.prompt_version == "dense-relations-v1":
            from .dense_relations_runtime import finalize_relation_batches

            finalize_relation_batches(temporary, destination, report)
        elif config.prompt_version in {"dense-qa-v1", "dense-qa-v2", "dense-qa-v3"}:
            from .dense_qa_runtime import finalize_qa_batches

            finalize_qa_batches(temporary, destination, report)
        else:
            temporary.replace(destination)
    finally:
        (attempt / "generation-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    skipped_journal = output / "skipped.records"
    if skipped_journal.exists():
        skipped_journal.replace(attempt / "generation-skipped-records.jsonl")
    (output / "resume-state.json").unlink(missing_ok=True)
    gc.collect()
    manifest = attempt / "dataset-records.jsonl.gz"
    analysis = DataImportAdapter._analyze(
        output,
        "generated",
        config.input_dataset_version_id,
        payload["output_dataset_version_id"],
        manifest,
        {},
    )
    files = analysis["source_snapshot"]["files"]
    analysis["source_snapshot"].update(
        generation={**payload["generation"], **provenance},
        storage_type="external",
        target_dir=str(output),
        generation_empty=analysis["document_count"] == 0,
    )
    emit(
        "result",
        **analysis,
        generation_summary=report,
        outputs=[
            {
                "ref": "dataset",
                "kind": "dataset_snapshot",
                "storage_type": "external",
                "uri": output.as_uri(),
                "content_hash": DataImportAdapter._manifest_hash(files),
                "size_bytes": analysis["raw_size_bytes"],
                "media_type": "application/vnd.llm-platform.dataset-directory",
                "availability_state": "available",
                "worker_id": context["worker_id"],
                "metadata": provenance,
            },
            {
                "ref": "record_manifest",
                "kind": "dataset_record_manifest",
                "path": str(manifest),
                "media_type": "application/gzip",
            },
            {
                "ref": "generation_report",
                "kind": "data_generation_report",
                "path": str(attempt / "generation-report.json"),
                "media_type": "application/json",
            },
        ],
    )


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
