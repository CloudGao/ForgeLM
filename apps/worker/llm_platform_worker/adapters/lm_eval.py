from __future__ import annotations

import asyncio
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time
from collections import defaultdict, deque
from importlib import metadata
from pathlib import Path
from typing import Any

from .command import Emit, GetDesiredState

_PROCESS_EXIT_STREAM_DRAIN_TIMEOUT_SECONDS = 2.0
_DETECTED_BATCH_SIZE_PATTERN = re.compile(
    r"determined\s+largest\s+batch\s+size\s*:\s*(\d+)", re.IGNORECASE
)
_CUDA_OOM_PATTERNS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cudaerrormemoryallocation",
    "torch.outofmemoryerror",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
    "hip out of memory",
)
MICROLM_NATIVE_BACKEND = "microlm-native"
_NATIVE_PROGRESS_EMIT_TIMEOUT_SECONDS = 10.0


def builtin_lm_eval_task_path() -> Path:
    return Path(__file__).resolve().parent.parent / "lm_eval_tasks"


class LmEvalProcessError(RuntimeError):
    def __init__(self, exit_code: int, output_tail: list[str]) -> None:
        super().__init__(f"lm-eval exited with {exit_code}")
        self.exit_code = exit_code
        self.output_tail = output_tail

    @property
    def is_device_oom(self) -> bool:
        output = "\n".join(self.output_tail).lower()
        return any(pattern in output for pattern in _CUDA_OOM_PATTERNS)

    @property
    def detected_batch_sizes(self) -> list[int]:
        return [
            int(match.group(1))
            for message in self.output_tail
            if (match := _DETECTED_BATCH_SIZE_PATTERN.search(message)) is not None
        ]


class LmEvalControlled(RuntimeError):
    def __init__(self, state: str) -> None:
        super().__init__(f"lm-eval task was {state}")
        self.state = state


def lm_eval_available() -> bool:
    return importlib.util.find_spec("lm_eval") is not None


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def stable_sample_indices(
    task_name: str, total: int, percentage: float, seed: int
) -> list[int]:
    if total <= 0:
        return []
    selected = min(total, max(1, math.ceil(total * percentage / 100)))
    ranked = sorted(
        range(total),
        key=lambda index: hashlib.sha256(
            f"stable_hash_v1\0{seed}\0{task_name}\0{index}".encode()
        ).digest(),
    )
    return sorted(ranked[:selected])


def build_sampling_plan(
    task_id: str, sampling: dict[str, Any]
) -> tuple[dict[str, list[int]] | None, dict[str, Any]]:
    from lm_eval.tasks import TaskManager

    if sampling.get("method") != "stable_hash_v1":
        raise ValueError("unsupported evaluation sampling method")
    percentage = float(sampling.get("percentage"))
    seed = int(sampling.get("seed"))
    loaded = TaskManager(include_path=builtin_lm_eval_task_path()).load([task_id])
    tasks = loaded.get("tasks") or {}
    if not tasks:
        raise ValueError(f"lm-eval task {task_id} did not resolve to any leaf tasks")
    samples: dict[str, list[int]] = {}
    task_counts: dict[str, dict[str, int]] = {}
    for leaf_name, leaf_task in sorted(tasks.items()):
        total = len(leaf_task.eval_docs)
        indices = stable_sample_indices(leaf_name, total, percentage, seed)
        samples[leaf_name] = indices
        task_counts[leaf_name] = {"total": total, "selected": len(indices)}
    plan_samples = None if percentage == 100 else samples
    identity_payload = {
        "sampling": sampling,
        "task_id": task_id,
        "tasks": task_counts,
        "samples": samples if plan_samples is not None else "all",
    }
    return plan_samples, {
        "plan_hash": canonical_hash(identity_payload),
        "leaf_task_count": len(task_counts),
        "total_documents": sum(item["total"] for item in task_counts.values()),
        "selected_documents": sum(item["selected"] for item in task_counts.values()),
        "tasks": task_counts,
    }


class LmEvalAdapter:
    """Run curated lm-evaluation-harness tasks without accepting an arbitrary command."""

    @staticmethod
    def framework_version() -> str:
        for package_name in ("lm_eval", "lm-eval"):
            try:
                return metadata.version(package_name)
            except metadata.PackageNotFoundError:
                continue
        return "unknown"

    @staticmethod
    def _is_api_backend(backend: str) -> bool:
        return backend in {"local-completions", "local-chat-completions"}

    @classmethod
    def batch_candidates(cls, requested: Any, backend: str) -> list[str | int]:
        if requested is None or (isinstance(requested, str) and not requested.strip()):
            requested = "auto"
        if backend == "local-chat-completions":
            return [1]
        if isinstance(requested, str) and requested.strip().lower() == "auto":
            if backend == MICROLM_NATIVE_BACKEND:
                # HFLM's auto detector runs five full forwards for every probe and
                # can spend minutes at 0% on this hybrid architecture. Start from
                # a bounded, known-safe throughput batch and reuse the adapter's
                # existing OOM backoff instead.
                return [16, 8, 4, 2, 1]
            return [1] if cls._is_api_backend(backend) else ["auto", 1]
        if isinstance(requested, bool):
            raise ValueError("batch_size must be auto or a positive integer")
        try:
            batch_size = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError("batch_size must be auto or a positive integer") from exc
        if batch_size < 1:
            raise ValueError("batch_size must be auto or a positive integer")
        if cls._is_api_backend(backend):
            return [batch_size]
        candidates: list[str | int] = []
        while batch_size >= 1:
            candidates.append(batch_size)
            if batch_size == 1:
                break
            batch_size = max(1, batch_size // 2)
        return candidates

    @staticmethod
    def _detected_batch_sizes(messages: list[str]) -> list[int]:
        return [
            int(match.group(1))
            for message in messages
            if (match := _DETECTED_BATCH_SIZE_PATTERN.search(message)) is not None
        ]

    @staticmethod
    def _release_native_harnesses(harnesses: dict[str | int, Any]) -> None:
        harnesses.clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except Exception:
            # Cleanup is best-effort and must not hide the evaluation result or
            # the original execution error on CPU-only/misconfigured Workers.
            return

    @classmethod
    def _auto_oom_fallbacks(
        cls, error: LmEvalProcessError, backend: str
    ) -> list[int]:
        detected = error.detected_batch_sizes
        if not detected:
            return [1]
        start = max(1, min(detected) // 2)
        return [
            int(candidate)
            for candidate in cls.batch_candidates(start, backend)
            if isinstance(candidate, int)
        ]

    @staticmethod
    def build_args(
        runtime: dict[str, Any],
        benchmark: dict[str, Any],
        inference_config: dict[str, Any],
        scoring_config: dict[str, Any],
        output_dir: Path,
    ) -> list[str]:
        backend = str(runtime.get("backend") or runtime.get("model") or "hf")
        model_args = str(runtime.get("model_args") or "").strip()
        task_id = str(benchmark.get("task_id") or "").strip()
        if not model_args:
            raise ValueError("lm-eval model_args is required")
        if not task_id:
            raise ValueError("benchmark task_id is required")
        if "trust_remote_code" in model_args.lower():
            raise ValueError("trust_remote_code is not supported by the built-in adapter")

        args = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            backend,
            "--model_args",
            model_args,
            "--tasks",
            task_id,
            "--include_path",
            str(builtin_lm_eval_task_path()),
            "--output_path",
            str(output_dir),
            "--batch_size",
            str(inference_config.get("batch_size") or "auto"),
        ]
        device = inference_config.get("device")
        if device and backend not in {"local-completions", "local-chat-completions"}:
            args.extend(["--device", str(device)])
        limit = inference_config.get("limit")
        sample_plan_path = inference_config.get("_sample_plan_path")
        if sample_plan_path:
            try:
                cli_samples = json.loads(Path(sample_plan_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("lm-eval sample plan file is not readable JSON") from exc
            if not isinstance(cli_samples, dict) or not cli_samples or any(
                not isinstance(indices, list)
                or not indices
                or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
                for indices in cli_samples.values()
            ):
                raise ValueError(
                    "lm-eval CLI sample plan must map task names directly to non-empty integer index lists"
                )
            args.extend(["--samples", str(sample_plan_path)])
        elif limit not in (None, "", 0, "0"):
            args.extend(["--limit", str(limit)])
        seed = inference_config.get("seed")
        if seed not in (None, ""):
            args.extend(["--seed", str(seed)])
        use_cache = inference_config.get("use_cache")
        if use_cache:
            args.extend(["--use_cache", str(use_cache)])
        cache_requests = inference_config.get("cache_requests")
        if cache_requests:
            args.extend(["--cache_requests", str(cache_requests)])
        if inference_config.get("check_integrity"):
            args.append("--check_integrity")

        execution = benchmark.get("execution") or {}
        num_fewshot = execution.get("num_fewshot")
        if num_fewshot is not None:
            args.extend(["--num_fewshot", str(num_fewshot)])
        system_instruction = execution.get("system_instruction") or inference_config.get(
            "system_instruction"
        )
        if system_instruction:
            args.extend(["--system_instruction", str(system_instruction)])
        apply_chat_template = execution.get(
            "apply_chat_template", inference_config.get("apply_chat_template")
        )
        if apply_chat_template:
            args.append("--apply_chat_template")
            if isinstance(apply_chat_template, str) and apply_chat_template.lower() != "true":
                args.append(apply_chat_template)
        if execution.get(
            "fewshot_as_multiturn", inference_config.get("fewshot_as_multiturn", False)
        ):
            args.append("--fewshot_as_multiturn")
        gen_kwargs = inference_config.get("gen_kwargs")
        if gen_kwargs:
            args.extend(["--gen_kwargs", str(gen_kwargs)])
        if scoring_config.get("save_samples", True):
            args.append("--log_samples")
        return args

    async def _run_process(
        self,
        args: list[str],
        emit: Emit,
        get_desired_state: GetDesiredState | None,
        env_overrides: dict[str, str] | None = None,
    ) -> list[int]:
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(__import__("subprocess"), "CREATE_NEW_PROCESS_GROUP", 0)
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
            env={**os.environ, **(env_overrides or {})},
        )
        assert process.stdout is not None and process.stderr is not None
        lines: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()

        async def pump(stream: asyncio.StreamReader, name: str) -> None:
            while line := await stream.readline():
                await lines.put((name, line))
            await lines.put((name, None))

        pumps = [
            asyncio.create_task(pump(process.stdout, "stdout")),
            asyncio.create_task(pump(process.stderr, "stderr")),
        ]
        next_line = asyncio.create_task(lines.get())
        finished = 0
        exit_observed_at: float | None = None
        close_lingering_streams = False
        output_tail: deque[str] = deque(maxlen=240)
        detected_batch_sizes: list[int] = []
        try:
            while finished < 2:
                loop = asyncio.get_running_loop()
                if process.returncode is not None and exit_observed_at is None:
                    exit_observed_at = loop.time()
                timeout = 1.0
                if exit_observed_at is not None:
                    remaining = (
                        _PROCESS_EXIT_STREAM_DRAIN_TIMEOUT_SECONDS
                        - (loop.time() - exit_observed_at)
                    )
                    if remaining <= 0 and not next_line.done():
                        await emit(
                            "log",
                            {
                                "stream": "platform",
                                "level": "warning",
                                "message": (
                                    "lm-eval 已退出，但输出管道未及时关闭；"
                                    "已按进程退出状态继续收尾"
                                ),
                            },
                        )
                        close_lingering_streams = True
                        break
                    timeout = min(timeout, max(0.0, remaining))
                done, _pending = await asyncio.wait(
                    {next_line},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if next_line not in done:
                    if process.returncode is not None and exit_observed_at is None:
                        exit_observed_at = loop.time()
                    if exit_observed_at is not None:
                        continue
                    desired = get_desired_state() if get_desired_state else "running"
                    if desired in {"paused", "cancelled"}:
                        process.terminate()
                        await process.wait()
                        raise RuntimeError(f"lm-eval task was {desired}") from None
                    continue
                stream_name, line = next_line.result()
                next_line = asyncio.create_task(lines.get())
                if line is None:
                    finished += 1
                    continue
                message = line.decode("utf-8", errors="replace").rstrip()
                if message:
                    output_tail.append(message)
                    match = _DETECTED_BATCH_SIZE_PATTERN.search(message)
                    if match is not None:
                        detected_batch_sizes.append(int(match.group(1)))
                    desired = await emit("log", {"stream": stream_name, "message": message})
                    if desired in {"paused", "cancelled"}:
                        process.terminate()
                        await process.wait()
                        raise RuntimeError(f"lm-eval task was {desired}")
        finally:
            for pump_task in pumps:
                if not pump_task.done():
                    pump_task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            if not next_line.done():
                next_line.cancel()
                await asyncio.gather(next_line, return_exceptions=True)
        if close_lingering_streams:
            for stream in (process.stdout, process.stderr):
                transport = getattr(stream, "_transport", None)
                if transport is not None:
                    transport.close()
            await process.wait()
        exit_code = process.returncode
        if exit_code is None:
            exit_code = await process.wait()
        if exit_code != 0:
            raise LmEvalProcessError(exit_code, list(output_tail))
        return detected_batch_sizes

    @staticmethod
    def _result_document(output_dir: Path) -> tuple[Path, dict[str, Any]]:
        candidates = sorted(
            output_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True
        )
        for path in candidates:
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(document, dict) and (
                isinstance(document.get("results"), dict)
                or isinstance(document.get("groups"), dict)
            ):
                return path, document
        raise RuntimeError(f"lm-eval did not produce a result JSON under {output_dir}")

    @staticmethod
    def _metric_value(metrics: dict[str, Any], primary_metric: str) -> float | None:
        candidates: list[tuple[str, float]] = []
        for raw_name, raw_value in metrics.items():
            if not isinstance(raw_value, int | float) or isinstance(raw_value, bool):
                continue
            name = str(raw_name).split(",", 1)[0]
            if "stderr" in name or name in {"alias", "samples"}:
                continue
            value = float(raw_value)
            if name == primary_metric:
                return value
            candidates.append((name, value))
        for fallback in ("acc_norm", "acc", "exact_match", "pass@1", "f1"):
            match = next((value for name, value in candidates if name == fallback), None)
            if match is not None:
                return match
        return candidates[0][1] if candidates else None

    @classmethod
    def _task_score(
        cls, document: dict[str, Any], task_id: str, primary_metric: str
    ) -> tuple[float, dict[str, Any]]:
        groups = document.get("groups") or {}
        results = document.get("results") or {}
        direct = groups.get(task_id) or results.get(task_id)
        if isinstance(direct, dict):
            score = cls._metric_value(direct, primary_metric)
            if score is not None:
                return score, direct
        values: list[float] = []
        for metrics in results.values():
            if not isinstance(metrics, dict):
                continue
            value = cls._metric_value(metrics, primary_metric)
            if value is not None:
                values.append(value)
        if not values:
            raise RuntimeError(
                f"lm-eval result for {task_id} does not contain metric {primary_metric}"
            )
        return sum(values) / len(values), {"aggregated_subtasks": len(values)}

    @staticmethod
    def _sample_rows(
        output_dir: Path,
        *,
        model_id: str,
        benchmark: dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        primary_metric = str(benchmark.get("primary_metric") or "acc")
        dimension = str((benchmark.get("dimensions") or ["标准评测"])[0])
        for path in sorted(output_dir.rglob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for index, line in enumerate(lines):
                if len(samples) >= limit:
                    return samples
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else raw
                value = LmEvalAdapter._metric_value(metrics, primary_metric)
                status = "observed" if value is None else "passed" if value >= 1 else "failed"
                samples.append(
                    {
                        "id": str(raw.get("doc_id") or raw.get("id") or f"{path.stem}-{index}"),
                        "model_version_id": model_id,
                        "benchmark_id": benchmark.get("id"),
                        "task_id": benchmark.get("task_id"),
                        "dimension": dimension,
                        "status": status,
                        "error_type": None if status == "passed" else f"{primary_metric}_miss",
                        "metric": primary_metric,
                        "score": value,
                    }
                )
        return samples

    @staticmethod
    def _persist_canonical_samples(successful_output_dir: Path, output_dir: Path) -> Path | None:
        """Copy one successful upstream sample log to the stable report contract."""
        candidates = [
            path
            for path in successful_output_dir.rglob("*.jsonl")
            if path.name == "samples.jsonl" or path.name.startswith("samples_")
        ]
        if not candidates:
            return None
        source = max(
            candidates,
            key=lambda path: (
                path.name == "samples.jsonl",
                path.stat().st_mtime,
                path.as_posix(),
            ),
        )
        destination = output_dir / "samples.jsonl"
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return destination

    @classmethod
    def normalize_results(
        cls,
        task: dict[str, Any],
        documents: list[tuple[str, dict[str, Any], dict[str, Any], Path]],
        framework_version: str,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        benchmark_config = payload.get("benchmark_config") or {}
        scoring_config = payload.get("scoring_config") or {}
        max_samples = max(0, min(1000, int(scoring_config.get("max_sample_results", 200))))
        models: dict[str, dict[str, Any]] = {}
        all_samples: list[dict[str, Any]] = []
        task_versions: dict[str, Any] = {}
        task_config_hashes: dict[str, str] = {}
        by_model: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = defaultdict(list)
        for model_id, benchmark, document, output_dir in documents:
            by_model[model_id].append((benchmark, document, output_dir))
            for task_name, version in (document.get("versions") or {}).items():
                task_versions[str(task_name)] = version
            configs = document.get("configs") or {}
            if configs:
                task_config_hashes[str(benchmark["task_id"])] = canonical_hash(configs)

        for model_id, items in by_model.items():
            weighted_total = 0.0
            total_weight = 0.0
            included_task_count = 0
            diagnostic_task_count = 0
            dimensions: dict[str, list[tuple[float, float]]] = defaultdict(list)
            tasks: dict[str, Any] = {}
            for benchmark, document, output_dir in items:
                primary_metric = str(benchmark.get("primary_metric") or "acc")
                score, raw_metrics = cls._task_score(
                    document, str(benchmark["task_id"]), primary_metric
                )
                weight = float(benchmark.get("weight", 1) or 1)
                include_in_overall = benchmark.get("include_in_overall", True) is not False
                if include_in_overall:
                    included_task_count += 1
                    weighted_total += score * weight
                    total_weight += weight
                    for dimension in benchmark.get("dimensions") or ["标准评测"]:
                        dimensions[str(dimension)].append((score, weight))
                else:
                    diagnostic_task_count += 1
                tasks[str(benchmark["task_id"])] = {
                    "benchmark_id": benchmark.get("id"),
                    "score": score,
                    "metric": primary_metric,
                    "weight": weight,
                    "higher_is_better": benchmark.get("higher_is_better", True),
                    "include_in_overall": include_in_overall,
                    "raw_metrics": raw_metrics,
                }
                remaining = max_samples - len(all_samples)
                if remaining > 0:
                    all_samples.extend(
                        cls._sample_rows(
                            output_dir,
                            model_id=model_id,
                            benchmark=benchmark,
                            limit=remaining,
                        )
                    )
            model_dimensions = {
                name: sum(score * weight for score, weight in values)
                / sum(weight for _, weight in values)
                for name, values in dimensions.items()
            }
            models[model_id] = {
                "overall": weighted_total / total_weight if total_weight else 0,
                "dimensions": model_dimensions,
                "tasks": tasks,
                "included_task_count": included_task_count,
                "diagnostic_task_count": diagnostic_task_count,
                "total_weight": total_weight,
            }

        primary_model_id = str(payload.get("primary_model_version_id") or "")
        if primary_model_id not in models:
            primary_model_id = next(iter(models), "")
        primary_model = models.get(primary_model_id) or {
            "overall": 0,
            "dimensions": {},
        }
        comparisons = {
            model_id: {
                "overall_delta": primary_model["overall"] - model["overall"],
                "dimensions": {
                    name: primary_model["dimensions"][name] - value
                    for name, value in model["dimensions"].items()
                    if name in primary_model["dimensions"]
                },
            }
            for model_id, model in models.items()
            if model_id != primary_model_id
        }
        summary = {
            "primary_model_version_id": primary_model_id or None,
            "overall": primary_model["overall"],
            "dimensions": primary_model["dimensions"],
            "comparisons": comparisons,
            "models": models,
            "aggregation": {
                "method": "weighted_mean",
                "unit": "benchmark_task",
                "formula": "sum(task_score * task_weight) / sum(task_weight)",
                "included_when": "include_in_overall != false",
            },
        }
        signature_payload = {
            "framework": "lm-eval",
            "framework_version": framework_version,
            "catalog_config_hash": benchmark_config.get("comparison_config_hash"),
            "task_versions": task_versions,
            "task_config_hashes": task_config_hashes,
        }
        signature = {
            **signature_payload,
            "compatibility_hash": canonical_hash(signature_payload),
            "status": "complete",
        }
        return {
            "summary": summary,
            "samples": all_samples,
            "benchmark_signature": signature,
        }

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        if not lm_eval_available():
            raise RuntimeError(
                'lm-eval is not installed on this Worker; install a backend such as "lm_eval[hf]"'
            )
        payload = task.get("payload") or {}
        benchmark_config = payload.get("benchmark_config") or {}
        benchmarks = benchmark_config.get("tasks") or []
        inference_config = payload.get("inference_config") or {}
        scoring_config = payload.get("scoring_config") or {}
        runtimes = inference_config.get("models") or {}
        if not benchmarks:
            raise ValueError("benchmark_config.tasks is required")
        if not runtimes:
            raise ValueError("inference_config.models is required")
        runtime_context = task.get("runtime_context") or {}
        run_root = Path(
            runtime_context.get("run_dir") or os.environ.get("LLM_PLATFORM_RUN_DIR", ".")
        ).resolve()
        report_root = run_root / "lm-eval-report"
        report_root.mkdir(parents=True, exist_ok=True)
        total = len(runtimes) * len(benchmarks)
        completed = 0
        documents: list[tuple[str, dict[str, Any], dict[str, Any], Path]] = []
        requested_batch_size = inference_config.get("batch_size") or "auto"
        execution_results: dict[str, dict[str, Any]] = {}
        sampling_config = inference_config.get("sampling")
        sampling_plans: dict[str, tuple[dict[str, list[int]] | None, dict[str, Any], Path]] = {}
        sampling_execution: dict[str, Any] = {}
        native_harnesses: dict[str | int, Any] = {}
        harness: Any = None
        for model_id, runtime in runtimes.items():
            if native_harnesses:
                harness = None
                await asyncio.to_thread(
                    self._release_native_harnesses, native_harnesses
                )
            if not isinstance(runtime, dict):
                raise ValueError(f"invalid runtime for model {model_id}")
            runtime_backend = str(runtime.get("backend") or runtime.get("model") or "hf")
            batch_candidates = self.batch_candidates(requested_batch_size, runtime_backend)
            model_execution: dict[str, Any] = {}
            execution_results[str(model_id)] = model_execution
            for benchmark in benchmarks:
                benchmark_task_id = str(benchmark["task_id"])
                if sampling_config and benchmark_task_id not in sampling_plans:
                    await emit(
                        "progress",
                        {
                            "progress": completed / total,
                            "phase": f"lm-eval:sampling:{benchmark_task_id}",
                            "model_version_id": model_id,
                        },
                    )
                    if float(sampling_config.get("percentage", 100)) < 100:
                        sample_plan, sample_metadata = await asyncio.to_thread(
                            build_sampling_plan, benchmark_task_id, sampling_config
                        )
                    else:
                        sample_plan = None
                        sample_metadata = {
                            "plan_hash": canonical_hash(
                                {
                                    "sampling": sampling_config,
                                    "task_id": benchmark_task_id,
                                    "samples": "all",
                                }
                            ),
                            "selection": "all",
                        }
                    plan_dir = report_root / "sampling-plans"
                    plan_dir.mkdir(parents=True, exist_ok=True)
                    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", benchmark_task_id)
                    plan_path = plan_dir / f"{safe_task_id}.json"
                    plan_path.write_text(
                        json.dumps(
                            {
                                "sampling": sampling_config,
                                "task_id": benchmark_task_id,
                                "metadata": sample_metadata,
                                "samples": sample_plan,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    cli_plan_path = Path()
                    if sample_plan:
                        cli_plan_path = plan_dir / f"{safe_task_id}.samples.json"
                        cli_plan_path.write_text(
                            json.dumps(sample_plan, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    sampling_plans[benchmark_task_id] = (
                        sample_plan,
                        sample_metadata,
                        cli_plan_path,
                    )
                    sampling_execution[benchmark_task_id] = sample_metadata
                    await emit(
                        "log",
                        {
                            "stream": "platform",
                            "level": "info",
                            "message": (
                                f"评测抽样计划已冻结：{benchmark_task_id} · "
                                f"{sampling_config['percentage']}% · Seed "
                                f"{sampling_config['seed']} · "
                                f"{sample_metadata['plan_hash'][:12]}"
                            ),
                            "phase": f"lm-eval:sampling:{benchmark_task_id}",
                        },
                    )
                sample_plan, _sample_metadata, sample_plan_path = sampling_plans.get(
                    benchmark_task_id, (None, {}, Path())
                )
                output_dir = report_root / str(model_id) / str(benchmark["task_id"])
                output_dir.mkdir(parents=True, exist_ok=True)
                await emit(
                    "progress",
                    {
                        "progress": completed / total,
                        "phase": f"lm-eval:{benchmark['task_id']}",
                        "model_version_id": model_id,
                    },
                )
                env_overrides: dict[str, str] = {}
                auth_env_var = str(runtime.get("auth_env_var") or "").strip()
                if auth_env_var:
                    auth_value = os.environ.get(auth_env_var)
                    if not auth_value:
                        raise RuntimeError(
                            f"endpoint credential environment variable {auth_env_var} is not set"
                        )
                    env_overrides["OPENAI_API_KEY"] = auth_value
                attempt_records: list[dict[str, Any]] = []
                successful_output_dir: Path | None = None
                detected_batch_sizes: list[int] = []
                candidate_queue = list(batch_candidates)
                batch_attempt = 0
                while batch_attempt < len(candidate_queue):
                    candidate = candidate_queue[batch_attempt]
                    batch_attempt += 1
                    candidate_label = str(candidate).replace(":", "-")
                    attempt_output_dir = (
                        output_dir / f"attempt-{batch_attempt}-batch-{candidate_label}"
                    )
                    attempt_output_dir.mkdir(parents=True, exist_ok=True)
                    effective_inference_config = {
                        **inference_config,
                        "batch_size": candidate,
                        "_sample_plan": sample_plan,
                        "_sample_plan_path": (
                            str(sample_plan_path) if sampling_config and sample_plan else None
                        ),
                    }
                    try:
                        if runtime_backend == MICROLM_NATIVE_BACKEND:
                            from .microlm_lm_eval import evaluate_task, load_harness

                            native_batch_size = candidate
                            harness = native_harnesses.get(native_batch_size)
                            if harness is None:
                                harness = await asyncio.to_thread(
                                    load_harness,
                                    str(runtime.get("model_args") or ""),
                                    batch_size=native_batch_size,
                                    device=str(effective_inference_config.get("device") or "cuda"),
                                    dtype_name=str(
                                        effective_inference_config.get("dtype") or "auto"
                                    ),
                                )
                                native_harnesses[native_batch_size] = harness
                            loop = asyncio.get_running_loop()
                            last_progress_emit_at = 0.0
                            last_emitted_progress = completed / total

                            def report_native_progress(
                                requests_completed: int,
                                requests_total: int,
                                operation: str,
                                *,
                                completed_before: int = completed,
                                benchmark_id: str = str(benchmark["task_id"]),
                                current_model_id: str = str(model_id),
                                event_loop: asyncio.AbstractEventLoop = loop,
                                batch_size_for_progress: int = (
                                    native_batch_size
                                    if isinstance(native_batch_size, int)
                                    else 1
                                ),
                            ) -> None:
                                nonlocal last_progress_emit_at, last_emitted_progress
                                desired = (
                                    get_desired_state() if get_desired_state else "running"
                                )
                                if desired in {"paused", "cancelled"}:
                                    raise LmEvalControlled(desired)
                                if requests_total <= 0:
                                    return
                                # HFLM updates its bar once per request after a
                                # whole GPU batch returns. Emit the completed
                                # batch, not its first request, so early progress
                                # and ETA are not understated by the batch size.
                                if (
                                    requests_completed < requests_total
                                    and requests_completed % batch_size_for_progress != 0
                                ):
                                    return
                                benchmark_progress = max(
                                    0.0,
                                    min(1.0, requests_completed / requests_total),
                                )
                                overall_progress = (
                                    completed_before + benchmark_progress
                                ) / total
                                now = time.monotonic()
                                if (
                                    requests_completed < requests_total
                                    and now - last_progress_emit_at < 2.0
                                    and overall_progress - last_emitted_progress < 0.001
                                ):
                                    return
                                future = asyncio.run_coroutine_threadsafe(
                                    emit(
                                        "progress",
                                        {
                                            "progress": overall_progress,
                                            "phase": f"lm-eval:{benchmark_id}",
                                            "model_version_id": current_model_id,
                                            "evaluation_operation": operation,
                                            "requests_completed": requests_completed,
                                            "requests_total": requests_total,
                                        },
                                    ),
                                    event_loop,
                                )
                                try:
                                    desired = future.result(
                                        timeout=_NATIVE_PROGRESS_EMIT_TIMEOUT_SECONDS
                                    )
                                except TimeoutError:
                                    # Progress is non-critical telemetry. A slow
                                    # control-plane request must not abort an
                                    # otherwise healthy GPU evaluation. Desired
                                    # state is also checked from the heartbeat-
                                    # synchronized callback at every batch boundary.
                                    future.cancel()
                                    desired = (
                                        get_desired_state()
                                        if get_desired_state
                                        else "running"
                                    )
                                last_progress_emit_at = now
                                last_emitted_progress = overall_progress
                                if desired in {"paused", "cancelled"}:
                                    raise LmEvalControlled(desired)

                            try:
                                detected_batch_sizes = await asyncio.to_thread(
                                    evaluate_task,
                                    harness,
                                    benchmark,
                                    effective_inference_config,
                                    scoring_config,
                                    attempt_output_dir,
                                    progress_callback=report_native_progress,
                                )
                            finally:
                                # The cache dictionary owns the reusable Harness;
                                # do not retain a second function-local reference
                                # when changing model/backend.
                                harness = None
                            detected_batch_sizes = detected_batch_sizes or (
                                [native_batch_size]
                                if isinstance(native_batch_size, int)
                                else []
                            )
                        else:
                            args = self.build_args(
                                runtime,
                                benchmark,
                                effective_inference_config,
                                scoring_config,
                                attempt_output_dir,
                            )
                            detected_batch_sizes = (
                                await self._run_process(
                                    args,
                                    emit,
                                    get_desired_state,
                                    env_overrides=env_overrides,
                                )
                                or []
                            )
                    except LmEvalControlled as exc:
                        harness = None
                        await asyncio.to_thread(
                            self._release_native_harnesses, native_harnesses
                        )
                        return {"controlled_state": exc.state}
                    except Exception as exc:
                        is_oom = (
                            exc.is_device_oom
                            if isinstance(exc, LmEvalProcessError)
                            else any(
                                pattern in str(exc).lower() for pattern in _CUDA_OOM_PATTERNS
                            )
                        )
                        failed_detected_batch_sizes = (
                            exc.detected_batch_sizes
                            if isinstance(exc, LmEvalProcessError)
                            else []
                        )
                        attempt_records.append(
                            {
                                "batch_size": candidate,
                                "status": "oom" if is_oom else "failed",
                                "exit_code": exc.exit_code
                                if isinstance(exc, LmEvalProcessError)
                                else None,
                                **(
                                    {"detected_batch_sizes": failed_detected_batch_sizes}
                                    if failed_detected_batch_sizes
                                    else {}
                                ),
                            }
                        )
                        if is_oom and runtime_backend == MICROLM_NATIVE_BACKEND:
                            harness = None
                            await asyncio.to_thread(
                                self._release_native_harnesses, native_harnesses
                            )
                        if (
                            is_oom
                            and candidate == "auto"
                            and isinstance(exc, LmEvalProcessError)
                        ):
                            fallbacks = self._auto_oom_fallbacks(exc, runtime_backend)
                            remaining = candidate_queue[batch_attempt:]
                            candidate_queue[batch_attempt:] = list(
                                dict.fromkeys([*fallbacks, *remaining])
                            )
                        has_fallback = batch_attempt < len(candidate_queue)
                        if not is_oom or not has_fallback:
                            harness = None
                            await asyncio.to_thread(
                                self._release_native_harnesses, native_harnesses
                            )
                            raise
                        next_candidate = candidate_queue[batch_attempt]
                        await emit(
                            "log",
                            {
                                "stream": "platform",
                                "level": "warning",
                                "message": (
                                    f"lm-eval batch_size={candidate} 发生 CUDA OOM；"
                                    f"自动降档为 {next_candidate} 后重试"
                                ),
                            },
                        )
                        continue
                    attempt_records.append(
                        {
                            "batch_size": candidate,
                            "status": "succeeded",
                            "exit_code": 0,
                        }
                    )
                    successful_output_dir = attempt_output_dir
                    break
                if successful_output_dir is None:
                    raise RuntimeError("lm-eval exhausted all batch-size candidates")
                if scoring_config.get("save_samples", True):
                    self._persist_canonical_samples(successful_output_dir, output_dir)
                _path, document = self._result_document(successful_output_dir)
                documents.append(
                    (str(model_id), benchmark, document, successful_output_dir)
                )
                successful_candidate = attempt_records[-1]["batch_size"]
                actual_batch_sizes = detected_batch_sizes or (
                    [successful_candidate] if isinstance(successful_candidate, int) else []
                )
                model_execution[str(benchmark["task_id"])] = {
                    "backend": runtime_backend,
                    "requested_batch_size": requested_batch_size,
                    "effective_batch_size": (
                        actual_batch_sizes[0]
                        if len(set(actual_batch_sizes)) == 1
                        else None
                    ),
                    "effective_batch_sizes": sorted(set(actual_batch_sizes)),
                    "detection_source": (
                        "microlm_bounded_oom_backoff"
                        if runtime_backend == MICROLM_NATIVE_BACKEND
                        and requested_batch_size == "auto"
                        else "microlm_native_fixed"
                        if runtime_backend == MICROLM_NATIVE_BACKEND
                        else "lm_eval_oom_backoff"
                        if any(
                            attempt["status"] == "oom" for attempt in attempt_records
                        )
                        else
                        "api_fixed_single_request"
                        if self._is_api_backend(runtime_backend)
                        and requested_batch_size == "auto"
                        else "lm_eval_runtime_probe"
                        if detected_batch_sizes
                        else "manual"
                        if requested_batch_size != "auto"
                        else "fallback_single_request"
                    ),
                    "oom_backoffs": sum(
                        1 for attempt in attempt_records if attempt["status"] == "oom"
                    ),
                    "attempts": attempt_records,
                }
                completed += 1
                await emit(
                    "progress",
                    {
                        "progress": completed / total,
                        "phase": f"lm-eval:{benchmark['task_id']}",
                        "model_version_id": model_id,
                    },
                )
        harness = None
        await asyncio.to_thread(self._release_native_harnesses, native_harnesses)
        normalized = self.normalize_results(task, documents, self.framework_version())
        normalized["summary"]["execution"] = {
            "batch_policy": "auto" if requested_batch_size == "auto" else "manual",
            "requested_batch_size": requested_batch_size,
            "sampling": (
                {"config": sampling_config, "benchmarks": sampling_execution}
                if sampling_config
                else None
            ),
            "models": execution_results,
        }
        return {
            **normalized,
            "outputs": [{"kind": "evaluation_report", "path": str(report_root)}],
        }
