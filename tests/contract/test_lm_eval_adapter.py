from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from llm_platform_worker.adapters.lm_eval import (
    LmEvalAdapter,
    LmEvalProcessError,
    builtin_lm_eval_task_path,
    stable_sample_indices,
)
from llm_platform_worker.adapters.microlm_lm_eval import _progress_tqdm


def benchmark(task_id: str = "hellaswag") -> dict:
    return {
        "id": f"lm-eval:{task_id}",
        "task_id": task_id,
        "primary_metric": "acc_norm",
        "dimensions": ["常识", "推理"],
        "weight": 1,
        "execution": {"num_fewshot": 3},
        "config_hash": "catalog-hash",
    }


def test_lm_eval_adapter_builds_a_structured_cli_without_remote_code(tmp_path: Path) -> None:
    cli_plan_path = tmp_path / "sampling-plan.json"
    cli_plan_path.write_text(json.dumps({"hellaswag": [1, 4]}), encoding="utf-8")
    args = LmEvalAdapter.build_args(
        {"backend": "hf", "model_args": "pretrained=local/model,dtype=float32"},
        benchmark(),
        {
            "batch_size": "auto",
            "device": "cpu",
            "limit": 12,
            "apply_chat_template": True,
            "seed": "0,1234,1234,1234",
        },
        {"save_samples": True},
        tmp_path,
    )

    assert args[1:3] == ["-m", "lm_eval"]
    assert args[args.index("--model") + 1] == "hf"
    assert args[args.index("--tasks") + 1] == "hellaswag"
    assert args[args.index("--include_path") + 1] == str(builtin_lm_eval_task_path())
    assert args[args.index("--num_fewshot") + 1] == "3"
    assert "--apply_chat_template" in args
    assert "--log_samples" in args
    assert all("shell" not in value for value in args)

    sampled_args = LmEvalAdapter.build_args(
        {"backend": "hf", "model_args": "pretrained=local/model"},
        benchmark(),
        {
            "batch_size": 4,
            "limit": 99,
            "_sample_plan_path": cli_plan_path,
        },
        {},
        tmp_path,
    )
    assert sampled_args[sampled_args.index("--samples") + 1].endswith(
        "sampling-plan.json"
    )
    assert "--limit" not in sampled_args

    audit_wrapper_path = tmp_path / "audit-wrapper.json"
    audit_wrapper_path.write_text(
        json.dumps({"sampling": {"percentage": 20}, "samples": {"hellaswag": [1, 4]}}),
        encoding="utf-8",
    )
    try:
        LmEvalAdapter.build_args(
            {"backend": "hf", "model_args": "pretrained=local/model"},
            benchmark(),
            {"batch_size": 4, "_sample_plan_path": audit_wrapper_path},
            {},
            tmp_path,
        )
    except ValueError as exc:
        assert "map task names directly" in str(exc)
    else:
        raise AssertionError("audit wrapper must not be accepted as lm-eval CLI samples")

    endpoint_args = LmEvalAdapter.build_args(
        {
            "backend": "local-completions",
            "model_args": (
                "model=test,base_url=http://localhost:11434/v1/completions,"
                "tokenizer=test/tokenizer"
            ),
        },
        benchmark(),
        {"batch_size": "auto", "device": "cuda"},
        {"save_samples": True},
        tmp_path,
    )
    assert "--device" not in endpoint_args

    default_args = LmEvalAdapter.build_args(
        {"backend": "hf", "model_args": "pretrained=local/model"},
        benchmark(),
        {"device": "cuda"},
        {},
        tmp_path,
    )
    assert default_args[default_args.index("--batch_size") + 1] == "auto"
    assert LmEvalAdapter.batch_candidates("auto", "local-completions") == [1]
    assert LmEvalAdapter.batch_candidates("auto", "microlm-native") == [16, 8, 4, 2, 1]
    assert LmEvalAdapter.batch_candidates(8, "local-chat-completions") == [1]
    assert LmEvalAdapter.batch_candidates(9, "hf") == [9, 4, 2, 1]


def test_stable_sample_indices_are_reusable_and_seeded() -> None:
    first = stable_sample_indices("mmlu_abstract_algebra", 101, 10, 42)
    repeated = stable_sample_indices("mmlu_abstract_algebra", 101, 10, 42)
    another_seed = stable_sample_indices("mmlu_abstract_algebra", 101, 10, 43)

    assert first == repeated
    assert len(first) == 11
    assert first == sorted(first)
    assert another_seed != first
    assert stable_sample_indices("task", 3, 0.01, 42)
    assert stable_sample_indices("task", 3, 100, 42) == [0, 1, 2]


def test_lm_eval_adapter_runs_microlm_native_bridge_without_cli(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.lm_eval_available", lambda: True
    )
    loaded: list[dict] = []
    evaluated_inference: list[dict] = []

    def fake_load(model_args: str, **kwargs):
        loaded.append({"model_args": model_args, **kwargs})
        return object()

    def fake_evaluate(
        harness,
        selected_benchmark,
        inference,
        scoring,
        output_dir,
        *,
        progress_callback=None,
    ):
        del harness, scoring
        evaluated_inference.append(inference)
        assert progress_callback is not None
        progress_callback(16, 100, "Running loglikelihood requests")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {selected_benchmark["task_id"]: {"acc_norm,none": 0.5}},
                    "versions": {selected_benchmark["task_id"]: 1},
                }
            ),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.load_harness", fake_load
    )
    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.evaluate_task", fake_evaluate
    )
    sampling_metadata = {
        "plan_hash": "sample-plan-hash",
        "leaf_task_count": 1,
        "total_documents": 10,
        "selected_documents": 2,
        "tasks": {"hellaswag": {"total": 10, "selected": 2}},
    }
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.build_sampling_plan",
        lambda *_args: ({"hellaswag": [1, 4]}, sampling_metadata),
    )

    events: list[tuple[str, dict]] = []

    async def emit(_event_type: str, _payload: dict) -> str:
        events.append((_event_type, _payload))
        return "running"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "primary_model_version_id": "microlm-1",
            "benchmark_config": {
                "comparison_config_hash": "comparison-hash",
                "tasks": [benchmark()],
            },
            "inference_config": {
                "models": {
                    "microlm-1": {
                        "backend": "microlm-native",
                        "model_args": f"pretrained={tmp_path / 'model'}",
                    }
                },
                "batch_size": "auto",
                "device": "cuda",
                "sampling": {
                    "schema_version": "forgelm.evaluation-sampling.v1",
                    "method": "stable_hash_v1",
                    "percentage": 20.0,
                    "seed": 42,
                },
            },
            "scoring_config": {},
        },
    }
    result = asyncio.run(LmEvalAdapter().run(task, emit))

    assert result["summary"]["overall"] == 0.5
    assert loaded == [
        {
            "model_args": f"pretrained={tmp_path / 'model'}",
            "batch_size": 16,
            "device": "cuda",
            "dtype_name": "auto",
        }
    ]
    execution = result["summary"]["execution"]["models"]["microlm-1"]["hellaswag"]
    assert execution["backend"] == "microlm-native"
    assert execution["effective_batch_sizes"] == [16]
    assert execution["detection_source"] == "microlm_bounded_oom_backoff"
    assert evaluated_inference[0]["_sample_plan"] == {"hellaswag": [1, 4]}
    assert result["summary"]["execution"]["sampling"]["benchmarks"][
        "hellaswag"
    ]["plan_hash"] == "sample-plan-hash"
    sampling_report = (
        tmp_path / "run" / "lm-eval-report" / "sampling-plans" / "hellaswag.json"
    )
    assert json.loads(sampling_report.read_text(encoding="utf-8"))["samples"] == {
        "hellaswag": [1, 4]
    }
    cli_sampling_plan = (
        tmp_path
        / "run"
        / "lm-eval-report"
        / "sampling-plans"
        / "hellaswag.samples.json"
    )
    assert json.loads(cli_sampling_plan.read_text(encoding="utf-8")) == {
        "hellaswag": [1, 4]
    }
    assert any(
        event_type == "progress"
        and payload.get("requests_completed") == 16
        and payload.get("requests_total") == 100
        and payload.get("progress") == 0.16
        for event_type, payload in events
    )


def test_lm_eval_adapter_stops_microlm_at_request_batch_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.lm_eval_available", lambda: True
    )
    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.load_harness",
        lambda *_args, **_kwargs: object(),
    )

    def fake_evaluate(*_args, progress_callback=None, **_kwargs):
        assert progress_callback is not None
        progress_callback(4, 100, "Running loglikelihood requests")
        raise AssertionError("control callback should have interrupted evaluation")

    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.evaluate_task", fake_evaluate
    )

    async def emit(_event_type: str, _payload: dict) -> str:
        return "cancelled"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "primary_model_version_id": "microlm-1",
            "benchmark_config": {"tasks": [benchmark()]},
            "inference_config": {
                "models": {
                    "microlm-1": {
                        "backend": "microlm-native",
                        "model_args": "pretrained=/model",
                    }
                },
                "batch_size": "auto",
                "device": "cuda",
            },
            "scoring_config": {},
        },
    }

    result = asyncio.run(
        LmEvalAdapter().run(task, emit, get_desired_state=lambda: "cancelled")
    )

    assert result == {"controlled_state": "cancelled"}


def test_lm_eval_adapter_does_not_fail_when_native_progress_delivery_times_out(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.lm_eval_available", lambda: True
    )
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval._NATIVE_PROGRESS_EMIT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.load_harness",
        lambda *_args, **_kwargs: object(),
    )

    def fake_evaluate(
        _harness,
        selected_benchmark,
        _inference,
        _scoring,
        output_dir,
        *,
        progress_callback=None,
    ):
        assert progress_callback is not None
        progress_callback(16, 100, "Running generate_until requests")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {
                        selected_benchmark["task_id"]: {"acc_norm,none": 0.5}
                    },
                    "versions": {selected_benchmark["task_id"]: 1},
                }
            ),
            encoding="utf-8",
        )
        return [16]

    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.evaluate_task", fake_evaluate
    )

    async def slow_emit(_event_type: str, _payload: dict) -> str:
        await asyncio.sleep(0.05)
        return "running"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "primary_model_version_id": "microlm-1",
            "benchmark_config": {"tasks": [benchmark()]},
            "inference_config": {
                "models": {
                    "microlm-1": {
                        "backend": "microlm-native",
                        "model_args": "pretrained=/model",
                    }
                },
                "batch_size": "auto",
                "device": "cuda",
            },
            "scoring_config": {},
        },
    }

    result = asyncio.run(LmEvalAdapter().run(task, slow_emit))

    assert result["summary"]["overall"] == 0.5


def test_lm_eval_adapter_releases_native_gpu_state_before_hf_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.lm_eval_available", lambda: True
    )
    adapter = LmEvalAdapter()
    operations: list[tuple[str, int]] = []

    def fake_release(harnesses: dict) -> None:
        operations.append(("release", len(harnesses)))
        harnesses.clear()

    def fake_evaluate(
        _harness,
        selected_benchmark,
        _inference,
        _scoring,
        output_dir,
        *,
        progress_callback=None,
    ):
        del progress_callback
        operations.append(("native", 1))
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {
                        selected_benchmark["task_id"]: {"acc_norm,none": 0.5}
                    },
                    "versions": {selected_benchmark["task_id"]: 1},
                }
            ),
            encoding="utf-8",
        )
        return [16]

    async def fake_run_process(
        args: list[str], emit, get_desired_state, env_overrides=None
    ) -> list[int]:
        del emit, get_desired_state, env_overrides
        operations.append(("hf", 1))
        output_dir = Path(args[args.index("--output_path") + 1])
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {"hellaswag": {"acc_norm,none": 0.6}},
                    "versions": {"hellaswag": 1},
                }
            ),
            encoding="utf-8",
        )
        return [8]

    monkeypatch.setattr(adapter, "_release_native_harnesses", fake_release)
    monkeypatch.setattr(adapter, "_run_process", fake_run_process)
    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.load_harness",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "llm_platform_worker.adapters.microlm_lm_eval.evaluate_task", fake_evaluate
    )

    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "primary_model_version_id": "native-model",
            "benchmark_config": {"tasks": [benchmark()]},
            "inference_config": {
                "models": {
                    "native-model": {
                        "backend": "microlm-native",
                        "model_args": "pretrained=/model",
                    },
                    "hf-model": {
                        "backend": "hf",
                        "model_args": "pretrained=fake/model",
                    },
                },
                "batch_size": "auto",
                "device": "cuda",
            },
            "scoring_config": {},
        },
    }

    result = asyncio.run(adapter.run(task, emit))

    assert list(result["summary"]["models"]) == ["native-model", "hf-model"]
    assert operations[:3] == [("native", 1), ("release", 1), ("hf", 1)]


def test_microlm_progress_tqdm_reports_only_harness_request_bars() -> None:
    updates: list[tuple[int, int, str]] = []

    class FakeTqdm:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def update(self, n=1):
            return n

    wrapped = _progress_tqdm(FakeTqdm, lambda *values: updates.append(values))
    tracked = wrapped(total=10, desc="Running loglikelihood requests", disable=True)
    ignored = wrapped(total=4, desc="Loading task documents", disable=True)

    assert tracked.update(3) == 3
    ignored.update(2)
    tracked.update(20)

    assert updates == [
        (3, 10, "Running loglikelihood requests"),
        (10, 10, "Running loglikelihood requests"),
    ]


def test_lm_eval_adapter_parses_auto_batch_probe_and_retries_manual_cuda_oom(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.lm_eval_available", lambda: True
    )
    adapter = LmEvalAdapter()
    attempted_batches: list[str] = []

    async def fake_run_process(
        args: list[str], emit, get_desired_state, env_overrides=None
    ) -> list[int]:
        batch_size = args[args.index("--batch_size") + 1]
        attempted_batches.append(batch_size)
        if len(attempted_batches) == 1:
            raise LmEvalProcessError(1, ["torch.OutOfMemoryError: CUDA out of memory"])
        output_dir = Path(args[args.index("--output_path") + 1])
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {"hellaswag": {"acc_norm,none": 0.5}},
                    "versions": {"hellaswag": 1},
                }
            ),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(adapter, "_run_process", fake_run_process)
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "primary_model_version_id": "model-1",
            "benchmark_config": {
                "comparison_config_hash": "comparison-hash",
                "tasks": [benchmark()],
            },
            "inference_config": {
                "models": {
                    "model-1": {
                        "backend": "hf",
                        "model_args": "pretrained=fake/model",
                    }
                },
                "batch_size": 8,
                "device": "cuda",
            },
            "scoring_config": {},
        },
    }

    result = asyncio.run(adapter.run(task, emit))

    assert attempted_batches == ["8", "4"]
    execution = result["summary"]["execution"]
    benchmark_execution = execution["models"]["model-1"]["hellaswag"]
    assert execution["batch_policy"] == "manual"
    assert benchmark_execution["effective_batch_size"] == 4
    assert benchmark_execution["oom_backoffs"] == 1
    assert any("自动降档为 4" in payload.get("message", "") for _, payload in events)


def test_lm_eval_adapter_restarts_auto_batch_at_half_detected_size_after_cuda_oom(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.lm_eval_available", lambda: True
    )
    adapter = LmEvalAdapter()
    attempted_batches: list[str] = []

    async def fake_run_process(
        args: list[str], emit, get_desired_state, env_overrides=None
    ) -> list[int]:
        batch_size = args[args.index("--batch_size") + 1]
        attempted_batches.append(batch_size)
        if batch_size == "auto":
            raise LmEvalProcessError(
                1,
                [
                    "Determined largest batch size: 32",
                    "torch.AcceleratorError: CUDA error: out of memory",
                    "cudaErrorMemoryAllocation",
                ],
            )
        output_dir = Path(args[args.index("--output_path") + 1])
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {"hellaswag": {"acc_norm,none": 0.5}},
                    "versions": {"hellaswag": 1},
                }
            ),
            encoding="utf-8",
        )
        return []

    monkeypatch.setattr(adapter, "_run_process", fake_run_process)
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "primary_model_version_id": "model-1",
            "benchmark_config": {"tasks": [benchmark()]},
            "inference_config": {
                "models": {
                    "model-1": {
                        "backend": "hf",
                        "model_args": "pretrained=fake/model",
                    }
                },
                "batch_size": "auto",
                "device": "cuda",
            },
            "scoring_config": {},
        },
    }

    result = asyncio.run(adapter.run(task, emit))

    assert attempted_batches == ["auto", "16"]
    benchmark_execution = result["summary"]["execution"]["models"]["model-1"][
        "hellaswag"
    ]
    assert benchmark_execution["effective_batch_size"] == 16
    assert benchmark_execution["detection_source"] == "lm_eval_oom_backoff"
    assert benchmark_execution["oom_backoffs"] == 1
    assert benchmark_execution["attempts"][0]["detected_batch_sizes"] == [32]
    assert any("自动降档为 16" in payload.get("message", "") for _, payload in events)


def test_lm_eval_adapter_recognizes_accelerator_cuda_oom_and_detected_batch() -> None:
    error = LmEvalProcessError(
        1,
        [
            "Determined largest batch size: 32",
            "torch.AcceleratorError: CUDA error: out of memory",
            "Search for cudaErrorMemoryAllocation in the CUDA Runtime API",
        ],
    )

    assert error.is_device_oom is True
    assert error.detected_batch_sizes == [32]


def test_lm_eval_adapter_parses_detected_batch_sizes() -> None:
    assert LmEvalAdapter._detected_batch_sizes(
        [
            "Passed argument batch_size = auto. Detecting largest batch size",
            "Determined Largest batch size: 16",
            "Determined largest batch size: 8",
        ]
    ) == [16, 8]


def test_lm_eval_adapter_normalizes_metrics_samples_and_signature(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    (output_dir / "samples.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"doc_id": 1, "metrics": {"acc_norm": 1}}),
                json.dumps({"doc_id": 2, "metrics": {"acc_norm": 0}}),
            ]
        ),
        encoding="utf-8",
    )
    document = {
        "results": {"hellaswag": {"acc_norm,none": 0.75, "acc_norm_stderr,none": 0.1}},
        "configs": {"hellaswag": {"num_fewshot": 3, "doc_to_text": "{{ctx}}"}},
        "versions": {"hellaswag": 2},
    }
    task = {
        "payload": {
            "benchmark_config": {
                "comparison_config_hash": "comparison-hash",
                "tasks": [benchmark()],
            },
            "scoring_config": {"max_sample_results": 10},
        }
    }

    result = LmEvalAdapter.normalize_results(
        task,
        [("model-1", benchmark(), document, output_dir)],
        "0.4.test",
    )

    assert result["summary"]["overall"] == 0.75
    assert result["summary"]["dimensions"] == {"常识": 0.75, "推理": 0.75}
    assert result["summary"]["aggregation"] == {
        "method": "weighted_mean",
        "unit": "benchmark_task",
        "formula": "sum(task_score * task_weight) / sum(task_weight)",
        "included_when": "include_in_overall != false",
    }
    assert result["summary"]["models"]["model-1"]["included_task_count"] == 1
    assert result["summary"]["models"]["model-1"]["total_weight"] == 1.0
    assert result["summary"]["models"]["model-1"]["tasks"]["hellaswag"]["score"] == 0.75
    assert [sample["status"] for sample in result["samples"]] == ["passed", "failed"]
    assert result["benchmark_signature"]["framework_version"] == "0.4.test"
    assert result["benchmark_signature"]["task_versions"] == {"hellaswag": 2}
    assert len(result["benchmark_signature"]["compatibility_hash"]) == 64


def test_lm_eval_adapter_keeps_diagnostic_perplexity_out_of_overall(tmp_path: Path) -> None:
    hellaswag = benchmark()
    wikitext = {
        "id": "lm-eval:wikitext",
        "task_id": "wikitext",
        "primary_metric": "word_perplexity",
        "dimensions": ["语言建模"],
        "include_in_overall": False,
        "higher_is_better": False,
        "weight": 1,
    }
    reports = []
    for name in ("hellaswag", "wikitext"):
        report = tmp_path / name
        report.mkdir()
        reports.append(report)
    task = {
        "payload": {
            "benchmark_config": {
                "comparison_config_hash": "base-preset",
                "tasks": [hellaswag, wikitext],
            },
            "scoring_config": {},
        }
    }

    result = LmEvalAdapter.normalize_results(
        task,
        [
            (
                "base-model",
                hellaswag,
                {"results": {"hellaswag": {"acc_norm,none": 0.4}}},
                reports[0],
            ),
            (
                "base-model",
                wikitext,
                {"results": {"wikitext": {"word_perplexity,none": 18.25}}},
                reports[1],
            ),
        ],
        "0.4.test",
    )

    model = result["summary"]["models"]["base-model"]
    assert model["overall"] == 0.4
    assert model["included_task_count"] == 1
    assert model["diagnostic_task_count"] == 1
    assert result["summary"]["dimensions"] == {"常识": 0.4, "推理": 0.4}
    assert model["tasks"]["wikitext"] == {
        "benchmark_id": "lm-eval:wikitext",
        "score": 18.25,
        "metric": "word_perplexity",
        "weight": 1.0,
        "higher_is_better": False,
        "include_in_overall": False,
        "raw_metrics": {"word_perplexity,none": 18.25},
    }


def test_lm_eval_adapter_persists_stable_samples_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "model" / "task"
    successful_dir = output_dir / "attempt-2-batch-8" / "model-path"
    successful_dir.mkdir(parents=True)
    upstream = successful_dir / "samples_task_2026-08-27.jsonl"
    upstream.write_text('{"doc_id": 1}\n', encoding="utf-8")

    canonical = LmEvalAdapter._persist_canonical_samples(successful_dir, output_dir)

    assert canonical == output_dir / "samples.jsonl"
    assert canonical.read_text(encoding="utf-8") == '{"doc_id": 1}\n'


def test_lm_eval_adapter_uses_primary_model_score_and_reports_baseline_delta(
    tmp_path: Path,
) -> None:
    primary_report = tmp_path / "primary"
    baseline_report = tmp_path / "baseline"
    primary_report.mkdir()
    baseline_report.mkdir()
    task = {
        "payload": {
            "primary_model_version_id": "candidate",
            "benchmark_config": {
                "comparison_config_hash": "comparison-hash",
                "tasks": [benchmark()],
            },
            "scoring_config": {},
        }
    }
    primary_document = {
        "results": {"hellaswag": {"acc_norm,none": 0.72}},
        "versions": {"hellaswag": 1},
    }
    baseline_document = {
        "results": {"hellaswag": {"acc_norm,none": 0.60}},
        "versions": {"hellaswag": 1},
    }

    result = LmEvalAdapter.normalize_results(
        task,
        [
            ("candidate", benchmark(), primary_document, primary_report),
            ("baseline", benchmark(), baseline_document, baseline_report),
        ],
        "0.4.test",
    )

    assert result["summary"]["primary_model_version_id"] == "candidate"
    assert result["summary"]["overall"] == 0.72
    assert result["summary"]["dimensions"] == {"常识": 0.72, "推理": 0.72}
    assert round(result["summary"]["comparisons"]["baseline"]["overall_delta"], 6) == 0.12


def test_lm_eval_adapter_resolves_endpoint_credential_only_for_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval.lm_eval_available", lambda: True
    )
    monkeypatch.setenv("VENDOR_EVAL_KEY", "top-secret-value")
    captured_env: list[dict[str, str]] = []
    adapter = LmEvalAdapter()

    async def fake_run_process(
        args: list[str], emit, get_desired_state, env_overrides=None
    ) -> None:
        captured_env.append(dict(env_overrides or {}))
        output_dir = Path(args[args.index("--output_path") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(
            json.dumps(
                {
                    "results": {"hellaswag": {"acc_norm,none": 0.5}},
                    "versions": {"hellaswag": 1},
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(adapter, "_run_process", fake_run_process)

    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "primary_model_version_id": "endpoint-model",
            "benchmark_config": {
                "comparison_config_hash": "comparison-hash",
                "tasks": [benchmark()],
            },
            "inference_config": {
                "models": {
                    "endpoint-model": {
                        "backend": "local-completions",
                        "model_args": "model=test,base_url=http://localhost/v1/completions",
                        "auth_env_var": "VENDOR_EVAL_KEY",
                    }
                }
            },
            "scoring_config": {},
        },
    }

    result = asyncio.run(adapter.run(task, emit))

    assert result["summary"]["overall"] == 0.5
    assert captured_env == [{"OPENAI_API_KEY": "top-secret-value"}]
    assert "top-secret-value" not in json.dumps(task)


def test_lm_eval_adapter_finishes_when_exited_process_streams_never_close(
    monkeypatch,
) -> None:
    class PipeTransport:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class HangingStream:
        def __init__(self) -> None:
            self._transport = PipeTransport()

        async def readline(self) -> bytes:
            await asyncio.Event().wait()
            return b""

    class ExitedProcess:
        def __init__(self) -> None:
            self.stdout = HangingStream()
            self.stderr = HangingStream()
            self.returncode: int | None = 0

        async def wait(self) -> int:
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -1

    process = ExitedProcess()

    async def fake_create_subprocess_exec(*_args, **_kwargs) -> ExitedProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(
        "llm_platform_worker.adapters.lm_eval._PROCESS_EXIT_STREAM_DRAIN_TIMEOUT_SECONDS",
        0.01,
    )
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    asyncio.run(
        asyncio.wait_for(
            LmEvalAdapter()._run_process(["lm-eval"], emit, None),
            timeout=0.5,
        )
    )

    assert any(
        event_type == "log"
        and payload.get("level") == "warning"
        and "输出管道未及时关闭" in payload.get("message", "")
        for event_type, payload in events
    )
    assert process.stdout._transport.closed is True
    assert process.stderr._transport.closed is True


def test_lm_eval_adapter_runs_an_installed_module_and_keeps_reports(
    tmp_path: Path, monkeypatch
) -> None:
    fake_root = tmp_path / "fake-package"
    module_root = fake_root / "lm_eval"
    module_root.mkdir(parents=True)
    (module_root / "__init__.py").write_text("", encoding="utf-8")
    (module_root / "__main__.py").write_text(
        """
import json
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output_path") + 1])
task = args[args.index("--tasks") + 1]
output.mkdir(parents=True, exist_ok=True)
(output / "results.json").write_text(json.dumps({
    "results": {task: {"acc_norm,none": 0.8}},
    "configs": {task: {"num_fewshot": 3}},
    "versions": {task: 1},
}), encoding="utf-8")
(output / "samples.jsonl").write_text(json.dumps({
    "doc_id": "sample-1", "metrics": {"acc_norm": 1}
}) + "\\n", encoding="utf-8")
print("fake lm-eval completed")
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(fake_root))
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(fake_root) + (os.pathsep + current_pythonpath if current_pythonpath else ""),
    )
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    task = {
        "runtime_context": {"run_dir": str(tmp_path / "run")},
        "payload": {
            "benchmark_config": {
                "comparison_config_hash": "comparison-hash",
                "tasks": [benchmark()],
            },
            "inference_config": {
                "models": {
                    "model-1": {
                        "backend": "hf",
                        "model_args": "pretrained=fake/model",
                    }
                },
                "device": "cpu",
            },
            "scoring_config": {"save_samples": True},
        },
    }

    result = asyncio.run(LmEvalAdapter().run(task, emit))

    assert result["summary"]["overall"] == 0.8
    assert result["samples"][0]["id"] == "sample-1"
    assert Path(result["outputs"][0]["path"]).is_dir()
    assert any(
        event_type == "log" and payload["message"] == "fake lm-eval completed"
        for event_type, payload in events
    )
