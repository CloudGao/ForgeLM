import asyncio
import io
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from llm_platform_core.data_generation import DataGenerationCreate, prompt_for
from llm_platform_worker.data_generation_runtime import (
    generate_records,
    generate_records_vllm_async,
    jobs,
    prepare_generation_checkpoint,
    submission_window,
    tune_batch,
    validate_output,
    vllm_scheduler_limits,
)
from pydantic import ValidationError

from llm_platform_worker.adapters.data_generation import generation_environment


def config(**kwargs):
    return DataGenerationCreate(
        name="test", input_dataset_version_id="data", model_weight_version_id="model", **kwargs
    )


def test_vllm_generation_uses_wsl_compatible_model_runner():
    environment = generation_environment({"payload": {}})
    assert environment["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_vllm_scheduler_keeps_decode_concurrency_and_cuda_graphs_enabled():
    limits = vllm_scheduler_limits(config(max_batch_size=32, max_input_tokens=6144))
    assert limits == {
        "max_num_seqs": 32,
        "max_num_batched_tokens": 8192,
        "enforce_eager": False,
    }
    assert vllm_scheduler_limits(config(max_batch_size=128))["max_num_seqs"] == 64
    assert submission_window(config(engine="vllm", max_batch_size=64), 64) == 128


@pytest.mark.parametrize(
    "patch",
    [
        {"enable_thinking": True},
        {"engine": "remote"},
        {"batch_size": 0},
        {"batch_size": 129},
        {"styles": []},
        {"styles": ["plain", "plain"]},
        {"resource_request": {"gpus": 2}},
        {"engine": "transformers", "quantization": "fp8"},
        {"engine": "transformers", "enable_prefix_caching": True},
        {"chinese_output_constraint": True},
        {"engine": "vllm", "chinese_output_constraint": True},
        {"temperature": float("nan")},
    ],
)
def test_invalid_config(patch):
    with pytest.raises(ValidationError):
        config(**patch)


def test_existing_generation_configs_do_not_silently_enable_language_constraint():
    assert config(engine="vllm", prompt_version="cpt-generation-v7", document_mode="whole",
                  styles=["facts"]).chinese_output_constraint is False


def test_chunk_styles_and_provenance(tmp_path):
    text = "知识" * 150
    (tmp_path / "source.jsonl").write_text(json.dumps({"body": text}), encoding="utf-8")
    result = list(
        jobs(
            {"runtime_path": str(tmp_path), "logical_files": ["source.jsonl"]},
            config(text_field="body", chunk_characters=128, styles=["plain", "causal"]),
            "knowledge",
        )
    )
    assert len(result) == 6
    assert [r["character_start"] for r in result] == [0, 128, 256, 0, 128, 256]
    assert len({r["source_sha256"] for r in result}) == 1
    assert [r["style"] for r in result] == ["plain"] * 3 + ["causal"] * 3
    assert result[0]["source_ordinal"] == 0
    assert "教材化" not in prompt_for("text", "translation", "translation")


def test_no_path_escape_or_silent_bad_json(tmp_path):
    with pytest.raises(ValueError):
        list(
            jobs(
                {"runtime_path": str(tmp_path), "logical_files": ["../bad.jsonl"]},
                config(),
                "translation",
            )
        )
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError):
        list(
            jobs(
                {"runtime_path": str(tmp_path), "logical_files": [bad.name]},
                config(),
                "translation",
            )
        )


def test_whole_articles_filter_before_generation_and_preserve_boundaries(tmp_path):
    rows = ["知" * 3000, "知" * 3001, "", "短文"]
    (tmp_path / "source.jsonl").write_text(
        "\n".join(json.dumps({"text": text}) for text in rows), encoding="utf-8"
    )
    report = {}
    result = list(
        jobs(
            {"runtime_path": str(tmp_path), "logical_files": ["source.jsonl"]},
            config(
                document_mode="whole",
                max_source_characters=3000,
                chunk_characters=128,
                styles=["textbook", "plain"],
                prompt_version="cpt-generation-v2",
            ),
            "knowledge",
            report,
        )
    )
    assert len(result) == 4
    assert [r["source_ordinal"] for r in result] == [0, 3, 0, 3]
    assert [r["character_start"] for r in result] == [0] * 4
    assert [r["character_end"] for r in result] == [3000, 2, 3000, 2]
    assert [r["style"] for r in result] == ["textbook", "textbook", "plain", "plain"]
    assert rows[0] in result[0]["prompt"]
    assert "禁止问答、QA" in result[0]["prompt"]
    assert report == dict(
        source_records=4, accepted_source_records=2, skipped_empty_records=1, skipped_long_records=1
    )


def test_fact_preserving_prompt_keeps_prose_rule_and_source_boundaries():
    source = "张三于1979年建立了甲村。"
    previous = prompt_for(source, "knowledge", "textbook", "cpt-generation-v2")
    current = prompt_for(source, "knowledge", "textbook", "cpt-generation-v3")
    assert "禁止问答、QA" in current
    assert "逐项保留原文中的所有具体数字、日期、专有名称和独立事实" in current
    assert current.endswith(f"<source>\n{source}\n</source>")
    assert "逐项保留原文" not in previous


@pytest.mark.parametrize(
    "text,truncated",
    [("", False), ("<think>秘密思考</think>中文", False), ("hello", False), ("中文", True)],
)
def test_reject_invalid_output(text, truncated):
    with pytest.raises(ValueError):
        validate_output(text, truncated)
    assert validate_output("<think>\n\n</think>\n中文") == "中文"


def test_prose_only_protocol_rejects_qa():
    for text in ["问：什么是物质？\n答：物质。", "**Q:** 什么是物质？\n**A:** 物质。"]:
        with pytest.raises(ValueError, match="QA"):
            validate_output(text, forbid_qa=True)
    assert validate_output("物质的性质可以通过实验研究。", forbid_qa=True)


class FakeEngine:
    def __init__(self, capacity=3):
        self.capacity, self.calls, self.cleanups = capacity, [], 0

    def generate(self, prompts, *, probe=False):
        self.calls.append((len(prompts), probe))
        if len(prompts) > self.capacity:
            raise RuntimeError("CUDA out of memory")
        return [("中文" + prompt, 10, False) for prompt in prompts]

    def cleanup_oom(self):
        self.cleanups += 1


class TruncatingVllmEngine(FakeEngine):
    def generate(self, prompts, *, probe=False, full_budget=False):
        self.calls.append((len(prompts), probe, full_budget))
        if full_budget:
            return [("中文完整", 20, False) for _ in prompts]
        return [
            ("中文截断" if index == 0 else "中文完整", 10, index == 0)
            for index, _ in enumerate(prompts)
        ]


def test_autotune_searches_boundary_and_measures_throughput(monkeypatch):
    # At constant elapsed time, batch=3 is fastest; batch=4 fails.
    clock = iter(range(100))
    monkeypatch.setattr(
        "llm_platform_worker.data_generation_runtime.time.perf_counter", lambda: next(clock)
    )
    engine, report = FakeEngine(), {"probes": []}
    assert tune_batch(engine, [{"prompt": "source"}], config(max_batch_size=8), report) == 3
    assert report["largest_tested_safe_batch"] == 3
    assert any(p["status"] == "oom" for p in report["probes"])
    assert engine.cleanups == 1


def test_vllm_autotune_uses_scheduler_ceiling_without_batch_barrier_probe():
    engine, report = FakeEngine(3), {"probes": []}
    selected = tune_batch(
        engine,
        [{"prompt": "source"}],
        config(engine="vllm", quantization="fp8", max_batch_size=8),
        report,
    )
    assert selected == 8
    assert report["probes"] == []
    assert report["selection"] == "vllm_scheduler_ceiling"
    assert engine.calls == []
    assert engine.cleanups == 0


def test_runtime_oom_replays_without_missing_or_duplicate_rows():
    engine, report, stream = FakeEngine(2), {"probes": [], "oom_backoffs": 0}, io.StringIO()
    source = iter({"prompt": str(i), "source_ordinal": i} for i in range(7))
    generate_records(engine, source, config(batch_size=4, max_batch_size=4), stream, {}, report)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [r["metadata"]["source_ordinal"] for r in rows] == list(range(7))
    assert all("prompt" not in r["metadata"] for r in rows)
    assert report["oom_backoffs"] == 1
    assert report["final_batch_size"] == 2


def test_vllm_queues_a_continuous_window_grouped_by_style():
    engine, report, stream = FakeEngine(256), {"probes": [], "oom_backoffs": 0}, io.StringIO()
    source = iter(
        {
            "prompt": str(i),
            "source_ordinal": i,
            "style": "textbook" if i % 2 == 0 else "plain",
        }
        for i in range(130)
    )
    generate_records(
        engine,
        source,
        config(
            engine="vllm",
            quantization="fp8",
            batch_size=32,
            max_batch_size=32,
            styles=["textbook", "plain"],
        ),
        stream,
        {},
        report,
    )
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert engine.calls == [(128, False), (2, False)]
    assert [row["metadata"]["style"] for row in rows[:64]] == ["textbook"] * 64
    assert [row["metadata"]["style"] for row in rows[64:128]] == ["plain"] * 64
    assert [row["metadata"]["style"] for row in rows[128:]] == ["textbook", "plain"]
    assert report["submission_window"] == 128
    assert report["output_records"] == 130


def test_vllm_retries_only_adaptively_truncated_records_with_full_budget():
    engine = TruncatingVllmEngine(256)
    report, stream = {"probes": [], "oom_backoffs": 0}, io.StringIO()
    source = iter(
        {"prompt": str(i), "source_ordinal": i, "style": "textbook"} for i in range(3)
    )
    generate_records(
        engine,
        source,
        config(
            engine="vllm",
            quantization="fp8",
            batch_size=8,
            max_batch_size=8,
            styles=["textbook"],
        ),
        stream,
        {},
        report,
    )
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert engine.calls == [(3, False, False), (1, False, True)]
    assert rows[0]["text"] == "中文完整"
    assert report["length_retries"] == 1


def test_generation_checkpoint_resumes_verified_rows_and_truncates_partial_tail(tmp_path):
    task = {
        "id": "task-1",
        "payload": {
            "output_dataset_version_id": "version-1",
            "generation": {"prompt_version": "cpt-generation-v3"},
            "generation_input": {"artifact_hash": "input-hash"},
            "generation_model": {"identity_hash": "model-hash"},
        },
    }
    provenance = {"input_artifact_hash": "input-hash", "model_identity_hash": "model-hash"}
    first, report = tmp_path / "version-1-first", {}
    keys, rows, tokens, skipped, skip_path = prepare_generation_checkpoint(
        task, tmp_path, first, provenance, report
    )
    assert (keys, rows, tokens, skipped) == (set(), 0, 0, 0)
    metadata = {
        **provenance,
        "source_file": "source.jsonl",
        "source_ordinal": 3,
        "source_sha256": "source-hash",
        "character_start": 0,
        "character_end": 3,
        "style": "textbook",
        "generation_tokens": 12,
    }
    (first / "data.partial").write_bytes(
        (json.dumps({"text": "中文正文", "metadata": metadata}) + "\n").encode()
        + b'{"text":"unfinished"'
    )
    skipped_metadata = {**metadata, "source_ordinal": 4}
    skip_path.write_text(
        json.dumps({"metadata": skipped_metadata, "reason": "output_length_limit"}) + "\n",
        encoding="utf-8",
    )
    second, resumed_report = tmp_path / "version-1-second", {}
    keys, rows, tokens, skipped, _ = prepare_generation_checkpoint(
        task, tmp_path, second, provenance, resumed_report
    )
    assert (rows, tokens, skipped) == (1, 12, 1)
    assert len(keys) == 2
    assert (second / "data.partial").read_bytes().endswith(b"\n")
    assert resumed_report["resumed_from"] == first.name
    changed_task = {**task, "id": "another-task"}
    _, rows, _, _, _ = prepare_generation_checkpoint(
        changed_task, tmp_path, tmp_path / "version-1-third", provenance, {}
    )
    assert rows == 0


def test_failed_generation_attempt_preserves_partial_and_retry_skips_completed_rows(
    tmp_path, monkeypatch
):
    from llm_platform_worker import data_generation_runtime as runtime

    workspace = tmp_path / "project"
    workspace.mkdir()
    source_file = tmp_path / "source.jsonl"
    source_file.write_text(
        "".join(json.dumps({"text": f"知识{i}"}, ensure_ascii=False) + "\n" for i in range(3)),
        encoding="utf-8",
    )
    task = {
        "id": "resume-task",
        "payload": {
            "generation": {
                "mode": "knowledge",
                "name": "resume",
                "input_dataset_version_id": "source-version",
                "model_weight_version_id": "model-version",
                "engine": "vllm",
                "document_mode": "whole",
                "styles": ["textbook"],
                "prompt_version": "cpt-generation-v3",
            },
            "generation_input": {
                "runtime_path": str(source_file),
                "logical_files": [source_file.name],
                "artifact_hash": "input-hash",
            },
            "generation_model": {"runtime_path": "fake", "identity_hash": "model-hash"},
            "output_dataset_version_id": "output-version",
        },
        "runtime_context": {
            "project_workspace": str(workspace),
            "worker_id": "worker-1",
            "attempt_dir": str(tmp_path / "first-attempt"),
        },
    }
    calls, emitted = [], []

    async def fake_generate(model_path, source, config, stream, provenance, report, keys, skip_stream, **kwargs):
        remaining = list(source)
        calls.append([item["source_ordinal"] for item in remaining])
        for item in remaining[:1] if len(calls) == 1 else remaining:
            stream.write(
                json.dumps(
                    {
                        "text": "中文正文",
                        "metadata": {
                            **provenance,
                            **{key: value for key, value in item.items() if key != "prompt"},
                            "generation_tokens": 4,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        if len(calls) == 1:
            raise RuntimeError("simulated interruption")
        report.update(output_records=3, generated_tokens=12, skipped_generation_records=0)

    monkeypatch.setattr(runtime, "generate_records_vllm_async", fake_generate)
    monkeypatch.setattr(runtime, "emit", lambda event, **payload: emitted.append((event, payload)))
    monkeypatch.setattr(
        runtime.DataImportAdapter,
        "_analyze",
        lambda *args, **kwargs: {
            "source_snapshot": {"files": [{"path": "data.jsonl", "sha256": "hash"}]},
            "raw_size_bytes": 100,
            "document_count": 3,
        },
    )
    monkeypatch.setattr(runtime.DataImportAdapter, "_manifest_hash", lambda files: "hash")
    (tmp_path / "first-attempt").mkdir()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runtime.run(task)
    root = workspace / ".llm-platform" / "data-assets" / "data-generation"
    first = root / "output-version-first-attempt"
    assert (first / "data.partial").is_file()
    task["runtime_context"]["attempt_dir"] = str(tmp_path / "second-attempt")
    (tmp_path / "second-attempt").mkdir()
    runtime.run(task)
    second = root / "output-version-second-attempt"
    rows = [json.loads(line) for line in (second / "data.jsonl").read_text(encoding="utf-8").splitlines()]
    assert calls == [[0, 1, 2], [1, 2]]
    assert [row["metadata"]["source_ordinal"] for row in rows] == [0, 1, 2]
    assert emitted[-1][0] == "result"


@pytest.mark.parametrize("failure", ["length", "surface", "surface_natural", "abstention", "surface_single"])
@pytest.mark.parametrize("job_count", [2, 16])
@pytest.mark.parametrize("batch_mode", ["auto", 1])
def test_vllm_default_skips_full_budget_length_output_and_records_source(
    tmp_path, monkeypatch, failure, job_count, batch_mode
):
    def module(name, **values):
        result = ModuleType(name)
        for key, value in values.items():
            setattr(result, key, value)
        monkeypatch.setitem(sys.modules, name, result)

    class Tokenizer:
        chat_template = "template"
        retries = []

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            if len(messages) > 1:
                Tokenizer.retries.append(messages)
            return messages[0]["content"]

        @staticmethod
        def encode(text, **kwargs):
            return list(text)

    class Engine:
        requests = []
        constraints = []

        @classmethod
        def from_engine_args(cls, args):
            assert args["max_num_seqs"] == (2 if batch_mode == "auto" else 1)
            return cls()

        async def generate(self, prompt, params, request_id):
            self.requests.append((prompt, params.max_tokens))
            self.constraints.append(getattr(params, "structured_outputs", None))
            bad = prompt == "bad"
            limited = bad and failure == "length"
            text = ("中文未完" if limited else "中文完整") if failure == "length" else (
                "中文2002" if bad else "中文2001"
            )
            if bad and failure == "abstention":
                text = "【无法可靠生成】"
            yield SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=text,
                        token_ids=[1] * 4,
                        finish_reason="length" if limited else "stop",
                    )
                ]
            )

        def shutdown(self):
            pass

        async def reset_prefix_cache(self):
            return True

    module("torch", cuda=SimpleNamespace(synchronize=lambda: None))
    module("vllm", __version__="test", SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
    module("transformers", AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: Tokenizer()))
    module("vllm.engine")
    module("vllm.engine.arg_utils", AsyncEngineArgs=lambda **kwargs: kwargs)
    module("vllm.sampling_params", RequestOutputKind=SimpleNamespace(FINAL_ONLY="final"),
           StructuredOutputsParams=lambda **kwargs: SimpleNamespace(**kwargs))
    module("vllm.v1")
    module("vllm.v1.engine")
    module("vllm.v1.engine.async_llm", AsyncLLM=Engine)
    style = "textbook" if failure == "length" else "thematic"
    source = iter(
        {
            "prompt": prompt,
            "source_file": "source.jsonl",
            "source_ordinal": index,
            "source_sha256": f"hash-{index}",
            "character_start": 0,
            "character_end": len(prompt),
            "style": style,
            **({"quality_anchors": {"numbers": ["2001"], "chinese_only": True}} if failure != "length" else {}),
        }
        for index, prompt in enumerate(["bad", *(["good"] * (job_count - 1))])
    )
    output, skipped = tmp_path / "data.partial", tmp_path / "skipped.records"
    report = {"length_retries": 0}
    with output.open("w", encoding="utf-8") as stream, skipped.open(
        "w", encoding="utf-8"
    ) as skip_stream:
        asyncio.run(
            generate_records_vllm_async(
                "fake-model",
                source,
                config(engine="vllm", chinese_output_constraint=failure == "surface", batch_size=batch_mode, max_batch_size=2, styles=[style], document_mode="whole",
                       validation_retry_limit=0 if failure == "surface_single" else 1,
                       prompt_version="cpt-generation-v1" if failure == "length" else "cpt-generation-v5"),
                stream,
                {},
                report,
                set(),
                skip_stream,
                telemetry_dir=tmp_path,
            )
        )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    skips = [json.loads(line) for line in skipped.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["metadata"]["source_ordinal"] for row in rows) == list(range(1, job_count))
    assert skips[0]["metadata"]["source_ordinal"] == 0
    assert skips[0]["reason"] == ("output_length_limit" if failure == "length" else "fact_surface_validation")
    assert report["output_records"] == job_count - 1
    retried = failure not in {"abstention", "surface_single"}
    assert report["raw_generated_tokens"] == (job_count + int(retried)) * 4
    assert report["retry_generated_tokens"] == (4 if retried else 0)
    assert report["rejected_generated_tokens"] == (8 if retried else 4)
    latest = json.loads((tmp_path / "generation-progress.json").read_text(encoding="utf-8"))
    assert latest["accepted_tokens_this_attempt"] == (job_count - 1) * 4
    assert latest["raw_generated_tokens"] == latest["accepted_tokens_this_attempt"] + latest["rejected_generated_tokens"]
    assert report["skipped_generation_records"] == 1
    assert report["length_retries"] == (1 if failure == "length" else 0)
    if failure == "abstention":
        assert report["validation_retries"] == 0
        assert report["quality_rejected_records"] == 1
        assert skips[0]["detail"] == "unresolved_source"
        assert skips[0]["rejected_text"] == "【无法可靠生成】"
        assert not Tokenizer.retries
        assert all(tokens == 512 for prompt, tokens in Engine.requests if prompt == "bad")
    elif failure == "surface_single":
        assert report["validation_retries"] == 0
        assert report["quality_rejected_records"] == 1
        assert not Tokenizer.retries
        assert all(tokens == 512 for prompt, tokens in Engine.requests if prompt == "bad")
    elif failure != "length":
        assert report["validation_retries"] == 1
        assert report["quality_rejected_records"] == 1
        assert skips[0]["rejected_text"] == "中文2002"
        assert any("数字必须与原始事实一致" in messages[-1]["content"] for messages in Tokenizer.retries)
        assert all(messages[0]["content"] == "bad" for messages in Tokenizer.retries)
    if retried:
        assert [tokens for prompt, tokens in Engine.requests if prompt == "bad"][-2:] == [512, 4096]
    assert report["selection"] == ("manual" if batch_mode != "auto" else
                                    "vllm_measured_continuous" if job_count >= 16 else "insufficient_probe_input")
    if batch_mode == "auto" and job_count >= 16 and failure != "surface":
        audit = [json.loads(line) for line in (tmp_path / "generation-validation.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(audit) == job_count
        assert report["validation_sample_saved"] is True
        assert report["fact_rules_implementation_sha256"]
        if failure == "surface_single":
            assert all(row["requests"] == 1 for row in audit)
    if failure == "surface":
        assert all(item.regex == r"[^A-Za-z]*" for item in Engine.constraints[-(job_count + 1):])
        if batch_mode == "auto" and job_count >= 16:
            audit = [json.loads(line) for line in (tmp_path / "generation-validation.jsonl").read_text(encoding="utf-8").splitlines()]
            assert len(audit) == job_count * 2
            assert len({row["policy"] for row in audit}) == 2
            assert report["language_guard_comparison"]["legacy"]["input_records"] == job_count


def test_batch_validation_is_atomic():
    engine = FakeEngine(2)
    engine.generate = lambda prompts: [("中文", 3, False), ("中文", 3, True)]
    stream = io.StringIO()
    with pytest.raises(ValueError):
        generate_records(
            engine,
            iter([{"prompt": "a"}, {"prompt": "b"}]),
            config(batch_size=2, output_length_limit_policy="fail"),
            stream,
            {},
            {"oom_backoffs": 0},
        )
    assert stream.getvalue() == ""


def test_default_length_policy_skips_only_truncated_sync_output():
    engine = FakeEngine(2)
    engine.generate = lambda prompts: [("中文完整", 3, False), ("中文截断", 3, True)]
    stream, skipped, report = io.StringIO(), io.StringIO(), {"oom_backoffs": 0}
    generate_records(
        engine,
        iter(
            [
                {"prompt": "a", "source_ordinal": 0, "style": "textbook"},
                {"prompt": "b", "source_ordinal": 1, "style": "textbook"},
            ]
        ),
        config(batch_size=2),
        stream,
        {},
        report,
        skipped,
    )
    assert [json.loads(line)["metadata"]["source_ordinal"] for line in stream.getvalue().splitlines()] == [0]
    assert json.loads(skipped.getvalue())["metadata"]["source_ordinal"] == 1
    assert report["skipped_generation_records"] == 1


def test_render_enforces_no_thinking_and_input_budget():
    from llm_platform_worker.data_generation_runtime import LocalEngine

    engine = object.__new__(LocalEngine)
    engine.config = config(max_input_tokens=256)

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert kwargs["add_generation_prompt"] is True
            return messages[0]["content"]

        def encode(self, text, **kwargs):
            return list(text)

    engine.tokenizer = Tokenizer()
    assert engine.render(["source"]) == ["source"]
    with pytest.raises(ValueError, match="max_input_tokens"):
        engine.render(["x" * 257])


def test_complete_worker_output_is_analyzed_and_registered(tmp_path, monkeypatch, capsys):
    from llm_platform_worker.data_generation_runtime import run

    cfg = config(batch_size=2, styles=["plain"])
    source = tmp_path / "source"
    source.mkdir()
    (source / "text.jsonl").write_text(
        json.dumps({"text": "Physics studies matter and motion."}), encoding="utf-8"
    )
    attempt = tmp_path / "attempt-1"
    attempt.mkdir()
    engine = FakeEngine(2)
    engine.version = "test"
    monkeypatch.setattr(
        "llm_platform_worker.data_generation_runtime.LocalEngine", lambda path, config: engine
    )
    task = {
        "id": "task",
        "payload": {
            "generation": {
                **cfg.model_dump(),
                "mode": "translation",
                "prompt_version": "cpt-generation-v1",
            },
            "output_dataset_version_id": "output",
            "generation_input": {
                "runtime_path": str(source),
                "artifact_hash": "source-hash",
                "logical_files": ["text.jsonl"],
            },
            "generation_model": {"runtime_path": str(tmp_path), "identity_hash": "model-hash"},
        },
        "runtime_context": {
            "attempt_dir": str(attempt),
            "project_workspace": str(tmp_path),
            "worker_id": "worker",
        },
    }
    run(task)
    result = next(
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if json.loads(line).get("event") == "result"
    )
    assert result["train_ready"] and result["document_count"] == 1
    assert [a["ref"] for a in result["outputs"]] == [
        "dataset",
        "record_manifest",
        "generation_report",
    ]
    assert result["source_snapshot"]["generation"]["enable_thinking"] is False
