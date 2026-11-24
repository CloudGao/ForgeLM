import asyncio
import re
import sys
from types import SimpleNamespace

import pytest
from llm_platform_core.data_generation import factual_surface_anchors
from llm_platform_worker.data_generation_runtime import (
    GenerationDiagnostics,
    GenerationRateWindow,
    chinese_sampling_constraint,
    log_vllm_stats,
    render_generation_request,
    select_continuous_window,
)


def test_chinese_guard_is_input_scoped_and_blocks_latin_before_decode(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(
        StructuredOutputsParams=lambda **kw: SimpleNamespace(**kw)))
    item = {"quality_anchors": {"chinese_only": True}}
    regex = chinese_sampling_constraint(item)["structured_outputs"].regex
    assert re.fullmatch(regex, "组件对象模型由微软开发。2026年。")
    assert re.fullmatch(regex, "组件对象模型（COM）。") is None
    assert chinese_sampling_constraint(item, enabled=False) == {}
    assert chinese_sampling_constraint({"quality_anchors": {"latin": ["COM"]}}) == {}
    anchors = factual_surface_anchors('[{"主体":"darest","关系":"词语由其派生","客体":"dare"}]', "facts")
    with pytest.raises(ValueError, match="不能硬译"):
        chinese_sampling_constraint({"quality_anchors": anchors})
    assert chinese_sampling_constraint({"quality_anchors": anchors}, enabled=False) == {}


def test_retry_feedback_preserves_facts_and_drops_only_optional_context():
    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            return "|".join(row["content"] for row in messages)

        @staticmethod
        def encode(text, **kwargs):
            return list(text)

    item = {"prompt": "完整事实：建于2001年。", "quality_anchors": {"numbers": ["2001"]}}
    rendered, _ = render_generation_request(Tokenizer(), item, SimpleNamespace(max_input_tokens=1000),
                                            "建于2002年。", "number_mismatch")
    assert "2001" in rendered and "2002" in rendered and "不新增或遗漏数字" in rendered
    rendered, _ = render_generation_request(Tokenizer(), item, SimpleNamespace(max_input_tokens=20),
                                            "很长的错误草稿" * 200, "number_mismatch")
    assert rendered == item["prompt"]
    item = {"prompt": "甲某的子女包括乙某。", "quality_anchors": {
        "obvious_relation_checks": [{"expected": "甲某 — 子女包括 — 乙某"}]}}
    rendered, _ = render_generation_request(Tokenizer(), item, SimpleNamespace(max_input_tokens=1000),
                                            "甲某是乙某的子女。", "obvious_relation_contradiction")
    assert "甲某 — 子女包括 — 乙某" in rendered and "关系方向写反" in rendered


def test_record_diagnostics_are_bounded_but_counts_are_exact(monkeypatch):
    clock = [0]
    monkeypatch.setattr("llm_platform_worker.data_generation_runtime.time.perf_counter", lambda: clock[0])
    emitted = []
    diagnostics = GenerationDiagnostics(lambda *args, **kwargs: emitted.append(kwargs))
    for _ in range(1000):
        diagnostics.emit("generation.record_skipped", reason="validation")
    assert len(emitted) == 1
    assert diagnostics.counts["generation.record_skipped"] == 1000
    clock[0] = 30
    diagnostics.emit("generation.record_skipped", reason="validation")
    assert emitted[-1]["total_occurrences"] == 1001


def test_recent_throughput_excludes_resumed_rows_and_exposes_slowdown():
    meter = GenerationRateWindow(0, records=19756, seconds=120)
    meter.update(60, 60000, 19936)
    assert meter.update(120, 120000, 20116)["recent_tokens_per_second"] == 1000
    meter.update(180, 126000, 20134)
    slow = meter.update(240, 132000, 20152)
    assert slow == {
        "recent_tokens_per_second": 100,
        "recent_documents_per_second": 0.3,
        "rate_window_seconds": 120,
    }


@pytest.mark.asyncio
async def test_stats_continue_without_completions_and_survive_collection_error(monkeypatch):
    calls, logs = [], []
    observed = asyncio.Event()

    async def stats():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("stats failure")
        observed.set()

    monkeypatch.setattr("llm_platform_worker.data_generation_runtime.emit",
                        lambda *args, **kwargs: logs.append(kwargs))
    task = asyncio.create_task(log_vllm_stats(SimpleNamespace(do_log_stats=stats), interval=0.001))
    try:
        await asyncio.wait_for(observed.wait(), timeout=1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert len(calls) >= 2
    assert logs == [{"message": "vLLM statistics unavailable: RuntimeError"}]


@pytest.mark.asyncio
async def test_continuous_tuner_measures_effective_output_and_keeps_noisy_baseline():
    async def trial(window):
        return {"accepted_tokens_per_second": {8: 70, 16: 140, 32: 105, 48: 102, 128: 100}[window],
                "raw_tokens_per_second": 2000 if window == 48 else 1000}

    report = {}
    assert await select_continuous_window(48, trial, report) == 16
    assert report["selection_metric"] == "accepted_tokens_per_second"
    assert len([p for p in report["probes"] if p["submission_window"] == 128]) == 2

    async def noisy(window):
        return {"accepted_tokens_per_second": 103 if window == 16 else 100,
                "raw_tokens_per_second": 1000}

    assert await select_continuous_window(48, noisy, {}) == 128


@pytest.mark.asyncio
async def test_continuous_tuner_does_not_hide_dead_engine():
    async def trial(window):
        raise RuntimeError("engine died")

    with pytest.raises(RuntimeError, match="engine died"):
        await select_continuous_window(48, trial, {})


@pytest.mark.asyncio
async def test_probe_timeout_is_explicit_and_preserves_known_baseline():
    async def trial(window):
        raise TimeoutError()

    report = {}
    assert await select_continuous_window(48, trial, report) == 128
    assert report["selection"] == "vllm_probe_timeout_baseline"
    assert report["probes"] == [{"submission_window": 128, "status": "timeout"}]


@pytest.mark.asyncio
async def test_qa_tuner_starts_one_and_confirms_winner():
    calls = []
    async def trial(window):
        calls.append(window)
        return {"accepted_tokens_per_second": 0, "raw_tokens_per_second": 100 if window == 2 else 50}
    report = {}
    assert await select_continuous_window(4, trial, report, start_small=True) == 2
    assert calls == [1, 2, 4, 4, 2, 1]
    assert report["selected_batch_size"] == 2


@pytest.mark.asyncio
async def test_capacity_failure_rolls_back_then_measures_downward():
    calls = []
    async def trial(window):
        calls.append(window)
        if window == 4:
            raise RuntimeError("Engine core initialization failed")
        return {"accepted_tokens_per_second": 0, "raw_tokens_per_second": 100 * window}
    report = {}
    assert await select_continuous_window(8, trial, report, start_small=True) == 2
    assert calls == [1, 2, 4, 2, 1]
    assert report["highest_verified_capacity"] == 2
    assert report["probes"][2]["status"] == "failed"


@pytest.mark.asyncio
async def test_capacity_one_failure_does_not_claim_success():
    async def trial(window):
        raise RuntimeError("cannot initialize")
    with pytest.raises(RuntimeError, match="batch 1"):
        await select_continuous_window(8, trial, {}, start_small=True)
