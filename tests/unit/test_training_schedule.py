import pytest
from llm_platform_core.training_schedule import (
    freeze_learning_rate_schedules,
    freeze_warmup_schedule,
    resolve_warmup_schedule,
)


def test_ratio_uses_batch_aligned_optimizer_steps_and_preserves_zero():
    stage = {"type": "pretrain", "target_tokens": 15_000_000_000,
             "effective_batch_tokens": 262144, "warmup_ratio": 0.015}
    schedule = freeze_learning_rate_schedules([stage])[0]
    assert schedule["total_steps"] == 57221
    assert schedule["resolved_warmup_steps"] == 859
    assert schedule["warmup_ratio"] == 0.015
    assert freeze_warmup_schedule({"warmup_ratio": 0}, total_steps=100)["resolved_warmup_steps"] == 0


def test_dynamic_sft_budget_is_resolved_by_engine_and_steps_override_ratio():
    stage = {"type": "sft", "budget": {"unit": "examples", "value": 100}, "warmup_ratio": 0.03}
    pending = freeze_learning_rate_schedules([stage])[0]
    assert pending["status"] == "awaiting_optimizer_budget"
    assert pending["resolved_warmup_steps"] is None
    assert resolve_warmup_schedule(pending, 40)["resolved_warmup_steps"] == 2
    assert pending["total_steps"] is None
    explicit = freeze_warmup_schedule({"warmup_ratio": 0.5, "warmup_steps": 7}, total_steps=40)
    assert explicit["resolved_warmup_steps"] == 7
    with pytest.raises(ValueError, match="differs from frozen"):
        resolve_warmup_schedule(explicit, 41)


@pytest.mark.parametrize("values", [
    {"warmup_ratio": -0.1}, {"warmup_ratio": 1}, {"warmup_ratio": float("nan")},
    {"warmup_ratio": True}, {"warmup_steps": -1}, {"warmup_steps": 1.5},
    {"warmup_steps": True}, {"warmup_steps": 101},
])
def test_invalid_schedule_is_rejected(values):
    with pytest.raises(ValueError):
        freeze_warmup_schedule(values, total_steps=100)


def test_step_budget_and_unconfigured_legacy_stages():
    schedules = freeze_learning_rate_schedules([
        {"type": "sft", "budget": {"unit": "steps", "value": 11}, "warmup_ratio": 0.1},
        {"name": "legacy"},
    ])
    assert len(schedules) == 1
    assert schedules[0]["resolved_warmup_steps"] == 2
