from __future__ import annotations

import math
from typing import Any

WARMUP_SCHEMA = "forgelm.training-warmup.v1"


def resolve_warmup_schedule(schedule: dict[str, Any], total_steps: int) -> dict[str, Any]:
    """Resolve a frozen policy against the actual optimizer budget, rejecting drift."""
    if schedule.get("schema_version") != WARMUP_SCHEMA:
        raise ValueError("unsupported training warmup contract")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("warmup requires positive integer optimizer steps")
    expected = schedule.get("total_steps")
    if expected is not None and expected != total_steps:
        raise ValueError("optimizer step budget differs from frozen warmup schedule")
    ratio, steps = _warmup_values(schedule)
    resolved = steps if steps > 0 else math.ceil(total_steps * ratio)
    if resolved > total_steps:
        raise ValueError("warmup steps exceed the optimizer step budget")
    return {
        **schedule,
        "total_steps": total_steps,
        "resolved_warmup_steps": resolved,
        "status": "resolved",
    }


def _warmup_values(stage: dict[str, Any]) -> tuple[float, int]:
    raw_ratio = stage.get("warmup_ratio")
    raw_steps = stage.get("warmup_steps")
    if isinstance(raw_ratio, bool) or isinstance(raw_steps, bool):
        raise ValueError("warmup ratio and steps must be numeric, not boolean")
    ratio = float(raw_ratio) if raw_ratio is not None else 0.0
    steps = int(raw_steps) if raw_steps is not None else 0
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("warmup_ratio must be finite and in [0, 1)")
    if raw_steps is not None and (steps < 0 or steps != float(raw_steps)):
        raise ValueError("warmup_steps must be a non-negative integer")
    return ratio, steps


def freeze_warmup_schedule(
    stage: dict[str, Any], *, total_steps: int | None = None, stage_index: int = 0
) -> dict[str, Any]:
    ratio, steps = _warmup_values(stage)
    frozen = {
        "schema_version": WARMUP_SCHEMA,
        "stage_index": stage_index,
        "stage_name": str(stage.get("name") or ""),
        "warmup_ratio": ratio,
        "warmup_steps": steps,
        "total_steps": total_steps,
        "resolved_warmup_steps": None,
        "status": "awaiting_optimizer_budget",
    }
    return resolve_warmup_schedule(frozen, total_steps) if total_steps is not None else frozen


def freeze_learning_rate_schedules(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schedules = []
    for index, stage in enumerate(stages):
        if stage.get("warmup_ratio") is None and stage.get("warmup_steps") is None:
            continue
        total_steps = None
        objective = str(stage.get("type") or "pretrain").lower()
        budget = stage.get("budget") or {}
        if objective in {"sft", "dpo"}:
            if budget.get("unit") == "steps" and int(budget.get("value") or 0) > 0:
                total_steps = int(budget["value"])
        else:
            target = int(stage.get("target_tokens") or 0)
            batch = int(stage.get("effective_batch_tokens") or 0)
            if target > 0 and batch > 0:
                total_steps = (target + batch - 1) // batch
            elif int(stage.get("target_steps") or 0) > 0:
                total_steps = int(stage["target_steps"])
        schedules.append(freeze_warmup_schedule(stage, total_steps=total_steps, stage_index=index))
    return schedules
