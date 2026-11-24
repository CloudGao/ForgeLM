"""Read training curves from their existing task events without rewriting history."""
from __future__ import annotations

import math

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from ..models import TaskEvent


def merge_training_metrics(current: dict | None, final: dict | None) -> dict:
    merged = {**(current or {}), **(final or {})}
    # The terminal summary is not a time-series replacement.
    if (current or {}).get("history"):
        merged["history"] = current["history"]
    return merged


def training_run_metrics_view(session: Session, run: dict, max_points: int = 500) -> dict:
    """Bound response size while covering the task's full recorded step range.

    Sampling uses event order and preserves both endpoints. It is not smoothing
    and may omit intermediate spikes. Original events remain authoritative.
    """
    task_id = run.get("task_id")
    if not task_id:
        return run
    metric = TaskEvent.payload["metrics"]
    filters = (
        TaskEvent.task_id == task_id,
        TaskEvent.event_type == "progress",
        or_(metric["loss"].as_float().is_not(None),
            metric["cross_entropy"].as_float().is_not(None)),
    )
    count = session.scalar(select(func.count()).select_from(TaskEvent).where(*filters)) or 0
    if not count:
        return run
    limit = max(2, max_points)
    ranks = (list(range(1, count + 1)) if count <= limit else
             [1 + i * (count - 1) // (limit - 1) for i in range(limit)])
    ranked = select(
        TaskEvent.payload.label("payload"),
        func.row_number().over(order_by=TaskEvent.sequence).label("rank"),
    ).where(*filters).subquery()
    payloads = session.scalars(
        select(ranked.c.payload).where(ranked.c.rank.in_(ranks)).order_by(ranked.c.rank)
    )
    history = []
    for payload in payloads:
        metrics = payload.get("metrics") or {}
        loss = metrics.get("loss", metrics.get("cross_entropy"))
        step = payload.get("step")
        if (isinstance(loss, bool) or not isinstance(loss, (int, float))
                or not math.isfinite(loss) or not isinstance(step, (int, float))
                or isinstance(step, bool) or not math.isfinite(step)):
            continue
        history.append({"step": step, "tokens": payload.get("tokens"), "loss": loss})
    if not history:
        return run
    return {
        **run,
        "latest_metrics": {
            **(run.get("latest_metrics") or {}),
            "loss": history[-1]["loss"],
            "history": history,
            "history_total_points": count,
            "history_sampled": count > limit,
        },
    }
