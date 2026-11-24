"""reconcile stage-local checkpoint token counts

Revision ID: 6a8f1c2d4e90
Revises: 3c6f4a2e8b10
Create Date: 2026-08-10 13:30:00
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "6a8f1c2d4e90"
down_revision: str | Sequence[str] | None = "3c6f4a2e8b10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MARKER = "stage_local_tokens_reconciled_v1"


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        parsed = json.loads(value or "{}")
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _database_value(value: dict[str, Any], original: Any) -> Any:
    return json.dumps(value, ensure_ascii=False) if isinstance(original, str) else value


def upgrade() -> None:
    bind = op.get_bind()
    tasks = sa.table(
        "tasks",
        sa.column("id", sa.String()),
        sa.column("kind", sa.String()),
        sa.column("observed_state", sa.String()),
        sa.column("payload", sa.JSON()),
        sa.column("result", sa.JSON()),
    )
    runs = sa.table(
        "training_runs",
        sa.column("id", sa.String()),
        sa.column("task_id", sa.String()),
        sa.column("current_tokens", sa.Integer()),
    )
    checkpoints = sa.table(
        "checkpoints",
        sa.column("id", sa.String()),
        sa.column("training_run_id", sa.String()),
        sa.column("token_count", sa.Integer()),
    )

    rows = bind.execute(
        sa.select(
            tasks.c.id.label("task_id"),
            tasks.c.payload,
            tasks.c.result,
            runs.c.id.label("run_id"),
            runs.c.current_tokens,
        )
        .select_from(tasks.join(runs, runs.c.task_id == tasks.c.id))
        .where(tasks.c.kind == "train", tasks.c.observed_state == "succeeded")
    ).mappings().all()
    for row in rows:
        payload = _json_object(row["payload"])
        result = _json_object(row["result"])
        resume = payload.get("resume_checkpoint")
        stages = payload.get("stages")
        if not isinstance(resume, dict) or not isinstance(stages, list):
            continue
        resume_tokens = int(resume.get("token_count") or 0)
        stage_tokens = sum(
            int(stage.get("target_tokens") or 0)
            for stage in stages
            if isinstance(stage, dict)
        )
        if (
            resume_tokens <= 0
            or stage_tokens <= 0
            or int(result.get("tokens") or 0) != stage_tokens
        ):
            continue

        cumulative_tokens = resume_tokens + stage_tokens
        checkpoint_rows = bind.execute(
            sa.select(checkpoints.c.id, checkpoints.c.token_count).where(
                checkpoints.c.training_run_id == row["run_id"]
            )
        ).mappings().all()
        for checkpoint_row in checkpoint_rows:
            local_tokens = int(checkpoint_row["token_count"] or 0)
            if 0 <= local_tokens <= stage_tokens:
                bind.execute(
                    checkpoints.update()
                    .where(checkpoints.c.id == checkpoint_row["id"])
                    .values(token_count=resume_tokens + local_tokens)
                )

        result["tokens"] = cumulative_tokens
        raw_checkpoints = result.get("checkpoints")
        if isinstance(raw_checkpoints, list):
            for checkpoint in raw_checkpoints:
                if not isinstance(checkpoint, dict):
                    continue
                key = (
                    "tokens"
                    if checkpoint.get("tokens") is not None
                    else "token_count"
                    if checkpoint.get("token_count") is not None
                    else None
                )
                if key is not None:
                    checkpoint[key] = resume_tokens + int(checkpoint[key] or 0)
        result[_MARKER] = {
            "resume_tokens": resume_tokens,
            "stage_tokens": stage_tokens,
        }
        bind.execute(
            tasks.update()
            .where(tasks.c.id == row["task_id"])
            .values(result=_database_value(result, row["result"]))
        )
        if int(row["current_tokens"] or 0) == stage_tokens:
            bind.execute(
                runs.update()
                .where(runs.c.id == row["run_id"])
                .values(current_tokens=cumulative_tokens)
            )


def downgrade() -> None:
    bind = op.get_bind()
    tasks = sa.table(
        "tasks",
        sa.column("id", sa.String()),
        sa.column("payload", sa.JSON()),
        sa.column("result", sa.JSON()),
    )
    runs = sa.table(
        "training_runs",
        sa.column("id", sa.String()),
        sa.column("task_id", sa.String()),
        sa.column("current_tokens", sa.Integer()),
    )
    checkpoints = sa.table(
        "checkpoints",
        sa.column("id", sa.String()),
        sa.column("training_run_id", sa.String()),
        sa.column("token_count", sa.Integer()),
    )
    rows = bind.execute(
        sa.select(
            tasks.c.id.label("task_id"),
            tasks.c.result,
            runs.c.id.label("run_id"),
            runs.c.current_tokens,
        ).select_from(tasks.join(runs, runs.c.task_id == tasks.c.id))
    ).mappings().all()
    for row in rows:
        result = _json_object(row["result"])
        marker = result.get(_MARKER)
        if not isinstance(marker, dict):
            continue
        resume_tokens = int(marker.get("resume_tokens") or 0)
        stage_tokens = int(marker.get("stage_tokens") or 0)
        if resume_tokens <= 0 or stage_tokens <= 0:
            continue
        checkpoint_rows = bind.execute(
            sa.select(checkpoints.c.id, checkpoints.c.token_count).where(
                checkpoints.c.training_run_id == row["run_id"]
            )
        ).mappings().all()
        for checkpoint_row in checkpoint_rows:
            cumulative = int(checkpoint_row["token_count"] or 0)
            if resume_tokens <= cumulative <= resume_tokens + stage_tokens:
                bind.execute(
                    checkpoints.update()
                    .where(checkpoints.c.id == checkpoint_row["id"])
                    .values(token_count=cumulative - resume_tokens)
                )
        result["tokens"] = stage_tokens
        raw_checkpoints = result.get("checkpoints")
        if isinstance(raw_checkpoints, list):
            for checkpoint in raw_checkpoints:
                if not isinstance(checkpoint, dict):
                    continue
                key = (
                    "tokens"
                    if checkpoint.get("tokens") is not None
                    else "token_count"
                    if checkpoint.get("token_count") is not None
                    else None
                )
                if key is not None:
                    checkpoint[key] = max(0, int(checkpoint[key] or 0) - resume_tokens)
        result.pop(_MARKER, None)
        bind.execute(
            tasks.update()
            .where(tasks.c.id == row["task_id"])
            .values(result=_database_value(result, row["result"]))
        )
        if int(row["current_tokens"] or 0) == resume_tokens + stage_tokens:
            bind.execute(
                runs.update()
                .where(runs.c.id == row["run_id"])
                .values(current_tokens=stage_tokens)
            )
