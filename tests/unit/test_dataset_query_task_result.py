from __future__ import annotations

import pytest
from llm_platform.models import Task, TaskAttempt
from llm_platform.services.tasks import finalize_task_domain


def _task() -> Task:
    return Task(
        project_id="project-1",
        kind="data_query",
        title="viewer",
        adapter_id="builtin.dataset-viewer",
        launcher_id="builtin.local-process",
        payload={},
    )


def test_data_query_result_stays_ephemeral_without_report_artifact() -> None:
    task = _task()
    result = {"columns": [{"name": "text", "type": "VARCHAR"}], "rows": [["hello"]]}

    finalize_task_domain(None, task, TaskAttempt(), result)  # type: ignore[arg-type]

    assert task.result == result
    assert "artifact_id" not in task.result


def test_data_query_result_has_a_control_plane_size_limit() -> None:
    task = _task()

    with pytest.raises(ValueError, match="10 MiB"):
        finalize_task_domain(  # type: ignore[arg-type]
            None,
            task,
            TaskAttempt(),
            {"rows": [["x" * (10 * 1024 * 1024)]]},
        )
