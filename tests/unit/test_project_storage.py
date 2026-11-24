from __future__ import annotations

from pathlib import Path

import pytest
from llm_platform_worker.project_storage import (
    project_run_root,
    project_training_run_root,
    scoped_runtime_path,
    worker_project_workspace,
)


def test_worker_run_root_is_scoped_to_project_workspace(tmp_path: Path) -> None:
    expected = tmp_path / ".llm-platform" / "runs"

    assert project_run_root(str(tmp_path)) == expected.resolve()
    assert project_run_root(tmp_path.as_uri()) == expected.resolve()
    assert expected.is_dir()


def test_worker_training_run_root_is_scoped_to_project_workspace(tmp_path: Path) -> None:
    expected = tmp_path / ".llm-platform" / "training-runs"

    assert project_training_run_root(str(tmp_path)) == expected.resolve()
    assert project_training_run_root(tmp_path.as_uri()) == expected.resolve()
    assert expected.is_dir()


def test_worker_run_root_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not accessible"):
        project_run_root(str(tmp_path / "missing"))

    with pytest.raises(ValueError, match="no configured workspace"):
        project_run_root("")

    with pytest.raises(ValueError, match="not accessible"):
        project_training_run_root(str(tmp_path / "missing"))


def test_scoped_runtime_path_rejects_directory_escape(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()

    assert scoped_runtime_path(root, "run-id", "attempt-id") == (
        root / "run-id" / "attempt-id"
    ).resolve()
    with pytest.raises(ValueError, match="escapes"):
        scoped_runtime_path(root, "..", "outside")


def test_worker_local_root_does_not_require_control_plane_path(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker"

    workspace = worker_project_workspace(
        "project-id",
        workspace_root=worker_root,
        legacy_workspace_uri="Z:/not-mounted/control-plane-project",
    )

    assert workspace == (worker_root / "projects" / "project-id").resolve()
    assert workspace.is_dir()


def test_worker_local_root_rejects_project_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        worker_project_workspace(
            "../outside",
            workspace_root=tmp_path / "worker",
            legacy_workspace_uri="ignored",
        )
