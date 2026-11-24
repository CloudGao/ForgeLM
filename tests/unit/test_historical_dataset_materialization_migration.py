from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _migration_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "apps"
        / "api"
        / "alembic"
        / "versions"
        / "c8f3a7d9e205_repair_historical_dataset_materialization.py"
    )
    spec = importlib.util.spec_from_file_location("historical_materialization_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_row() -> dict:
    return {
        "version_id": "version-1",
        "artifact_id": "artifact-1",
        "content_hash": "a" * 64,
        "task_id": "task-1",
        "parent_version_id": None,
        "conversion_recipe_version_id": None,
        "processing_recipe_version_id": None,
        "source_snapshot": {"files": [{"path": "train.parquet", "sha256": "b" * 64}]},
        "materialization_spec": {},
        "source_type": "huggingface",
        "source_uri": "BAAI/COIG",
        "dataset_source_config": {"revision": "main"},
        "task_adapter_id": "builtin.data-import",
        "task_status": "succeeded",
        "task_payload": {
            "dataset_id": "dataset-1",
            "dataset_version_id": "version-1",
            "source_type": "huggingface",
            "source_uri": "BAAI/COIG",
            "source_config": {
                "revision": "9f25758ec94f82762fb9c09a5c60e908cfb83632",
                "commit_hash": "9f25758ec94f82762fb9c09a5c60e908cfb83632",
                "selected_paths": ["train.parquet"],
            },
            "resource_request": {"gpus": 0},
            "command": {},
        },
    }


def test_historical_hf_download_restores_frozen_replay_payload() -> None:
    migration = _migration_module()

    row = _base_row()
    repaired = migration._repair_spec(row)

    assert migration._materialization_kind(row) == "download"
    assert repaired["kind"] == "download"
    assert repaired["replay_adapter_id"] == "builtin.data-import"
    assert repaired["replay_payload"]["source_config"]["commit_hash"] == (
        "9f25758ec94f82762fb9c09a5c60e908cfb83632"
    )
    assert repaired["expected_content_hash"] == "a" * 64
    assert repaired["expected_snapshot_files"][0]["path"] == "train.parquet"


def test_historical_derived_version_restores_recipe_task() -> None:
    migration = _migration_module()
    row = _base_row()
    row.update(
        {
            "source_type": "local",
            "parent_version_id": "parent-version",
            "conversion_recipe_version_id": "recipe-version",
            "task_adapter_id": "builtin.data-conversion",
            "task_payload": {
                "dataset_version_id": "version-1",
                "data_conversion_input": {"artifact_id": "parent-artifact"},
                "recipe": {"operators": [{"normalize_whitespace": {}}]},
            },
        }
    )

    repaired = migration._repair_spec(row)

    assert migration._materialization_kind(row) == "derived"
    assert repaired["replay_adapter_id"] == "builtin.data-conversion"
    assert repaired["replay_payload"]["recipe"]["operators"]
    assert repaired["parent_dataset_version_id"] == "parent-version"


def test_existing_replay_spec_is_not_removed_when_history_task_is_missing() -> None:
    migration = _migration_module()
    row = _base_row()
    frozen_payload = row["task_payload"]
    row.update(
        {
            "task_id": None,
            "task_adapter_id": None,
            "task_status": None,
            "task_payload": {},
            "materialization_spec": {
                "schema_version": 1,
                "kind": "download",
                "replay_adapter_id": "builtin.data-import",
                "replay_payload": frozen_payload,
            },
        }
    )

    repaired = migration._repair_spec(row)

    assert repaired["replay_adapter_id"] == "builtin.data-import"
    assert repaired["replay_payload"] == frozen_payload
