from __future__ import annotations

import asyncio

import pytest
from llm_platform_core.checkpoint_compatibility import (
    build_checkpoint_compatibility,
    compare_checkpoint_compatibility,
)
from llm_platform_worker.main import AgentConfig, WorkerAgent


def _descriptor(**changes: object) -> dict:
    configuration = {
        "hidden_size": 960,
        "num_hidden_layers": 28,
        "use_cache": True,
        "runtime": {"backend": "auto"},
    }
    configuration.update(changes)
    return build_checkpoint_compatibility(
        configuration,
        contract="example-model",
        runtime_only_fields=["runtime.backend", "use_cache"],
    )


def test_runtime_only_differences_are_visible_but_do_not_block() -> None:
    expected = _descriptor()
    actual = _descriptor(use_cache=False, runtime={"backend": "eager"})

    report = compare_checkpoint_compatibility(expected, actual)

    assert report["compatible"] is True
    assert report["differences"] == []
    assert {item["field"] for item in report["runtime_differences"]} == {
        "runtime.backend",
        "use_cache",
    }


def test_structural_difference_blocks_with_the_exact_field() -> None:
    report = compare_checkpoint_compatibility(
        _descriptor(),
        _descriptor(hidden_size=1024),
    )

    assert report["compatible"] is False
    assert report["reason"] == "configuration_mismatch"
    assert report["differences"] == [
        {
            "field": "hidden_size",
            "expected_present": True,
            "actual_present": True,
            "expected": 960,
            "actual": 1024,
        }
    ]


def test_contract_policy_cannot_be_changed_by_the_checkpoint() -> None:
    expected = _descriptor()
    actual = build_checkpoint_compatibility(
        expected["configuration"],
        contract="example-model",
        runtime_only_fields=["hidden_size", "runtime.backend", "use_cache"],
    )

    report = compare_checkpoint_compatibility(expected, actual)

    assert report["compatible"] is False
    assert report["reason"] == "contract_mismatch"
    assert report["differences"][0]["field"] == "runtime_only_fields"


def test_tampered_descriptor_is_rejected() -> None:
    actual = _descriptor()
    actual["configuration"]["hidden_size"] = 1024

    report = compare_checkpoint_compatibility(_descriptor(), actual)

    assert report["compatible"] is False
    assert report["reason"] == "invalid_actual_signature"


def test_worker_rechecks_the_frozen_contract_before_launching_user_code() -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    expected = _descriptor()
    actual = _descriptor(use_cache=False)
    task = {
        "payload": {
            "checkpoint_compatibility": expected,
            "resume_checkpoint": {"model_compatibility": actual},
        }
    }
    asyncio.run(worker.validate_training_checkpoint_compatibility(task, emit))

    assert events == [
        (
            "system.validation",
            {
                "validation_id": "resume_checkpoint_compatible",
                "status": "passed",
                "phase": "worker",
                "message": "checkpoint model is compatible with the plan",
                "reason": "compatible",
                "differences": [],
                "runtime_differences": [
                    {
                        "field": "use_cache",
                        "expected_present": True,
                        "actual_present": True,
                        "expected": True,
                        "actual": False,
                    }
                ],
            },
        )
    ]


def test_worker_blocks_structural_checkpoint_drift() -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))

    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    task = {
        "payload": {
            "checkpoint_compatibility": _descriptor(),
            "resume_checkpoint": {
                "model_compatibility": _descriptor(hidden_size=1024)
            },
        }
    }
    with pytest.raises(ValueError, match="hidden_size"):
        asyncio.run(worker.validate_training_checkpoint_compatibility(task, emit))


def test_native_post_training_refuses_missing_frozen_start_checkpoint() -> None:
    task = {
        "kind": "train",
        "payload": {
            "input_model_ref": {"type": "checkpoint_finetune", "checkpoint_id": "cp-1"},
            "supervision_training_view_inputs": [{"version_id": "view-1"}],
        },
    }

    with pytest.raises(ValueError, match="refusing random initialization"):
        WorkerAgent.validate_training_initialization_binding(task)

    task["payload"]["resume_checkpoint"] = {"checkpoint_id": "cp-1"}
    WorkerAgent.validate_training_initialization_binding(task)
