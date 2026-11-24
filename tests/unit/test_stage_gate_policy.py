from __future__ import annotations

import pytest
from fastapi import HTTPException
from llm_platform.routers.training import _post_training_configuration, _tokenized_requirements
from llm_platform.stage_gates import normalize_stage_gate_policy


def _stage(mode: str) -> dict:
    return {
        "name": "pretrain",
        "data_ref_id": "mixture-1",
        "target_tokens": 12_000,
        "cumulative_target_tokens": 12_000,
        "sequence_length": 100,
        "effective_batch_tokens": 1_000,
        "gate_policy": {
            "transition": "automatic",
            "data_consumption": mode,
            "input_rules": [],
            "output_rules": [],
        },
    }


def test_unique_data_policy_requires_the_full_training_range() -> None:
    requirement = _tokenized_requirements([_stage("unique_only")])[0]
    assert requirement["coverage_mode"] == "unique_only"
    assert requirement["required_tokens"] == 12_000


def test_repeat_epochs_policy_only_requires_a_nonempty_compatible_artifact() -> None:
    requirement = _tokenized_requirements([_stage("repeat_epochs")])[0]
    assert requirement["coverage_mode"] == "repeat_epochs"
    assert requirement["required_tokens"] == 100


def test_system_validation_ids_cannot_be_reintroduced_as_user_gates() -> None:
    stage = _stage("unique_only")
    stage["gate_policy"]["input_rules"] = [
        {
            "id": "resume_checkpoint_compatible",
            "label": "pretend this is configurable",
            "kind": "named_check",
            "source": "user",
        }
    ]

    with pytest.raises(ValueError, match="reserved for a system validation"):
        normalize_stage_gate_policy(stage)


def test_native_post_training_runs_one_objective_stage_at_a_time() -> None:
    stage = {
        "name": "SFT",
        "type": "sft",
        "data_ref_id": "supervision-mixture-version-1",
        "chat_template_version_id": "chat-template-version-1",
        "objective": {"schema_version": 1, "kind": "sft", "loss_scope": "assistant_only"},
        "budget": {"unit": "supervised_tokens", "value": 1024},
    }

    with pytest.raises(HTTPException) as error:
        _post_training_configuration([stage, {**stage, "name": "SFT continuation"}])

    assert error.value.status_code == 422
    assert error.value.detail["code"] == "native_post_training_requires_single_stage"
