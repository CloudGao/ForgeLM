from __future__ import annotations

import pytest
from llm_platform.schemas import PlanRevisionCreate, TrainingPlanCreate
from pydantic import ValidationError


@pytest.mark.parametrize(
    ("schema", "extra"),
    [(TrainingPlanCreate, {"name": "Plan"}), (PlanRevisionCreate, {})],
)
def test_training_stage_names_reject_utf8_latin1_mojibake(schema, extra) -> None:
    with pytest.raises(ValidationError, match="verify that the JSON client uses UTF-8"):
        schema(
            stages=[{"name": "Gate 0 \u00c2\u00b7 10M \u00e6\u0095\u00b0\u00e6\u008d\u00ae"}],
            **extra,
        )


@pytest.mark.parametrize(
    ("schema", "extra"),
    [(TrainingPlanCreate, {"name": "Plan"}), (PlanRevisionCreate, {})],
)
def test_training_stage_names_accept_unicode(schema, extra) -> None:
    payload = schema(stages=[{"name": "Gate 0 · 10M 数据链路验证"}], **extra)
    assert payload.stages[0]["name"] == "Gate 0 · 10M 数据链路验证"


def test_stage_gate_policy_rejects_reserved_system_validation_ids() -> None:
    with pytest.raises(ValidationError, match="reserved for a system validation"):
        TrainingPlanCreate(
            name="Plan",
            stages=[
                {
                    "name": "Gate 0",
                    "gate_policy": {
                        "transition": "automatic",
                        "data_consumption": "repeat_epochs",
                        "input_rules": [
                            {
                                "id": "tokenizer_locked",
                                "label": "attempted override",
                                "severity": "warning",
                            }
                        ],
                        "output_rules": [],
                    },
                }
            ],
        )


def test_legacy_template_gate_rules_become_editable_and_can_be_deleted() -> None:
    legacy = TrainingPlanCreate(
        name="Plan",
        stages=[
            {
                "name": "Gate 0",
                "gate": {"release": "manual_review", "checks": ["loss_decreases"]},
            }
        ],
    )
    legacy_rule = next(
        rule
        for rule in legacy.stages[0]["gate_policy"]["output_rules"]
        if rule["id"] == "loss_decreases"
    )
    assert legacy_rule["locked"] is False
    assert legacy_rule["source"] == "template"

    revised = PlanRevisionCreate(
        stages=[
            {
                **legacy.stages[0],
                "gate_policy": {
                    **legacy.stages[0]["gate_policy"],
                    "output_rules": [
                        rule
                        for rule in legacy.stages[0]["gate_policy"]["output_rules"]
                        if rule["locked"]
                    ],
                },
            }
        ]
    )
    assert not any(
        rule["id"] == "loss_decreases"
        for rule in revised.stages[0]["gate_policy"]["output_rules"]
    )


def test_native_post_training_requires_complete_versioned_inputs() -> None:
    with pytest.raises(ValidationError, match="requires a supervision mixture"):
        TrainingPlanCreate(
            name="SFT",
            stages=[{"name": "SFT", "type": "sft", "objective": {"schema_version": 1, "kind": "sft"}}],
        )
    with pytest.raises(ValidationError, match="DPO beta must be positive"):
        TrainingPlanCreate(
            name="DPO",
            stages=[
                {
                    "name": "DPO",
                    "type": "dpo",
                    "data_ref_id": "mixture-v1",
                    "chat_template_version_id": "template-v1",
                    "objective": {"schema_version": 1, "kind": "dpo", "beta": 0},
                    "budget": {"unit": "preference_pairs", "value": 10},
                }
            ],
        )


def test_type_only_post_training_stage_remains_legacy_custom_execution() -> None:
    payload = TrainingPlanCreate(name="Legacy SFT", stages=[{"name": "SFT", "type": "sft"}])
    assert payload.stages[0]["type"] == "sft"
