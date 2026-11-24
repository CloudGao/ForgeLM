from __future__ import annotations

import pytest
from llm_platform_core.training_resources import (
    POLICY_SCHEMA,
    RESOLVED_SCHEMA,
    freeze_resource_batch_policy,
    resolve_resource_batch_plan,
)


def policy_contract() -> dict:
    return {
        "resource_batch_planning": {
            "schema_version": POLICY_SCHEMA,
            "contract_version": 1,
            "profiles": {
                "sft": {
                    "candidates": [
                        {
                            "name": "16k",
                            "minimum_gpu_memory_mb": 12000,
                            "minimum_gpu_free_mb": 9500,
                            "values": {
                                "micro_batch_tokens": 16384,
                                "micro_batch_size": 8,
                                "gradient_accumulation_steps": 2,
                                "packing_strategy": "token_budget_v1",
                            },
                        },
                        {
                            "name": "8k",
                            "minimum_gpu_memory_mb": 8000,
                            "minimum_gpu_free_mb": 6500,
                            "values": {
                                "micro_batch_tokens": 8192,
                                "micro_batch_size": 4,
                                "gradient_accumulation_steps": 4,
                                "packing_strategy": "token_budget_v1",
                            },
                        },
                    ]
                }
            },
        }
    }


def test_freeze_and_resolve_resource_batch_policy_uses_allocated_gpu_capacity() -> None:
    policy = freeze_resource_batch_policy([{"type": "sft"}], policy_contract())
    assert policy is not None
    resolved = resolve_resource_batch_plan(
        policy,
        {
            "gpus": [
                {"index": 0, "memory_total_mb": 12282, "memory_free_mb": 10000},
                {"index": 1, "memory_total_mb": 24564, "memory_free_mb": 22000},
            ]
        },
        {"worker_id": "worker-1", "gpu_indexes": [0]},
    )
    assert resolved["schema_version"] == RESOLVED_SCHEMA
    assert resolved["selected"]["name"] == "16k"
    assert resolved["selected"]["values"]["micro_batch_tokens"] == 16384
    assert resolved["resource_basis"]["limiting_gpu_free_mb"] == 10000
    assert [item["name"] for item in resolved["fallback_candidates"]] == ["8k"]


def test_resource_batch_policy_falls_back_deterministically() -> None:
    policy = freeze_resource_batch_policy([{"type": "sft"}], policy_contract())
    assert policy is not None
    resolved = resolve_resource_batch_plan(
        policy,
        {"gpus": [{"index": 0, "memory_total_mb": 10000, "memory_free_mb": 7000}]},
        {"gpu_indexes": [0]},
    )
    assert resolved["selected"]["name"] == "8k"
    assert resolved["selected_candidate_index"] == 1


def test_resource_batch_policy_rejects_insufficient_capacity() -> None:
    policy = freeze_resource_batch_policy([{"type": "sft"}], policy_contract())
    assert policy is not None
    with pytest.raises(ValueError, match="no resource batch candidate fits"):
        resolve_resource_batch_plan(
            policy,
            {"gpus": [{"index": 0, "memory_total_mb": 6000, "memory_free_mb": 5000}]},
            {"gpu_indexes": [0]},
        )


def test_resource_batch_policy_rejects_unknown_engine_values() -> None:
    contract = policy_contract()
    contract["resource_batch_planning"]["profiles"]["sft"]["candidates"][0][
        "values"
    ]["private_magic"] = True
    with pytest.raises(ValueError, match="unsupported resource batch values"):
        freeze_resource_batch_policy([{"type": "sft"}], contract)
