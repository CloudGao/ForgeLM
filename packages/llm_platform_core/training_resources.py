from __future__ import annotations

import copy
from typing import Any

POLICY_SCHEMA = "forgelm.resource-batch-planning.v1"
RESOLVED_SCHEMA = "forgelm.resolved-resource-batch-plan.v1"
PLANNER_VERSION = "worker-static-capacity-v1"

_ALLOWED_VALUES = {
    "micro_batch_size",
    "micro_batch_tokens",
    "gradient_accumulation_steps",
    "packing_strategy",
    "dataloader_num_workers",
    "dataloader_prefetch_factor",
    "dataloader_persistent_workers",
    "dataloader_pin_memory",
    "optimizer_implementation",
}


def freeze_resource_batch_policy(
    stages: list[dict[str, Any]], training_contract: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Freeze an engine-declared policy for the selected training stage."""

    contract = (training_contract or {}).get("resource_batch_planning")
    if contract is None:
        return None
    if not isinstance(contract, dict) or contract.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"training resource batch policy must use {POLICY_SCHEMA}")
    if len(stages) != 1:
        raise ValueError("resource-aware batch planning currently requires one selected stage")
    stage_type = str(stages[0].get("type") or "").strip().lower()
    profiles = contract.get("profiles")
    profile = profiles.get(stage_type) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        raise ValueError(f"training contract has no resource batch profile for {stage_type or '<empty>'}")
    candidates = profile.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"resource batch profile {stage_type} requires candidates")
    frozen_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"resource batch candidate {index} must be an object")
        values = candidate.get("values")
        if not isinstance(values, dict) or not values:
            raise ValueError(f"resource batch candidate {index} requires values")
        unknown = sorted(set(values) - _ALLOWED_VALUES)
        if unknown:
            raise ValueError("unsupported resource batch values: " + ", ".join(unknown))
        minimum_total = int(candidate.get("minimum_gpu_memory_mb") or 0)
        minimum_free = int(candidate.get("minimum_gpu_free_mb") or minimum_total)
        if minimum_total <= 0 or minimum_free <= 0:
            raise ValueError("resource batch candidate GPU memory requirements must be positive")
        frozen_candidates.append(
            {
                "name": str(candidate.get("name") or f"candidate-{index + 1}"),
                "minimum_gpu_memory_mb": minimum_total,
                "minimum_gpu_free_mb": minimum_free,
                "values": copy.deepcopy(values),
            }
        )
    return {
        "schema_version": POLICY_SCHEMA,
        "mode": "auto",
        "profile": stage_type,
        "engine_contract_version": str(contract.get("contract_version") or "1"),
        "candidates": frozen_candidates,
    }


def resolve_resource_batch_plan(
    policy: dict[str, Any],
    resources: dict[str, Any],
    allocation: dict[str, Any],
) -> dict[str, Any]:
    """Select the first declared candidate that fits every allocated GPU."""

    if policy.get("schema_version") != POLICY_SCHEMA:
        raise ValueError(f"resource batch policy must use {POLICY_SCHEMA}")
    allocated = [int(value) for value in allocation.get("gpu_indexes") or []]
    if not allocated:
        raise ValueError("resource-aware training requires an allocated GPU")
    gpu_by_index = {
        int(gpu["index"]): gpu
        for gpu in resources.get("gpus") or []
        if isinstance(gpu, dict) and gpu.get("index") is not None
    }
    missing = [index for index in allocated if index not in gpu_by_index]
    if missing:
        raise ValueError("allocated GPU resources are unavailable: " + ", ".join(map(str, missing)))
    selected_gpus = [gpu_by_index[index] for index in allocated]
    limiting_total = min(int(gpu.get("memory_total_mb") or 0) for gpu in selected_gpus)
    limiting_free = min(int(gpu.get("memory_free_mb") or 0) for gpu in selected_gpus)
    candidates = policy.get("candidates") or []
    selected_index = next(
        (
            index
            for index, candidate in enumerate(candidates)
            if limiting_total >= int(candidate["minimum_gpu_memory_mb"])
            and limiting_free >= int(candidate["minimum_gpu_free_mb"])
        ),
        None,
    )
    if selected_index is None:
        minimum = candidates[-1] if candidates else {}
        raise ValueError(
            "no resource batch candidate fits allocated GPU capacity "
            f"(total={limiting_total} MiB, free={limiting_free} MiB, "
            f"minimum_total={minimum.get('minimum_gpu_memory_mb')} MiB, "
            f"minimum_free={minimum.get('minimum_gpu_free_mb')} MiB)"
        )
    candidate = copy.deepcopy(candidates[selected_index])
    cpu_limit = int(allocation.get("cpus") or resources.get("cpu_count") or 1)
    if "dataloader_num_workers" in candidate["values"]:
        candidate["values"]["dataloader_num_workers"] = min(
            max(0, int(candidate["values"]["dataloader_num_workers"])), max(1, cpu_limit)
        )
    return {
        "schema_version": RESOLVED_SCHEMA,
        "planner_version": PLANNER_VERSION,
        "profile": policy.get("profile"),
        "selected_candidate_index": selected_index,
        "selected": copy.deepcopy(candidate),
        "fallback_candidates": copy.deepcopy(candidates[selected_index + 1 :]),
        "resource_basis": {
            "worker_id": allocation.get("worker_id"),
            "gpu_indexes": allocated,
            "limiting_gpu_memory_mb": limiting_total,
            "limiting_gpu_free_mb": limiting_free,
        },
    }
