from __future__ import annotations

from copy import deepcopy
from typing import Any

from .training_system_validations import TRAINING_SYSTEM_VALIDATIONS

TRANSITIONS = {"automatic", "manual"}
DATA_CONSUMPTION_MODES = {"unique_only", "repeat_epochs"}
RULE_SEVERITIES = {"blocker", "warning", "info"}
FAILURE_ACTIONS = {"stop", "notify_and_continue", "record_only"}
RULE_KINDS = {"named_check", "metric_threshold", "artifact_required", "manual_approval"}
OPERATORS = {"exists", "eq", "ne", "gt", "gte", "lt", "lte"}

# V1 stored platform invariants as locked gate rules. V2 reports those checks as
# system validations and removes them from editable user policy.
LEGACY_SYSTEM_RULE_IDS = {
    "data_mixture_ready",
    "tokenizer_locked",
    "tokenizer_model_compatible",
    "validation_input_ready",
    "resume_checkpoint_compatible",
    "worker_capability_available",
    "tokenized_dataset_integrity",
    "finite_training_state",
    "output_artifact_integrity",
}
RESERVED_SYSTEM_RULE_IDS = LEGACY_SYSTEM_RULE_IDS | {
    str(validation["id"]) for validation in TRAINING_SYSTEM_VALIDATIONS
}


def _legacy_transition(stage: dict[str, Any]) -> str:
    release = str((stage.get("gate") or {}).get("release") or "")
    return "manual" if release in {"manual_review", "final_acceptance"} else "automatic"


def _legacy_output_rules(stage: dict[str, Any]) -> list[dict[str, Any]]:
    checks = (stage.get("gate") or {}).get("checks") or []
    return [
        {
            "id": str(check),
            "label": str(check).replace("_", " "),
            "kind": "named_check",
            "source": "template",
            "locked": False,
            "enabled": True,
            "severity": "blocker",
            "failure_action": "stop",
            "operator": "exists",
        }
        for check in checks
        if str(check).strip()
    ]


def _normalize_rule(raw: Any, *, index: int, phase: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{phase} gate rule {index + 1} must be an object")
    rule = deepcopy(raw)
    rule_id = str(rule.get("id") or "").strip()
    if not rule_id:
        raise ValueError(f"{phase} gate rule {index + 1} must have an id")
    kind = str(rule.get("kind") or "named_check")
    if kind not in RULE_KINDS:
        raise ValueError(f"unsupported gate rule kind: {kind}")
    severity = str(rule.get("severity") or "blocker")
    if severity not in RULE_SEVERITIES:
        raise ValueError(f"unsupported gate rule severity: {severity}")
    failure_action = str(rule.get("failure_action") or "stop")
    if failure_action not in FAILURE_ACTIONS:
        raise ValueError(f"unsupported gate failure action: {failure_action}")
    operator = str(rule.get("operator") or "exists")
    if operator not in OPERATORS:
        raise ValueError(f"unsupported gate rule operator: {operator}")
    return {
        **rule,
        "id": rule_id,
        "label": str(rule.get("label") or rule_id.replace("_", " ")).strip(),
        "kind": kind,
        "source": "template" if rule.get("source") == "template" else "user",
        "locked": False,
        "enabled": bool(rule.get("enabled", True)),
        "severity": severity,
        "failure_action": failure_action,
        "operator": operator,
    }


def _normalize_rules(raw_rules: Any, *, phase: str) -> list[dict[str, Any]]:
    if raw_rules is None:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise ValueError(f"{phase}_rules must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rules):
        if isinstance(raw, dict) and (
            raw.get("source") == "system"
            or (
                raw.get("locked") is True
                and str(raw.get("id") or "") in LEGACY_SYSTEM_RULE_IDS
            )
        ):
            continue
        if isinstance(raw, dict) and str(raw.get("id") or "") in RESERVED_SYSTEM_RULE_IDS:
            raise ValueError(
                f"{phase} gate rule id is reserved for a system validation: {raw.get('id')}"
            )
        rule = _normalize_rule(raw, index=index, phase=phase)
        if rule["id"] in seen:
            raise ValueError(f"duplicate {phase} gate rule id: {rule['id']}")
        seen.add(rule["id"])
        normalized.append(rule)
    return normalized


def normalize_stage_gate_policy(stage: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(stage)
    raw_policy = normalized.get("gate_policy")
    if raw_policy is None:
        raw_policy = {}
    if not isinstance(raw_policy, dict):
        raise ValueError("gate_policy must be an object")
    transition = str(raw_policy.get("transition") or _legacy_transition(normalized))
    if transition not in TRANSITIONS:
        raise ValueError(f"unsupported stage transition: {transition}")
    data_consumption = str(raw_policy.get("data_consumption") or "unique_only")
    if data_consumption not in DATA_CONSUMPTION_MODES:
        raise ValueError(f"unsupported data consumption mode: {data_consumption}")
    output_rules = raw_policy.get("output_rules")
    if output_rules is None:
        output_rules = _legacy_output_rules(normalized)
    normalized["gate_policy"] = {
        **raw_policy,
        "schema_version": 2,
        "transition": transition,
        "data_consumption": data_consumption,
        "input_rules": _normalize_rules(raw_policy.get("input_rules"), phase="input"),
        "output_rules": _normalize_rules(output_rules, phase="output"),
    }
    normalized.pop("gate", None)
    return normalized


def normalize_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_stage_gate_policy(stage) for stage in stages]


def data_consumption_mode(stage: dict[str, Any]) -> str:
    return normalize_stage_gate_policy(stage)["gate_policy"]["data_consumption"]
