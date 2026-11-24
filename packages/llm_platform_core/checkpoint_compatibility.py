"""Versioned, field-level compatibility contracts for checkpoint continuation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

SCHEMA_VERSION = "forgelm.checkpoint-compatibility.v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_paths(paths: Iterable[str]) -> list[str]:
    normalized = sorted({str(path).strip() for path in paths if str(path).strip()})
    for path in normalized:
        if any(not part for part in path.split(".")):
            raise ValueError(f"invalid runtime-only configuration path: {path}")
    return normalized


def _without_paths(configuration: dict[str, Any], paths: Iterable[str]) -> dict[str, Any]:
    comparable = deepcopy(configuration)
    for path in paths:
        parent: Any = comparable
        parts = path.split(".")
        for part in parts[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                parent = None
                break
            parent = parent[part]
        if isinstance(parent, dict):
            parent.pop(parts[-1], None)
    return comparable


def _signature_payload(descriptor: dict[str, Any]) -> dict[str, Any]:
    configuration = descriptor.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint compatibility configuration must be an object")
    runtime_only_fields = _normalized_paths(descriptor.get("runtime_only_fields") or [])
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": str(descriptor.get("contract") or "").strip(),
        "contract_version": int(descriptor.get("contract_version") or 0),
        "runtime_only_fields": runtime_only_fields,
        "configuration": _without_paths(configuration, runtime_only_fields),
    }


def checkpoint_compatibility_signature(descriptor: dict[str, Any]) -> str:
    """Return the canonical SHA-256 for a compatibility descriptor."""

    payload = _signature_payload(descriptor)
    if not payload["contract"]:
        raise ValueError("checkpoint compatibility contract is required")
    if payload["contract_version"] <= 0:
        raise ValueError("checkpoint compatibility contract_version must be positive")
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_checkpoint_compatibility(
    configuration: dict[str, Any],
    *,
    contract: str,
    contract_version: int = 1,
    runtime_only_fields: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a signed descriptor without discarding runtime-only values.

    Keeping the complete configuration lets callers report runtime differences
    while the signature and compatibility decision use only structural/semantic
    fields.
    """

    descriptor = {
        "schema_version": SCHEMA_VERSION,
        "contract": str(contract).strip(),
        "contract_version": int(contract_version),
        "runtime_only_fields": _normalized_paths(runtime_only_fields),
        "configuration": deepcopy(configuration),
    }
    descriptor["signature"] = checkpoint_compatibility_signature(descriptor)
    return descriptor


def _value_at(configuration: dict[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = configuration
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _differences(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append(
                    {
                        "field": child_path,
                        "expected_present": False,
                        "actual_present": True,
                        "expected": None,
                        "actual": actual[key],
                    }
                )
            elif key not in actual:
                differences.append(
                    {
                        "field": child_path,
                        "expected_present": True,
                        "actual_present": False,
                        "expected": expected[key],
                        "actual": None,
                    }
                )
            else:
                differences.extend(_differences(expected[key], actual[key], child_path))
        return differences
    if expected == actual:
        return []
    return [
        {
            "field": path or "$",
            "expected_present": True,
            "actual_present": True,
            "expected": expected,
            "actual": actual,
        }
    ]


def compare_checkpoint_compatibility(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any]:
    """Compare two signed descriptors and return an explainable decision."""

    try:
        expected_signature = checkpoint_compatibility_signature(expected)
        actual_signature = checkpoint_compatibility_signature(actual)
    except (TypeError, ValueError) as exc:
        return {
            "compatible": False,
            "reason": "invalid_descriptor",
            "message": str(exc),
            "differences": [],
            "runtime_differences": [],
        }
    if expected.get("schema_version") != SCHEMA_VERSION or actual.get(
        "schema_version"
    ) != SCHEMA_VERSION:
        return {
            "compatible": False,
            "reason": "unsupported_schema",
            "message": f"compatibility descriptors must use {SCHEMA_VERSION}",
            "differences": [],
            "runtime_differences": [],
        }
    if str(expected.get("signature") or "") != expected_signature:
        return {
            "compatible": False,
            "reason": "invalid_expected_signature",
            "message": "expected checkpoint compatibility signature is invalid",
            "differences": [],
            "runtime_differences": [],
        }
    if str(actual.get("signature") or "") != actual_signature:
        return {
            "compatible": False,
            "reason": "invalid_actual_signature",
            "message": "checkpoint compatibility signature is invalid",
            "differences": [],
            "runtime_differences": [],
        }
    contract_fields = ("contract", "contract_version", "runtime_only_fields")
    contract_differences = [
        {
            "field": field,
            "expected": expected.get(field),
            "actual": actual.get(field),
            "expected_present": field in expected,
            "actual_present": field in actual,
        }
        for field in contract_fields
        if expected.get(field) != actual.get(field)
    ]
    if contract_differences:
        return {
            "compatible": False,
            "reason": "contract_mismatch",
            "message": "checkpoint and plan use different compatibility contracts",
            "differences": contract_differences,
            "runtime_differences": [],
        }

    runtime_only_fields = _normalized_paths(expected.get("runtime_only_fields") or [])
    expected_configuration = expected.get("configuration") or {}
    actual_configuration = actual.get("configuration") or {}
    comparable_expected = _without_paths(expected_configuration, runtime_only_fields)
    comparable_actual = _without_paths(actual_configuration, runtime_only_fields)
    differences = _differences(comparable_expected, comparable_actual)
    runtime_differences = []
    for field in runtime_only_fields:
        expected_present, expected_value = _value_at(expected_configuration, field)
        actual_present, actual_value = _value_at(actual_configuration, field)
        if expected_present != actual_present or expected_value != actual_value:
            runtime_differences.append(
                {
                    "field": field,
                    "expected_present": expected_present,
                    "actual_present": actual_present,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return {
        "compatible": not differences,
        "reason": "compatible" if not differences else "configuration_mismatch",
        "message": (
            "checkpoint model is compatible with the plan"
            if not differences
            else "checkpoint model has incompatible structural or semantic fields"
        ),
        "expected_signature": expected_signature,
        "actual_signature": actual_signature,
        "differences": differences,
        "runtime_differences": runtime_differences,
    }
