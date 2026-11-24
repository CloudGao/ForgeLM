"""Shared constraints and quality gates for Tokenizer execution."""

import math
from typing import Any

DISK_BPE_DEFAULTS = {
    "min_word_frequency": 2,
    "max_unique_words": 250_000,
    "max_candidate_characters": 4_000_000,
    "max_word_length": 128,
    "merge_occurrence_budget": 2_000_000,
}

DISK_BPE_LIMITS = {
    "min_word_frequency": (1, 1_000_000),
    "max_unique_words": (1_000, 1_000_000),
    "max_candidate_characters": (100_000, 16_000_000),
    "max_word_length": (1, 2_048),
    "merge_occurrence_budget": (1_000, 20_000_000),
}

TOKENIZER_GATE_POLICY_VERSION = "tokenizer-quality-gate-v1"
TOKENIZER_GATE_DEFINITIONS = {
    "round_trip": {
        "label": "Round-trip 成功率",
        "operator": "gte",
        "default_enabled": True,
        "default_threshold": 1.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "unk_rate": {
        "label": "Unknown Token 比例",
        "operator": "lte",
        "default_enabled": True,
        "default_threshold": 0.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "byte_fallback_rate": {
        "label": "Byte fallback 比例",
        "operator": "lte",
        "default_enabled": False,
        "default_threshold": 0.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "newline_round_trip": {
        "label": "换行保真率",
        "operator": "gte",
        "default_enabled": False,
        "default_threshold": 1.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "indentation_round_trip": {
        "label": "缩进保真率",
        "operator": "gte",
        "default_enabled": False,
        "default_threshold": 1.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "special_token_integrity": {
        "label": "特殊 Token 完整性",
        "operator": "gte",
        "default_enabled": False,
        "default_threshold": 1.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "vocab_garbage_rate": {
        "label": "垃圾词表比例",
        "operator": "lte",
        "default_enabled": False,
        "default_threshold": 0.0,
        "minimum": 0.0,
        "maximum": 1.0,
    },
    "zh_chars_per_token": {
        "label": "中文压缩率",
        "operator": "gte",
        "default_enabled": False,
        "default_threshold": 1.5,
        "minimum": 0.0,
        "maximum": 100.0,
    },
    "en_chars_per_token": {
        "label": "英文压缩率",
        "operator": "gte",
        "default_enabled": False,
        "default_threshold": 3.0,
        "minimum": 0.0,
        "maximum": 100.0,
    },
    "code_chars_per_token": {
        "label": "代码压缩率",
        "operator": "gte",
        "default_enabled": False,
        "default_threshold": 2.0,
        "minimum": 0.0,
        "maximum": 100.0,
    },
    "math_chars_per_token": {
        "label": "数学压缩率",
        "operator": "gte",
        "default_enabled": False,
        "default_threshold": 1.5,
        "minimum": 0.0,
        "maximum": 100.0,
    },
}


def default_tokenizer_gate_policy() -> dict[str, Any]:
    """Return a fresh, explicit default policy suitable for version snapshots."""
    return {
        "version": TOKENIZER_GATE_POLICY_VERSION,
        "rules": {
            metric: {
                "enabled": definition["default_enabled"],
                "operator": definition["operator"],
                "threshold": definition["default_threshold"],
            }
            for metric, definition in TOKENIZER_GATE_DEFINITIONS.items()
        },
    }


def normalize_tokenizer_gate_policy(value: object) -> dict[str, Any]:
    """Validate a user policy and expand omitted rules to explicit defaults."""
    if value is None or value == {}:
        return default_tokenizer_gate_policy()
    if not isinstance(value, dict):
        raise ValueError("gate_policy must be an object")
    version = value.get("version", TOKENIZER_GATE_POLICY_VERSION)
    if version != TOKENIZER_GATE_POLICY_VERSION:
        raise ValueError(f"unsupported gate_policy version: {version}")
    raw_rules = value.get("rules", {})
    if not isinstance(raw_rules, dict):
        raise ValueError("gate_policy.rules must be an object")
    normalized = default_tokenizer_gate_policy()
    for metric, raw_rule in raw_rules.items():
        if (
            not isinstance(metric, str)
            or not metric
            or len(metric) > 100
            or not metric.replace("_", "").replace("-", "").replace(".", "").isalnum()
        ):
            raise ValueError("gate rule metric must be a safe non-empty identifier")
        if not isinstance(raw_rule, dict):
            raise ValueError(f"gate rule {metric} must be an object")
        definition = TOKENIZER_GATE_DEFINITIONS.get(metric)
        default_enabled = definition["default_enabled"] if definition else False
        default_operator = definition["operator"] if definition else "gte"
        default_threshold = definition["default_threshold"] if definition else 0.0
        enabled = raw_rule.get("enabled", default_enabled)
        if not isinstance(enabled, bool):
            raise ValueError(f"gate rule {metric}.enabled must be a boolean")
        operator = raw_rule.get("operator", default_operator)
        if operator not in {"gte", "lte"}:
            raise ValueError(f"gate rule {metric}.operator must be gte or lte")
        if definition and operator != default_operator:
            raise ValueError(f"gate rule {metric}.operator must be {default_operator}")
        threshold = raw_rule.get("threshold", default_threshold)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"gate rule {metric}.threshold must be a number")
        numeric_threshold = float(threshold)
        if not math.isfinite(numeric_threshold):
            raise ValueError(f"gate rule {metric}.threshold must be finite")
        minimum = float(definition["minimum"]) if definition else -1_000_000_000_000.0
        maximum = float(definition["maximum"]) if definition else 1_000_000_000_000.0
        if numeric_threshold < minimum or numeric_threshold > maximum:
            raise ValueError(
                f"gate rule {metric}.threshold must be between {minimum:g} and {maximum:g}"
            )
        normalized["rules"][metric] = {
            "enabled": enabled,
            "operator": operator,
            "threshold": numeric_threshold,
            **(
                {"label": str(raw_rule.get("label") or metric)[:180]}
                if definition is None
                else {}
            ),
        }
    return normalized


def evaluate_tokenizer_gate_policy(policy: object, metrics: object) -> dict[str, Any]:
    """Evaluate frozen quality rules against Worker metrics with per-rule evidence."""
    normalized = normalize_tokenizer_gate_policy(policy)
    metric_values = metrics if isinstance(metrics, dict) else {}
    results: list[dict[str, Any]] = []
    passed = True
    for metric, rule in normalized["rules"].items():
        definition = TOKENIZER_GATE_DEFINITIONS.get(metric)
        enabled = bool(rule["enabled"])
        raw_actual = metric_values.get(metric)
        actual = (
            float(raw_actual)
            if not isinstance(raw_actual, bool)
            and isinstance(raw_actual, (int, float))
            and math.isfinite(float(raw_actual))
            else None
        )
        rule_passed: bool | None = None
        reason = "disabled"
        if enabled:
            if actual is None:
                rule_passed = False
                reason = "missing_metric"
            elif rule["operator"] == "gte":
                rule_passed = actual >= float(rule["threshold"])
                reason = "passed" if rule_passed else "threshold_not_met"
            else:
                rule_passed = actual <= float(rule["threshold"])
                reason = "passed" if rule_passed else "threshold_exceeded"
            passed = passed and rule_passed
        results.append(
            {
                "metric": metric,
                "label": definition["label"] if definition else rule.get("label", metric),
                "enabled": enabled,
                "operator": rule["operator"],
                "threshold": rule["threshold"],
                "actual": actual,
                "passed": rule_passed,
                "reason": reason,
            }
        )
    return {
        "gate_policy": normalized,
        "gate_results": results,
        "gate_passed": passed,
    }


def validate_builtin_tokenizer_sample_tokens(value: object) -> int:
    sample_tokens = int(value or 0)
    if sample_tokens <= 0:
        raise ValueError("内置 Tokenizer 必须设置大于 0 的抽样 Tokens")
    return sample_tokens


def validate_disk_bpe_config(config: dict[str, Any]) -> dict[str, int]:
    """Return validated working-set bounds for the disk-backed BPE adapter."""
    validated: dict[str, int] = {}
    for name, default in DISK_BPE_DEFAULTS.items():
        value = int(config.get(name, default))
        minimum, maximum = DISK_BPE_LIMITS[name]
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        validated[name] = value
    if validated["merge_occurrence_budget"] < validated["max_unique_words"]:
        raise ValueError("merge_occurrence_budget must be at least max_unique_words")
    return validated
