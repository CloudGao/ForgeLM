import pytest
from llm_platform_core.tokenizers import (
    default_tokenizer_gate_policy,
    evaluate_tokenizer_gate_policy,
    normalize_tokenizer_gate_policy,
)


def test_default_tokenizer_gate_policy_is_explicit_and_passes_legacy_metrics() -> None:
    policy = default_tokenizer_gate_policy()

    assert policy["version"] == "tokenizer-quality-gate-v1"
    assert policy["rules"]["round_trip"] == {
        "enabled": True,
        "operator": "gte",
        "threshold": 1.0,
    }
    assert policy["rules"]["zh_chars_per_token"]["enabled"] is False

    evaluated = evaluate_tokenizer_gate_policy(policy, {"round_trip": 1, "unk_rate": 0})
    assert evaluated["gate_passed"] is True
    assert [item["passed"] for item in evaluated["gate_results"][:2]] == [True, True]


def test_tokenizer_gate_policy_uses_frozen_thresholds_and_reports_evidence() -> None:
    policy = default_tokenizer_gate_policy()
    policy["rules"]["round_trip"]["threshold"] = 0.99
    policy["rules"]["unk_rate"]["threshold"] = 0.001
    policy["rules"]["zh_chars_per_token"]["enabled"] = True
    policy["rules"]["zh_chars_per_token"]["threshold"] = 1.7

    evaluated = evaluate_tokenizer_gate_policy(
        policy,
        {"round_trip": 0.995, "unk_rate": 0.0005, "zh_chars_per_token": 1.67},
    )

    assert evaluated["gate_passed"] is False
    zh_result = next(
        item for item in evaluated["gate_results"] if item["metric"] == "zh_chars_per_token"
    )
    assert zh_result == {
        "metric": "zh_chars_per_token",
        "label": "中文压缩率",
        "enabled": True,
        "operator": "gte",
        "threshold": 1.7,
        "actual": 1.67,
        "passed": False,
        "reason": "threshold_not_met",
    }


def test_enabled_tokenizer_gate_fails_when_worker_omits_metric() -> None:
    policy = default_tokenizer_gate_policy()
    evaluated = evaluate_tokenizer_gate_policy(policy, {"round_trip": 1})

    assert evaluated["gate_passed"] is False
    unknown_result = next(
        item for item in evaluated["gate_results"] if item["metric"] == "unk_rate"
    )
    assert unknown_result["actual"] is None
    assert unknown_result["passed"] is False
    assert unknown_result["reason"] == "missing_metric"


@pytest.mark.parametrize(
    "policy, message",
    [
        ({"version": "v2"}, "unsupported gate_policy version"),
        (
            {"rules": {"round_trip": {"enabled": True, "threshold": 1.1}}},
            "must be between 0 and 1",
        ),
        (
            {"rules": {"unsafe metric!": {"enabled": True, "threshold": 1}}},
            "safe non-empty identifier",
        ),
    ],
)
def test_invalid_tokenizer_gate_policy_is_rejected(policy: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_tokenizer_gate_policy(policy)


def test_historical_custom_gate_is_preserved_and_evaluated() -> None:
    policy = normalize_tokenizer_gate_policy(
        {
            "rules": {
                "legacy_symbol_coverage": {
                    "label": "历史符号覆盖率",
                    "enabled": True,
                    "operator": "gte",
                    "threshold": 0.95,
                }
            }
        }
    )

    assert policy["rules"]["legacy_symbol_coverage"] == {
        "label": "历史符号覆盖率",
        "enabled": True,
        "operator": "gte",
        "threshold": 0.95,
    }
    evaluated = evaluate_tokenizer_gate_policy(
        policy, {"round_trip": 1, "unk_rate": 0, "legacy_symbol_coverage": 0.96}
    )
    custom = next(
        item
        for item in evaluated["gate_results"]
        if item["metric"] == "legacy_symbol_coverage"
    )
    assert custom["passed"] is True
    assert custom["label"] == "历史符号覆盖率"
