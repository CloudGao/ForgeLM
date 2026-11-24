from __future__ import annotations

from llm_platform.training_system_validations import (
    annotate_system_validation_issues,
    build_training_system_validations,
)


def _by_id(results: list[dict]) -> dict[str, dict]:
    return {item["id"]: item for item in results}


def test_unexecuted_system_validations_are_never_reported_as_passed() -> None:
    results = _by_id(
        build_training_system_validations(
            [],
            executed={"tokenizer_locked"},
            not_applicable={"resume_checkpoint_compatible"},
        )
    )

    assert results["tokenizer_locked"]["status"] == "passed"
    assert results["resume_checkpoint_compatible"]["status"] == "not_applicable"
    assert results["runtime_training_invariants"]["status"] == "pending"
    assert results["finite_training_state"]["status"] == "pending"
    assert results["output_artifact_integrity"]["status"] == "pending"


def test_warning_keeps_a_system_validation_pending_and_blocker_blocks() -> None:
    issues = [
        {
            "severity": "warning",
            "code": "resume_checkpoint_model_compatibility_unverified",
            "message": "deferred",
        },
        {
            "severity": "blocker",
            "code": "tokenizer_model_vocab_mismatch",
            "message": "mismatch",
        },
    ]
    annotate_system_validation_issues(issues)
    results = _by_id(
        build_training_system_validations(
            issues,
            executed={"resume_checkpoint_compatible", "tokenizer_model_compatible"},
            not_applicable=set(),
        )
    )

    assert results["resume_checkpoint_compatible"]["status"] == "pending"
    assert results["tokenizer_model_compatible"]["status"] == "blocked"
    assert issues[0]["configurable"] is False
