from __future__ import annotations

from llm_platform.services.tasks import normalize_training_result_tokens


def test_stage_local_training_result_is_normalized_to_cumulative_tokens() -> None:
    result = normalize_training_result_tokens(
        {
            "resume_checkpoint": {"token_count": 10_000_000},
            "stages": [{"target_tokens": 90_000_000}],
        },
        {
            "tokens": 90_000_000,
            "checkpoints": [
                {"ref": "middle", "tokens": 45_000_000},
                {"ref": "final", "tokens": 90_000_000},
            ],
        },
    )

    assert result["tokens"] == 100_000_000
    assert [item["tokens"] for item in result["checkpoints"]] == [55_000_000, 100_000_000]


def test_cumulative_training_result_is_not_double_counted() -> None:
    original = {
        "tokens": 100_000_000,
        "checkpoints": [{"ref": "final", "tokens": 100_000_000}],
    }

    result = normalize_training_result_tokens(
        {
            "resume_checkpoint": {"token_count": 10_000_000},
            "stages": [{"target_tokens": 90_000_000}],
        },
        original,
    )

    assert result is original


def test_unproven_stage_local_result_is_left_unchanged() -> None:
    original = {"tokens": 80_000_000}

    result = normalize_training_result_tokens(
        {
            "resume_checkpoint": {"token_count": 10_000_000},
            "stages": [{"target_tokens": 90_000_000}],
        },
        original,
    )

    assert result is original
