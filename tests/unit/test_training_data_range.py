from __future__ import annotations

import pytest
from llm_platform.routers.training import _training_data_range


def test_training_data_range_rounds_target_to_full_batches() -> None:
    result = _training_data_range(
        [
            {
                "name": "Bootstrap",
                "target_tokens": 100_000_000,
                "effective_batch_tokens": 262_144,
            }
        ],
        2_048,
    )

    assert result["planned_steps"] == 382
    assert result["planned_train_tokens"] == 100_139_008
    assert result["sequence_start"] == 0
    assert result["sequence_count"] == 48_896
    assert result["sequence_end"] == 48_896


def test_training_data_range_adds_checkpoint_cursor_to_capacity_requirement() -> None:
    result = _training_data_range(
        [{"target_tokens": 100_000_000, "effective_batch_tokens": 262_144}],
        2_048,
        sequence_start=48_896,
    )

    assert result["sequence_start"] == 48_896
    assert result["sequence_count"] == 48_896
    assert result["sequence_end"] == 97_792


def test_training_data_range_rejects_batches_that_split_packed_rows() -> None:
    with pytest.raises(
        ValueError, match="effective batch tokens must align to Tokenized Dataset rows"
    ):
        _training_data_range(
            [{"target_tokens": 10_000, "effective_batch_tokens": 3_000}],
            2_048,
        )
