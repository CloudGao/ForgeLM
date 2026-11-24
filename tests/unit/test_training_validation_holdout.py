from __future__ import annotations

import pytest
from fastapi import HTTPException
from llm_platform.models import DataMixtureVersion
from llm_platform.routers.training import (
    _effective_validation_data_input,
    _validation_holdout_configuration,
)


def mixture(
    version_id: str,
    datasets: list[str],
    *,
    target_tokens: int = 2_000_000,
    seed: int = 42,
) -> DataMixtureVersion:
    return DataMixtureVersion(
        id=version_id,
        mixture_id=f"mixture-{version_id}",
        version=1,
        entries=[
            {
                "dataset_version_id": dataset,
                "weight_units": index + 1,
                "selection": {
                    "sample_ratio": 1,
                    "max_tokens": None,
                    "min_length": None,
                    "max_length": None,
                    "include_paths": [],
                    "exclude_paths": [],
                    "seed": 42,
                },
            }
            for index, dataset in enumerate(datasets)
        ],
        target_tokens=target_tokens,
        seed=seed,
        manifest_hash="a" * 64,
    )


def test_same_source_validation_freezes_document_holdout() -> None:
    training = mixture("train-v1", ["dataset-a", "dataset-b"])
    validation = mixture("validation-v1", ["dataset-a", "dataset-b"])

    holdout = _validation_holdout_configuration(training, validation)

    assert holdout == {
        "strategy": "weighted_random_tokenized_document_holdout_v1",
        "target_tokens": 2_000_000,
        "seed": 42,
        "validation_mixture_version_id": "validation-v1",
        "validation_manifest_hash": "a" * 64,
        "sources": [
            {"dataset_version_id": "dataset-a", "weight_units": 1},
            {"dataset_version_id": "dataset-b", "weight_units": 2},
        ],
    }


def test_disjoint_validation_keeps_independent_materialization() -> None:
    assert (
        _validation_holdout_configuration(
            mixture("train-v1", ["dataset-a"]),
            mixture("validation-v1", ["dataset-b"]),
        )
        is None
    )


def test_partial_overlap_is_rejected() -> None:
    with pytest.raises(HTTPException) as raised:
        _validation_holdout_configuration(
            mixture("train-v1", ["dataset-a", "dataset-b"]),
            mixture("validation-v1", ["dataset-b", "dataset-c"]),
        )
    assert raised.value.detail["code"] == "validation_holdout_partial_overlap"


def test_effective_validation_input_binds_materialized_view_holdout() -> None:
    request = {
        "strategy": "weighted_random_tokenized_document_holdout_v1",
        "target_tokens": 2_000_000,
        "seed": 42,
        "validation_mixture_version_id": "validation-v1",
    }
    actual = {
        **request,
        "identity_hash": "b" * 64,
        "sequence_count": 977,
        "token_count": 2_000_896,
    }
    binding = _effective_validation_data_input(
        {"mixture_version_id": "validation-v1"},
        [
            {
                "version_id": "view-v1",
                "identity_hash": "c" * 64,
                "configuration": {"validation_holdout": request},
                "validation_holdout": actual,
            }
        ],
    )

    assert binding == {
        "format": "forgelm-validation-token-holdout-v1",
        "usage": "training_validation",
        "target_tokens": 2_000_000,
        "seed": 42,
        "mixture_version_id": "validation-v1",
        "training_data_view_version_id": "view-v1",
        "training_data_view_identity_hash": "c" * 64,
        "holdout_identity_hash": "b" * 64,
        "sequence_count": 977,
        "token_count": 2_000_896,
    }
