from __future__ import annotations

from llm_platform.services import data_sources


def test_huggingface_generic_preflight_requires_worker_plan() -> None:
    result = data_sources.preflight_source(
        "huggingface",
        "org/corpus",
        {"revision": "main"},
    )

    assert result["accessible"] is False
    assert "Worker" in str(result["error"])
    assert result["files"] == []
