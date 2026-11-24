from __future__ import annotations

from llm_platform.services.model_specifications import (
    architecture_catalog,
    resolve_model_specification,
)


def resolve(family_id: str, config: dict | None = None) -> dict:
    return resolve_model_specification(
        source_type="standard",
        architecture_family_id=family_id,
        preset_id=None,
        config=config or {},
        initialization={"type": "random"},
    )


def test_dense_model_specification_has_exact_deterministic_parameters() -> None:
    first = resolve("llama_dense")
    second = resolve("llama_dense")

    assert first["status"] == "ready"
    assert first["parameter_summary"]["method"] == "architecture_formula_v2"
    assert first["parameter_summary"]["verified"] is True
    assert first["parameter_summary"]["total_parameters"] > 100_000_000
    assert (
        first["parameter_summary"]["active_parameters_per_token"]
        == first["parameter_summary"]["total_parameters"]
    )
    assert first["config_signature"] == second["config_signature"]
    assert "LlamaConfig" in first["generated_code"]


def test_moe_model_specification_separates_total_and_active_parameters() -> None:
    result = resolve("mixtral_moe")

    assert result["status"] == "ready"
    assert (
        result["parameter_summary"]["total_parameters"]
        > result["parameter_summary"]["active_parameters_per_token"]
    )
    assert result["parameter_summary"]["breakdown"]["routed_experts"] > 0
    assert "MixtralConfig" in result["generated_code"]


def test_model_specification_rejects_incompatible_dimensions() -> None:
    result = resolve(
        "llama_dense",
        {"hidden_size": 2000, "num_attention_heads": 16, "num_key_value_heads": 3},
    )

    assert result["status"] == "blocked"
    codes = {issue["code"] for issue in result["issues"]}
    assert "heads_not_divisible_by_kv_heads" in codes


def test_custom_code_statistics_are_explicitly_unverified() -> None:
    result = resolve_model_specification(
        source_type="custom_code",
        architecture_family_id="custom_code",
        preset_id=None,
        code_version={
            "id": "code-v1",
            "version": 1,
            "entrypoints": {"model_factory": "model_factory.py"},
            "config_schema": {"type": "object", "properties": {}},
        },
        config={
            "architecture_name": "ResearchDecoder",
            "vocab_size": 64000,
            "total_parameters": 4_000_000_000,
            "active_parameters": 1_000_000_000,
            "trainable_parameters": 4_000_000_000,
        },
        initialization={"type": "random"},
    )

    assert result["status"] == "ready"
    assert result["parameter_summary"]["method"] == "user_declared"
    assert result["parameter_summary"]["verified"] is False
    assert "build_frozen_model" in result["generated_code"]


def test_catalog_exposes_real_implementations_without_size_presets() -> None:
    catalog = architecture_catalog()
    standard = [item for item in catalog if item["source_type"] == "standard"]

    assert {item["id"] for item in standard} == {
        "llama_dense",
        "mistral",
        "qwen2",
        "gemma2",
        "phi3",
        "gpt_neox",
        "mixtral_moe",
        "qwen2_moe",
        "deepseek_v3",
    }
    assert all(not item["presets"] for item in catalog)
    assert all(item["implementation"]["config_class"] for item in standard)


def test_size_profiles_are_rejected_instead_of_pretending_to_be_presets() -> None:
    result = resolve_model_specification(
        source_type="standard",
        architecture_family_id="llama_dense",
        preset_id="dense_1b",
        config={},
        initialization={"type": "random"},
    )

    assert result["status"] == "blocked"
    assert {issue["code"] for issue in result["issues"]} == {
        "model_profiles_not_supported"
    }
