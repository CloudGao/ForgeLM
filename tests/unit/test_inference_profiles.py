from __future__ import annotations

from llm_platform_core.inference import (
    MICROLM_INFERENCE_ADAPTER,
    MICROLM_INFERENCE_FORMAT,
    MICROLM_W8A16_INFERENCE_FORMAT,
    microlm_inference_metadata,
    resolve_inference_profile,
)


def test_historical_microlm_format_derives_a_native_gpu_profile() -> None:
    profile = resolve_inference_profile(
        {"inference": {"format": MICROLM_INFERENCE_FORMAT}}
    )

    assert profile["adapter_id"] == MICROLM_INFERENCE_ADAPTER
    assert profile["resource_request"]["gpus"] == 1
    assert profile["resource_request"]["worker_selector"]["capabilities"] == [
        MICROLM_INFERENCE_ADAPTER
    ]


def test_explicit_project_adapter_overrides_the_builtin_format_mapping() -> None:
    profile = resolve_inference_profile(
        {
            "inference": {
                "format": MICROLM_INFERENCE_FORMAT,
                "adapter_id": "project.microlm-inference.v2",
                "command": {"executable": "python"},
            }
        }
    )

    assert profile["adapter_id"] == "project.microlm-inference.v2"
    assert "resource_request" not in profile


def test_new_microlm_metadata_is_self_describing() -> None:
    profile = microlm_inference_metadata()

    assert profile["format"] == MICROLM_INFERENCE_FORMAT
    assert profile["adapter_id"] == MICROLM_INFERENCE_ADAPTER


def test_quantized_microlm_format_uses_the_same_trusted_adapter() -> None:
    profile = resolve_inference_profile(
        {"inference": {"format": MICROLM_W8A16_INFERENCE_FORMAT}}
    )

    assert profile["adapter_id"] == MICROLM_INFERENCE_ADAPTER
    assert profile["resource_request"]["gpus"] == 1


def test_malformed_historical_resource_metadata_falls_back_to_safe_gpu_profile() -> None:
    profile = resolve_inference_profile(
        {
            "inference": {
                "format": MICROLM_INFERENCE_FORMAT,
                "resource_request": {"gpus": "invalid", "worker_selector": []},
            }
        }
    )

    assert profile["resource_request"] == {
        "gpus": 1,
        "worker_selector": {"capabilities": [MICROLM_INFERENCE_ADAPTER]},
    }
