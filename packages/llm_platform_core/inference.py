from __future__ import annotations

from typing import Any

MICROLM_INFERENCE_ADAPTER = "builtin.microlm-inference"
MICROLM_INFERENCE_FORMAT = "microlm-hybrid-safetensors"
MICROLM_W8A16_INFERENCE_FORMAT = "microlm-w8a16-safetensors"
MICROLM_INFERENCE_FORMATS = frozenset(
    {MICROLM_INFERENCE_FORMAT, MICROLM_W8A16_INFERENCE_FORMAT}
)
MICROLM_INFERENCE_DISPLAY_NAME = "MicroLM 原生推理"


def resolve_inference_profile(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return the explicit or built-in profile for a frozen model format.

    Historical MicroLM model versions only recorded their immutable output format.
    Deriving the built-in adapter here keeps those versions runnable without mutating
    their metadata while still allowing an explicit non-MicroLM adapter to override it.
    """

    source = metadata if isinstance(metadata, dict) else {}
    raw = source.get("inference")
    inference = dict(raw) if isinstance(raw, dict) else {}
    adapter_id = str(inference.get("adapter_id") or "").strip()
    if inference.get("format") not in MICROLM_INFERENCE_FORMATS:
        return inference
    if adapter_id and adapter_id != MICROLM_INFERENCE_ADAPTER:
        return inference

    raw_request = inference.get("resource_request")
    request = dict(raw_request) if isinstance(raw_request, dict) else {}
    try:
        request["gpus"] = max(1, int(request.get("gpus", 0) or 0))
    except (TypeError, ValueError):
        request["gpus"] = 1
    raw_selector = request.get("worker_selector")
    selector = dict(raw_selector) if isinstance(raw_selector, dict) else {}
    raw_capabilities = selector.get("capabilities")
    capabilities_before = (
        [value for value in raw_capabilities if isinstance(value, str) and value]
        if isinstance(raw_capabilities, list)
        else []
    )
    capabilities = list(
        dict.fromkeys(
            [
                *capabilities_before,
                MICROLM_INFERENCE_ADAPTER,
            ]
        )
    )
    selector["capabilities"] = capabilities
    request["worker_selector"] = selector
    return {
        **inference,
        "adapter_id": MICROLM_INFERENCE_ADAPTER,
        "display_name": str(
            inference.get("display_name") or MICROLM_INFERENCE_DISPLAY_NAME
        ),
        "resource_request": request,
    }


def microlm_inference_metadata() -> dict[str, Any]:
    """Return the self-describing inference metadata written by new training runs."""

    return resolve_inference_profile({"inference": {"format": MICROLM_INFERENCE_FORMAT}})
