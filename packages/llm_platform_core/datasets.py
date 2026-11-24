from __future__ import annotations

from typing import Any

DATASET_MODALITIES = ("text", "image", "audio", "video", "multimodal", "other")

TEXT_PARSER_ID = "builtin.text-parser"
TEXT_PARSER_VERSION = "1.0.0"

SOURCE_ARCHIVE_CAPABILITY = "source_archive"
TEXT_USAGE_CAPABILITIES = (
    SOURCE_ARCHIVE_CAPABILITY,
    "record_explorer",
    "text_pretraining",
    "tokenizer_training",
    "text_evaluation",
    "supervision_mapping",
)


def parser_request_for_modality(modality: str) -> dict[str, str] | None:
    """Return the parser contract frozen into a newly created import task."""
    if modality == "text":
        return {"id": TEXT_PARSER_ID, "version": TEXT_PARSER_VERSION}
    return None


def version_supports_capability(version: Any, capability: str) -> bool:
    """Check the new capability contract with a safe legacy fallback."""
    capabilities = (
        version.get("usage_capabilities", [])
        if isinstance(version, dict)
        else getattr(version, "usage_capabilities", [])
    ) or []
    if capabilities:
        return capability in capabilities
    train_ready = (
        bool(version.get("train_ready"))
        if isinstance(version, dict)
        else bool(getattr(version, "train_ready", False))
    )
    if capability in {"text_pretraining", "tokenizer_training", "text_evaluation"}:
        return train_ready
    if capability == "record_explorer":
        manifest_id = (
            version.get("record_manifest_artifact_id")
            if isinstance(version, dict)
            else getattr(version, "record_manifest_artifact_id", None)
        )
        return bool(manifest_id)
    return False
