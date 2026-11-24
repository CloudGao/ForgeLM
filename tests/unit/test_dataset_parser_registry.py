from __future__ import annotations

from pathlib import Path

import pytest
from llm_platform_worker.dataset_parsers import (
    DatasetParserContext,
    DatasetParserRegistry,
    FunctionParserAdapter,
    ObjectParserAdapter,
    ParserContractError,
)


def context(modality: str = "text") -> DatasetParserContext:
    return DatasetParserContext(
        root=Path("."),
        source_type="local",
        source_uri="/data",
        dataset_version_id="version-1",
        modality=modality,
        detected_formats=("jsonl",),
    )


def test_registry_resolves_frozen_function_parser() -> None:
    registry = DatasetParserRegistry()
    parser = FunctionParserAdapter(
        parser_id="test.text",
        parser_version="2.1.0",
        modalities=frozenset({"text"}),
        handler=lambda value: {"modality": value.modality},
    )
    registry.register(parser)

    resolved = registry.resolve("text", {"id": "test.text", "version": "2.1.0"})

    assert resolved is parser
    assert resolved.parse(context()) == {"modality": "text"}


def test_object_adapter_wraps_sdk_parser() -> None:
    class SDK:
        def parse(self, value: DatasetParserContext) -> dict:
            return {"formats": list(value.detected_formats)}

    parser = ObjectParserAdapter(
        parser_id="vendor.audio",
        parser_version="1",
        modalities=frozenset({"audio"}),
        client=SDK(),
    )

    assert parser.parse(context("audio")) == {"formats": ["jsonl"]}
    assert parser.implementation == "sdk"


def test_registry_rejects_duplicate_or_wrong_frozen_contract() -> None:
    registry = DatasetParserRegistry()
    parser = FunctionParserAdapter(
        parser_id="test.text",
        parser_version="1",
        modalities=frozenset({"text"}),
        handler=lambda _: {},
    )
    registry.register(parser)

    with pytest.raises(ParserContractError, match="already registered"):
        registry.register(parser)
    with pytest.raises(ParserContractError, match="unavailable"):
        registry.resolve("text", {"id": "test.text", "version": "2"})
    with pytest.raises(ParserContractError, match="does not support audio"):
        registry.resolve("audio", {"id": "test.text", "version": "1"})


def test_registry_returns_none_when_modality_has_no_parser() -> None:
    assert DatasetParserRegistry().resolve("video") is None


def test_registry_requires_explicit_choice_when_multiple_versions_are_available() -> None:
    registry = DatasetParserRegistry()
    for version in ("1", "2"):
        registry.register(
            FunctionParserAdapter(
                parser_id="test.text",
                parser_version=version,
                modalities=frozenset({"text"}),
                handler=lambda _: {},
            )
        )

    with pytest.raises(ParserContractError, match="freeze one explicitly"):
        registry.resolve("text")
