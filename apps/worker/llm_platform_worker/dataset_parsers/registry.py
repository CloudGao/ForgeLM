from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class ParserContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetParserContext:
    root: Path
    source_type: str
    source_uri: str
    dataset_version_id: str
    modality: str
    detected_formats: tuple[str, ...]
    parser_config: Mapping[str, Any] = field(default_factory=dict)
    record_manifest_path: Path | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    get_desired_state: Callable[[], str] | None = None


class DatasetParser(Protocol):
    parser_id: str
    parser_version: str
    modalities: frozenset[str]
    implementation: str

    def parse(self, context: DatasetParserContext) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FunctionParserAdapter:
    parser_id: str
    parser_version: str
    modalities: frozenset[str]
    handler: Callable[[DatasetParserContext], dict[str, Any]]
    implementation: str = "function"

    def parse(self, context: DatasetParserContext) -> dict[str, Any]:
        return self.handler(context)


@dataclass(frozen=True, slots=True)
class ObjectParserAdapter:
    """Adapts an SDK/client object exposing parse(context) to the same contract."""

    parser_id: str
    parser_version: str
    modalities: frozenset[str]
    client: Any
    implementation: str = "sdk"

    def parse(self, context: DatasetParserContext) -> dict[str, Any]:
        result = self.client.parse(context)
        if not isinstance(result, dict):
            raise ParserContractError("Dataset parser SDK must return an object")
        return result


class DatasetParserRegistry:
    def __init__(self) -> None:
        self._parsers: dict[tuple[str, str], DatasetParser] = {}

    def register(self, parser: DatasetParser) -> None:
        key = (parser.parser_id, parser.parser_version)
        if key in self._parsers:
            raise ParserContractError(
                f"Dataset parser is already registered: {parser.parser_id}@{parser.parser_version}"
            )
        if not parser.modalities:
            raise ParserContractError("Dataset parser must declare at least one modality")
        self._parsers[key] = parser

    def resolve(
        self,
        modality: str,
        requested: Mapping[str, Any] | None = None,
    ) -> DatasetParser | None:
        requested = requested or {}
        parser_id = str(requested.get("id") or "").strip()
        parser_version = str(requested.get("version") or "").strip()
        if parser_id or parser_version:
            if not parser_id or not parser_version:
                raise ParserContractError("Frozen parser request requires both id and version")
            parser = self._parsers.get((parser_id, parser_version))
            if parser is None:
                raise ParserContractError(
                    f"Frozen dataset parser is unavailable: {parser_id}@{parser_version}"
                )
            if modality not in parser.modalities:
                raise ParserContractError(
                    f"Dataset parser {parser_id}@{parser_version} does not support {modality}"
                )
            return parser
        candidates = [
            parser for parser in self._parsers.values() if modality in parser.modalities
        ]
        if not candidates:
            return None
        if len(candidates) > 1:
            names = ", ".join(
                sorted(f"{item.parser_id}@{item.parser_version}" for item in candidates)
            )
            raise ParserContractError(
                f"Multiple dataset parsers support {modality}; freeze one explicitly: {names}"
            )
        return candidates[0]


def detect_formats(root: Path) -> tuple[str, ...]:
    formats: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or ({".hfd", ".cache", ".git"} & set(path.relative_to(root).parts)):
            continue
        suffixes = [suffix.lower().lstrip(".") for suffix in path.suffixes]
        if suffixes:
            formats.add(".".join(suffixes[-2:]) if suffixes[-1] in {"gz", "zst"} else suffixes[-1])
        else:
            formats.add("unknown")
    return tuple(sorted(formats))
