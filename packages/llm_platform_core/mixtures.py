from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, TextIO

MIXTURE_MANIFEST_FORMAT = "llm-mixture-manifest-v1"
MIXTURE_INDEX_FORMAT = "llm-mixture-index-v2"
RECORD_MANIFEST_FORMAT = "llm-dataset-records-v1"
RESOLVER_VERSION = "stable_hash_selector_v1"
WEIGHT_NORMALIZER_VERSION = "integer_weight_normalizer_v1"
SAMPLER_VERSION = "weighted_hash_sampler_v1"
REFERENCE_RESOLVER_VERSION = "identity_reference_resolver_v2"
LEGACY_REFERENCE_SAMPLER_VERSION = "weighted_manifest_order_sampler_v2"
REFERENCE_SAMPLER_VERSION = "weighted_manifest_block_sampler_v3"
REFERENCE_SAMPLE_BLOCK_SIZE = 256
WEIGHT_SCALE = 1_000_000


def is_identity_selection(selection: dict[str, Any] | None) -> bool:
    """Return whether every record is selected without filter index state."""
    rule = selection or {}
    return (
        not rule.get("include_paths")
        and not rule.get("exclude_paths")
        and not int(rule.get("min_length") or 0)
        and not int(rule.get("max_length") or 0)
        and not int(rule.get("max_tokens") or 0)
        and float(rule.get("sample_ratio", 1.0)) == 1.0
    )


@contextmanager
def open_text_manifest(path: str | Path) -> Iterator[TextIO]:
    """Open plain or gzip-compressed JSON Lines manifests by content magic."""
    from .storage_layout import resolve_relocated_path
    path = resolve_relocated_path(str(path))
    raw = Path(path).open("rb")
    try:
        magic = raw.read(2)
        raw.seek(0)
        binary = gzip.GzipFile(fileobj=raw, mode="rb") if magic == b"\x1f\x8b" else raw
        text = io.TextIOWrapper(binary, encoding="utf-8")
        try:
            yield text
        finally:
            text.close()
    finally:
        if not raw.closed:
            raw.close()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_digest(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_record_id(
    dataset_version_id: str, logical_path: str, record_index: int, content_hash: str
) -> str:
    return stable_digest(dataset_version_id, logical_path, record_index, content_hash)


def normalize_weight_units(weights: list[object], scale: int = WEIGHT_SCALE) -> list[int]:
    try:
        decimal_weights = [Decimal(str(value)) for value in weights]
    except InvalidOperation as exc:
        raise ValueError("mixture weights must be finite numbers") from exc
    if not decimal_weights or any(not value.is_finite() or value < 0 for value in decimal_weights):
        raise ValueError("mixture weights must be finite non-negative numbers")
    total = sum(decimal_weights, Decimal(0))
    if total <= 0:
        raise ValueError("mixture weights must sum to more than zero")
    exact = [value * scale / total for value in decimal_weights]
    units = [int(value) for value in exact]
    remainder = scale - sum(units)
    ranking = sorted(
        range(len(exact)),
        key=lambda index: (exact[index] - units[index], -index),
        reverse=True,
    )
    for index in ranking[:remainder]:
        units[index] += 1
    return units


def _matches_any(path: str, patterns: list[str]) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    return any(
        fnmatch.fnmatchcase(normalized, pattern.replace("\\", "/").lstrip("/"))
        for pattern in patterns
    )


def select_records(
    records: list[dict[str, Any]],
    selection: dict[str, Any] | None,
    *,
    dataset_version_id: str,
    default_seed: int,
) -> list[dict[str, Any]]:
    rule = selection or {}
    include_paths = [str(value) for value in rule.get("include_paths") or [] if str(value)]
    exclude_paths = [str(value) for value in rule.get("exclude_paths") or [] if str(value)]
    minimum = max(int(rule.get("min_length") or 0), 0)
    maximum = max(int(rule.get("max_length") or 0), 0)
    ratio = min(max(float(rule.get("sample_ratio", 1.0)), 0.0), 1.0)
    max_tokens = max(int(rule.get("max_tokens") or 0), 0)
    seed = int(rule.get("seed", default_seed))
    ratio_limit = int(ratio * (1 << 256))
    selected: list[tuple[str, dict[str, Any]]] = []
    for raw_record in records:
        record = dict(raw_record)
        path = str(record.get("logical_path") or record.get("path") or "")
        length = int(record.get("character_length") or record.get("length") or 0)
        if include_paths and not _matches_any(path, include_paths):
            continue
        if exclude_paths and _matches_any(path, exclude_paths):
            continue
        if length < minimum or (maximum and length > maximum):
            continue
        record_id = str(record.get("record_id") or "")
        if not record_id:
            continue
        order_key = stable_digest(seed, dataset_version_id, record_id)
        if int(order_key, 16) >= ratio_limit:
            continue
        selected.append((order_key, record))
    selected.sort(key=lambda item: (item[0], str(item[1]["record_id"])))
    if not max_tokens:
        return [record for _, record in selected]
    result: list[dict[str, Any]] = []
    consumed = 0
    for _, record in selected:
        tokens = max(int(record.get("token_estimate") or 0), 0)
        if consumed + tokens > max_tokens:
            continue
        result.append(record)
        consumed += tokens
    return result


def stream_key(header_without_stream_key: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(header_without_stream_key).encode("utf-8")).hexdigest()


def sample_reference(
    header: dict[str, Any], records_by_source: list[list[dict[str, Any]]], sample_index: int
) -> tuple[int, int]:
    """Return a stable (source index, source record index) for a global sample position."""
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    sources = header.get("sources") or []
    if len(sources) != len(records_by_source) or not sources:
        raise ValueError("mixture manifest sources do not match record groups")
    units = [int(source.get("weight_units") or 0) for source in sources]
    total = sum(units)
    if total <= 0:
        raise ValueError("mixture manifest has no positive source weight")
    key = str(header.get("stream_key") or "")
    seed = int(header.get("seed") or 0)
    sampler = str(header.get("sampler") or LEGACY_REFERENCE_SAMPLER_VERSION)
    if sampler not in {LEGACY_REFERENCE_SAMPLER_VERSION, REFERENCE_SAMPLER_VERSION}:
        raise ValueError(f"unsupported mixture reference sampler: {sampler}")
    if sampler == REFERENCE_SAMPLER_VERSION:
        block_size = int(header.get("sample_block_size") or REFERENCE_SAMPLE_BLOCK_SIZE)
        if block_size <= 0 or block_size > 1_000_000:
            raise ValueError("mixture sample block size must be between 1 and 1000000")
        block_index, block_offset = divmod(sample_index, block_size)
        draw_index = block_index
        draw_label = "block-source"
    else:
        block_index = sample_index
        block_offset = 0
        draw_index = sample_index
        draw_label = "source"
    draw = int(stable_digest(key, seed, draw_index, draw_label)[:16], 16) % total
    cumulative = 0
    source_index: int | None = None
    for candidate_index, weight in enumerate(units):
        cumulative += weight
        if draw < cumulative:
            source_index = candidate_index
            break
    if source_index is None:
        raise ValueError("mixture sampler could not select a source")
    records = records_by_source[source_index]
    if not records:
        raise ValueError("mixture manifest source has no selected records")
    if sampler == REFERENCE_SAMPLER_VERSION:
        block_start = int(
            stable_digest(key, seed, block_index, source_index, "block-start")[:16], 16
        ) % len(records)
        record_index = (block_start + block_offset) % len(records)
    else:
        record_index = int(
            stable_digest(key, seed, sample_index, source_index, "record")[:16], 16
        ) % len(records)
    return source_index, record_index
