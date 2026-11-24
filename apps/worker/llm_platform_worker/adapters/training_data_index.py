from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shutil
import struct
from bisect import bisect_right
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from llm_platform_core.storage_layout import derived_asset_root
from llm_platform_core.training_data import (
    materialized_sequence_size_bytes,
    sequence_order_strategy,
)

from .command import Emit, GetDesiredState
from .corpus_selection import selected_ordinals
from .data_import import AnalysisInterrupted

TRAINING_DATA_VIEW_FORMAT = "forgelm-training-index-v1"


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _file_spec(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _manifest_file(
    root: Path, manifest: dict[str, Any], key: str, *, verify_hash: bool = True
) -> Path:
    spec = (manifest.get("files") or {}).get(key) or {}
    path = (root / str(spec.get("path") or "")).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise ValueError(f"Training Data View file {key} is missing or unsafe")
    if path.stat().st_size != int(spec.get("size_bytes") or -1):
        raise ValueError(f"Training Data View file {key} size does not match manifest")
    if verify_hash and _file_spec(path)["sha256"] != spec.get("sha256"):
        raise ValueError(f"Training Data View file {key} hash does not match manifest")
    return path


def _require_file_prefix(base_path: Path, expanded_path: Path, label: str) -> None:
    if base_path.stat().st_size > expanded_path.stat().st_size:
        raise ValueError(f"expanded {label} is shorter than its frozen prefix")
    with base_path.open("rb") as base_stream, expanded_path.open("rb") as expanded_stream:
        while chunk := base_stream.read(8 * 1024 * 1024):
            if expanded_stream.read(len(chunk)) != chunk:
                raise ValueError(f"expanded {label} does not preserve its frozen prefix")


def _copy_with_digest(source: Path, target: BinaryIO, digest: Any) -> None:
    with source.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            target.write(chunk)
            digest.update(chunk)


def _document_lengths(root: Path, manifest: dict[str, Any]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for shard in manifest.get("shards") or []:
        spec = (shard.get("files") or {}).get("documents") or {}
        path = (root / str(spec.get("path") or "")).resolve()
        if root.resolve() not in path.parents or not path.is_file():
            raise ValueError("Tokenized Corpus document index path is missing or unsafe")
        rows = np.fromfile(path, dtype="<u8")
        if rows.size % 2:
            raise ValueError("Tokenized Corpus document index is truncated")
        chunks.append(rows.reshape(-1, 2)[:, 1].copy())
    return np.concatenate(chunks) if chunks else np.asarray([], dtype="<u8")


class _TokenCorpusReader:
    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.token_dtype = np.dtype(
            str((manifest.get("resolved_encoding") or {}).get("numpy_dtype") or "")
        )
        if self.token_dtype not in {np.dtype("<u2"), np.dtype("<u4")}:
            raise ValueError("Tokenized Corpus token dtype is unsupported")
        self.shards = list(manifest.get("shards") or [])
        self.document_ends: list[int] = []
        total = 0
        for shard in self.shards:
            total += int(shard.get("document_count") or 0)
            self.document_ends.append(total)
        self._maps: dict[int, tuple[np.memmap, np.memmap]] = {}

    def _open_shard(self, shard_index: int) -> tuple[np.memmap, np.memmap]:
        cached = self._maps.get(shard_index)
        if cached is not None:
            return cached
        shard = self.shards[shard_index]
        files = shard.get("files") or {}
        token_path = (
            self.root / str((files.get("tokens") or {}).get("path") or "")
        ).resolve()
        document_path = (
            self.root / str((files.get("documents") or {}).get("path") or "")
        ).resolve()
        if self.root not in token_path.parents or self.root not in document_path.parents:
            raise ValueError("Tokenized Corpus shard path escaped its root")
        token_map = np.memmap(token_path, dtype=self.token_dtype, mode="r")
        document_map = np.memmap(
            document_path,
            dtype=np.dtype([("offset", "<u8"), ("length", "<u8")]),
            mode="r",
        )
        self._maps[shard_index] = (token_map, document_map)
        return token_map, document_map

    def tokens(self, ordinal: int, offset: int, count: int) -> np.ndarray:
        shard_index = bisect_right(self.document_ends, ordinal)
        shard_start = 0 if shard_index == 0 else self.document_ends[shard_index - 1]
        tokens, documents = self._open_shard(shard_index)
        record = documents[ordinal - shard_start]
        start = int(record["offset"]) + offset
        return np.asarray(tokens[start : start + count], dtype=self.token_dtype)


def _shuffle_seed(namespace: str, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{namespace}:{seed}".encode()).digest()[:16], "little"
    )


def _materialize_global_shuffle(
    *,
    partial: Path,
    order_path: Path,
    sequence_path: Path,
    lengths: list[np.ndarray],
    readers: list[_TokenCorpusReader],
    sequence_count: int,
    sequence_length: int,
    padding_token_id: int,
    seed: int,
    get_desired_state: GetDesiredState | None,
    report: Any,
    file_key_prefix: str = "materialized",
    filename_prefix: str = "materialized",
    shuffle_namespace: str = "materialized-global-order",
    progress_phase: str = "materializing_global_shuffle",
    progress_start: float = 0.45,
    progress_span: float = 0.5,
    prefix_root: Path | None = None,
    prefix_manifest: dict[str, Any] | None = None,
    prefix_sequence_count: int = 0,
    prefix_identity_hash: str = "",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    token_dtypes = {reader.token_dtype for reader in readers}
    token_dtype = (
        np.dtype("<u4") if np.dtype("<u4") in token_dtypes else np.dtype("<u2")
    )
    if not 0 <= padding_token_id <= int(np.iinfo(token_dtype).max):
        raise ValueError("padding_token_id does not fit the materialized token dtype")
    boundary_bytes = (sequence_length + 7) // 8
    estimated_size = materialized_sequence_size_bytes(
        sequence_count=sequence_count,
        sequence_length=sequence_length,
        token_bytes=token_dtype.itemsize,
    )
    free_bytes = shutil.disk_usage(partial).free
    safety_bytes = (
        max(1_073_741_824, math.ceil(estimated_size * 0.05))
        if estimated_size
        else 0
    )
    if free_bytes < estimated_size + safety_bytes:
        raise ValueError(
            "materialized global shuffle has insufficient free disk space: "
            f"requires {estimated_size + safety_bytes:,} bytes including safety margin, "
            f"available {free_bytes:,}"
        )

    if sequence_count == 0:
        empty_digest = hashlib.sha256(b"").hexdigest()
        files: dict[str, dict[str, Any]] = {}
        for suffix, filename_suffix in (
            ("tokens", "tokens.bin"),
            ("document_starts", "document-starts.bin"),
            ("valid_lengths", "valid-lengths.idx"),
        ):
            key = f"{file_key_prefix}_{suffix}"
            path = partial / f"{filename_prefix}-{filename_suffix}"
            path.write_bytes(b"")
            files[key] = {
                "path": path.name,
                "size_bytes": 0,
                "sha256": empty_digest,
            }
        return files, {
            f"{file_key_prefix}_token_dtype": token_dtype.str,
            f"{file_key_prefix}_boundary_bytes_per_sequence": boundary_bytes,
            f"{file_key_prefix}_sequence_order_algorithm": (
                "sha256_seeded_numpy_pcg64_permutation_v1"
            ),
            f"{file_key_prefix}_sequence_size_bytes": 0,
        }

    order = np.memmap(
        order_path,
        dtype=np.dtype([("source", "<u4"), ("ordinal", "<u8")]),
        mode="r",
    )
    sequences = np.memmap(
        sequence_path,
        dtype=np.dtype([("start", "<u8"), ("valid", "<u4"), ("reserved", "<u4")]),
        mode="r",
    )
    selected_lengths = np.fromiter(
        (
            lengths[int(record["source"])][int(record["ordinal"])]
            for record in order
        ),
        dtype=np.uint64,
        count=len(order),
    )
    document_ends = np.cumsum(selected_lengths, dtype=np.uint64)
    if not 0 <= prefix_sequence_count <= sequence_count:
        raise ValueError("materialized View prefix sequence count is out of range")
    if prefix_sequence_count:
        if prefix_root is None or prefix_manifest is None:
            raise ValueError("materialized View prefix files are unavailable")
        if prefix_manifest.get("materialized_token_dtype") != token_dtype.str:
            raise ValueError("materialized View prefix token dtype changed")
        if int(prefix_manifest.get("materialized_boundary_bytes_per_sequence") or 0) != (
            boundary_bytes
        ):
            raise ValueError("materialized View prefix boundary width changed")
        new_physical_order = np.random.Generator(
            np.random.PCG64(
                _shuffle_seed(
                    (
                        f"{shuffle_namespace}-extension:"
                        f"{prefix_identity_hash}:{prefix_sequence_count}"
                    ),
                    seed,
                )
            )
        ).permutation(sequence_count - prefix_sequence_count) + prefix_sequence_count
    else:
        new_physical_order = np.random.Generator(
            np.random.PCG64(_shuffle_seed(shuffle_namespace, seed))
        ).permutation(sequence_count)

    token_path = partial / f"{filename_prefix}-tokens.bin"
    starts_path = partial / f"{filename_prefix}-document-starts.bin"
    valid_path = partial / f"{filename_prefix}-valid-lengths.idx"
    token_digest = hashlib.sha256()
    starts_digest = hashlib.sha256()
    valid_digest = hashlib.sha256()
    token_buffer = bytearray()
    starts_buffer = bytearray()
    valid_buffer = bytearray()
    flush_rows = max(1, min(256, 4 * 1024 * 1024 // (sequence_length * token_dtype.itemsize)))

    with (
        token_path.open("wb") as token_stream,
        starts_path.open("wb") as starts_stream,
        valid_path.open("wb") as valid_stream,
    ):
        if prefix_sequence_count:
            assert prefix_root is not None and prefix_manifest is not None
            prefix_files = (
                ("materialized_tokens", token_stream, token_digest),
                (
                    "materialized_document_starts",
                    starts_stream,
                    starts_digest,
                ),
                ("materialized_valid_lengths", valid_stream, valid_digest),
            )
            expected_prefix_sizes = {
                "materialized_tokens": (
                    prefix_sequence_count * sequence_length * token_dtype.itemsize
                ),
                "materialized_document_starts": (
                    prefix_sequence_count * boundary_bytes
                ),
                "materialized_valid_lengths": prefix_sequence_count * 4,
            }
            for key, stream, digest in prefix_files:
                prefix_path = _manifest_file(prefix_root, prefix_manifest, key)
                if prefix_path.stat().st_size != expected_prefix_sizes[key]:
                    raise ValueError(
                        f"materialized View prefix file {key} has an invalid row count"
                    )
                _copy_with_digest(prefix_path, stream, digest)
        for extension_index, physical_index_value in enumerate(new_physical_order):
            logical_index = prefix_sequence_count + extension_index
            desired = get_desired_state() if get_desired_state else "running"
            if desired in {"paused", "cancelled"}:
                raise AnalysisInterrupted(desired)
            physical_index = int(physical_index_value)
            sequence_record = sequences[physical_index]
            logical_offset = int(sequence_record["start"])
            valid_length = int(sequence_record["valid"])
            tokens = np.full(sequence_length, padding_token_id, dtype=token_dtype)
            starts = np.zeros(sequence_length, dtype=np.bool_)
            starts[0] = True
            written = 0
            document_position = bisect_right(document_ends, logical_offset)
            previous_end = (
                0 if document_position == 0 else int(document_ends[document_position - 1])
            )
            document_offset = logical_offset - previous_end
            while written < valid_length:
                order_record = order[document_position]
                source = int(order_record["source"])
                ordinal = int(order_record["ordinal"])
                length = int(lengths[source][ordinal])
                take = min(valid_length - written, length - document_offset)
                tokens[written : written + take] = readers[source].tokens(
                    ordinal, document_offset, take
                )
                if written:
                    starts[written] = True
                written += take
                document_position += 1
                document_offset = 0
            if 0 < valid_length < sequence_length:
                starts[valid_length] = True

            token_buffer.extend(tokens.tobytes(order="C"))
            starts_buffer.extend(
                np.packbits(starts, bitorder="little").tobytes(order="C")
            )
            valid_buffer.extend(struct.pack("<I", valid_length))
            if (extension_index + 1) % flush_rows == 0 or logical_index + 1 == sequence_count:
                token_stream.write(token_buffer)
                starts_stream.write(starts_buffer)
                valid_stream.write(valid_buffer)
                token_digest.update(token_buffer)
                starts_digest.update(starts_buffer)
                valid_digest.update(valid_buffer)
                token_buffer.clear()
                starts_buffer.clear()
                valid_buffer.clear()
            if (logical_index + 1) % 4096 == 0 or logical_index + 1 == sequence_count:
                report(
                    {
                        "phase": progress_phase,
                        "progress": progress_start
                        + progress_span * (logical_index + 1) / max(sequence_count, 1),
                        "materialized_sequences": logical_index + 1,
                        "sequence_count": sequence_count,
                        "estimated_materialized_bytes": estimated_size,
                    }
                )

    def written_spec(path: Path, digest: Any) -> dict[str, Any]:
        return {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }

    files = {
        f"{file_key_prefix}_tokens": written_spec(token_path, token_digest),
        f"{file_key_prefix}_document_starts": written_spec(starts_path, starts_digest),
        f"{file_key_prefix}_valid_lengths": written_spec(valid_path, valid_digest),
    }
    metadata = {
        f"{file_key_prefix}_token_dtype": token_dtype.str,
        f"{file_key_prefix}_boundary_bytes_per_sequence": boundary_bytes,
        f"{file_key_prefix}_sequence_order_algorithm": (
            "prefix_copy_then_sha256_seeded_numpy_pcg64_extension_v1"
            if prefix_sequence_count
            else "sha256_seeded_numpy_pcg64_permutation_v1"
        ),
        f"{file_key_prefix}_sequence_size_bytes": estimated_size,
    }
    return files, metadata


def _random_ordinals(count: int, size: int, seed: int) -> list[int]:
    if not 0 <= size <= count or count > 2**64:
        raise ValueError("validation holdout document count is out of range")
    if size == count:
        return list(range(count))
    select_complement = size > count // 2
    target = count - size if select_complement else size
    selected: set[int] = set()
    bit_generator = np.random.PCG64(seed)
    uint64_range = 2**64
    unbiased_limit = uint64_range - (uint64_range % count)
    while len(selected) < target:
        value = int(bit_generator.random_raw())
        if value < unbiased_limit:
            selected.add(value % count)
    if select_complement:
        return [ordinal for ordinal in range(count) if ordinal not in selected]
    return sorted(selected)


def _build_validation_holdout(
    *,
    partial: Path,
    configuration: dict[str, Any],
    source_specs: list[dict[str, Any]],
    lengths: list[np.ndarray],
    readers: list[_TokenCorpusReader],
    sequence_length: int,
    padding_token_id: int,
    get_desired_state: GetDesiredState | None,
    report: Any,
    allowed_ordinals: list[np.ndarray] | None = None,
) -> tuple[list[set[int]], dict[str, dict[str, Any]], dict[str, Any]]:
    holdout = configuration.get("validation_holdout")
    if not isinstance(holdout, dict):
        return [set() for _source in source_specs], {}, {}
    if holdout.get("strategy") != "weighted_random_tokenized_document_holdout_v1":
        raise ValueError("training data view validation holdout strategy is unsupported")
    target_tokens = int(holdout.get("target_tokens") or 0)
    seed = int(holdout.get("seed") or 0)
    validation_manifest_hash = str(holdout.get("validation_manifest_hash") or "")
    validation_mixture_version_id = str(
        holdout.get("validation_mixture_version_id") or ""
    )
    configured_sources = list(holdout.get("sources") or [])
    if (
        target_tokens <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", validation_manifest_hash)
        or not validation_mixture_version_id
        or len(configured_sources) != len(source_specs)
    ):
        raise ValueError("training data view validation holdout configuration is invalid")
    for configured, source in zip(configured_sources, source_specs, strict=True):
        if str(configured.get("dataset_version_id") or "") != str(
            source.get("dataset_version_id") or ""
        ):
            raise ValueError("validation holdout sources do not match Tokenized Corpora")

    target_sequence_count = math.ceil(target_tokens / sequence_length)
    packed_target_tokens = target_sequence_count * sequence_length
    weights = [max(int(source.get("weight_units") or 0), 0) for source in configured_sources]
    if not any(weights):
        weights = [1 for _source in configured_sources]
    total_weight = sum(weights)
    request_payload = {
        "strategy": holdout["strategy"],
        "target_tokens": target_tokens,
        "packed_target_tokens": packed_target_tokens,
        "sequence_length": sequence_length,
        "seed": seed,
        "validation_mixture_version_id": validation_mixture_version_id,
        "validation_manifest_hash": validation_manifest_hash,
        "sources": [
            {
                "dataset_version_id": source["dataset_version_id"],
                "weight_units": weight,
                "corpus_identity_hash": source["identity_hash"],
                "document_count": source.get("selected_document_count", source["document_count"]),
                "token_count": source.get("selected_token_count", source["token_count"]),
                **({"selection_manifest_hash": source["selection_input"]["manifest_hash"]} if source.get("selection_input") else {}),
            }
            for source, weight in zip(source_specs, weights, strict=True)
        ],
    }
    request_identity_hash = hashlib.sha256(_canonical_json(request_payload)).hexdigest()
    excluded: list[set[int]] = []
    selected_records: list[tuple[bytes, int, int, int]] = []
    source_results: list[dict[str, Any]] = []
    assigned_tokens = 0
    assigned_weight = 0
    for source_index, (source_lengths, source, weight) in enumerate(
        zip(lengths, source_specs, weights, strict=True)
    ):
        assigned_weight += weight
        next_assigned_tokens = packed_target_tokens * assigned_weight // total_weight
        quota = next_assigned_tokens - assigned_tokens
        assigned_tokens = next_assigned_tokens
        if quota <= 0:
            excluded.append(set())
            source_results.append(
                {
                    "source_index": source_index,
                    "dataset_version_id": source["dataset_version_id"],
                    "document_count": 0,
                    "token_count": 0,
                }
            )
            continue
        available = allowed_ordinals[source_index] if allowed_ordinals is not None else np.arange(len(source_lengths))
        document_count = len(available)
        total_source_tokens = int(source_lengths[available].sum(dtype=np.uint64))
        if document_count <= 0 or total_source_tokens < quota:
            raise ValueError("Tokenized Corpus cannot satisfy validation holdout quota")
        average_tokens = total_source_tokens / document_count
        selected_count = min(
            document_count,
            max(1, math.ceil(quota / max(average_tokens, 1) * 1.25) + 16),
        )
        selected: list[tuple[bytes, int, int, int]] = []
        while True:
            ordinals = _random_ordinals(
                document_count,
                selected_count,
                _shuffle_seed(
                    f"validation-holdout-select:{request_identity_hash}:{source_index}",
                    seed,
                ),
            )
            ordinals = available[ordinals]
            candidates = sorted(
                (
                    hashlib.sha256(
                        (
                            "validation-holdout-order:"
                            f"{request_identity_hash}:{source_index}:{ordinal}"
                        ).encode()
                    ).digest(),
                    source_index,
                    ordinal,
                    int(source_lengths[ordinal]),
                )
                for ordinal in ordinals
            )
            selected = []
            selected_tokens = 0
            for candidate in candidates:
                selected.append(candidate)
                selected_tokens += candidate[3]
                if selected_tokens >= quota:
                    break
            if selected_tokens >= quota:
                break
            if selected_count == document_count:
                raise ValueError("Tokenized Corpus cannot fill validation holdout target")
            selected_count = min(
                document_count,
                max(selected_count + 1, selected_count * 2),
            )
        source_excluded = {record[2] for record in selected}
        excluded.append(source_excluded)
        selected_records.extend(selected)
        source_results.append(
            {
                "source_index": source_index,
                "dataset_version_id": source["dataset_version_id"],
                "document_count": len(selected),
                "token_count": sum(record[3] for record in selected),
            }
        )

    selected_records.sort(key=lambda record: (record[0], record[1], record[2]))
    selected_token_count = sum(record[3] for record in selected_records)
    if selected_token_count < packed_target_tokens:
        raise ValueError("validation holdout selection is shorter than its packed target")
    order_path = partial / "validation-holdout-document-order.idx"
    with order_path.open("wb") as stream:
        for _key, source_index, ordinal, _length in selected_records:
            stream.write(struct.pack("<IQ", source_index, ordinal))
    sequence_path = partial / "validation-holdout-sequence-index.idx"
    with sequence_path.open("wb") as stream:
        for index in range(target_sequence_count):
            stream.write(struct.pack("<QII", index * sequence_length, sequence_length, 0))
    materialized_files, materialized_metadata = _materialize_global_shuffle(
        partial=partial,
        order_path=order_path,
        sequence_path=sequence_path,
        lengths=lengths,
        readers=readers,
        sequence_count=target_sequence_count,
        sequence_length=sequence_length,
        padding_token_id=padding_token_id,
        seed=seed,
        get_desired_state=get_desired_state,
        report=report,
        file_key_prefix="validation_holdout",
        filename_prefix="validation-holdout",
        shuffle_namespace=f"validation-holdout-sequence-order:{request_identity_hash}",
        progress_phase="materializing_validation_holdout",
        progress_start=0.02,
        progress_span=0.08,
    )
    files = {
        "validation_holdout_document_order": _file_spec(order_path),
        "validation_holdout_sequence_index": _file_spec(sequence_path),
        **materialized_files,
    }
    content_payload = {
        "request_identity_hash": request_identity_hash,
        "files": {
            key: {"size_bytes": spec["size_bytes"], "sha256": spec["sha256"]}
            for key, spec in files.items()
        },
    }
    identity_hash = hashlib.sha256(_canonical_json(content_payload)).hexdigest()
    metadata = {
        "strategy": holdout["strategy"],
        "validation_mixture_version_id": validation_mixture_version_id,
        "request_identity_hash": request_identity_hash,
        "identity_hash": identity_hash,
        "target_tokens": target_tokens,
        "token_count": packed_target_tokens,
        "selected_token_count": selected_token_count,
        "sequence_count": target_sequence_count,
        "document_count": len(selected_records),
        "sources": source_results,
        "files": files,
        **materialized_metadata,
    }
    return excluded, files, metadata


class TrainingDataIndexAdapter:
    """Build a deterministic logical sample index over reusable token corpora."""

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        runtime = task.get("runtime_context") or {}
        identity_hash = str(payload.get("identity_hash") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", identity_hash):
            raise ValueError("training data view requires a frozen SHA-256 identity")
        config = dict(payload.get("configuration") or {})
        sequence_length = int(config.get("sequence_length") or 0)
        requested_tokens = int(config.get("requested_tokens") or 0)
        consumption = str(config.get("consumption_policy") or "")
        tail = str(config.get("tail_policy") or "")
        packing = str(config.get("packing_policy") or "")
        sampling = str(config.get("sampling_policy") or "")
        sequence_order = str(
            config.get("sequence_order_policy") or "sequential_v1"
        )
        sequence_strategy = sequence_order_strategy(sequence_order)
        shuffle_block_sequences = config.get("shuffle_block_sequences")
        seed = int(config.get("seed", 42))
        if sequence_length <= 0 or requested_tokens <= 0:
            raise ValueError("training data view dimensions must be positive")
        if consumption not in {"unique_only", "repeat_epochs"}:
            raise ValueError("unsupported data consumption policy")
        if tail not in {"drop", "pad"}:
            raise ValueError("unsupported tail policy")
        if packing != "document_stream_v1":
            raise ValueError("unsupported packing policy")
        if sampling != "weighted_deficit_v1":
            raise ValueError("unsupported sampling policy")
        if sequence_strategy["requires_block_size"]:
            shuffle_block_sequences = int(shuffle_block_sequences or 0)
            if not 1 <= shuffle_block_sequences <= 4096:
                raise ValueError(
                    "block shuffle requires shuffle_block_sequences in [1, 4096]"
                )
        if tail == "pad" and config.get("padding_token_id") is None:
            raise ValueError("tail policy pad requires an explicit padding_token_id")
        bindings = list(payload.get("tokenized_corpora") or [])
        if not bindings:
            raise ValueError("training data view has no Tokenized Corpus bindings")
        frozen_selections = {binding["source_index"]: binding["selection_input"] for binding in bindings if binding.get("selection_input")}
        runtime_selections = payload.get("training_selection_inputs") or []
        if len(runtime_selections) != len(frozen_selections) or {item["source_index"] for item in runtime_selections} != set(frozen_selections):
            raise ValueError("Training selection bindings are missing or duplicated")
        for item in runtime_selections:
            if any(item.get(key) != value for key, value in frozen_selections[item["source_index"]].items()):
                raise ValueError("Training selection differs from the frozen Corpus binding")
        prefix_binding = dict(payload.get("prefix_training_data_view") or {})

        project_workspace = Path(str(runtime.get("project_workspace") or "")).resolve()
        if not project_workspace.is_dir():
            raise ValueError("training data view project workspace is unavailable")
        storage_root = (
            derived_asset_root(project_workspace) / "training-data-views"
        ).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        destination = (storage_root / identity_hash).resolve()
        if destination.parent != storage_root:
            raise ValueError("training data view destination escaped project storage")
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if manifest.get("identity_hash") != identity_hash:
                raise ValueError("existing training data view has a different identity")
            return self._result(destination, manifest, hashlib.sha256(manifest_bytes).hexdigest())
        if destination.exists():
            raise ValueError("incomplete training data view occupies the immutable destination")

        partial = (storage_root / f".{identity_hash}.{task['id']}.partial").resolve()
        if partial.parent != storage_root:
            raise ValueError("training data view partial path escaped project storage")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        loop = asyncio.get_running_loop()

        def report(event: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(emit("progress", event), loop)
            future.result()

        def build() -> tuple[dict[str, Any], str]:
            lengths: list[np.ndarray] = []
            allowed_ordinals: list[np.ndarray] = []
            selection_files = {}
            readers: list[_TokenCorpusReader] = []
            source_specs: list[dict[str, Any]] = []
            for binding in sorted(bindings, key=lambda item: int(item["source_index"])):
                root = Path(str(binding.get("runtime_path") or "")).resolve()
                raw_manifest = (root / "manifest.json").read_bytes()
                manifest = json.loads(raw_manifest)
                if hashlib.sha256(raw_manifest).hexdigest() != binding["content_hash"]:
                    raise ValueError("Tokenized Corpus manifest hash changed during index build")
                source_lengths = _document_lengths(root, manifest)
                if len(source_lengths) != int(binding.get("document_count") or 0):
                    raise ValueError("Tokenized Corpus document count does not match its index")
                lengths.append(source_lengths)
                selection_binding = next((item for item in payload.get("training_selection_inputs") or [] if item["source_index"] == binding["source_index"]), None)
                allowed = selected_ordinals(root, manifest, selection_binding, partial / f"selection-{binding['source_index']}", get_desired_state) if selection_binding else np.arange(len(source_lengths))
                allowed_ordinals.append(allowed)
                if selection_binding:
                    selection_path = partial / f"selected-source-{binding['source_index']:05d}.idx"
                    allowed.astype("<u8").tofile(selection_path)
                    selection_files[f"selected_source_{binding['source_index']}"] = _file_spec(selection_path)
                readers.append(_TokenCorpusReader(root, manifest))
                source_specs.append(
                    {
                        key: binding[key]
                        for key in (
                            "source_index",
                            "dataset_version_id",
                            "weight_units",
                            "normalized_weight",
                            "version_id",
                            "identity_hash",
                            "document_count",
                            "token_count",
                            "artifact_id",
                            "content_hash",
                        )
                    }
                )
                if selection_binding:
                    source_specs[-1].update(selection_input=binding["selection_input"], selected_document_count=len(allowed), selected_token_count=int(source_lengths[allowed].sum(dtype=np.uint64)))

            prefix_root: Path | None = None
            prefix_manifest: dict[str, Any] | None = None
            prefix_sequence_count = 0
            prefix_identity_hash = ""
            if prefix_binding:
                prefix_root = Path(
                    str(prefix_binding.get("runtime_path") or "")
                ).resolve()
                raw_prefix_manifest = (prefix_root / "manifest.json").read_bytes()
                if hashlib.sha256(raw_prefix_manifest).hexdigest() != prefix_binding.get(
                    "content_hash"
                ):
                    raise ValueError("Training Data View prefix manifest hash changed")
                prefix_manifest = json.loads(raw_prefix_manifest)
                prefix_identity_hash = str(
                    prefix_binding.get("identity_hash") or ""
                )
                if (
                    prefix_manifest.get("format") != TRAINING_DATA_VIEW_FORMAT
                    or prefix_manifest.get("identity_hash") != prefix_identity_hash
                ):
                    raise ValueError("Training Data View prefix identity is invalid")
                prefix_config = dict(prefix_manifest.get("configuration") or {})
                comparable_prefix = {
                    key: value
                    for key, value in prefix_config.items()
                    if key != "requested_tokens"
                }
                comparable_config = {
                    key: value
                    for key, value in config.items()
                    if key != "requested_tokens"
                }
                prefix_sources = [
                    {
                        key: source.get(key)
                        for key in (
                            "source_index",
                            "version_id",
                            "identity_hash",
                            "content_hash",
                        )
                    }
                    for source in (prefix_manifest.get("sources") or [])
                ]
                current_sources = [
                    {
                        key: source.get(key)
                        for key in (
                            "source_index",
                            "version_id",
                            "identity_hash",
                            "content_hash",
                        )
                    }
                    for source in source_specs
                ]
                if comparable_prefix != comparable_config or prefix_sources != current_sources:
                    raise ValueError(
                        "Training Data View prefix data or ordering semantics changed"
                    )
                prefix_sequence_count = int(
                    prefix_manifest.get("sequence_count") or 0
                )
                if prefix_sequence_count != int(
                    prefix_binding.get("sequence_count") or 0
                ):
                    raise ValueError("Training Data View prefix sequence count changed")

            holdout_excluded, holdout_files, holdout_metadata = (
                _build_validation_holdout(
                    partial=partial,
                    configuration=config,
                    source_specs=source_specs,
                    lengths=lengths,
                    allowed_ordinals=allowed_ordinals,
                    readers=readers,
                    sequence_length=sequence_length,
                    padding_token_id=int(config.get("padding_token_id") or 0),
                    get_desired_state=get_desired_state,
                    report=report,
                )
            )
            weights = [max(int(item.get("weight_units") or 0), 0) for item in source_specs]
            if not any(weights):
                weights = [max(int(float(item.get("normalized_weight") or 0) * 1_000_000), 1) for item in source_specs]
            cursors = [0] * len(source_specs)
            total_configured_weight = max(sum(weights), 1)
            scores = [
                int.from_bytes(
                    hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8],
                    "big",
                )
                % total_configured_weight
                for index in range(len(source_specs))
            ]
            logical_tokens = 0
            selected_documents = 0
            target_sequences = math.ceil(requested_tokens / sequence_length)
            epoch_sequences = int(config.get("repeat_sequence_count") or 0)
            if epoch_sequences and (
                consumption != "repeat_epochs" or sequence_order != "block_shuffle_v1"
                or epoch_sequences % int(shuffle_block_sequences or 1)
                or target_sequences % epoch_sequences or prefix_binding
            ):
                raise ValueError("fixed repeated epochs require complete block-aligned epochs without a prefix")
            target_stream_tokens = (epoch_sequences or target_sequences) * sequence_length
            order_path = partial / "document-order.idx"
            order_stream: BinaryIO
            try:
                with order_path.open("wb") as order_stream:
                    while logical_tokens < target_stream_tokens:
                        desired = get_desired_state() if get_desired_state else "running"
                        if desired in {"paused", "cancelled"}:
                            raise AnalysisInterrupted(desired)
                        eligible: list[int] = []
                        for index, source_ordinals in enumerate(allowed_ordinals):
                            usable_documents = len(source_ordinals) - len(
                                holdout_excluded[index]
                            )
                            if usable_documents <= 0:
                                continue
                            if consumption == "unique_only":
                                while (
                                    cursors[index] < len(source_ordinals)
                                    and int(source_ordinals[cursors[index]]) in holdout_excluded[index]
                                ):
                                    cursors[index] += 1
                                if cursors[index] >= len(source_ordinals):
                                    continue
                            eligible.append(index)
                        if not eligible:
                            break
                        total_weight = sum(weights[index] for index in eligible)
                        for index in eligible:
                            scores[index] += weights[index]
                        chosen = max(eligible, key=lambda index: (scores[index], -index))
                        scores[chosen] -= total_weight
                        ordinal = int(allowed_ordinals[chosen][cursors[chosen] % len(allowed_ordinals[chosen])])
                        while ordinal in holdout_excluded[chosen]:
                            cursors[chosen] += 1
                            ordinal = int(allowed_ordinals[chosen][cursors[chosen] % len(allowed_ordinals[chosen])])
                        cursors[chosen] += 1
                        length = int(lengths[chosen][ordinal])
                        order_stream.write(struct.pack("<IQ", chosen, ordinal))
                        logical_tokens += length
                        selected_documents += 1
                        if selected_documents % 4096 == 0:
                            progress_start = 0.1 if holdout_metadata else 0.0
                            progress_ceiling = (
                                0.45
                                if sequence_strategy["materializes_sequences"]
                                else 0.95
                            )
                            report(
                                {
                                    "phase": "building_document_order",
                                    "progress": progress_start
                                    + min(
                                        logical_tokens / target_stream_tokens,
                                        1.0,
                                    )
                                    * (progress_ceiling - progress_start),
                                    "selected_documents": selected_documents,
                                    "logical_tokens": logical_tokens,
                                    "requested_tokens": requested_tokens,
                                    "configuration": config,
                                }
                            )
            except BaseException:
                order_path.unlink(missing_ok=True)
                raise

            full_sequences = logical_tokens // sequence_length
            remainder = logical_tokens % sequence_length
            sequence_count = min(full_sequences, target_sequences)
            if tail == "pad" and sequence_count < target_sequences and remainder:
                sequence_count += 1
            valid_tokens = min(logical_tokens, sequence_count * sequence_length)
            if epoch_sequences:
                if full_sequences < epoch_sequences:
                    raise ValueError("corpus does not cover the fixed epoch")
                sequence_count = target_sequences
                valid_tokens = sequence_count * sequence_length
            sequence_path = partial / "sequence-index.idx"
            with sequence_path.open("wb") as stream:
                for index in range(sequence_count):
                    start = (index % epoch_sequences if epoch_sequences else index) * sequence_length
                    valid_length = min(sequence_length, max(valid_tokens - start, 0))
                    stream.write(struct.pack("<QII", start, valid_length, 0))

            if prefix_manifest is not None and prefix_root is not None:
                if sequence_count < prefix_sequence_count:
                    raise ValueError(
                        "expanded Training Data View is shorter than its frozen prefix"
                    )
                _require_file_prefix(
                    _manifest_file(prefix_root, prefix_manifest, "document_order"),
                    order_path,
                    "document order",
                )
                _require_file_prefix(
                    _manifest_file(prefix_root, prefix_manifest, "sequence_index"),
                    sequence_path,
                    "sequence index",
                )

            block_order_path: Path | None = None
            full_sequence_block_count = 0
            sequence_block_shuffle_algorithm = ""
            if sequence_order == "block_shuffle_v1":
                block_order_path = partial / "sequence-block-order.idx"
                full_sequence_block_count = sequence_count // int(
                    shuffle_block_sequences
                )
                if prefix_manifest is not None and prefix_root is not None:
                    block_size = int(shuffle_block_sequences)
                    if prefix_sequence_count % block_size:
                        raise ValueError(
                            "block-shuffled View prefix is not block aligned"
                        )
                    prefix_block_count = prefix_sequence_count // block_size
                    if int(prefix_manifest.get("full_sequence_block_count") or -1) != (
                        prefix_block_count
                    ):
                        raise ValueError(
                            "block-shuffled View prefix block count is invalid"
                        )
                    prefix_block_path = _manifest_file(
                        prefix_root, prefix_manifest, "sequence_block_order"
                    )
                    prefix_block_order = np.fromfile(
                        prefix_block_path, dtype="<u8"
                    )
                    if (
                        len(prefix_block_order) != prefix_block_count
                        or not np.array_equal(
                            np.sort(prefix_block_order),
                            np.arange(prefix_block_count, dtype="<u8"),
                        )
                    ):
                        raise ValueError(
                            "block-shuffled View prefix is not a complete permutation"
                        )
                    extension_block_count = (
                        full_sequence_block_count - prefix_block_count
                    )
                    extension_order = np.random.Generator(
                        np.random.PCG64(
                            _shuffle_seed(
                                (
                                    "sequence-block-order-extension:"
                                    f"{prefix_identity_hash}:{prefix_block_count}"
                                ),
                                seed,
                            )
                        )
                    ).permutation(extension_block_count) + prefix_block_count
                    block_order = np.concatenate(
                        (prefix_block_order, extension_order)
                    )
                    sequence_block_shuffle_algorithm = (
                        "prefix_copy_then_sha256_seeded_numpy_pcg64_extension_v1"
                    )
                elif epoch_sequences:
                    blocks_per_epoch = epoch_sequences // int(shuffle_block_sequences)
                    block_order = np.concatenate([
                        np.random.Generator(np.random.PCG64(
                            _shuffle_seed(f"sequence-block-order-epoch:{epoch}", seed)
                        )).permutation(blocks_per_epoch) + epoch * blocks_per_epoch
                        for epoch in range(sequence_count // epoch_sequences)
                    ])
                    sequence_block_shuffle_algorithm = "sha256_seeded_numpy_pcg64_per_epoch_v1"
                else:
                    block_order = np.random.Generator(
                        np.random.PCG64(
                            _shuffle_seed("sequence-block-order", seed)
                        )
                    ).permutation(full_sequence_block_count)
                    sequence_block_shuffle_algorithm = (
                        "sha256_seeded_numpy_pcg64_permutation_v1"
                    )
                np.asarray(block_order, dtype="<u8").tofile(block_order_path)

            materialized_files: dict[str, dict[str, Any]] = {}
            materialized_metadata: dict[str, Any] = {}
            if sequence_strategy["materializes_sequences"]:
                materialized_files, materialized_metadata = _materialize_global_shuffle(
                    partial=partial,
                    order_path=order_path,
                    sequence_path=sequence_path,
                    lengths=lengths,
                    readers=readers,
                    sequence_count=sequence_count,
                    sequence_length=sequence_length,
                    padding_token_id=int(config.get("padding_token_id") or 0),
                    seed=seed,
                    get_desired_state=get_desired_state,
                    report=report,
                    prefix_root=prefix_root,
                    prefix_manifest=prefix_manifest,
                    prefix_sequence_count=prefix_sequence_count,
                    prefix_identity_hash=prefix_identity_hash,
                )

            warnings = [
                warning
                for warning in ((payload.get("preflight") or {}).get("warnings") or [])
                if warning.get("code") != "selection_capacity_pending"
            ]
            if sequence_count < target_sequences:
                warnings.append(
                    {
                        "code": "view_shorter_than_request",
                        "message": (
                            f"唯一消费策略生成 {sequence_count:,} 个序列；请求为 "
                            f"{target_sequences:,}。制品保留实际结果，未判定为失败。"
                        ),
                    }
                )
            files = {
                "document_order": _file_spec(order_path),
                "sequence_index": _file_spec(sequence_path),
                **holdout_files,
                **selection_files,
            }
            if block_order_path is not None:
                files["sequence_block_order"] = _file_spec(block_order_path)
            files.update(materialized_files)
            manifest = {
                "format": TRAINING_DATA_VIEW_FORMAT,
                "format_version": 1,
                "identity_hash": identity_hash,
                "identity_snapshot": payload.get("identity_snapshot") or {},
                "configuration": config,
                "sources": source_specs,
                "document_order_record": "little-endian uint32 source_index + uint64 document_ordinal",
                "sequence_index_record": "little-endian uint64 logical_offset + uint32 valid_length + uint32 reserved",
                "selected_document_count": selected_documents,
                "logical_token_count": logical_tokens,
                "token_count": valid_tokens,
                "padded_token_count": sequence_count * sequence_length,
                "sequence_count": sequence_count,
                "requested_sequence_count": target_sequences,
                "warnings": warnings,
                "files": files,
            }
            if holdout_metadata:
                manifest["validation_holdout"] = holdout_metadata
            if block_order_path is not None:
                manifest.update(
                    {
                        "sequence_block_order_record": (
                            "little-endian uint64 physical full-block index"
                        ),
                        "sequence_block_shuffle_algorithm": (
                            sequence_block_shuffle_algorithm
                        ),
                        "full_sequence_block_count": full_sequence_block_count,
                    }
                )
            if materialized_metadata:
                manifest.update(materialized_metadata)
            if prefix_manifest is not None:
                manifest["prefix_lineage"] = payload.get("identity_snapshot", {}).get(
                    "prefix_lineage"
                )
            manifest_bytes = _canonical_json(manifest)
            (partial / "manifest.json").write_bytes(manifest_bytes)
            partial.replace(destination)
            return manifest, hashlib.sha256(manifest_bytes).hexdigest()

        try:
            manifest, manifest_hash = await asyncio.to_thread(build)
        except AnalysisInterrupted as exc:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            await emit("progress", {"phase": exc.state, "progress": 0.0})
            return {"controlled_state": exc.state}
        except Exception:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            raise
        await emit(
            "progress",
            {
                "phase": "completed",
                "progress": 1.0,
                "sequence_count": manifest["sequence_count"],
                "token_count": manifest["token_count"],
                "warnings": manifest["warnings"],
            },
        )
        return self._result(destination, manifest, manifest_hash)

    @staticmethod
    def _result(destination: Path, manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
        size_bytes = sum(int(spec["size_bytes"]) for spec in manifest["files"].values())
        return {
            "identity_hash": manifest["identity_hash"],
            "format": manifest["format"],
            "token_count": int(manifest["token_count"]),
            "sequence_count": int(manifest["sequence_count"]),
            "size_bytes": size_bytes,
            "warnings": list(manifest.get("warnings") or []),
            "outputs": [
                {
                    "ref": "training_data_view",
                    "kind": "training_data_index",
                    "storage_type": "external",
                    "uri": destination.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": size_bytes,
                    "media_type": "application/vnd.forgelm.training-index-v1",
                    "availability_state": "available",
                    "metadata": {
                        "format": manifest["format"],
                        "identity_hash": manifest["identity_hash"],
                        "sequence_count": manifest["sequence_count"],
                        "token_count": manifest["token_count"],
                        "configuration": manifest["configuration"],
                        "validation_holdout": manifest.get("validation_holdout"),
                        "warnings": manifest.get("warnings") or [],
                    },
                }
            ],
        }
