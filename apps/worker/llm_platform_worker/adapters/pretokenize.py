from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from llm_platform_core.storage_layout import derived_asset_root

from .command import Emit, GetDesiredState
from .data_import import AnalysisInterrupted
from .hf_tokenizer import HfTokenizerAdapter

TOKENIZED_DATA_FORMAT = "forgelm-packed-tokens-v1"


def pretokenize_available() -> bool:
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError:
        return False
    return True


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


class _ShardWriter:
    def __init__(
        self,
        root: Path,
        *,
        sequence_length: int,
        shard_sequences: int,
        token_dtype: np.dtype,
    ) -> None:
        self.root = root
        self.sequence_length = sequence_length
        self.shard_sequences = shard_sequences
        self.token_dtype = token_dtype
        self.boundary_bytes = math.ceil(sequence_length / 8)
        self.shards: list[dict[str, Any]] = []
        self.total_size = 0
        self._index = -1
        self._count = 0
        self._files: dict[str, BinaryIO] = {}
        self._hashes: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}

    def _open(self) -> None:
        self._index += 1
        self._count = 0
        stem = f"shard-{self._index:05d}"
        suffixes = {
            "tokens": f"{stem}.tokens.bin",
            "document_starts": f"{stem}.document_starts.bin",
        }
        self._paths = {key: self.root / name for key, name in suffixes.items()}
        self._files = {key: path.open("wb") for key, path in self._paths.items()}
        self._hashes = {key: hashlib.sha256() for key in self._files}

    def append(self, tokens: list[int], document_starts: list[bool]) -> None:
        if len(tokens) != self.sequence_length or len(document_starts) != self.sequence_length:
            raise ValueError("packed token rows must exactly match sequence_length")
        if not document_starts[0]:
            raise ValueError("every packed row must reset document state at position zero")
        if not self._files:
            self._open()
        token_bytes = np.asarray(tokens, dtype=self.token_dtype).tobytes(order="C")
        boundary_bytes = np.packbits(
            np.asarray(document_starts, dtype=np.uint8), bitorder="little"
        ).tobytes()
        if len(boundary_bytes) != self.boundary_bytes:
            raise RuntimeError("packed document boundary width is invalid")
        for key, payload in (
            ("tokens", token_bytes),
            ("document_starts", boundary_bytes),
        ):
            self._files[key].write(payload)
            self._hashes[key].update(payload)
        self._count += 1
        if self._count >= self.shard_sequences:
            self._close()

    def _close(self) -> None:
        if not self._files:
            return
        for stream in self._files.values():
            stream.flush()
            stream.close()
        files = {
            key: {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": self._hashes[key].hexdigest(),
            }
            for key, path in self._paths.items()
        }
        size_bytes = sum(int(value["size_bytes"]) for value in files.values())
        self.total_size += size_bytes
        self.shards.append(
            {
                "index": self._index,
                "sequence_count": self._count,
                "files": files,
                "size_bytes": size_bytes,
            }
        )
        self._files = {}
        self._hashes = {}
        self._paths = {}
        self._count = 0

    def finish(self) -> None:
        self._close()


def _batched_token_ids(texts: Iterator[str], tokenizer: Any) -> Iterator[list[int]]:
    batch: list[str] = []
    characters = 0

    def encode(values: list[str]) -> list[list[int]]:
        encoded = tokenizer(
            values,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        rows = encoded["input_ids"]
        if len(rows) != len(values):
            raise RuntimeError("batched tokenizer returned the wrong document count")
        return [[int(token) for token in row] for row in rows]

    for text in texts:
        if batch and (len(batch) >= 256 or characters + len(text) > 4_000_000):
            yield from encode(batch)
            batch = []
            characters = 0
        batch.append(text)
        characters += len(text)
    if batch:
        yield from encode(batch)


def _packed_rows(
    token_rows: Iterator[list[int]],
    *,
    eos_token_id: int,
    sequence_length: int,
    target_sequences: int,
    get_desired_state: GetDesiredState | None,
    report: Callable[[int, int], None],
) -> Iterator[tuple[list[int], list[bool]]]:
    tokens: list[int] = []
    starts: list[bool] = []
    emitted = 0
    documents = 0
    for document_tokens in token_rows:
        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            raise AnalysisInterrupted(desired)
        if not document_tokens:
            continue
        document_tokens = [*document_tokens, eos_token_id]
        offset = 0
        documents += 1
        while offset < len(document_tokens):
            available = sequence_length - len(tokens)
            take = min(available, len(document_tokens) - offset)
            chunk = document_tokens[offset : offset + take]
            chunk_starts = [False] * take
            if not tokens:
                # A model batch has no state from the previous packed row, including
                # when a very long source document continues across the boundary.
                chunk_starts[0] = True
            if offset == 0:
                chunk_starts[0] = True
            tokens.extend(chunk)
            starts.extend(chunk_starts)
            offset += take
            if len(tokens) == sequence_length:
                yield tokens, starts
                emitted += 1
                report(emitted, documents)
                if emitted >= target_sequences:
                    return
                tokens = []
                starts = []
    raise ValueError(
        f"frozen corpus produced only {emitted:,} packed sequences; "
        f"{target_sequences:,} are required"
    )


class PretokenizeAdapter:
    """Create immutable, document-aware packed Token ID shards for formal training."""

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        from transformers import AutoTokenizer

        payload = task.get("payload") or {}
        runtime = task.get("runtime_context") or {}
        identity_hash = str(payload.get("identity_hash") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", identity_hash):
            raise ValueError("pretokenization requires a frozen SHA-256 identity")
        sequence_length = int(payload.get("sequence_length") or 0)
        target_tokens = int(payload.get("target_tokens") or 0)
        shard_sequences = int(payload.get("shard_sequences") or 8192)
        if sequence_length <= 0 or target_tokens <= 0 or shard_sequences <= 0:
            raise ValueError("pretokenization dimensions must be positive")
        tokenizer_binding = payload.get("tokenizer") or {}
        tokenizer_path = Path(str(tokenizer_binding.get("runtime_path") or ""))
        if not tokenizer_path.is_dir():
            raise ValueError("pretokenization tokenizer was not materialized")
        data_input = payload.get("data_input") or {}
        project_workspace = Path(str(runtime.get("project_workspace") or "")).resolve()
        if not project_workspace.is_dir():
            raise ValueError("pretokenization project workspace is unavailable")
        storage_root = (
            derived_asset_root(project_workspace) / "tokenized"
        ).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        destination = (storage_root / identity_hash).resolve()
        if destination.parent != storage_root:
            raise ValueError("pretokenization destination escaped the project storage root")
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if manifest.get("identity_hash") != identity_hash:
                raise ValueError("existing tokenized dataset has a different identity")
            return self._result(destination, manifest, hashlib.sha256(manifest_bytes).hexdigest())
        if destination.exists():
            raise ValueError("incomplete tokenized dataset occupies the immutable destination")

        target_sequences = math.ceil(target_tokens / sequence_length)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        special_tokens = list(tokenizer_binding.get("special_tokens") or [])
        eos_token = str(special_tokens[2]) if len(special_tokens) >= 3 else "</s>"
        if tokenizer.eos_token_id is None and eos_token in tokenizer.get_vocab():
            tokenizer.eos_token = eos_token
        if tokenizer.eos_token_id is None:
            raise ValueError("pretokenization tokenizer must define eos_token_id")
        vocab_size = len(tokenizer)
        token_dtype = np.dtype("<u2" if vocab_size <= 65_536 else "<u4")
        expected_bytes = target_sequences * (
            sequence_length * token_dtype.itemsize + math.ceil(sequence_length / 8)
        )
        free_bytes = shutil.disk_usage(storage_root).free
        if free_bytes < math.ceil(expected_bytes * 1.05) + 256 * 1024 * 1024:
            raise OSError(
                f"tokenized shards need about {expected_bytes / (1024**3):.1f} GiB, "
                f"but only {free_bytes / (1024**3):.1f} GiB is free"
            )

        run_dir = Path(str(runtime.get("run_dir") or ".")).resolve()
        partial = (storage_root / f".{identity_hash}.{task['id']}.partial").resolve()
        if partial.parent != storage_root:
            raise ValueError("pretokenization partial path escaped the project storage root")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        loop = asyncio.get_running_loop()
        last_reported = 0

        def report(sequence_count: int, document_count: int) -> None:
            nonlocal last_reported
            if sequence_count < target_sequences and sequence_count - last_reported < 256:
                return
            last_reported = sequence_count
            future = asyncio.run_coroutine_threadsafe(
                emit(
                    "progress",
                    {
                        "phase": "packing_tokens",
                        "progress": 0.40
                        + min(sequence_count / target_sequences, 1.0) * 0.56,
                        "sequence_count": sequence_count,
                        "target_sequences": target_sequences,
                        "document_count": document_count,
                        "token_count": sequence_count * sequence_length,
                    },
                ),
                loop,
            )
            future.result()

        def report_corpus(payload: dict[str, Any]) -> None:
            phase = str(payload.get("phase") or "")
            if phase == "preparing_corpus":
                scanned = max(int(payload.get("scanned_manifest_records") or 0), 0)
                total = max(int(data_input.get("resolved_document_count") or 0), 1)
                progress = min(scanned / total, 1.0) * 0.05
            elif phase == "spooling_corpus":
                spooled = max(int(payload.get("spooled_documents") or 0), 0)
                total = max(int(payload.get("total_documents") or 0), 1)
                progress = 0.05 + min(spooled / total, 1.0) * 0.35
            else:
                return
            future = asyncio.run_coroutine_threadsafe(
                emit("progress", {**payload, "progress": progress}), loop
            )
            future.result()

        def build() -> tuple[dict[str, Any], str]:
            corpus_work_dir = run_dir / "corpus-index"
            corpus, _evaluation, _stats = HfTokenizerAdapter._corpus(
                data_input,
                math.ceil(target_tokens * 1.10),
                corpus_work_dir,
                report_progress=report_corpus,
                get_desired_state=get_desired_state,
                spool_path=corpus_work_dir / "ordered-corpus.sqlite3",
            )
            writer = _ShardWriter(
                partial,
                sequence_length=sequence_length,
                shard_sequences=shard_sequences,
                token_dtype=token_dtype,
            )
            try:
                rows = _packed_rows(
                    _batched_token_ids(corpus, tokenizer),
                    eos_token_id=int(tokenizer.eos_token_id),
                    sequence_length=sequence_length,
                    target_sequences=target_sequences,
                    get_desired_state=get_desired_state,
                    report=report,
                )
                for token_row, document_starts in rows:
                    writer.append(token_row, document_starts)
            finally:
                writer.finish()
            manifest = {
                "format": TOKENIZED_DATA_FORMAT,
                "format_version": 1,
                "identity_hash": identity_hash,
                "packing_version": str(payload.get("packing_version") or "document-pack-v1"),
                "sequence_length": sequence_length,
                "target_tokens": target_tokens,
                "token_count": target_sequences * sequence_length,
                "sequence_count": target_sequences,
                "token_dtype": token_dtype.str,
                "boundary_bitorder": "little",
                "boundary_bytes_per_sequence": math.ceil(sequence_length / 8),
                "vocab_size": vocab_size,
                "eos_token_id": int(tokenizer.eos_token_id),
                "mixture_manifest_hash": str(data_input.get("manifest_hash") or ""),
                "tokenizer_content_hash": str(tokenizer_binding.get("content_hash") or ""),
                "shards": writer.shards,
            }
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
            },
        )
        return self._result(destination, manifest, manifest_hash)

    @staticmethod
    def _result(destination: Path, manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
        size_bytes = int(sum(int(shard.get("size_bytes") or 0) for shard in manifest["shards"]))
        return {
            "identity_hash": manifest["identity_hash"],
            "format": manifest["format"],
            "token_count": int(manifest["token_count"]),
            "sequence_count": int(manifest["sequence_count"]),
            "shard_count": len(manifest["shards"]),
            "size_bytes": size_bytes,
            "outputs": [
                {
                    "ref": "tokenized_dataset",
                    "kind": "tokenized_dataset",
                    "storage_type": "external",
                    "uri": destination.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": size_bytes,
                    "media_type": "application/vnd.forgelm.packed-tokens-v1",
                    "availability_state": "available",
                    "metadata": {
                        "format": manifest["format"],
                        "identity_hash": manifest["identity_hash"],
                        "sequence_length": manifest["sequence_length"],
                        "token_count": manifest["token_count"],
                        "shard_count": len(manifest["shards"]),
                    },
                }
            ],
        }
