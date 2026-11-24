from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import re
import shutil
import struct
import time
from collections import deque
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from llm_platform_core.storage_layout import derived_asset_root

from .command import Emit, GetDesiredState
from .corpus_selection import ordered_records, selected_records
from .data_import import AnalysisInterrupted
from .hf_tokenizer import HfTokenizerAdapter
from .pretokenize import _batched_token_ids

TOKENIZED_CORPUS_FORMAT = "forgelm-token-corpus-v2"


def tokenize_corpus_available() -> bool:
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


class _CorpusShardWriter:
    """Write contiguous token IDs and one offset/length row per source document."""

    def __init__(self, root: Path, *, token_dtype: np.dtype, shard_token_limit: int) -> None:
        self.root = root
        self.token_dtype = token_dtype
        self.shard_token_limit = shard_token_limit
        self.shards: list[dict[str, Any]] = []
        self.total_size = 0
        self.token_count = 0
        self.document_count = 0
        self._index = -1
        self._tokens = 0
        self._documents = 0
        self._files: dict[str, BinaryIO] = {}
        self._hashes: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}

    def _open(self) -> None:
        self._index += 1
        self._tokens = 0
        self._documents = 0
        stem = f"shard-{self._index:05d}"
        self._paths = {
            "tokens": self.root / f"{stem}.tokens.bin",
            "documents": self.root / f"{stem}.documents.idx",
        }
        self._files = {key: path.open("wb") for key, path in self._paths.items()}
        self._hashes = {key: hashlib.sha256() for key in self._files}

    def append(self, tokens: list[int]) -> None:
        if self._files and self._tokens and self._tokens + len(tokens) > self.shard_token_limit:
            self._close()
        if not self._files:
            self._open()
        token_bytes = np.asarray(tokens, dtype=self.token_dtype).tobytes(order="C")
        index_bytes = struct.pack("<QQ", self._tokens, len(tokens))
        self._files["tokens"].write(token_bytes)
        self._files["documents"].write(index_bytes)
        self._hashes["tokens"].update(token_bytes)
        self._hashes["documents"].update(index_bytes)
        self._tokens += len(tokens)
        self._documents += 1
        self.token_count += len(tokens)
        self.document_count += 1

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
        size_bytes = sum(int(item["size_bytes"]) for item in files.values())
        self.total_size += size_bytes
        self.shards.append(
            {
                "index": self._index,
                "document_count": self._documents,
                "token_count": self._tokens,
                "files": files,
                "size_bytes": size_bytes,
            }
        )
        self._files = {}
        self._hashes = {}
        self._paths = {}

    def finish(self) -> None:
        self._close()


class TokenizedCorpusAdapter:
    """Encode a frozen dataset into reusable, document-indexed Token IDs."""

    PROGRESS_INTERVAL_SECONDS = 5.0

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
            raise ValueError("tokenized corpus requires a frozen SHA-256 identity")
        selection = dict(payload.get("selection") or {})
        encoding = dict(payload.get("encoding") or {})
        selection_mode = str(selection.get("mode") or "")
        if selection_mode not in {"all", "estimated_token_limit", "mixture"}:
            raise ValueError("unsupported corpus selection mode")
        estimated_limit = int(selection.get("estimated_token_limit") or 0)
        if selection_mode == "estimated_token_limit" and estimated_limit <= 0:
            raise ValueError("estimated token selection limit must be positive")
        eos_policy = str(encoding.get("eos_policy") or "")
        empty_policy = str(encoding.get("empty_document_policy") or "")
        dtype_policy = str(encoding.get("token_dtype") or "")
        shard_token_limit = int(encoding.get("shard_token_limit") or 0)
        if eos_policy not in {"append_required", "append_if_available", "none"}:
            raise ValueError("unsupported EOS policy")
        if empty_policy not in {"skip", "eos_only"}:
            raise ValueError("unsupported empty document policy")
        if dtype_policy not in {"auto", "uint16", "uint32"}:
            raise ValueError("unsupported token dtype policy")
        if shard_token_limit <= 0:
            raise ValueError("shard token limit must be positive")

        tokenizer_binding = payload.get("tokenizer") or {}
        tokenizer_path = Path(str(tokenizer_binding.get("runtime_path") or ""))
        if not tokenizer_path.is_dir():
            raise ValueError("tokenized corpus tokenizer was not materialized")
        data_input = payload.get("data_input") or {}
        project_workspace = Path(str(runtime.get("project_workspace") or "")).resolve()
        if not project_workspace.is_dir():
            raise ValueError("tokenized corpus project workspace is unavailable")
        storage_root = (
            derived_asset_root(project_workspace) / "tokenized-corpora"
        ).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        destination = (storage_root / identity_hash).resolve()
        if destination.parent != storage_root:
            raise ValueError("tokenized corpus destination escaped the project storage root")
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if manifest.get("identity_hash") != identity_hash:
                raise ValueError("existing tokenized corpus has a different identity")
            return self._result(destination, manifest, hashlib.sha256(manifest_bytes).hexdigest())
        if destination.exists():
            raise ValueError("incomplete tokenized corpus occupies the immutable destination")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
        special_tokens = list(tokenizer_binding.get("special_tokens") or [])
        eos_token = str(special_tokens[2]) if len(special_tokens) >= 3 else "</s>"
        if tokenizer.eos_token_id is None and eos_token in tokenizer.get_vocab():
            tokenizer.eos_token = eos_token
        warnings: list[dict[str, Any]] = []
        if eos_policy == "append_required" and tokenizer.eos_token_id is None:
            raise ValueError("EOS policy append_required needs a tokenizer eos_token_id")
        if eos_policy == "append_if_available" and tokenizer.eos_token_id is None:
            warnings.append(
                {
                    "code": "eos_unavailable",
                    "message": "Tokenizer 没有 EOS token；已按公开策略保留原始文档 token。",
                }
            )
        vocab_size = len(tokenizer)
        if dtype_policy == "uint16" and vocab_size > 65_536:
            raise ValueError("uint16 cannot represent this tokenizer vocabulary")
        resolved_dtype = (
            "uint16"
            if dtype_policy == "uint16" or (dtype_policy == "auto" and vocab_size <= 65_536)
            else "uint32"
        )
        token_dtype = np.dtype("<u2" if resolved_dtype == "uint16" else "<u4")

        run_dir = Path(str(runtime.get("run_dir") or ".")).resolve()
        partial = (storage_root / f".{identity_hash}.{task['id']}.partial").resolve()
        if partial.parent != storage_root:
            raise ValueError("tokenized corpus partial path escaped the project storage root")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        loop = asyncio.get_running_loop()
        progress_token_target = estimated_limit or max(
            int(data_input.get("token_estimate") or 0), 0
        )
        progress_document_target = max(int(data_input.get("document_count") or 0), 0)
        max_task_progress = 0.02
        last_emitted_progress = max_task_progress
        last_progress_report_at = time.monotonic()
        last_progress_phase = "preparing_corpus"

        await emit(
            "progress",
            {
                "phase": last_progress_phase,
                "progress": max_task_progress,
                "progress_basis": "frozen_dataset_statistics",
                "target_token_estimate": progress_token_target or None,
                "target_document_count": progress_document_target or None,
                "configuration": {"selection": selection, "encoding": encoding},
            },
        )

        def report_corpus(event: dict[str, Any]) -> None:
            nonlocal last_emitted_progress, last_progress_phase
            nonlocal last_progress_report_at, max_task_progress
            phase = str(event.get("phase") or "encoding_documents")
            processed_token_estimate = max(
                int(
                    event.get("sampled_tokens")
                    or event.get("source_token_estimate")
                    or 0
                ),
                0,
            )
            processed_documents = max(
                int(
                    event.get("source_documents_seen")
                    or event.get("sampled_documents")
                    or event.get("document_count")
                    or 0
                ),
                0,
            )
            ratio: float | None = None
            if progress_token_target:
                ratio = processed_token_estimate / progress_token_target
            elif progress_document_target:
                ratio = processed_documents / progress_document_target
            if ratio is not None:
                max_task_progress = max(
                    max_task_progress,
                    min(0.95, 0.03 + 0.92 * min(max(ratio, 0.0), 1.0)),
                )

            now = time.monotonic()
            should_emit = (
                phase != last_progress_phase
                or now - last_progress_report_at >= self.PROGRESS_INTERVAL_SECONDS
                or max_task_progress - last_emitted_progress >= 0.005
                or (last_emitted_progress < 0.95 <= max_task_progress)
            )
            if not should_emit:
                return
            future = asyncio.run_coroutine_threadsafe(
                emit(
                    "progress",
                    {
                        **event,
                        "progress": max_task_progress,
                        "target_token_estimate": progress_token_target or None,
                        "target_document_count": progress_document_target or None,
                        "configuration": {"selection": selection, "encoding": encoding},
                    },
                ),
                loop,
            )
            future.result()
            last_emitted_progress = max_task_progress
            last_progress_report_at = now
            last_progress_phase = phase

        def build() -> tuple[dict[str, Any], str]:
            corpus_work_dir = run_dir / "corpus-index"
            pending_records = deque()
            record_callback = pending_records.append if data_input.get("format") == "llm-dataset-input-v1" else None
            corpus, _evaluation, source_stats = HfTokenizerAdapter._corpus(
                data_input,
                estimated_limit if selection_mode == "estimated_token_limit" else 0,
                corpus_work_dir,
                report_progress=report_corpus,
                get_desired_state=get_desired_state,
                spool_path=corpus_work_dir / "ordered-corpus.sqlite3",
                record_callback=record_callback,
            )
            if selection_mode == "mixture":
                selection_binding = payload["training_selection_inputs"][0]
                source_stats = {"documents": 0, "token_estimate": 0}

                def filtered_documents():
                    with selected_records(selection_binding, corpus_work_dir, get_desired_state) as connection:
                        records = ordered_records(connection)
                        source_root = Path(data_input["runtime_sources"][0]["runtime_path"])
                        for text, record in HfTokenizerAdapter._iter_selected_texts(source_root, records, get_desired_state):
                            source_stats["documents"] += 1
                            source_stats["token_estimate"] += max(int(record.get("token_estimate") or 0), 1)
                            pending_records.append(record)
                            yield text

                corpus = filtered_documents()
                record_callback = pending_records.append
            identities_path = partial / "document-identities.jsonl.gz"
            identities = gzip.GzipFile(filename=str(identities_path), mode="wb", mtime=0) if record_callback else None

            def write_identity(record, skipped=False):
                if identities is not None:
                    identities.write(_canonical_json({
                        "record_id": record["record_id"], "content_hash": record["content_hash"],
                        **({"skipped": True} if skipped else {}),
                    }))
            writer = _CorpusShardWriter(
                partial,
                token_dtype=token_dtype,
                shard_token_limit=shard_token_limit,
            )
            skipped_empty = 0
            try:
                for document_tokens in _batched_token_ids(corpus, tokenizer):
                    record = pending_records.popleft() if identities is not None else None
                    desired = get_desired_state() if get_desired_state else "running"
                    if desired in {"paused", "cancelled"}:
                        raise AnalysisInterrupted(desired)
                    tokens = list(document_tokens)
                    if not tokens and empty_policy == "skip":
                        skipped_empty += 1
                        write_identity(record, skipped=True)
                        continue
                    if eos_policy != "none" and tokenizer.eos_token_id is not None:
                        tokens.append(int(tokenizer.eos_token_id))
                    if not tokens:
                        skipped_empty += 1
                        write_identity(record, skipped=True)
                        continue
                    writer.append(tokens)
                    write_identity(record)
                    if writer.document_count % 256 == 0:
                        report_corpus(
                            {
                                "phase": "encoding_documents",
                                "document_count": writer.document_count,
                                "token_count": writer.token_count,
                                "source_documents_seen": source_stats["documents"],
                                "source_token_estimate": source_stats["token_estimate"],
                                "progress_basis": "actual_documents_and_tokens",
                            }
                        )
            finally:
                writer.finish()
                if identities is not None:
                    identities.close()
            if skipped_empty:
                warnings.append(
                    {
                        "code": "empty_documents_skipped",
                        "count": skipped_empty,
                        "message": f"按公开策略跳过了 {skipped_empty:,} 个空文档。",
                    }
                )
            manifest = {
                "format": TOKENIZED_CORPUS_FORMAT,
                "format_version": 2,
                "identity_hash": identity_hash,
                "identity_snapshot": payload.get("identity_snapshot") or {},
                "selection": selection,
                "encoding": encoding,
                "resolved_encoding": {
                    "token_dtype": resolved_dtype,
                    "numpy_dtype": token_dtype.str,
                    "vocab_size": vocab_size,
                    "eos_token_id": (
                        int(tokenizer.eos_token_id)
                        if tokenizer.eos_token_id is not None
                        else None
                    ),
                    "document_index_record": "little-endian uint64 offset + uint64 length",
                },
                "document_count": writer.document_count,
                "token_count": writer.token_count,
                "source_document_count": source_stats["documents"],
                "source_token_estimate": source_stats["token_estimate"],
                "warnings": warnings,
                "shards": writer.shards,
            }
            if identities is not None:
                with identities_path.open("rb") as stream:
                    identity_hash_value = hashlib.file_digest(stream, "sha256").hexdigest()
                manifest["files"] = {"document_identities": {
                    "path": identities_path.name, "size_bytes": identities_path.stat().st_size,
                    "sha256": identity_hash_value,
                }}
            manifest_bytes = _canonical_json(manifest)
            (partial / "manifest.json").write_bytes(manifest_bytes)
            partial.replace(destination)
            return manifest, hashlib.sha256(manifest_bytes).hexdigest()

        try:
            manifest, manifest_hash = await asyncio.to_thread(build)
        except AnalysisInterrupted as exc:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            await emit("progress", {"phase": exc.state, "progress": max_task_progress})
            return {"controlled_state": exc.state}
        except Exception:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            raise
        await emit(
            "progress",
            {
                "phase": "completed",
                "progress": 1.0,
                "document_count": manifest["document_count"],
                "token_count": manifest["token_count"],
                "warnings": manifest["warnings"],
            },
        )
        return self._result(destination, manifest, manifest_hash)

    @staticmethod
    def _result(destination: Path, manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
        size_bytes = int(sum(int(shard.get("size_bytes") or 0) for shard in manifest["shards"]))
        size_bytes += sum(int(spec["size_bytes"]) for spec in (manifest.get("files") or {}).values())
        return {
            "identity_hash": manifest["identity_hash"],
            "format": manifest["format"],
            "document_count": int(manifest["document_count"]),
            "token_count": int(manifest["token_count"]),
            "shard_count": len(manifest["shards"]),
            "size_bytes": size_bytes,
            "warnings": list(manifest.get("warnings") or []),
            "resolved_encoding": manifest.get("resolved_encoding") or {},
            "outputs": [
                {
                    "ref": "tokenized_corpus",
                    "kind": "tokenized_corpus",
                    "storage_type": "external",
                    "uri": destination.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": size_bytes,
                    "media_type": "application/vnd.forgelm.token-corpus-v2",
                    "availability_state": "available",
                    "metadata": {
                        "format": manifest["format"],
                        "identity_hash": manifest["identity_hash"],
                        "document_count": manifest["document_count"],
                        "token_count": manifest["token_count"],
                        "shard_count": len(manifest["shards"]),
                        "encoding": manifest["encoding"],
                        "resolved_encoding": manifest["resolved_encoding"],
                        **({"document_identity_format": "record-id-content-hash-v1"} if (manifest.get("files") or {}).get("document_identities") else {}),
                        "warnings": manifest.get("warnings") or [],
                    },
                }
            ],
        }
