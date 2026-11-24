from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from llm_platform_core.mixtures import (
    MIXTURE_INDEX_FORMAT,
    MIXTURE_MANIFEST_FORMAT,
    RECORD_MANIFEST_FORMAT,
    REFERENCE_RESOLVER_VERSION,
    REFERENCE_SAMPLE_BLOCK_SIZE,
    REFERENCE_SAMPLER_VERSION,
    RESOLVER_VERSION,
    SAMPLER_VERSION,
    WEIGHT_NORMALIZER_VERSION,
    canonical_json,
    is_identity_selection,
    normalize_weight_units,
    open_text_manifest,
    stream_key,
)

from .command import Emit, GetDesiredState
from .mixture_resolve import progress_snapshot, resolve_manifest_source


class DataMixtureAdapter:
    """Resolve immutable dataset record manifests into a stable mixture manifest."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def _download_manifest(
        self,
        artifact_id: str,
        expected_hash: str,
        project_id: str,
        target: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> str:
        digest = hashlib.sha256()
        downloaded = 0
        last_emit = 0.0
        desired = "running"
        async with self.client.stream(
            "GET",
            f"/api/agent/artifacts/{artifact_id}/payload",
            params={"project_id": project_id},
            timeout=None,
        ) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length") or 0)
            with target.open("wb") as output:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    desired = get_desired_state() if get_desired_state else "running"
                    if desired in {"paused", "cancelled"}:
                        break
                    digest.update(chunk)
                    output.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_emit >= 1:
                        desired = await emit(
                            "progress",
                            {
                                "progress": 0.05,
                                "phase": "downloading_manifest",
                                "processed_bytes": downloaded,
                                "total_bytes": total_bytes,
                            },
                        )
                        last_emit = now
                        if desired in {"paused", "cancelled"}:
                            break
        if desired in {"paused", "cancelled"}:
            target.unlink(missing_ok=True)
            return desired
        if digest.hexdigest() != expected_hash:
            target.unlink(missing_ok=True)
            raise ValueError("dataset record manifest failed SHA-256 verification")
        return "running"

    @staticmethod
    def _validate_header(path: Path, dataset_version_id: str) -> None:
        with open_text_manifest(path) as stream:
            first = stream.readline()
            if not first:
                raise ValueError("dataset record manifest is empty")
            header = json.loads(first)
            if header.get("type") != "header" or header.get("format") != RECORD_MANIFEST_FORMAT:
                raise ValueError("dataset record manifest has an unsupported format")
            if str(header.get("dataset_version_id") or "") != dataset_version_id:
                raise ValueError("dataset record manifest belongs to a different dataset version")

    async def _resolve_source(
        self,
        *,
        input_path: Path,
        body_path: Path,
        work_dir: Path,
        dataset_version_id: str,
        selection: dict[str, Any],
        seed: int,
        source_index: int,
        source_count: int,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> tuple[dict[str, int] | None, str]:
        cancel_path = work_dir.parent / f"cancel-source-{source_index:05d}"
        cancel_path.unlink(missing_ok=True)
        progress: dict[str, Any] = {}
        progress_lock = threading.Lock()
        resolver = asyncio.create_task(
            asyncio.to_thread(
                resolve_manifest_source,
                input_path=input_path,
                output_path=body_path,
                work_dir=work_dir,
                dataset_version_id=dataset_version_id,
                selection=selection,
                default_seed=seed,
                source_index=source_index,
                cancel_path=cancel_path,
                progress=progress,
                progress_lock=progress_lock,
            )
        )
        controlled_state = "running"
        try:
            while not resolver.done():
                await asyncio.wait({resolver}, timeout=1)
                desired = get_desired_state() if get_desired_state else "running"
                snapshot = progress_snapshot(progress, progress_lock)
                if snapshot:
                    total_rows = max(int(snapshot.get("total_rows") or 0), 0)
                    processed_rows = max(int(snapshot.get("processed_rows") or 0), 0)
                    total_bytes = max(int(snapshot.get("total_bytes") or 0), 0)
                    processed_bytes = max(int(snapshot.get("processed_bytes") or 0), 0)
                    if total_rows:
                        source_fraction = min(processed_rows / total_rows, 1)
                    elif total_bytes:
                        source_fraction = min(processed_bytes / total_bytes, 1)
                    else:
                        source_fraction = 0
                    desired = await emit(
                        "progress",
                        {
                            **snapshot,
                            "progress": 0.1 + 0.7 * (source_index + source_fraction) / source_count,
                            "source_index": source_index,
                        },
                    )
                if desired in {"paused", "cancelled"}:
                    controlled_state = desired
                    cancel_path.parent.mkdir(parents=True, exist_ok=True)
                    cancel_path.touch(exist_ok=True)
            try:
                result = await resolver
            except Exception:
                if controlled_state in {"paused", "cancelled"}:
                    return None, controlled_state
                raise
            return result, controlled_state
        finally:
            if controlled_state in {"paused", "cancelled"} and not resolver.done():
                cancel_path.touch(exist_ok=True)
                try:
                    await resolver
                except Exception:
                    pass
            cancel_path.unlink(missing_ok=True)

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        mixture_version_id = str(payload.get("mixture_version_id") or "")
        entries = [dict(entry) for entry in payload.get("entries") or []]
        seed = int(payload.get("seed") or 0)
        if not mixture_version_id or not entries:
            raise ValueError("mixture resolution requires a version and at least one source")
        run_dir = Path((task.get("runtime_context") or {}).get("run_dir") or ".").resolve()
        input_dir = run_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = run_dir / "mixture-shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        weight_units = normalize_weight_units([entry.get("weight", 0) for entry in entries])
        if all(is_identity_selection(entry.get("selection")) for entry in entries):
            resolved_entries: list[dict[str, Any]] = []
            for entry, units in zip(entries, weight_units, strict=True):
                selected_count = int(entry.get("document_count") or 0)
                selected_tokens = int(entry.get("token_estimate") or 0)
                if selected_count <= 0 or selected_tokens <= 0:
                    raise ValueError(
                        "identity mixture sources require exact document and token counts"
                    )
                resolved_entries.append(
                    {
                        **entry,
                        "weight_units": units,
                        "normalized_weight": units / 1_000_000,
                        "selected_document_count": selected_count,
                        "selected_tokens": selected_tokens,
                    }
                )
            sources = [
                {
                    "dataset_version_id": str(entry["dataset_version_id"]),
                    "dataset_name": entry.get("name"),
                    "dataset_artifact_id": entry.get("dataset_artifact_id"),
                    "dataset_artifact_hash": entry.get("dataset_artifact_hash"),
                    "record_manifest_artifact_id": entry["record_manifest_artifact_id"],
                    "record_manifest_hash": entry["record_manifest_hash"],
                    "weight": entry.get("weight"),
                    "weight_units": entry["weight_units"],
                    "selection": entry.get("selection") or {},
                    "selected_document_count": entry["selected_document_count"],
                    "selected_token_estimate": entry["selected_tokens"],
                }
                for entry in resolved_entries
            ]
            header = {
                "type": "header",
                "format": MIXTURE_INDEX_FORMAT,
                "mixture_version_id": mixture_version_id,
                "seed": seed,
                "target_tokens": int(payload.get("target_tokens") or 0),
                "resolver": REFERENCE_RESOLVER_VERSION,
                "weight_normalizer": WEIGHT_NORMALIZER_VERSION,
                "sampler": REFERENCE_SAMPLER_VERSION,
                "sample_block_size": REFERENCE_SAMPLE_BLOCK_SIZE,
                "sampling_mode": "deterministic_with_replacement",
                "record_order": "record_manifest",
                "sources": sources,
            }
            header["stream_key"] = stream_key(header)
            output_path = run_dir / "data-mixture-index.jsonl"
            output_path.write_text(canonical_json(header) + "\n", encoding="utf-8")
            await emit("progress", {"progress": 1.0, "phase": "reference_index_ready"})
            return {
                "format": MIXTURE_INDEX_FORMAT,
                "resolver_version": REFERENCE_RESOLVER_VERSION,
                "sampler_version": REFERENCE_SAMPLER_VERSION,
                "stream_key": header["stream_key"],
                "resolved_document_count": sum(
                    int(entry["selected_document_count"]) for entry in resolved_entries
                ),
                "resolved_token_estimate": sum(
                    int(entry["selected_tokens"]) for entry in resolved_entries
                ),
                "resolved_entries": resolved_entries,
                "outputs": [
                    {
                        "ref": "mixture_manifest",
                        "path": str(output_path),
                        "kind": "data_mixture_manifest",
                        "media_type": "application/x-ndjson",
                        "metadata": {"format": MIXTURE_INDEX_FORMAT},
                    }
                ],
            }
        body_paths: list[Path] = []
        resolved_entries: list[dict[str, Any]] = []

        for index, (entry, units) in enumerate(zip(entries, weight_units, strict=True)):
            desired = get_desired_state() if get_desired_state else "running"
            if desired in {"paused", "cancelled"}:
                return {"controlled_state": desired}
            dataset_version_id = str(entry.get("dataset_version_id") or "")
            artifact_id = str(entry.get("record_manifest_artifact_id") or "")
            expected_hash = str(entry.get("record_manifest_hash") or "")
            if not dataset_version_id or not artifact_id or not expected_hash:
                raise ValueError("mixture source is missing its dataset record manifest")
            local_path = input_dir / f"source-{index:05d}.download"
            desired = await self._download_manifest(
                artifact_id,
                expected_hash,
                str(task.get("project_id") or ""),
                local_path,
                emit,
                get_desired_state,
            )
            if desired in {"paused", "cancelled"}:
                return {"controlled_state": desired}
            with local_path.open("rb") as stream:
                suffix = ".jsonl.gz" if stream.read(2) == b"\x1f\x8b" else ".jsonl"
            local_path = local_path.replace(local_path.with_suffix(suffix))
            await asyncio.to_thread(self._validate_header, local_path, dataset_version_id)
            body_path = shard_dir / f"source-{index:05d}.jsonl.gz"
            resolved, desired = await self._resolve_source(
                input_path=local_path,
                body_path=body_path,
                work_dir=shard_dir / f"duckdb-{index:05d}",
                dataset_version_id=dataset_version_id,
                selection=entry.get("selection") or {},
                seed=seed,
                source_index=index,
                source_count=len(entries),
                emit=emit,
                get_desired_state=get_desired_state,
            )
            if desired in {"paused", "cancelled"}:
                return {"controlled_state": desired}
            assert resolved is not None
            selected_count = int(resolved["selected_document_count"])
            if not selected_count:
                raise ValueError(f"mixture source {dataset_version_id} selected no records")
            selected_tokens = int(resolved["selected_tokens"])
            body_paths.append(body_path)
            resolved_entries.append(
                {
                    **entry,
                    "weight_units": units,
                    "normalized_weight": units / 1_000_000,
                    "selected_document_count": selected_count,
                    "selected_tokens": selected_tokens,
                }
            )
            await emit(
                "progress",
                {
                    "progress": 0.1 + 0.7 * (index + 1) / len(entries),
                    "phase": "resolving_sources",
                    "source_index": index,
                    "processed_rows": int(resolved["candidate_document_count"]),
                    "selected_documents": selected_count,
                },
            )

        sources = [
            {
                "dataset_version_id": str(entry["dataset_version_id"]),
                "dataset_name": entry.get("name"),
                "dataset_artifact_id": entry.get("dataset_artifact_id"),
                "dataset_artifact_hash": entry.get("dataset_artifact_hash"),
                "record_manifest_artifact_id": entry["record_manifest_artifact_id"],
                "record_manifest_hash": entry["record_manifest_hash"],
                "weight": entry.get("weight"),
                "weight_units": entry["weight_units"],
                "selection": entry.get("selection") or {},
                "selected_document_count": entry["selected_document_count"],
                "selected_token_estimate": entry["selected_tokens"],
            }
            for entry in resolved_entries
        ]
        header = {
            "type": "header",
            "format": MIXTURE_MANIFEST_FORMAT,
            "mixture_version_id": mixture_version_id,
            "seed": seed,
            "target_tokens": int(payload.get("target_tokens") or 0),
            "resolver": RESOLVER_VERSION,
            "weight_normalizer": WEIGHT_NORMALIZER_VERSION,
            "sampler": SAMPLER_VERSION,
            "sampling_mode": "deterministic_with_replacement",
            "sources": sources,
        }
        header["stream_key"] = stream_key(header)
        output_path = run_dir / "data-mixture-manifest.jsonl.gz"
        with output_path.open("wb") as output:
            output.write(
                gzip.compress((canonical_json(header) + "\n").encode("utf-8"), mtime=0)
            )
            for body_path in body_paths:
                with body_path.open("rb") as body:
                    shutil.copyfileobj(body, output, length=8 * 1024 * 1024)
        await emit("progress", {"progress": 1.0, "phase": "manifest_ready"})
        return {
            "format": MIXTURE_MANIFEST_FORMAT,
            "resolver_version": RESOLVER_VERSION,
            "sampler_version": SAMPLER_VERSION,
            "stream_key": header["stream_key"],
            "resolved_document_count": sum(
                int(entry["selected_document_count"]) for entry in resolved_entries
            ),
            "resolved_token_estimate": sum(
                int(entry["selected_tokens"]) for entry in resolved_entries
            ),
            "resolved_entries": resolved_entries,
            "outputs": [
                {
                    "ref": "mixture_manifest",
                    "path": str(output_path),
                    "kind": "data_mixture_manifest",
                    "media_type": "application/gzip",
                    "metadata": {"format": MIXTURE_MANIFEST_FORMAT},
                }
            ],
        }
