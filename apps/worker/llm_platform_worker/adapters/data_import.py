from __future__ import annotations

import asyncio
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import unquote, urlparse

import httpx
from llm_platform_core.datasets import (
    SOURCE_ARCHIVE_CAPABILITY,
    TEXT_PARSER_ID,
    TEXT_PARSER_VERSION,
    TEXT_USAGE_CAPABILITIES,
)
from llm_platform_core.mixtures import (
    RECORD_MANIFEST_FORMAT,
    canonical_json,
    stable_record_id,
)

from ..dataset_parsers import DatasetParserContext, DatasetParserRegistry, FunctionParserAdapter
from ..dataset_parsers.registry import detect_formats
from .command import Emit, GetDesiredState
from .data_transfer import ProcessInterrupted, ScriptTransferRunner
from .parquet_analysis import (
    PARQUET_ENGINE,
    TOKEN_ESTIMATE_METHOD,
    analyze_parquet_file,
    recommended_processes,
)

TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown", ".jsonl", ".ndjson", ".json", ".csv", ".tsv"}
PARQUET_SUFFIXES = {".parquet"}
DATA_SUFFIXES = TEXT_SUFFIXES | {".parquet", ".arrow", ".feather", ".gz", ".zst"}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
DOWNLOAD_RETRY_ATTEMPTS = 5
DOWNLOAD_RETRY_BASE_SECONDS = 1.0
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}
HF_METADATA_POLL_SECONDS = 0.25
HF_METADATA_TERMINATE_TIMEOUT_SECONDS = 5.0
HF_METADATA_TIMEOUT_SECONDS = 300.0
QUALITY_SAMPLE_LIMIT = 100_000
# Arrow buffers remain columnar, but converting a 65k-row batch to Python objects
# multiplies memory substantially. Keep the conversion window deliberately small.
PARQUET_BATCH_SIZE = 8_192
PREFERRED_TEXT_FIELDS = [
    "text",
    "content",
    "document",
    "body",
    "instruction",
    "input",
    "output",
    "response",
]

PARSER_REGISTRY = DatasetParserRegistry()


class AnalysisInterrupted(RuntimeError):
    def __init__(self, state: str) -> None:
        self.state = state
        super().__init__(f"Dataset analysis was {state}")


class SnapshotDriftError(RuntimeError):
    pass


@contextmanager
def _gzip_text_writer(path: Path) -> Iterator[TextIO]:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0
        ) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                yield text


def _huggingface_endpoint(source_config: dict[str, Any]) -> str:
    endpoint = (
        str(
            source_config.get("hf_endpoint") or os.getenv("HF_ENDPOINT") or "https://huggingface.co"
        )
        .strip()
        .rstrip("/")
    )
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Hugging Face mirror endpoint must be a complete HTTP(S) URL")
    return endpoint


class DataImportAdapter:
    """Imports structured data and registers a real analyzed snapshot."""

    @staticmethod
    def _hf_metadata_timeout_seconds() -> float:
        configured = os.getenv("LLM_PLATFORM_HF_METADATA_TIMEOUT_SECONDS", "").strip()
        if not configured:
            return HF_METADATA_TIMEOUT_SECONDS
        try:
            timeout = float(configured)
        except ValueError as exc:
            raise ValueError(
                "LLM_PLATFORM_HF_METADATA_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if timeout <= 0:
            raise ValueError("LLM_PLATFORM_HF_METADATA_TIMEOUT_SECONDS must be positive")
        return timeout

    @staticmethod
    async def _run_hf_metadata_operation(
        operation: str,
        repo_id: str,
        source_config: dict[str, Any],
        run_dir: Path,
        get_desired_state: GetDesiredState | None,
        *,
        destination: Path | None = None,
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / f"huggingface-{operation}-request.json"
        result_path = run_dir / f"huggingface-{operation}-result.json"
        result_path.unlink(missing_ok=True)
        request_path.write_text(
            json.dumps(
                {
                    "operation": operation,
                    "repo_id": repo_id,
                    "source_config": source_config,
                    "destination": str(destination) if destination is not None else None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            raise ProcessInterrupted(desired)
        process_kwargs: dict[str, Any] = {}
        if os.name == "nt":
            process_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "llm_platform_worker.hf_connector",
            "--metadata-request-file",
            str(request_path),
            "--result-file",
            str(result_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_kwargs,
        )
        communication = asyncio.create_task(process.communicate())

        async def terminate() -> tuple[bytes, bytes]:
            if process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
            try:
                return await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=HF_METADATA_TERMINATE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                if process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                return await communication

        started_at = time.monotonic()
        try:
            while True:
                try:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.shield(communication), timeout=HF_METADATA_POLL_SECONDS
                    )
                    break
                except TimeoutError:
                    desired = get_desired_state() if get_desired_state else "running"
                    if desired in {"paused", "cancelled"}:
                        await terminate()
                        raise ProcessInterrupted(desired) from None
                    if time.monotonic() - started_at >= DataImportAdapter._hf_metadata_timeout_seconds():
                        await terminate()
                        raise TimeoutError(
                            "Hugging Face metadata operation exceeded its time limit"
                        ) from None
        except BaseException:
            if process.returncode is None:
                try:
                    await terminate()
                except Exception:
                    pass
            raise
        if process.returncode:
            detail = stderr.decode("utf-8", errors="replace").strip()[-4_000:]
            raise RuntimeError(
                f"Hugging Face metadata process exited with code {process.returncode}: "
                f"{detail or 'No diagnostic output was produced.'}"
            )
        if not result_path.is_file():
            detail = stdout.decode("utf-8", errors="replace").strip()[-1_000:]
            raise RuntimeError(
                "Hugging Face metadata process did not produce a result"
                + (f": {detail}" if detail else "")
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError("Hugging Face metadata result must be an object")
        return result

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        source_type = str(payload.get("source_type") or "").lower()
        source_uri = str(payload.get("source_uri") or "").strip()
        source_config = payload.get("source_config") or {}
        run_dir = Path((task.get("runtime_context") or {}).get("run_dir") or ".").resolve()
        worker_id = str((task.get("runtime_context") or {}).get("worker_id") or "") or None
        operation = str(payload.get("operation") or "").strip()
        if operation == "hf_scan":
            await emit("progress", {"progress": 0.1, "phase": "resolving_revision"})
            try:
                scan = await self._run_hf_metadata_operation(
                    "scan", source_uri, source_config, run_dir, get_desired_state
                )
            except ProcessInterrupted as exc:
                await emit("progress", {"progress": 0.1, "phase": exc.state})
                return {"controlled_state": exc.state, "operation": operation}
            reported_state = await emit(
                "progress",
                {
                    "progress": 1.0,
                    "phase": "repository_resolved",
                    "file_count_total": scan["file_count"],
                    "total_bytes": scan["total_size_bytes"],
                },
            )
            desired = get_desired_state() if get_desired_state else reported_state
            if desired in {"paused", "cancelled"}:
                return {"controlled_state": desired, "operation": operation}
            return {**scan, "worker_id": worker_id, "operation": operation}
        if operation == "hf_plan":
            project_workspace_uri = str(payload.get("project_workspace_uri") or "").strip()
            destination = self._hf_asset_destination(
                str(payload.get("asset_key") or ""), project_workspace_uri
            )
            await emit("progress", {"progress": 0.1, "phase": "checking_cache"})
            try:
                plan = await self._run_hf_metadata_operation(
                    "plan",
                    source_uri,
                    source_config,
                    run_dir,
                    get_desired_state,
                    destination=destination,
                )
            except ProcessInterrupted as exc:
                await emit("progress", {"progress": 0.1, "phase": exc.state})
                return {"controlled_state": exc.state, "operation": operation}
            reported_state = await emit(
                "progress",
                {
                    "progress": 1.0,
                    "phase": "download_planned",
                    "file_count_total": plan["file_count"],
                    "total_bytes": plan["selected_bytes"],
                    "downloaded_bytes": plan["cached_bytes"],
                },
            )
            desired = get_desired_state() if get_desired_state else reported_state
            if desired in {"paused", "cancelled"}:
                return {"controlled_state": desired, "operation": operation}
            return {
                **plan,
                "worker_id": worker_id,
                "operation": operation,
                "project_workspace_uri": project_workspace_uri,
            }
        # Attempts of the same task have sibling run directories. Keep network download
        # state at task scope so a user-triggered retry can reuse completed and partial files.
        planned_hf_asset = source_type == "huggingface" and bool(
            source_config.get("hf_plan_task_id")
        )
        destination = (
            self._hf_asset_destination(
                str(source_config.get("asset_key") or ""),
                str(source_config.get("project_workspace_uri") or ""),
            )
            if planned_hf_asset
            else run_dir.parent / "dataset"
        )
        from llm_platform_core.storage_layout import unified_storage_root
        unified_root = unified_storage_root()
        if unified_root and not planned_hf_asset:
            from uuid import UUID
            destination = unified_root / "dataset" / "objects" / "external" / "imports" / str(UUID(payload["dataset_version_id"]))
        destination.mkdir(parents=True, exist_ok=True)
        transfer_runner = ScriptTransferRunner(run_dir)

        await emit("progress", {"progress": 0.01, "phase": "preparing", "source": source_uri})
        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            await emit("progress", {"progress": task.get("progress", 0.0), "phase": desired})
            return {"controlled_state": desired, "transfer": {}}
        analysis_only = bool(payload.get("analysis_only"))
        reuse_artifact_id = str(payload.get("reuse_artifact_id") or "").strip()
        transfer: dict[str, Any] | None = None
        try:
            if analysis_only:
                if not reuse_artifact_id:
                    raise ValueError("Analysis-only mode requires a reusable artifact")
                analysis_root = self._local_path(source_uri)
                if not analysis_root.exists():
                    raise FileNotFoundError(
                        f"Local dataset source does not exist on this Worker: {analysis_root}"
                    )
                await emit(
                    "progress",
                    {"progress": 0.12, "phase": "reading_existing", "source": str(analysis_root)},
                )
            elif source_type in {"http", "https"}:
                transfer = await transfer_runner.download_http(
                    source_uri, destination, emit, get_desired_state
                )
            elif source_type == "huggingface":
                if planned_hf_asset:
                    transfer = await self._download_planned_huggingface(
                        transfer_runner,
                        source_uri,
                        source_config,
                        destination,
                        run_dir,
                        emit,
                        get_desired_state,
                    )
                else:
                    transfer = await transfer_runner.download_huggingface(
                        source_uri, source_config, destination, emit, get_desired_state
                    )
            elif source_type == "git":
                transfer = await transfer_runner.clone_git(
                    source_uri, source_config, destination, emit, get_desired_state
                )
            elif source_type == "local":
                await self._copy_local(source_uri, destination, emit)
                transfer = {"tool": "python-copy", "source_type": "local"}
            else:
                raise ValueError(f"Unsupported built-in dataset source type: {source_type}")
        except ProcessInterrupted as exc:
            await emit("progress", {"progress": 0.0, "phase": exc.state})
            return {"controlled_state": exc.state, "transfer": transfer or {}}

        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            await emit("progress", {"progress": 0.0, "phase": desired})
            return {"controlled_state": desired, "transfer": transfer or {}}
        analysis_base = 0.12 if analysis_only else 0.82
        analysis_span = 0.84 if analysis_only else 0.14
        await emit("progress", {"progress": analysis_base, "phase": "analyzing"})
        record_manifest_path = run_dir / "dataset-records.jsonl.gz"
        analysis_root = analysis_root if analysis_only else destination
        modality = str(payload.get("modality") or "text").strip().lower()
        detected_formats = detect_formats(analysis_root)
        parser = PARSER_REGISTRY.resolve(modality, payload.get("parser"))
        loop = asyncio.get_running_loop()

        def report_analysis_progress(progress: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(
                emit(
                    "progress",
                    {
                        **progress,
                        "progress": analysis_base
                        + analysis_span * float(progress.get("progress") or 0),
                        "phase": "analyzing",
                    },
                ),
                loop,
            )
            future.result()

        try:
            parser_config = {
                **dict(payload.get("parser_config") or {}),
                **source_config,
                "known_files": payload.get("known_files") or [],
            }
            context = DatasetParserContext(
                root=analysis_root,
                source_type=source_type,
                source_uri=source_uri,
                dataset_version_id=str(payload.get("dataset_version_id") or ""),
                modality=modality,
                detected_formats=detected_formats,
                parser_config=parser_config,
                record_manifest_path=record_manifest_path if parser else None,
                progress_callback=report_analysis_progress,
                get_desired_state=get_desired_state,
            )
            analysis = await asyncio.to_thread(
                parser.parse if parser else self._inventory_unparsed,
                context,
            )
            document_count = int(analysis.get("document_count") or 0)
            parser_status = "succeeded" if parser else "unavailable"
            usage_capabilities = (
                list(TEXT_USAGE_CAPABILITIES)
                if parser and modality == "text" and document_count > 0
                else [SOURCE_ARCHIVE_CAPABILITY]
            )
            analysis = {
                **analysis,
                "modality": modality,
                "record_count": int(
                    analysis.get("record_count")
                    if analysis.get("record_count") is not None
                    else document_count
                ),
                "parser_id": parser.parser_id if parser else None,
                "parser_version": parser.parser_version if parser else None,
                "parser_status": parser_status,
                "parser_implementation": parser.implementation if parser else None,
                "parser_config": dict(payload.get("parser_config") or {}),
                "detected_formats": list(detected_formats),
                "data_schema": analysis.get("data_schema")
                or (
                    {"type": "text_records", "record_manifest_format": RECORD_MANIFEST_FORMAT}
                    if parser and modality == "text"
                    else {"type": "opaque_files", "semantic_parser": False}
                ),
                "statistics": analysis.get("statistics")
                or {
                    "record_count": int(analysis.get("record_count") or document_count),
                    "document_count": document_count,
                    "token_estimate": int(analysis.get("token_estimate") or 0),
                    "raw_size_bytes": int(analysis.get("raw_size_bytes") or 0),
                    "file_count": int(
                        (analysis.get("source_snapshot") or {}).get("file_count") or 0
                    ),
                },
                "usage_capabilities": usage_capabilities,
                "train_ready": "text_pretraining" in usage_capabilities,
            }
        except AnalysisInterrupted as exc:
            record_manifest_path.unlink(missing_ok=True)
            await emit("progress", {"progress": analysis_base, "phase": exc.state})
            return {"controlled_state": exc.state, "transfer": transfer or {}}
        expected_snapshot_files = payload.get("expected_snapshot_files") or []
        if expected_snapshot_files:
            def identity(files: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
                return sorted(
                    (
                        str(item.get("path") or "").replace("\\", "/"),
                        int(item.get("size_bytes") or 0),
                        str(item.get("sha256") or ""),
                    )
                    for item in files
                    if isinstance(item, dict) and item.get("path")
                )

            expected_identity = identity(expected_snapshot_files)
            actual_identity = identity((analysis.get("source_snapshot") or {}).get("files") or [])
            if actual_identity != expected_identity:
                record_manifest_path.unlink(missing_ok=True)
                expected_paths = {item[0] for item in expected_identity}
                actual_paths = {item[0] for item in actual_identity}
                changed_paths = sorted(
                    path
                    for path in expected_paths & actual_paths
                    if next(item[1:] for item in expected_identity if item[0] == path)
                    != next(item[1:] for item in actual_identity if item[0] == path)
                )
                raise SnapshotDriftError(
                    "snapshot_drift: 源文件已变化，请作为新的数据版本重新导入；"
                    f"缺失 {sorted(expected_paths - actual_paths)[:10]}，"
                    f"新增 {sorted(actual_paths - expected_paths)[:10]}，"
                    f"内容变化 {changed_paths[:10]}"
                )
        record_manifest_output = (
            {
                "ref": "record_manifest",
                "path": str(record_manifest_path),
                "kind": "dataset_record_manifest",
                "media_type": "application/gzip",
                "metadata": {
                    "index_kind": "record_manifest",
                    "format": RECORD_MANIFEST_FORMAT,
                    "parser_id": analysis["parser_id"],
                    "parser_version": analysis["parser_version"],
                },
            }
            if record_manifest_path.is_file()
            else None
        )
        analysis["indexes"] = (
            [
                {
                    "ref": "record_manifest",
                    "kind": "record_manifest",
                    "format": RECORD_MANIFEST_FORMAT,
                }
            ]
            if record_manifest_output
            else []
        )
        if transfer:
            analysis["source_snapshot"] = {
                **analysis["source_snapshot"],
                "transfer": transfer,
            }
        await emit(
            "progress",
            {
                "progress": 0.98,
                "phase": "finalizing" if analysis_only else "uploading",
                "document_count": analysis["document_count"],
                "bytes": analysis["raw_size_bytes"],
            },
        )
        if analysis_only:
            analysis["source_snapshot"] = {
                **analysis["source_snapshot"],
                "analysis_only": True,
                "reused_artifact_id": reuse_artifact_id,
            }
            return {
                **analysis,
                "output_artifact_ids": [reuse_artifact_id],
                "outputs": [record_manifest_output] if record_manifest_output else [],
            }
        if planned_hf_asset or unified_root:
            files = analysis["source_snapshot"]["files"]
            manifest_hash = self._manifest_hash(files)
            selected_paths = set(str(value) for value in source_config.get("selected_paths") or [])
            actual_paths = {str(item["path"]) for item in files}
            if planned_hf_asset and actual_paths != selected_paths:
                raise ValueError(
                    "Downloaded Hugging Face files do not match the frozen selection: "
                    f"missing={sorted(selected_paths - actual_paths)}, "
                    f"unexpected={sorted(actual_paths - selected_paths)}"
                )
            analysis["source_snapshot"] = {
                **analysis["source_snapshot"],
                "requested_revision": source_config.get("requested_revision"),
                "commit_hash": source_config.get("commit_hash"),
                "selected_paths": list(source_config.get("selected_paths") or []),
                "target_dir": str(destination),
                "storage_type": "external",
            }
            return {
                **analysis,
                "outputs": [
                    {
                        "ref": "dataset",
                        "kind": "dataset_snapshot",
                        "storage_type": "external",
                        "uri": destination.as_uri(),
                        "content_hash": manifest_hash,
                        "size_bytes": analysis["raw_size_bytes"],
                        "media_type": "application/vnd.llm-platform.dataset-directory",
                        "availability_state": "available",
                        "metadata": {
                            "source_type": source_type,
                            "source_uri": source_uri,
                            **({"repo_id": source_uri, "commit_hash": source_config.get("commit_hash")} if planned_hf_asset else {}),
                            "file_count": len(files),
                        },
                    },
                    *([record_manifest_output] if record_manifest_output else []),
                ],
            }
        return {
            **analysis,
            "outputs": [
                {
                    "ref": "dataset",
                    "path": str(destination),
                    "kind": "dataset_snapshot",
                    "media_type": "application/gzip",
                },
                *([record_manifest_output] if record_manifest_output else []),
            ],
        }

    @classmethod
    def _hf_asset_destination(cls, asset_key: str, project_workspace_uri: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9-]{8,80}", asset_key):
            raise ValueError("A server-generated Hugging Face asset key is required")
        if not project_workspace_uri.strip():
            raise ValueError("A project workspace path is required for Hugging Face assets")
        workspace_root = cls._local_path(project_workspace_uri)
        if not workspace_root.exists() or not workspace_root.is_dir():
            raise FileNotFoundError(
                f"Project workspace is not a directory on this Worker: {workspace_root}"
            )
        from llm_platform_core.storage_layout import data_asset_root
        asset_root = data_asset_root(workspace_root)
        asset_root.mkdir(parents=True, exist_ok=True)
        return asset_root / asset_key

    @staticmethod
    async def _download_planned_huggingface(
        transfer_runner: ScriptTransferRunner,
        repo_id: str,
        source_config: dict[str, Any],
        destination: Path,
        run_dir: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> dict[str, Any]:
        execution_plan = await DataImportAdapter._run_hf_metadata_operation(
            "plan",
            repo_id,
            source_config,
            run_dir,
            get_desired_state,
            destination=destination,
        )
        planned_file_count = int(source_config.get("planned_file_count") or 0)
        planned_selected_bytes = int(source_config.get("planned_selected_bytes") or 0)
        if (
            execution_plan["file_count"] != planned_file_count
            or execution_plan["selected_bytes"] != planned_selected_bytes
        ):
            raise ValueError(
                "Hugging Face repository contents no longer match the frozen download plan"
            )
        if not execution_plan["disk_space_sufficient"]:
            raise OSError(
                "Hugging Face target disk does not have enough free space for the frozen plan"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        plan_file = run_dir / "huggingface-download-plan.json"
        plan_file.write_text(json.dumps(execution_plan, ensure_ascii=False), encoding="utf-8")
        completed_bytes, completed_files = DataImportAdapter._planned_hf_completed(
            destination, execution_plan["files"]
        )
        await emit(
            "progress",
            {
                "progress": min(
                    0.78,
                    0.05 + 0.73 * completed_bytes / execution_plan["selected_bytes"],
                ),
                "phase": "downloading",
                "tool": "huggingface_hub.snapshot_download",
                "downloaded_bytes": completed_bytes,
                "total_bytes": execution_plan["selected_bytes"],
                "file_count_completed": completed_files,
                "file_count_total": execution_plan["file_count"],
                "progress_basis": "completed_file_bytes",
            },
        )
        download = asyncio.create_task(
            transfer_runner._run_process(
                [
                    sys.executable,
                    "-m",
                    "llm_platform_worker.hf_connector",
                    "--plan-file",
                    str(plan_file),
                ],
                tool="huggingface_hub.snapshot_download",
                emit=emit,
                get_desired_state=get_desired_state,
                env=os.environ.copy(),
                parse_progress=False,
            )
        )
        while not download.done():
            completed_bytes, completed_files = await asyncio.to_thread(
                DataImportAdapter._planned_hf_completed,
                destination,
                execution_plan["files"],
            )
            await emit(
                "progress",
                {
                    "progress": min(
                        0.78,
                        0.05 + 0.73 * completed_bytes / execution_plan["selected_bytes"],
                    ),
                    "phase": "downloading",
                    "tool": "huggingface_hub.snapshot_download",
                    "downloaded_bytes": completed_bytes,
                    "total_bytes": execution_plan["selected_bytes"],
                    "file_count_completed": completed_files,
                    "file_count_total": execution_plan["file_count"],
                    "progress_basis": "completed_file_bytes",
                },
            )
            await asyncio.wait({download}, timeout=2)
        await download
        completed_bytes, completed_files = await asyncio.to_thread(
            DataImportAdapter._planned_hf_completed,
            destination,
            execution_plan["files"],
        )
        await emit(
            "progress",
            {
                "progress": min(
                    0.78,
                    0.05 + 0.73 * completed_bytes / execution_plan["selected_bytes"],
                ),
                "phase": "downloaded",
                "tool": "huggingface_hub.snapshot_download",
                "downloaded_bytes": completed_bytes,
                "total_bytes": execution_plan["selected_bytes"],
                "file_count_completed": completed_files,
                "file_count_total": execution_plan["file_count"],
                "progress_basis": "completed_file_bytes",
            },
        )
        return {
            "tool": "huggingface_hub.snapshot_download",
            "endpoint": execution_plan["endpoint"],
            "requested_revision": execution_plan["requested_revision"],
            "commit_hash": execution_plan["commit_hash"],
            "selected_paths": execution_plan["selected_paths"],
            "selected_bytes": execution_plan["selected_bytes"],
            "cached_bytes_at_start": execution_plan["cached_bytes"],
            "planned_download_bytes_at_start": execution_plan["download_bytes"],
        }

    @staticmethod
    def _planned_hf_completed(destination: Path, files: list[dict[str, Any]]) -> tuple[int, int]:
        root = destination.resolve()
        completed_bytes = 0
        completed_files = 0
        for item in files:
            relative = Path(str(item["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid Hugging Face plan path: {relative}")
            try:
                # A concurrent download may still be creating parent directories.
                # Non-strict Windows resolution can return a partially normalized
                # short-name path; only inspect files that fully resolve.
                target = (root / relative).resolve(strict=True)
            except FileNotFoundError:
                continue
            if root.resolve(strict=True) not in target.parents:
                raise ValueError(f"Hugging Face plan path escapes target directory: {relative}")
            expected_size = int(item["size_bytes"])
            try:
                actual_size = target.stat().st_size
            except OSError:
                continue
            if actual_size == expected_size:
                completed_bytes += expected_size
                completed_files += 1
        return completed_bytes, completed_files

    @staticmethod
    def _manifest_hash(files: list[dict[str, Any]]) -> str:
        digest = hashlib.sha256()
        for item in sorted(files, key=lambda value: str(value["path"])):
            digest.update(str(item["path"]).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(int(item["size_bytes"])).encode("ascii"))
            digest.update(b"\0")
            digest.update(str(item["sha256"]).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _local_path(source_uri: str) -> Path:
        from llm_platform_core.storage_layout import resolve_relocated_path
        source_uri = resolve_relocated_path(source_uri)
        # ``urlparse`` treats ``C:\\data`` as a URI with scheme ``c``. A Windows
        # drive path is still a plain local path and must be handled before URI parsing.
        if re.match(r"^[A-Za-z]:[\\/]", source_uri):
            return Path(source_uri).expanduser().resolve()
        parsed = urlparse(source_uri)
        if parsed.scheme not in {"", "file"}:
            raise ValueError("Local dataset source must be a path or file:// URI")
        raw_path = unquote(parsed.path) if parsed.scheme else source_uri
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        if os.name == "nt" and len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        return Path(raw_path).expanduser().resolve()

    @staticmethod
    async def _check_control(get_desired_state: GetDesiredState | None) -> None:
        desired = get_desired_state() if get_desired_state else "running"
        if desired in {"paused", "cancelled"}:
            raise RuntimeError(f"Dataset import interrupted: {desired}")

    async def _download_http(
        self,
        source_uri: str,
        destination: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
        *,
        relative_path: str | None = None,
        progress_start: float = 0.05,
        progress_span: float = 0.7,
        headers: dict[str, str] | None = None,
        expected_size: int | None = None,
    ) -> None:
        parsed = urlparse(source_uri)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("HTTP dataset source must use http:// or https://")
        raw_name = relative_path or unquote(Path(parsed.path).name) or "download.bin"
        safe_parts = [part for part in Path(raw_name).parts if part not in {"", ".", ".."}]
        target = destination.joinpath(*safe_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.part")
        if target.exists() and (expected_size is None or target.stat().st_size == expected_size):
            await emit(
                "progress",
                {
                    "progress": progress_start + progress_span,
                    "phase": "downloaded",
                    "current_file": raw_name,
                    "downloaded_bytes": target.stat().st_size,
                    "total_bytes": expected_size or target.stat().st_size,
                    "reused": True,
                },
            )
            return
        if target.exists():
            target.replace(partial)
        if (
            expected_size is not None
            and partial.exists()
            and partial.stat().st_size > expected_size
        ):
            partial.unlink()

        for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
            await self._check_control(get_desired_state)
            resume_from = partial.stat().st_size if partial.exists() else 0
            request_headers = dict(headers or {})
            if resume_from:
                request_headers["Range"] = f"bytes={resume_from}-"
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=DOWNLOAD_TIMEOUT,
                    headers=request_headers,
                ) as client:
                    async with client.stream("GET", source_uri) as response:
                        if response.status_code == 416 and resume_from:
                            total = self._content_range_total(response.headers.get("content-range"))
                            if total is not None and total == resume_from:
                                partial.replace(target)
                                return
                            partial.unlink(missing_ok=True)
                            raise httpx.RemoteProtocolError("远端拒绝续传且本地临时文件长度不匹配")
                        response.raise_for_status()
                        resumed = resume_from > 0 and response.status_code == 206
                        if not resumed:
                            resume_from = 0
                        response_length = self._positive_int(response.headers.get("content-length"))
                        total = self._content_range_total(response.headers.get("content-range"))
                        if total is None:
                            total = (
                                resume_from + response_length
                                if response_length is not None
                                else expected_size
                            )
                        downloaded = resume_from
                        last_reported = resume_from
                        with partial.open("ab" if resumed else "wb") as output:
                            async for chunk in response.aiter_raw(1024 * 256):
                                await self._check_control(get_desired_state)
                                output.write(chunk)
                                downloaded += len(chunk)
                                if downloaded - last_reported >= 1024 * 1024 or downloaded == total:
                                    ratio = downloaded / total if total else 0
                                    await emit(
                                        "progress",
                                        {
                                            "progress": min(
                                                progress_start + progress_span * ratio,
                                                progress_start + progress_span,
                                            ),
                                            "phase": "downloading",
                                            "current_file": raw_name,
                                            "downloaded_bytes": downloaded,
                                            "total_bytes": total,
                                            "retry_attempt": attempt,
                                            "resumed_from_bytes": resume_from,
                                        },
                                    )
                                    last_reported = downloaded
                        if total is not None and downloaded != total:
                            raise httpx.RemoteProtocolError(
                                f"响应体不完整：收到 {downloaded} 字节，预期 {total} 字节"
                            )
                partial.replace(target)
                return
            except (httpx.HTTPError, OSError) as exc:
                retryable = self._retryable_download_error(exc)
                if not retryable or attempt >= DOWNLOAD_RETRY_ATTEMPTS:
                    detail = str(exc).strip() or repr(exc)
                    raise RuntimeError(
                        f"下载 {raw_name} 失败（已尝试 {attempt} 次）："
                        f"{type(exc).__name__}: {detail}"
                    ) from exc
                delay = DOWNLOAD_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                await emit(
                    "progress",
                    {
                        "progress": progress_start,
                        "phase": "retrying_download",
                        "current_file": raw_name,
                        "retry_attempt": attempt + 1,
                        "resumed_from_bytes": partial.stat().st_size if partial.exists() else 0,
                        "error_type": type(exc).__name__,
                        "retry_in_seconds": delay,
                    },
                )
                await asyncio.sleep(delay)

    @staticmethod
    def _positive_int(value: str | None) -> int | None:
        return int(value) if value and value.isdigit() else None

    @staticmethod
    def _content_range_total(value: str | None) -> int | None:
        if not value or "/" not in value:
            return None
        raw_total = value.rpartition("/")[2]
        return int(raw_total) if raw_total.isdigit() else None

    @staticmethod
    def _retryable_download_error(exc: httpx.HTTPError | OSError) -> bool:
        if isinstance(exc, httpx.TransportError):
            return True
        return (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code in RETRYABLE_HTTP_STATUS_CODES
        )

    async def _download_huggingface(
        self,
        repo_id: str,
        source_config: dict[str, Any],
        destination: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> None:
        if not repo_id or "/" not in repo_id:
            raise ValueError("Hugging Face source must be a dataset repository ID such as org/name")
        revision = str(source_config.get("revision") or "main")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        endpoint = _huggingface_endpoint(source_config)
        api_url = f"{endpoint}/api/datasets/{repo_id}/tree/{revision}"
        entries: list[dict[str, Any]] = []
        for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
            entries = []
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=DOWNLOAD_TIMEOUT,
                    headers=headers,
                ) as client:
                    next_url: str | None = api_url
                    params: dict[str, Any] | None = {
                        "recursive": "true",
                        "expand": "false",
                        "limit": 100,
                    }
                    while next_url:
                        response = await client.get(next_url, params=params)
                        response.raise_for_status()
                        page = response.json()
                        if not isinstance(page, list):
                            raise RuntimeError("Unexpected Hugging Face tree response")
                        entries.extend(item for item in page if item.get("type") == "file")
                        next_link = response.links.get("next", {}).get("url")
                        next_url = str(next_link) if next_link else None
                        params = None
                break
            except httpx.HTTPError as exc:
                if not self._retryable_download_error(exc) or attempt >= DOWNLOAD_RETRY_ATTEMPTS:
                    detail = str(exc).strip() or repr(exc)
                    raise RuntimeError(
                        f"读取 Hugging Face 文件列表失败（已尝试 {attempt} 次）："
                        f"{type(exc).__name__}: {detail}"
                    ) from exc
                delay = DOWNLOAD_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
                await emit(
                    "progress",
                    {
                        "progress": 0.01,
                        "phase": "retrying_source_listing",
                        "retry_attempt": attempt + 1,
                        "error_type": type(exc).__name__,
                        "retry_in_seconds": delay,
                    },
                )
                await asyncio.sleep(delay)

        include = source_config.get("include_patterns") or []
        if isinstance(include, str):
            include = [value.strip() for value in include.split(",") if value.strip()]
        config_name = str(source_config.get("config") or "").strip().lower()
        split = str(source_config.get("split") or "").strip().lower()
        candidates: list[tuple[str, int | None]] = []
        for entry in entries:
            path = str(entry.get("path") or "")
            lowered = path.lower()
            if Path(path).suffix.lower() not in DATA_SUFFIXES:
                continue
            if include and not any(Path(path).match(pattern) for pattern in include):
                continue
            if config_name and config_name not in lowered:
                continue
            if split and split not in lowered:
                continue
            candidates.append((path, self._positive_int(str(entry.get("size") or ""))))
        if not candidates:
            raise FileNotFoundError("No matching data files found in the Hugging Face repository")
        for index, (path, size) in enumerate(candidates):
            await self._download_http(
                f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{path}",
                destination,
                emit,
                get_desired_state,
                relative_path=path,
                progress_start=0.05 + 0.72 * index / len(candidates),
                progress_span=0.72 / len(candidates),
                headers=headers,
                expected_size=size,
            )

    async def _clone_git(
        self,
        source_uri: str,
        source_config: dict[str, Any],
        destination: Path,
        emit: Emit,
    ) -> None:
        if not source_uri:
            raise ValueError("Git repository address is required")
        revision = str(source_config.get("revision") or "").strip()
        shutil.rmtree(destination)
        args = ["git", "clone", "--depth", "1"]
        if revision:
            args.extend(["--branch", revision])
        args.extend([source_uri, str(destination)])
        await emit(
            "progress", {"progress": 0.08, "phase": "cloning", "revision": revision or "default"}
        )
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            detail = (stderr or stdout).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"git clone failed: {detail}")
        shutil.rmtree(destination / ".git", ignore_errors=True)
        await emit("progress", {"progress": 0.78, "phase": "cloned"})

    async def _copy_local(self, source_uri: str, destination: Path, emit: Emit) -> None:
        source = self._local_path(source_uri)
        if not source.exists():
            raise FileNotFoundError(f"Local dataset source does not exist: {source}")
        if source.is_dir() and destination.is_relative_to(source):
            raise ValueError("Dataset destination cannot be inside the imported source directory")
        await emit("progress", {"progress": 0.08, "phase": "copying", "source": str(source)})
        if source.is_dir():
            await asyncio.to_thread(shutil.copytree, source, destination, dirs_exist_ok=True)
        else:
            await asyncio.to_thread(shutil.copy2, source, destination / source.name)
        await emit("progress", {"progress": 0.78, "phase": "copied"})

    @staticmethod
    def _configured_text_fields(source_config: dict[str, Any]) -> list[str]:
        configured = source_config.get("text_fields") or source_config.get("text_field")
        field_mapping = source_config.get("field_mapping") or {}
        if not configured and isinstance(field_mapping, dict):
            configured = field_mapping.get("text")
        if isinstance(configured, str):
            return [configured.strip()] if configured.strip() else []
        if isinstance(configured, list):
            return [str(value).strip() for value in configured if str(value).strip()]
        return []

    @classmethod
    def _parquet_text_fields(
        cls, path: Path, source_config: dict[str, Any] | None = None
    ) -> list[str]:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet analysis requires pyarrow on this Worker") from exc
        schema = pq.read_schema(path)
        configured = cls._configured_text_fields(source_config or {})
        if configured:
            missing = [field for field in configured if field not in schema.names]
            if missing:
                raise ValueError(f"configured Parquet text fields are missing: {missing}")
            invalid = [
                field
                for field in configured
                if not (
                    pa.types.is_string(schema.field(field).type)
                    or pa.types.is_large_string(schema.field(field).type)
                )
            ]
            if invalid:
                raise ValueError(f"configured Parquet text fields are not strings: {invalid}")
            return configured
        preferred = [field for field in PREFERRED_TEXT_FIELDS if field in schema.names]
        if preferred:
            return preferred
        string_fields = [
            field.name
            for field in schema
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
        ]
        if len(string_fields) == 1:
            return string_fields
        if not string_fields:
            raise ValueError("Parquet file has no string column that can be used as document text")
        raise ValueError("Parquet text column is ambiguous; configure text_field or text_fields")

    @classmethod
    def _iter_parquet_documents(
        cls,
        path: Path,
        source_config: dict[str, Any] | None = None,
        get_desired_state: GetDesiredState | None = None,
    ) -> Iterator[str]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet analysis requires pyarrow on this Worker") from exc
        fields = cls._parquet_text_fields(path, source_config)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=PARQUET_BATCH_SIZE,
            columns=fields,
            use_threads=True,
        ):
            desired = get_desired_state() if get_desired_state else "running"
            if desired in {"paused", "cancelled"}:
                raise AnalysisInterrupted(desired)
            columns = [batch.column(index) for index in range(len(fields))]
            for row_index in range(batch.num_rows):
                values = [column[row_index].as_py() for column in columns]
                if not any(value not in {None, ""} for value in values):
                    continue
                text = "\n".join(
                    str(value) for value in values if value not in {None, ""}
                )
                if text:
                    yield text

    @classmethod
    def _iter_documents(
        cls,
        path: Path,
        source_config: dict[str, Any] | None = None,
        get_desired_state: GetDesiredState | None = None,
    ) -> Iterator[str]:
        suffix = path.suffix.lower()
        if suffix in PARQUET_SUFFIXES:
            yield from cls._iter_parquet_documents(path, source_config, get_desired_state)
        elif suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        yield line
                        continue
                    yield cls._record_text(value)
        elif suffix == ".json":
            if path.stat().st_size > 64 * 1024 * 1024:
                return
            value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            records = value if isinstance(value, list) else [value]
            for record in records:
                yield cls._record_text(record)
        elif suffix in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ",")
                for row in reader:
                    yield cls._record_text(row)
        else:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    text = line.strip()
                    if text:
                        yield text

    @staticmethod
    def _record_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            parts = [
                str(value[key])
                for key in PREFERRED_TEXT_FIELDS
                if value.get(key) is not None and value.get(key) != ""
            ]
            if parts:
                return "\n".join(parts)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _language(text: str) -> str:
        chinese = sum("\u4e00" <= char <= "\u9fff" for char in text)
        latin = sum(char.isascii() and char.isalpha() for char in text)
        code = sum(token in text for token in ("def ", "class ", "function ", "=>", "{", "}"))
        if code >= 2:
            return "code"
        if chinese > latin * 0.25:
            return "zh"
        if latin:
            return "en"
        return "other"

    @staticmethod
    def _token_estimate(text: str) -> int:
        chinese = sum("\u4e00" <= char <= "\u9fff" for char in text)
        other = max(len(text) - chinese, 0)
        return chinese + (other + 3) // 4

    @staticmethod
    def _known_files(source_config: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        return {
            str(item.get("path") or "").replace("\\", "/"): dict(item)
            for item in (source_config or {}).get("known_files") or []
            if item.get("path") and item.get("sha256")
        }

    @staticmethod
    def _file_record(
        path: Path,
        relative: str,
        known_files: dict[str, dict[str, Any]],
        get_desired_state: GetDesiredState | None,
    ) -> dict[str, Any]:
        size = path.stat().st_size
        known = known_files.get(relative) or {}
        if str(known.get("sha256") or "") and int(known.get("size_bytes") or -1) == size:
            return {
                "path": relative,
                "size_bytes": size,
                "sha256": str(known["sha256"]),
                "hash_reused": True,
            }
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                desired = get_desired_state() if get_desired_state else "running"
                if desired in {"paused", "cancelled"}:
                    raise AnalysisInterrupted(desired)
                digest.update(chunk)
        return {
            "path": relative,
            "size_bytes": size,
            "sha256": digest.hexdigest(),
            "hash_reused": False,
        }

    @classmethod
    def _inventory_unparsed(cls, context: DatasetParserContext) -> dict[str, Any]:
        files = sorted(
            path
            for path in context.root.rglob("*")
            if path.is_file()
            and not ({".hfd", ".cache", ".git"} & set(path.relative_to(context.root).parts))
        )
        known_files = cls._known_files(dict(context.parser_config))
        records: list[dict[str, Any]] = []
        for index, path in enumerate(files):
            desired = context.get_desired_state() if context.get_desired_state else "running"
            if desired in {"paused", "cancelled"}:
                raise AnalysisInterrupted(desired)
            relative = path.relative_to(context.root).as_posix()
            records.append(cls._file_record(path, relative, known_files, context.get_desired_state))
            if context.progress_callback:
                context.progress_callback(
                    {
                        "progress": (index + 1) / max(len(files), 1),
                        "file_count_completed": index + 1,
                        "file_count_total": len(files),
                        "document_count": 0,
                    }
                )
        raw_size = sum(int(item["size_bytes"]) for item in records)
        return {
            "raw_size_bytes": raw_size,
            "document_count": 0,
            "record_count": len(records),
            "token_estimate": 0,
            "automatic_quality_score": None,
            "quality_report": {
                "semantic_analysis": "unavailable",
                "message": f"No semantic parser is registered for modality: {context.modality}",
            },
            "sample_preview": [],
            "source_snapshot": {
                "source_type": context.source_type,
                "source_uri": context.source_uri,
                "fetched": True,
                "fetched_at": datetime.now(UTC).isoformat(),
                "file_count": len(records),
                "files": records,
                "raw_size_bytes": raw_size,
            },
            "data_schema": {"type": "opaque_files", "semantic_parser": False},
            "statistics": {
                "record_count": len(records),
                "file_count": len(records),
                "raw_size_bytes": raw_size,
            },
            "train_ready": False,
        }

    @classmethod
    def _analyze_with_parallel_parquet(
        cls,
        root: Path,
        source_type: str,
        source_uri: str,
        dataset_version_id: str,
        record_manifest_path: Path,
        source_config: dict[str, Any] | None,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        get_desired_state: GetDesiredState | None,
    ) -> dict[str, Any]:
        import pyarrow.parquet as pq

        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not ({".hfd", ".cache", ".git"} & set(path.relative_to(root).parts))
        )
        known_files = cls._known_files(source_config)
        shard_root = record_manifest_path.parent / "analysis-shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        cancel_path = shard_root / ".cancel"
        cancel_path.unlink(missing_ok=True)
        file_records: dict[str, dict[str, Any]] = {}
        parsed_file_details: list[dict[str, Any]] = []
        unsupported_files: list[str] = []
        results: dict[str, dict[str, Any]] = {}
        jobs: list[dict[str, Any]] = []
        total_bytes = sum(path.stat().st_size for path in files)
        total_rows = 0
        completed_bytes = 0
        completed_rows = 0
        quality_remaining = QUALITY_SAMPLE_LIMIT

        for file_index, path in enumerate(files):
            relative = path.relative_to(root).as_posix()
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                try:
                    text_fields = cls._parquet_text_fields(path, source_config)
                    source_rows = int(pq.ParquetFile(path).metadata.num_rows)
                except (OSError, ValueError, RuntimeError) as exc:
                    unsupported_files.append(f"{relative}: {type(exc).__name__}: {exc}")
                    file_records[relative] = cls._file_record(
                        path, relative, known_files, get_desired_state
                    )
                    completed_bytes += path.stat().st_size
                    continue
                total_rows += source_rows
                known = known_files.get(relative) or {}
                quality_budget = min(quality_remaining, source_rows)
                quality_remaining -= quality_budget
                jobs.append(
                    {
                        "path": str(path),
                        "logical_path": relative,
                        "dataset_version_id": dataset_version_id,
                        "text_fields": text_fields,
                        "source_rows": source_rows,
                        "quality_budget": quality_budget,
                        "known_hash": known.get("sha256"),
                        "known_size": known.get("size_bytes"),
                        "shard_path": str(shard_root / f"{file_index:06d}.jsonl.gz"),
                        "temp_dir": str(shard_root / f"duckdb-{file_index:06d}"),
                        "cancel_path": str(cancel_path),
                    }
                )
            elif suffix not in TEXT_SUFFIXES:
                unsupported_files.append(relative)
                file_records[relative] = cls._file_record(
                    path, relative, known_files, get_desired_state
                )
                completed_bytes += path.stat().st_size

        worker_count = recommended_processes(
            len(jobs), (source_config or {}).get("analysis_workers")
        )
        started = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                pending = {executor.submit(analyze_parquet_file, job): job for job in jobs}
                while pending:
                    desired = get_desired_state() if get_desired_state else "running"
                    if desired in {"paused", "cancelled"}:
                        cancel_path.touch(exist_ok=True)
                        for future in pending:
                            future.cancel()
                        raise AnalysisInterrupted(desired)
                    done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    for future in done:
                        job = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            if cancel_path.exists():
                                desired = get_desired_state() if get_desired_state else "cancelled"
                                raise AnalysisInterrupted(desired) from exc
                            relative = str(job["logical_path"])
                            unsupported_files.append(f"{relative}: {type(exc).__name__}: {exc}")
                            path = Path(str(job["path"]))
                            file_records[relative] = cls._file_record(
                                path, relative, known_files, get_desired_state
                            )
                            completed_bytes += path.stat().st_size
                            continue
                        relative = str(result["logical_path"])
                        results[relative] = result
                        file_records[relative] = dict(result["file_record"])
                        parsed_file_details.append(dict(result["file_detail"]))
                        completed_bytes += int(result["file_record"]["size_bytes"])
                        completed_rows += int(result["document_count"])
                        elapsed = max(time.perf_counter() - started, 0.001)
                        if progress_callback:
                            progress_callback(
                                {
                                    "progress": completed_bytes / max(total_bytes, 1),
                                    "progress_basis": "processed_source_bytes",
                                    "processed_bytes": completed_bytes,
                                    "total_bytes": total_bytes,
                                    "processed_rows": completed_rows,
                                    "total_rows": total_rows,
                                    "documents_per_second": round(completed_rows / elapsed, 2),
                                    "bytes_per_second": round(completed_bytes / elapsed, 2),
                                    "file_count_completed": len(file_records),
                                    "file_count_total": len(files),
                                    "parallel_workers": worker_count,
                                }
                            )

            # Text sidecars are usually README/license files. Keep their legacy semantics
            # without putting the Parquet hot path back through Python.
            for file_index, path in enumerate(files):
                if path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                relative = path.relative_to(root).as_posix()
                file_record = cls._file_record(path, relative, known_files, get_desired_state)
                shard_path = shard_root / f"text-{file_index:06d}.jsonl.gz"
                language_counts: Counter[str] = Counter()
                length_counts: Counter[str] = Counter()
                document_count = token_estimate = total_chars = 0
                samples: list[dict[str, Any]] = []
                with _gzip_text_writer(shard_path) as shard:
                    for index, text in enumerate(
                        cls._iter_documents(path, source_config, get_desired_state)
                    ):
                        if not text:
                            continue
                        document_count += 1
                        length = len(text)
                        total_chars += length
                        record_tokens = cls._token_estimate(text)
                        token_estimate += record_tokens
                        language = cls._language(text)
                        language_counts[language] += 1
                        bucket = (
                            "0-128"
                            if length <= 128
                            else "129-512"
                            if length <= 512
                            else "513-2048"
                            if length <= 2048
                            else "2049+"
                        )
                        length_counts[bucket] += 1
                        encoded = text.encode("utf-8", errors="replace")
                        content_hash = hashlib.sha256(encoded).hexdigest()
                        shard.write(
                            canonical_json(
                                {
                                    "type": "record",
                                    "record_id": stable_record_id(
                                        dataset_version_id, relative, index, content_hash
                                    ),
                                    "logical_path": relative,
                                    "record_index": index,
                                    "content_hash": content_hash,
                                    "character_length": length,
                                    "byte_length": len(encoded),
                                    "token_estimate": record_tokens,
                                    "language": language,
                                }
                            )
                            + "\n"
                        )
                result = {
                    "logical_path": relative,
                    "shard_path": str(shard_path),
                    "file_record": file_record,
                    "file_detail": {
                        "path": relative,
                        "format": path.suffix.lower().lstrip("."),
                        "document_count": document_count,
                        "engine": "python_text_v1",
                    },
                    "document_count": document_count,
                    "token_estimate": token_estimate,
                    "total_chars": total_chars,
                    "language_counts": dict(language_counts),
                    "length_counts": dict(length_counts),
                    "quality": {
                        "evaluated_documents": 0,
                        "duplicates": 0,
                        "pii_documents": 0,
                        "ad_documents": 0,
                        "replacement_chars": 0,
                        "total_chars": 0,
                    },
                    "samples": samples,
                }
                results[relative] = result
                file_records[relative] = file_record
                parsed_file_details.append(result["file_detail"])
                completed_bytes += int(file_record["size_bytes"])
                completed_rows += document_count

            record_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with _gzip_text_writer(record_manifest_path) as record_manifest:
                record_manifest.write(
                    canonical_json(
                        {
                            "type": "header",
                            "format": RECORD_MANIFEST_FORMAT,
                            "dataset_version_id": dataset_version_id,
                            "analysis_engine": PARQUET_ENGINE,
                            "token_estimate_method": TOKEN_ESTIMATE_METHOD,
                        }
                    )
                    + "\n"
                )
            with record_manifest_path.open("ab") as output:
                for relative in sorted(results):
                    with Path(str(results[relative]["shard_path"])).open("rb") as shard:
                        shutil.copyfileobj(shard, output, length=8 * 1024 * 1024)
        except BaseException:
            cancel_path.touch(exist_ok=True)
            record_manifest_path.unlink(missing_ok=True)
            raise
        finally:
            shutil.rmtree(shard_root, ignore_errors=True)

        document_count = sum(int(result["document_count"]) for result in results.values())
        token_estimate = sum(int(result["token_estimate"]) for result in results.values())
        language_counts: Counter[str] = Counter()
        length_counts: Counter[str] = Counter()
        quality = Counter()
        samples: list[dict[str, Any]] = []
        for relative in sorted(results):
            result = results[relative]
            language_counts.update(result["language_counts"])
            length_counts.update(result["length_counts"])
            quality.update(result["quality"])
            for sample in result["samples"]:
                if len(samples) >= 50:
                    break
                text = str(sample["raw_text"])
                language = cls._language(text)
                flags = [
                    name
                    for name, active in (
                        ("pii", bool(EMAIL_RE.search(text) or PHONE_RE.search(text))),
                        ("ad_template", bool(URL_RE.search(text))),
                        ("garbled", "\ufffd" in text),
                    )
                    if active
                ]
                samples.append(
                    {
                        **sample,
                        "language": language,
                        "length": len(text),
                        "flags": flags,
                        "rejection_reason": None,
                    }
                )

        evaluated = int(quality["evaluated_documents"])
        divisor = max(evaluated, 1)
        garbled_rate = int(quality["replacement_chars"]) / max(int(quality["total_chars"]), 1)
        duplicate_rate = int(quality["duplicates"]) / divisor
        pii_rate = int(quality["pii_documents"]) / divisor
        ad_rate = int(quality["ad_documents"]) / divisor
        score = (
            round(
                max(
                    0.0,
                    100.0
                    - garbled_rate * 1000
                    - duplicate_rate * 35
                    - pii_rate * 25
                    - ad_rate * 15,
                ),
                2,
            )
            if document_count
            else None
        )

        def percentages(counter: Counter[str]) -> dict[str, float]:
            total = sum(counter.values()) or 1
            return {key: round(value * 100 / total, 2) for key, value in counter.items()}

        ordered_file_records = [file_records[path.relative_to(root).as_posix()] for path in files]
        raw_size = sum(int(record["size_bytes"]) for record in ordered_file_records)
        return {
            "raw_size_bytes": raw_size,
            "document_count": document_count,
            "token_estimate": token_estimate,
            "automatic_quality_score": score,
            "quality_report": {
                "analyzed_document_count": document_count,
                "parsed_file_count": len(parsed_file_details),
                "parsed_files": sorted(parsed_file_details, key=lambda item: item["path"]),
                "unsupported_files": unsupported_files,
                "quality_evaluated_document_count": evaluated,
                "quality_sample_limit": QUALITY_SAMPLE_LIMIT,
                "quality_sample_strategy": "deterministic_path_prefix_v1",
                "analysis_engine": PARQUET_ENGINE,
                "parallel_workers": worker_count,
                "token_estimate_method": TOKEN_ESTIMATE_METHOD,
                "garbled_rate": round(garbled_rate, 8),
                "duplicate_rate": round(duplicate_rate, 8),
                "pii_rate": round(pii_rate, 8),
                "ad_template_rate": round(ad_rate, 8),
                "language_distribution": percentages(language_counts),
                "length_distribution": percentages(length_counts),
                "education_value": None,
                "factual_density": None,
                "code_quality": None,
                "license_check": "declared_only",
                "contamination": None,
                "quality_score_method": "transparent_heuristic_v1",
            },
            "sample_preview": samples,
            "source_snapshot": {
                "source_type": source_type,
                "source_uri": source_uri,
                "fetched": True,
                "fetched_at": datetime.now(UTC).isoformat(),
                "file_count": len(files),
                "files": ordered_file_records,
                "raw_size_bytes": raw_size,
                "analysis_engine": PARQUET_ENGINE,
            },
            "train_ready": document_count > 0,
        }
    @classmethod
    def _analyze(
        cls,
        root: Path,
        source_type: str,
        source_uri: str,
        dataset_version_id: str,
        record_manifest_path: Path,
        source_config: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        if any(
            path.is_file() and path.suffix.lower() in PARQUET_SUFFIXES for path in root.rglob("*")
        ):
            return cls._analyze_with_parallel_parquet(
                root,
                source_type,
                source_uri,
                dataset_version_id,
                record_manifest_path,
                source_config,
                progress_callback,
                get_desired_state,
            )
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and not ({".hfd", ".cache", ".git"} & set(path.relative_to(root).parts))
        )
        file_records: list[dict[str, Any]] = []
        document_count = token_estimate = total_chars = replacement_chars = 0
        duplicates = pii_documents = ad_documents = 0
        language_counts: Counter[str] = Counter()
        length_counts: Counter[str] = Counter()
        seen_hashes: set[str] = set()
        samples: list[dict[str, Any]] = []
        parsed_files = 0
        parsed_file_details: list[dict[str, Any]] = []
        unsupported_files: list[str] = []
        quality_evaluated_documents = 0
        record_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with _gzip_text_writer(record_manifest_path) as record_manifest:
            record_manifest.write(
                canonical_json(
                    {
                        "type": "header",
                        "format": RECORD_MANIFEST_FORMAT,
                        "dataset_version_id": dataset_version_id,
                    }
                )
                + "\n"
            )
            for file_index, path in enumerate(files):
                desired = get_desired_state() if get_desired_state else "running"
                if desired in {"paused", "cancelled"}:
                    raise AnalysisInterrupted(desired)
                relative = path.relative_to(root).as_posix()
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        desired = get_desired_state() if get_desired_state else "running"
                        if desired in {"paused", "cancelled"}:
                            raise AnalysisInterrupted(desired)
                        digest.update(chunk)
                file_records.append(
                    {
                        "path": relative,
                        "size_bytes": path.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
                if path.suffix.lower() not in TEXT_SUFFIXES | PARQUET_SUFFIXES:
                    unsupported_files.append(relative)
                    if progress_callback:
                        progress_callback(
                            {
                                "progress": (file_index + 1) / max(len(files), 1),
                                "file_count_completed": file_index + 1,
                                "file_count_total": len(files),
                                "document_count": document_count,
                            }
                        )
                    continue
                try:
                    text_fields = (
                        cls._parquet_text_fields(path, source_config)
                        if path.suffix.lower() in PARQUET_SUFFIXES
                        else None
                    )
                    iterator = cls._iter_documents(path, source_config, get_desired_state)
                    parsed_files += 1
                    file_document_count = 0
                    for index, text in enumerate(iterator):
                        if not text:
                            continue
                        file_document_count += 1
                        document_count += 1
                        total_chars += len(text)
                        replacement_chars += text.count("\ufffd")
                        record_tokens = cls._token_estimate(text)
                        token_estimate += record_tokens
                        language = cls._language(text)
                        language_counts[language] += 1
                        length = len(text)
                        bucket = (
                            "0-128"
                            if length <= 128
                            else "129-512"
                            if length <= 512
                            else "513-2048"
                            if length <= 2048
                            else "2049+"
                        )
                        length_counts[bucket] += 1
                        encoded = text.encode("utf-8", errors="replace")
                        content_hash = hashlib.sha256(encoded).hexdigest()
                        record_manifest.write(
                            canonical_json(
                                {
                                    "type": "record",
                                    "record_id": stable_record_id(
                                        dataset_version_id, relative, index, content_hash
                                    ),
                                    "logical_path": relative,
                                    "record_index": index,
                                    "content_hash": content_hash,
                                    "character_length": length,
                                    "byte_length": len(encoded),
                                    "token_estimate": record_tokens,
                                    "language": language,
                                }
                            )
                            + "\n"
                        )
                        quality_active = quality_evaluated_documents < QUALITY_SAMPLE_LIMIT
                        quality_evaluated_documents += int(quality_active)
                        duplicate = quality_active and content_hash in seen_hashes
                        duplicates += int(duplicate)
                        if quality_active:
                            seen_hashes.add(content_hash)
                        pii = quality_active and bool(
                            EMAIL_RE.search(text) or PHONE_RE.search(text)
                        )
                        pii_documents += int(pii)
                        ad_like = (
                            quality_active
                            and URL_RE.search(text) is not None
                            and any(
                                word in text.lower()
                                for word in ("buy", "sale", "discount", "点击", "优惠", "购买")
                            )
                        )
                        ad_documents += int(ad_like)
                        if len(samples) < 50:
                            flags = [
                                name
                                for name, active in (
                                    ("duplicate", duplicate),
                                    ("pii", pii),
                                    ("ad_template", ad_like),
                                    ("garbled", "\ufffd" in text),
                                )
                                if active
                            ]
                            samples.append(
                                {
                                    "sample_key": f"{relative}:{index + 1}",
                                    "source_file": relative,
                                    "raw_text": text[:4000],
                                    "cleaned_text": text[:4000],
                                    "language": language,
                                    "length": length,
                                    "flags": flags,
                                    "rejection_reason": None,
                                }
                            )
                    parsed_file_details.append(
                        {
                            "path": relative,
                            "format": path.suffix.lower().lstrip("."),
                            "document_count": file_document_count,
                            **({"text_fields": text_fields} if text_fields else {}),
                        }
                    )
                except AnalysisInterrupted:
                    raise
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    csv.Error,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    unsupported_files.append(f"{relative}: {type(exc).__name__}: {exc}")
                if progress_callback:
                    progress_callback(
                        {
                            "progress": (file_index + 1) / max(len(files), 1),
                            "file_count_completed": file_index + 1,
                            "file_count_total": len(files),
                            "document_count": document_count,
                        }
                    )

        raw_size = sum(record["size_bytes"] for record in file_records)
        divisor = max(quality_evaluated_documents, 1)
        garbled_rate = replacement_chars / max(total_chars, 1)
        duplicate_rate = duplicates / divisor
        pii_rate = pii_documents / divisor
        ad_rate = ad_documents / divisor
        score = (
            round(
                max(
                    0.0,
                    100.0
                    - garbled_rate * 1000
                    - duplicate_rate * 35
                    - pii_rate * 25
                    - ad_rate * 15,
                ),
                2,
            )
            if document_count
            else None
        )

        def percentages(counter: Counter[str]) -> dict[str, float]:
            total = sum(counter.values()) or 1
            return {key: round(value * 100 / total, 2) for key, value in counter.items()}

        source_snapshot = {
            "source_type": source_type,
            "source_uri": source_uri,
            "fetched": True,
            "fetched_at": datetime.now(UTC).isoformat(),
            "file_count": len(files),
            "files": file_records,
            "raw_size_bytes": raw_size,
        }
        quality_report = {
            "analyzed_document_count": document_count,
            "parsed_file_count": parsed_files,
            "parsed_files": parsed_file_details,
            "unsupported_files": unsupported_files,
            "quality_evaluated_document_count": quality_evaluated_documents,
            "quality_sample_limit": QUALITY_SAMPLE_LIMIT,
            "garbled_rate": round(garbled_rate, 8),
            "duplicate_rate": round(duplicate_rate, 8),
            "pii_rate": round(pii_rate, 8),
            "ad_template_rate": round(ad_rate, 8),
            "language_distribution": percentages(language_counts),
            "length_distribution": percentages(length_counts),
            "education_value": None,
            "factual_density": None,
            "code_quality": None,
            "license_check": "declared_only",
            "contamination": None,
            "quality_score_method": "transparent_heuristic_v1",
        }
        return {
            "raw_size_bytes": raw_size,
            "document_count": document_count,
            "token_estimate": token_estimate,
            "automatic_quality_score": score,
            "quality_report": quality_report,
            "sample_preview": samples,
            "source_snapshot": source_snapshot,
            "train_ready": document_count > 0,
        }


def _run_builtin_text_parser(context: DatasetParserContext) -> dict[str, Any]:
    if context.record_manifest_path is None:
        raise ValueError("The built-in text parser requires a record manifest output path")
    return DataImportAdapter._analyze(
        context.root,
        context.source_type,
        context.source_uri,
        context.dataset_version_id,
        context.record_manifest_path,
        dict(context.parser_config),
        context.progress_callback,
        context.get_desired_state,
    )


PARSER_REGISTRY.register(
    FunctionParserAdapter(
        parser_id=TEXT_PARSER_ID,
        parser_version=TEXT_PARSER_VERSION,
        modalities=frozenset({"text"}),
        handler=_run_builtin_text_parser,
    )
)
