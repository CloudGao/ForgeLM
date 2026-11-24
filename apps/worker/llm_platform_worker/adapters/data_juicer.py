from __future__ import annotations

import asyncio
import codecs
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
from collections import Counter
from collections.abc import Awaitable, Callable
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any

from ..project_storage import project_data_asset_root, scoped_runtime_path
from .data_import import AnalysisInterrupted, DataImportAdapter

Emit = Callable[[str, dict[str, Any]], Awaitable[Any]]
GetDesiredState = Callable[[], str]
CONVERSATION_PROJECTION_VERSION = "role-content-v1"
MAX_PROCESS_LOG_CHARS = 32_768


def _validated_export_files(export_path: Path) -> list[Path]:
    """Reject missing/truncated shards before publishing the dataset directory."""
    if export_path.suffix == ".jsonl":
        if not export_path.is_file():
            raise FileNotFoundError("Data-Juicer completed without creating dataset.jsonl")
        return [export_path]

    import pyarrow.parquet as pq

    shards = sorted(export_path.parent.glob("dataset-*-of-*.parquet"))
    if export_path.is_file():
        if shards:
            raise ValueError("Data-Juicer export mixes single-file and sharded layouts")
        pq.read_metadata(export_path)
        return [export_path]
    if not shards:
        raise FileNotFoundError("Data-Juicer completed without creating Parquet shards")
    counts: set[int] = set()
    indexes: list[int] = []
    for path in shards:
        match = re.fullmatch(r"dataset-(\d+)-of-(\d+)\.parquet", path.name)
        if match is None:
            raise ValueError(f"Invalid Data-Juicer shard name: {path.name}")
        indexes.append(int(match[1]))
        counts.add(int(match[2]))
        # Data-Juicer's asynchronous exporter may exit without surfacing a
        # child exception. A readable footer and complete shard set are required.
        pq.read_metadata(path)
    if counts != {len(shards)} or indexes != list(range(len(shards))):
        raise ValueError("Data-Juicer export has missing or inconsistent Parquet shards")
    return shards


def _conversation_source_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in {".jsonl", ".ndjson"} else []
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".jsonl", ".ndjson"}
    ]


def _project_conversation_input(
    source_root: Path,
    destination: Path,
    messages_key: str,
    get_desired_state: GetDesiredState | None,
) -> dict[str, Any]:
    """Write a uniform role/content JSONL that Arrow can infer deterministically."""

    source_files = _conversation_source_files(source_root)
    if not source_files:
        raise ValueError("conversation projection requires JSONL or NDJSON input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    rejected = 0
    reasons: Counter[str] = Counter()
    with destination.open("wb") as output:
        for source_path in source_files:
            logical_path = (
                source_path.name
                if source_root.is_file()
                else source_path.relative_to(source_root).as_posix()
            )
            with source_path.open("r", encoding="utf-8", errors="replace") as stream:
                for record_index, line in enumerate(stream):
                    if record_index % 4096 == 0 and get_desired_state:
                        desired = get_desired_state()
                        if desired in {"paused", "cancelled"}:
                            raise AnalysisInterrupted(desired)
                    try:
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("record_not_object")
                        values = record.get(messages_key)
                        if not isinstance(values, list) or not values:
                            raise ValueError("empty_messages")
                        messages: list[dict[str, str]] = []
                        for value in values:
                            if not isinstance(value, dict):
                                raise ValueError("message_not_object")
                            role = str(value.get("role") or "").strip()
                            content = value.get("content")
                            if role not in {"system", "user", "assistant"}:
                                raise ValueError("unknown_role")
                            if not isinstance(content, str) or not content.strip():
                                raise ValueError("empty_content")
                            messages.append({"role": role, "content": content})
                        if any(message["role"] == "system" for message in messages[1:]):
                            raise ValueError("system_not_first")
                        conversational = [
                            message["role"]
                            for message in messages
                            if message["role"] != "system"
                        ]
                        if "assistant" not in conversational:
                            raise ValueError("assistant_missing")
                        if any(
                            previous == current
                            for previous, current in zip(
                                conversational, conversational[1:], strict=False
                            )
                        ):
                            raise ValueError("invalid_role_order")
                        projected = {
                            messages_key: messages,
                            "__forgelm_source": {
                                "logical_path": logical_path,
                                "record_index": record_index,
                            },
                        }
                        output.write(
                            (
                                json.dumps(
                                    projected,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            ).encode("utf-8")
                        )
                        accepted += 1
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        rejected += 1
                        reasons[str(exc) or "invalid_record"] += 1
    if accepted == 0:
        raise ValueError("conversation projection produced no valid records")
    return {
        "version": CONVERSATION_PROJECTION_VERSION,
        "accepted_record_count": accepted,
        "rejected_record_count": rejected,
        "rejection_reasons": dict(reasons),
    }


async def _forward_process_stream(
    stream: asyncio.StreamReader | None,
    name: str,
    emit: Emit,
) -> None:
    """Forward logs without letting carriage-return progress lines grow unbounded."""

    if stream is None:
        return
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buffer = ""
    latest_progress = ""
    loop = asyncio.get_running_loop()
    last_progress_emit = loop.time()

    async def send(message: str, *, continued: bool = False) -> None:
        for offset in range(0, len(message), MAX_PROCESS_LOG_CHARS):
            part = message[offset : offset + MAX_PROCESS_LOG_CHARS]
            await emit(
                "log",
                {
                    "stream": name,
                    "message": part,
                    **(
                        {"continued": True}
                        if continued or offset + MAX_PROCESS_LOG_CHARS < len(message)
                        else {}
                    ),
                },
            )

    while chunk := await stream.read(16_384):
        buffer += decoder.decode(chunk)
        cursor = 0
        for index, character in enumerate(buffer):
            if character not in {"\r", "\n"}:
                continue
            message = buffer[cursor:index].strip()
            cursor = index + 1
            if character == "\r":
                if message:
                    latest_progress = message
                continue
            message = message or latest_progress
            latest_progress = ""
            await send(message)
        buffer = buffer[cursor:]
        while len(buffer) > MAX_PROCESS_LOG_CHARS:
            await send(buffer[:MAX_PROCESS_LOG_CHARS], continued=True)
            buffer = buffer[MAX_PROCESS_LOG_CHARS:]
        if latest_progress and loop.time() - last_progress_emit >= 5:
            await send(latest_progress)
            latest_progress = ""
            last_progress_emit = loop.time()

    buffer += decoder.decode(b"", final=True)
    await send(buffer.strip() or latest_progress)


def _terminate_process_tree(process: asyncio.subprocess.Process, *, force: bool = False) -> None:
    if os.name == "nt":
        (process.kill if force else process.terminate)()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def data_juicer_executable() -> str | None:
    executable_name = "dj-process.exe" if os.name == "nt" else "dj-process"
    # Keep the virtual-environment path. Resolving the Python symlink first points
    # at /usr/bin on POSIX and misses console scripts installed beside the venv's
    # Python executable.
    beside_python = Path(os.sys.executable).parent / executable_name
    if beside_python.is_file():
        return str(beside_python)
    return shutil.which("dj-process")


def data_juicer_version() -> str | None:
    for distribution in ("py-data-juicer", "data-juicer"):
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return None


def data_juicer_available() -> bool:
    return (
        importlib.util.find_spec("data_juicer") is not None
        and data_juicer_executable() is not None
        and data_juicer_version() is not None
    )


class DataJuicerAdapter:
    """Run a frozen Data-Juicer recipe and register a new analyzed dataset snapshot."""

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        runtime_context = task.get("runtime_context") or {}
        run_dir = Path(runtime_context.get("run_dir") or ".").resolve()
        attempt_dir = Path(runtime_context.get("attempt_dir") or run_dir).resolve()
        project_workspace = str(runtime_context.get("project_workspace") or "")
        worker_id = str(runtime_context.get("worker_id") or "") or None
        processing_run_id = str(payload.get("processing_run_id") or "").strip()
        output_version_id = str(payload.get("output_dataset_version_id") or "").strip()
        data_input = payload.get("data_processing_input") or {}
        recipe = payload.get("recipe") or {}
        input_path = Path(str(data_input.get("runtime_path") or "")).resolve()
        operators = recipe.get("operators")
        execution_config = recipe.get("execution_config") or {}
        if not processing_run_id or not output_version_id:
            raise ValueError("Data processing task is missing its frozen run identity")
        if not input_path.exists():
            raise FileNotFoundError("Frozen Data-Juicer input was not materialized")
        if not isinstance(operators, list) or not operators:
            raise ValueError("Frozen Data-Juicer recipe has no operators")
        effective_operators = deepcopy(operators)
        file_selection = data_input.get("file_selection")
        text_keys: list[str] | None = None
        conversation_messages_key: str | None = None
        first_operator = effective_operators[0]
        if (
            isinstance(first_operator, dict)
            and list(first_operator) == ["agent_dialog_normalize_mapper"]
        ):
            first_parameters = first_operator["agent_dialog_normalize_mapper"]
            if first_parameters is None:
                first_parameters = {}
                first_operator["agent_dialog_normalize_mapper"] = first_parameters
            if not isinstance(first_parameters, dict):
                raise ValueError("agent_dialog_normalize_mapper parameters must be an object")
            messages_key = str(first_parameters.get("messages_key") or "messages").strip()
            if file_selection:
                raise ValueError("File-selected conversation projection is not supported")
            if not messages_key:
                raise ValueError("agent_dialog_normalize_mapper messages_key cannot be empty")
            text_keys = [messages_key]
            conversation_messages_key = messages_key
            for operator in effective_operators:
                operator_name, parameters = next(iter(operator.items()))
                if parameters is None:
                    parameters = {}
                    operator[operator_name] = parameters
                parameters.setdefault("text_key", "text")
                if operator_name in {
                    "agent_dialog_normalize_mapper",
                    "general_field_filter",
                }:
                    parameters.setdefault("turbo", True)
        from ..resource_capacity import (
            dataset_loading_parallelism,
            effective_capacity,
            processing_parallelism,
        )

        capacity = effective_capacity()
        processes = processing_parallelism(
            execution_config.get("np", "auto"), runtime_context.get("allocation") or {}, capacity
        )
        loading_processes = dataset_loading_parallelism(processes, capacity)
        for operator in effective_operators:
            name, parameters = next(iter(operator.items()))
            if parameters is None:
                parameters = {}
                operator[name] = parameters
            parameters["num_proc"] = processing_parallelism(
                parameters.get("num_proc", processes), {"cpus": processes}, capacity
            )
        await emit("log", {"stream": "stdout", "message":
            f"Data-Juicer parallelism: requested={execution_config.get('np', 'auto')}, "
            f"allocated_cpus={(runtime_context.get('allocation') or {}).get('cpus')}, "
            f"effective_cpus={capacity['cpu_count']}, operator_np={processes}, "
            f"loading_np={loading_processes}; export=single-file native writer"})
        executable = data_juicer_executable()
        engine_version = data_juicer_version()
        if executable is None or engine_version is None:
            raise RuntimeError(
                'Data-Juicer is not installed on this Worker; install the "data-juicer" extra'
            )

        data_juicer_root = scoped_runtime_path(
            project_data_asset_root(project_workspace), "data-juicer"
        )
        data_juicer_root.mkdir(parents=True, exist_ok=True)
        output_root = scoped_runtime_path(data_juicer_root, processing_run_id)
        marker_path = scoped_runtime_path(data_juicer_root, f"{processing_run_id}.success.json")
        attempt_id = str(runtime_context.get("attempt_dir") or attempt_dir.name).replace("\\", "/").split("/")[-1]
        staging_root = scoped_runtime_path(
            data_juicer_root, f".{processing_run_id}-{attempt_id}.staging"
        )
        export_settings = (
            {"format": "jsonl", "export_shard_size": 0}
            if conversation_messages_key is not None
            else {"format": "parquet", "export_shard_size": 0}
        )
        # The platform analyzes the published body to generate its quality
        # report. Exporting DJ's unused stats sidecar adds a full serial map.
        export_settings["export_stats"] = False
        recipe_identity = hashlib.sha256(
            json.dumps(
                {
                    "input_artifact_hash": data_input.get("artifact_hash"),
                    **({"file_selection": file_selection} if file_selection else {}),
                    "operators": effective_operators,
                    "text_keys": text_keys,
                    "conversation_projection_version": (
                        CONVERSATION_PROJECTION_VERSION
                        if conversation_messages_key is not None
                        else None
                    ),
                    "execution_config": {"np": processes, "loading_np": loading_processes},
                    "export_layout": export_settings,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        reusable = False
        if marker_path.is_file() and output_root.is_dir():
            try:
                reusable = json.loads(marker_path.read_text(encoding="utf-8")).get(
                    "recipe_identity"
                ) == recipe_identity
            except (OSError, ValueError):
                reusable = False

        config_path = attempt_dir / "data-juicer-process.yaml"
        scratch_root = attempt_dir / "data-juicer-scratch"
        retain_cache = (payload.get("storage_policy") or {}).get("cache_cleanup") == "retain"
        export_root = attempt_dir / "data-juicer-export"
        export_path = export_root / f"dataset.{export_settings['format']}"
        projected_input_root = attempt_dir / "data-juicer-conversation-input"
        effective_input_path = (
            projected_input_root if conversation_messages_key is not None else input_path
        )
        config = {
            "project_name": f"llm-platform-{processing_run_id}",
            "dataset_path": str(effective_input_path),
            "np": loading_processes,
            "export_path": str(export_path),
            "work_dir": str(attempt_dir / "data-juicer-work"),
            "export_shard_size": export_settings["export_shard_size"],
            # Data-Juicer's sharded exporter forks multiprocessing.Pool after
            # Arrow has initialized threads; on Linux it can hang in pool.join.
            # The native single-file Parquet writer avoids that fork entirely.
            "export_in_parallel": False,
            "export_extra_args": {"export_stats": export_settings["export_stats"]},
            "ds_cache_dir": str(scratch_root / "datasets"),
            "temp_dir": str(scratch_root / "tmp"),
            "cache_compress": "zstd",
            "process": effective_operators,
        }
        if text_keys is not None:
            config["text_keys"] = text_keys
        if file_selection:
            from ..processing_selection import selected_processing_files

            await emit("progress", {"phase": "verifying_selected_files", "progress": 0.02})
            try:
                selected_files = await asyncio.to_thread(
                    selected_processing_files, input_path, file_selection, get_desired_state,
                )
            except AnalysisInterrupted as exc:
                return {"controlled_state": exc.state}
            config["dataset_path"] = ""
            config["dataset"] = {"configs": [{"type": "local", "path": path} for path in selected_files]}
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        projection_summary: dict[str, Any] | None = None
        if reusable:
            await emit(
                "progress",
                {
                    "progress": 0.72,
                    "phase": "reusing_processed_output",
                    "engine": "Data-Juicer",
                    "engine_version": engine_version,
                },
            )
        else:
            scratch_root.mkdir(parents=True, exist_ok=True)
            (scratch_root / "owner.json").write_text(json.dumps({
                "task_id": task.get("id"), "attempt_id": attempt_id,
                "category": "computation_cache", "retention": "retain" if retain_cache else "auto",
            }), encoding="utf-8")
            if export_root.exists():
                shutil.rmtree(export_root)
            export_root.mkdir(parents=True)
            if conversation_messages_key is not None:
                if projected_input_root.exists():
                    shutil.rmtree(projected_input_root)
                await emit(
                    "progress",
                    {
                        "progress": 0.04,
                        "phase": "projecting_conversation_input",
                        "projection_version": CONVERSATION_PROJECTION_VERSION,
                    },
                )
                try:
                    projection_summary = await asyncio.to_thread(
                        _project_conversation_input,
                        input_path,
                        projected_input_root / "dataset.jsonl",
                        conversation_messages_key,
                        get_desired_state,
                    )
                except AnalysisInterrupted as exc:
                    await asyncio.to_thread(shutil.rmtree, projected_input_root, True)
                    return {"controlled_state": exc.state}
                except Exception:
                    await asyncio.to_thread(shutil.rmtree, projected_input_root, True)
                    raise
                await emit(
                    "progress",
                    {
                        "progress": 0.07,
                        "phase": "conversation_input_projected",
                        **projection_summary,
                    },
                )
            if staging_root.exists():
                shutil.rmtree(staging_root)
            staging_root.mkdir(parents=True, exist_ok=True)
            await emit(
                "progress",
                {
                    "progress": 0.08,
                    "phase": "starting_data_juicer",
                    "engine": "Data-Juicer",
                    "engine_version": engine_version,
                    "operator_count": len(operators),
                    "processes": processes,
                },
            )
            try:
                controlled_state = await self._run_process(
                    executable, config_path, attempt_dir, emit, get_desired_state
                )
            except BaseException:
                await asyncio.to_thread(shutil.rmtree, export_root, True)
                await asyncio.to_thread(shutil.rmtree, staging_root, True)
                raise
            finally:
                if not retain_cache:
                    await asyncio.to_thread(shutil.rmtree, scratch_root, True)
                if conversation_messages_key is not None:
                    await asyncio.to_thread(shutil.rmtree, projected_input_root, True)
            if controlled_state:
                shutil.rmtree(export_root, ignore_errors=True)
                shutil.rmtree(staging_root, ignore_errors=True)
                return {"controlled_state": controlled_state}
            exported_files = await asyncio.to_thread(_validated_export_files, export_path)
            for exported_file in exported_files:
                exported_file.replace(staging_root / exported_file.name)
            if output_root.exists():
                shutil.rmtree(output_root)
            staging_root.replace(output_root)
            marker_path.write_text(
                json.dumps(
                    {
                        "recipe_identity": recipe_identity,
                        "engine_version": engine_version,
                        "output_dataset_version_id": output_version_id,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            await emit(
                "progress",
                {"progress": 0.76, "phase": "data_juicer_completed"},
            )

        record_manifest_path = attempt_dir / "dataset-records.jsonl.gz"
        loop = asyncio.get_running_loop()

        def report_analysis_progress(progress: dict[str, Any]) -> None:
            future = asyncio.run_coroutine_threadsafe(
                emit(
                    "progress",
                    {
                        **progress,
                        "progress": 0.78 + 0.18 * float(progress.get("progress") or 0),
                        "phase": "analyzing_processed_output",
                    },
                ),
                loop,
            )
            future.result()

        try:
            analysis = await asyncio.to_thread(
                DataImportAdapter._analyze,
                output_root,
                "data_juicer",
                output_root.as_uri(),
                output_version_id,
                record_manifest_path,
                {"analysis_workers": processes},
                report_analysis_progress,
                get_desired_state,
            )
        except AnalysisInterrupted as exc:
            record_manifest_path.unlink(missing_ok=True)
            return {"controlled_state": exc.state}
        files = analysis.get("source_snapshot", {}).get("files") or []
        snapshot_hash = DataImportAdapter._manifest_hash(files)
        input_documents = int(data_input.get("document_count") or 0)
        input_tokens = int(data_input.get("token_estimate") or 0)
        output_documents = int(analysis.get("document_count") or 0)
        output_tokens = int(analysis.get("token_estimate") or 0)
        processing_summary = {
            "input_document_count": input_documents,
            "output_document_count": output_documents,
            "document_delta": output_documents - input_documents,
            "filtered_document_count": max(input_documents - output_documents, 0),
            "filter_rate": (
                round(max(input_documents - output_documents, 0) / input_documents, 8)
                if input_documents
                else None
            ),
            "input_token_estimate": input_tokens,
            "output_token_estimate": output_tokens,
            "token_delta": output_tokens - input_tokens,
            "operator_count": len(effective_operators),
            "processes": processes,
            "export_layout": export_settings,
            **(
                {"conversation_projection": projection_summary}
                if not reusable and projection_summary is not None
                else {}
            ),
        }
        analysis["source_snapshot"] = {
            **(analysis.get("source_snapshot") or {}),
            "source_type": "data_juicer",
            "parent_dataset_version_id": payload.get("input_dataset_version_id"),
            "processing_run_id": processing_run_id,
            "processing_recipe_version_id": recipe.get("recipe_version_id"),
            "engine": "Data-Juicer",
            "engine_version": engine_version,
            "recipe_identity": recipe_identity,
            **({"input_selection": file_selection} if file_selection else {}),
            "storage_type": "external",
        }
        await emit(
            "progress",
            {
                "progress": 0.98,
                "phase": "registering_processed_dataset",
                "document_count": output_documents,
                "token_estimate": output_tokens,
            },
        )
        return {
            **analysis,
            "engine": "Data-Juicer",
            "engine_version": engine_version,
            "recipe_identity": recipe_identity,
            "processing_summary": processing_summary,
            "outputs": [
                {
                    "ref": "dataset",
                    "kind": "dataset_snapshot",
                    "storage_type": "external",
                    "uri": output_root.as_uri(),
                    "content_hash": snapshot_hash,
                    "size_bytes": int(analysis.get("raw_size_bytes") or 0),
                    "media_type": "application/vnd.llm-platform.dataset-directory",
                    "availability_state": "available",
                    "worker_id": worker_id,
                    "metadata": {
                        "processing_run_id": processing_run_id,
                        "engine": "Data-Juicer",
                        "engine_version": engine_version,
                    },
                },
                {
                    "ref": "record_manifest",
                    "path": str(record_manifest_path),
                    "kind": "dataset_record_manifest",
                    "media_type": "application/gzip",
                },
                {
                    "ref": "processing_recipe",
                    "path": str(config_path),
                    "kind": "data_processing_recipe",
                    "media_type": "application/yaml",
                },
            ],
        }

    @staticmethod
    async def _run_process(
        executable: str,
        config_path: Path,
        cwd: Path,
        emit: Emit,
        get_desired_state: GetDesiredState | None,
    ) -> str | None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        process = await asyncio.create_subprocess_exec(
            executable,
            "--config",
            str(config_path),
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )

        stdout_task = asyncio.create_task(_forward_process_stream(process.stdout, "stdout", emit))
        stderr_task = asyncio.create_task(_forward_process_stream(process.stderr, "stderr", emit))
        controlled_state: str | None = None
        storage_error = False
        while process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=0.5)
            except TimeoutError:
                desired = get_desired_state() if get_desired_state else "running"
                usage = shutil.disk_usage(cwd)
                storage_error = usage.free < min(8 * 1024**3, usage.total // 100)
                if desired in {"paused", "cancelled"} or storage_error:
                    controlled_state = desired if not storage_error else None
                    _terminate_process_tree(process)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        _terminate_process_tree(process, force=True)
                        await process.wait()
                    break
        await asyncio.gather(stdout_task, stderr_task)
        if storage_error:
            raise RuntimeError("Data-Juicer stopped before exhausting the data disk; reclaim storage and retry")
        if controlled_state:
            return controlled_state
        if process.returncode != 0:
            raise RuntimeError(f"dj-process exited with code {process.returncode}")
        return None
