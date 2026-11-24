from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from llm_platform_core.mixtures import open_text_manifest, stable_record_id
from llm_platform_worker.adapters import data_import as data_import_module
from llm_platform_worker.adapters.data_import import (
    AnalysisInterrupted,
    DataImportAdapter,
    SnapshotDriftError,
)


def test_planned_download_rechecks_root_after_alias_resolution(tmp_path, monkeypatch):
    destination = tmp_path / "download"
    destination.mkdir()
    (destination / "train.jsonl").write_bytes(b"abc")
    alias = tmp_path / "DOWNLO~1"
    resolve = Path.resolve
    first = True

    def resolve_alias(path, *args, **kwargs):
        nonlocal first
        if path == destination and first:
            first = False
            return alias
        if path == alias or alias in path.parents:
            path = destination / path.relative_to(alias)
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_alias)
    assert DataImportAdapter._planned_hf_completed(
        destination, [{"path": "train.jsonl", "size_bytes": 3}]
    ) == (3, 1)


def test_planned_download_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError, match="Invalid Hugging Face plan path"):
        DataImportAdapter._planned_hf_completed(
            tmp_path, [{"path": "../outside", "size_bytes": 3}]
        )


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


class InterruptingRangeHandler(BaseHTTPRequestHandler):
    payload = ("可恢复下载数据\n" * 32_768).encode("utf-8")
    interrupted = False
    ranges: list[str | None] = []

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        range_header = self.headers.get("Range")
        type(self).ranges.append(range_header)
        if not type(self).interrupted:
            type(self).interrupted = True
            cutoff = len(self.payload) // 2
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload[:cutoff])
            self.wfile.flush()
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return

        start = int((range_header or "bytes=0-").removeprefix("bytes=").partition("-")[0])
        remaining = self.payload[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(remaining)))
        if start:
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        self.end_headers()
        self.wfile.write(remaining)


class HuggingFaceMirrorHandler(BaseHTTPRequestHandler):
    payload = b'{"text":"mirror dataset"}\n'
    paths: list[str] = []

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def do_GET(self) -> None:
        path = self.path.partition("?")[0]
        type(self).paths.append(path)
        if path == "/api/datasets/org/corpus/tree/main":
            body = json.dumps(
                [{"type": "file", "path": "train.jsonl", "size": len(self.payload)}]
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/datasets/org/corpus/resolve/main/train.jsonl":
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
            return
        self.send_error(404)


async def run_adapter(
    tmp_path: Path, source_type: str, source_uri: str
) -> tuple[dict, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    run_dir = tmp_path / "run"
    result = await DataImportAdapter().run(
        {
            "payload": {
                "source_type": source_type,
                "source_uri": source_uri,
                "source_config": {},
            },
            "runtime_context": {"run_dir": str(run_dir)},
        },
        emit,
    )
    return result, events


@pytest.mark.asyncio
async def test_hf_scan_stop_terminates_metadata_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished = asyncio.Event()

    class FakeProcess:
        returncode: int | None = None
        terminated = False
        killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await finished.wait()
            return b"", b""

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15
            finished.set()

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9
            finished.set()

    process = FakeProcess()

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        return process

    checks = 0

    def desired_state() -> str:
        nonlocal checks
        checks += 1
        return "cancelled" if checks >= 3 else "running"

    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return desired_state()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(data_import_module, "HF_METADATA_POLL_SECONDS", 0.01)
    result = await DataImportAdapter().run(
        {
            "payload": {
                "operation": "hf_scan",
                "source_type": "huggingface",
                "source_uri": "org/corpus",
                "source_config": {"revision": "main"},
            },
            "runtime_context": {"run_dir": str(tmp_path / "run")},
        },
        emit,
        desired_state,
    )

    assert result == {"controlled_state": "cancelled", "operation": "hf_scan"}
    assert process.terminated is True
    assert process.killed is False
    assert any(payload.get("phase") == "cancelled" for _, payload in events)


def test_parquet_analysis_streams_text_and_writes_gzip_record_manifest(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.table({"text": ["第一条语料", None, "second document"], "score": [1, 2, 3]}),
        source / "train.parquet",
    )
    manifest_path = tmp_path / "records.jsonl.gz"

    result = DataImportAdapter._analyze(
        source,
        "local",
        str(source),
        "dataset-version-1",
        manifest_path,
    )

    assert manifest_path.read_bytes().startswith(b"\x1f\x8b")
    with open_text_manifest(manifest_path) as stream:
        lines = [json.loads(line) for line in stream]
    assert [line["type"] for line in lines] == ["header", "record", "record"]
    assert [line["record_index"] for line in lines[1:]] == [0, 1]
    assert lines[1]["record_id"] == stable_record_id(
        "dataset-version-1",
        "train.parquet",
        0,
        lines[1]["content_hash"],
    )
    assert result["document_count"] == 2
    assert result["train_ready"] is True
    assert result["quality_report"]["parsed_files"] == [
        {
            "path": "train.parquet",
            "format": "parquet",
            "document_count": 2,
            "text_fields": ["text"],
            "engine": "duckdb_native_v1",
        }
    ]


def test_parallel_parquet_manifest_is_deterministic_and_reuses_known_hash(
    tmp_path: Path,
) -> None:
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    source.mkdir()
    path = source / "train.parquet"
    pq.write_table(pa.table({"text": ["a", "b", "c"]}), path)
    other_path = source / "other.parquet"
    pq.write_table(pa.table({"text": ["d", "e"]}), other_path)
    known_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    other_hash = hashlib.sha256(other_path.read_bytes()).hexdigest()
    config = {
        "known_files": [
            {
                "path": "train.parquet",
                "size_bytes": path.stat().st_size,
                "sha256": known_hash,
            },
            {
                "path": "other.parquet",
                "size_bytes": other_path.stat().st_size,
                "sha256": other_hash,
            },
        ]
    }

    first = DataImportAdapter._analyze(
        source,
        "local",
        str(source),
        "dataset-version-1",
        tmp_path / "first.jsonl.gz",
        config,
    )
    second = DataImportAdapter._analyze(
        source,
        "local",
        str(source),
        "dataset-version-1",
        tmp_path / "second.jsonl.gz",
        config,
    )

    assert (tmp_path / "first.jsonl.gz").read_bytes() == (tmp_path / "second.jsonl.gz").read_bytes()
    assert all(item["hash_reused"] for item in first["source_snapshot"]["files"])
    assert {item["sha256"] for item in second["source_snapshot"]["files"]} == {
        known_hash,
        other_hash,
    }
    with open_text_manifest(tmp_path / "first.jsonl.gz") as stream:
        records = [json.loads(line) for line in stream if json.loads(line)["type"] == "record"]
    assert [record["logical_path"] for record in records] == [
        "other.parquet",
        "other.parquet",
        "train.parquet",
        "train.parquet",
        "train.parquet",
    ]


def test_parquet_analysis_requires_mapping_for_ambiguous_string_columns(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.table({"left": ["a"], "right": ["b"]}),
        source / "ambiguous.parquet",
    )

    result = DataImportAdapter._analyze(
        source,
        "local",
        str(source),
        "dataset-version-1",
        tmp_path / "records.jsonl.gz",
    )

    assert result["document_count"] == 0
    assert result["train_ready"] is False
    assert "configure text_field or text_fields" in result["quality_report"]["unsupported_files"][0]


def test_parallel_parquet_analysis_cancels_without_publishing_manifest(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(pa.table({"text": ["cancel me"] * 10_000}), source / "train.parquet")
    manifest_path = tmp_path / "records.jsonl.gz"

    with pytest.raises(AnalysisInterrupted, match="cancelled"):
        DataImportAdapter._analyze(
            source,
            "local",
            str(source),
            "dataset-version-1",
            manifest_path,
            get_desired_state=lambda: "cancelled",
        )

    assert not manifest_path.exists()
    assert not (tmp_path / "analysis-shards").exists()


@pytest.mark.asyncio
async def test_local_import_copies_and_analyzes_real_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    records = [
        {"text": "第一条真实中文语料。"},
        {"text": "第一条真实中文语料。"},
        {"text": "联系 test@example.com 获取资料。"},
    ]
    (source / "train.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    result, events = await run_adapter(tmp_path, "local", str(source))

    assert result["document_count"] == 3
    assert result["raw_size_bytes"] == (source / "train.jsonl").stat().st_size
    assert result["token_estimate"] > 0
    assert result["train_ready"] is True
    assert result["quality_report"]["duplicate_rate"] == pytest.approx(1 / 3)
    assert result["quality_report"]["pii_rate"] == pytest.approx(1 / 3)
    assert len(result["sample_preview"]) == 3
    assert result["source_snapshot"]["files"][0]["sha256"]
    assert (
        Path(result["outputs"][0]["path"], "train.jsonl").read_bytes()
        == (source / "train.jsonl").read_bytes()
    )
    assert any(payload.get("phase") == "analyzing" for _, payload in events)


@pytest.mark.asyncio
async def test_non_text_import_preserves_content_without_claiming_text_capabilities(
    tmp_path: Path,
) -> None:
    source = tmp_path / "images"
    source.mkdir()
    (source / "sample.png").write_bytes(b"not-a-real-image-but-real-source-bytes")
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result = await DataImportAdapter().run(
        {
            "payload": {
                "source_type": "local",
                "source_uri": str(source),
                "source_config": {},
                "modality": "image",
                "parser": None,
                "dataset_version_id": "image-version-1",
            },
            "runtime_context": {"run_dir": str(tmp_path / "image-run")},
        },
        emit,
    )

    assert result["modality"] == "image"
    assert result["parser_status"] == "unavailable"
    assert result["record_count"] == 1
    assert result["document_count"] == 0
    assert result["train_ready"] is False
    assert result["usage_capabilities"] == ["source_archive"]
    assert result["detected_formats"] == ["png"]
    assert [output["ref"] for output in result["outputs"]] == ["dataset"]
    assert result["source_snapshot"]["files"][0]["sha256"]
@pytest.mark.asyncio
async def test_http_import_downloads_the_address_instead_of_registering_it(tmp_path: Path) -> None:
    served = tmp_path / "served"
    served.mkdir()
    payload = "first real document\nsecond real document\n"
    (served / "corpus.txt").write_text(payload, encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(served)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result, _ = await run_adapter(
            tmp_path, "http", f"http://127.0.0.1:{server.server_port}/corpus.txt"
        )
    finally:
        await asyncio.to_thread(server.shutdown)
        thread.join(timeout=2)
        server.server_close()

    assert result["document_count"] == 2
    downloaded = Path(result["outputs"][0]["path"]) / "corpus.txt"
    assert downloaded.read_text(encoding="utf-8") == payload
    assert result["source_snapshot"]["raw_size_bytes"] == (served / "corpus.txt").stat().st_size


@pytest.mark.asyncio
@pytest.mark.skip(reason="replaced by script-driven Hugging Face CLI adapter tests")
async def test_huggingface_import_uses_configured_mirror_endpoint(tmp_path: Path) -> None:
    HuggingFaceMirrorHandler.paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), HuggingFaceMirrorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    try:
        result = await DataImportAdapter().run(
            {
                "payload": {
                    "source_type": "huggingface",
                    "source_uri": "org/corpus",
                    "source_config": {"hf_endpoint": f"http://127.0.0.1:{server.server_port}/"},
                },
                "runtime_context": {"run_dir": str(tmp_path / "run")},
            },
            emit,
        )
    finally:
        await asyncio.to_thread(server.shutdown)
        thread.join(timeout=2)
        server.server_close()

    assert result["document_count"] == 1
    assert HuggingFaceMirrorHandler.paths == [
        "/api/datasets/org/corpus/tree/main",
        "/datasets/org/corpus/resolve/main/train.jsonl",
    ]
    assert (Path(result["outputs"][0]["path"]) / "train.jsonl").read_bytes() == (
        HuggingFaceMirrorHandler.payload
    )


@pytest.mark.asyncio
async def test_planned_huggingface_import_rechecks_and_registers_external_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_workspace = tmp_path / "project"
    project_workspace.mkdir()
    calls: list[tuple[str, dict, Path]] = []

    async def fake_metadata_operation(
        operation: str,
        repo_id: str,
        source_config: dict,
        run_dir: Path,
        get_desired_state,
        *,
        destination: Path | None = None,
    ) -> dict:
        del run_dir, get_desired_state
        assert operation == "plan"
        assert destination is not None
        calls.append((repo_id, source_config, destination))
        return {
            "repo_id": repo_id,
            "requested_revision": "main",
            "commit_hash": "a" * 40,
            "endpoint": "https://huggingface.co",
            "selected_paths": ["data/train.jsonl"],
            "file_count": 1,
            "selected_bytes": 25,
            "cached_bytes": 0,
            "download_bytes": 25,
            "disk_free_bytes": 1_000_000,
            "disk_space_sufficient": True,
            "target_dir": str(destination),
            "files": [
                {
                    "path": "data/train.jsonl",
                    "size_bytes": 25,
                    "is_cached": False,
                    "will_download": True,
                }
            ],
            "config": None,
            "split": "train",
        }

    async def fake_run_process(self, command: list[str], **kwargs: object) -> None:
        del self
        assert kwargs["parse_progress"] is False
        plan = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        output = Path(plan["target_dir"]) / "data" / "train.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'{"text":"planned asset"}\n')

    monkeypatch.setattr(
        DataImportAdapter,
        "_run_hf_metadata_operation",
        staticmethod(fake_metadata_operation),
    )
    monkeypatch.setattr(
        data_import_module.ScriptTransferRunner,
        "_run_process",
        fake_run_process,
    )
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    source_config = {
        "hf_plan_task_id": "plan-task-id",
        "asset_key": "asset-key-1234",
        "requested_revision": "main",
        "revision": "a" * 40,
        "commit_hash": "a" * 40,
        "selected_paths": ["data/train.jsonl"],
        "allow_patterns": ["data/train.jsonl"],
        "split": "train",
        "planned_file_count": 1,
        "planned_selected_bytes": 25,
        "project_workspace_uri": str(project_workspace),
    }
    result = await DataImportAdapter().run(
        {
            "payload": {
                "source_type": "huggingface",
                "source_uri": "org/corpus",
                "source_config": source_config,
            },
            "runtime_context": {"run_dir": str(tmp_path / "run")},
        },
        emit,
    )

    destination = (project_workspace / ".llm-platform" / "data-assets" / "asset-key-1234").resolve()
    assert calls == [("org/corpus", source_config, destination)]
    assert result["outputs"][0]["storage_type"] == "external"
    assert result["outputs"][0]["uri"] == destination.as_uri()
    assert result["outputs"][0]["content_hash"]
    assert result["source_snapshot"]["commit_hash"] == "a" * 40
    assert result["source_snapshot"]["selected_paths"] == ["data/train.jsonl"]
    assert result["source_snapshot"]["transfer"]["tool"] == ("huggingface_hub.snapshot_download")
    assert any(payload.get("phase") == "downloading" for _, payload in events)
    assert any(
        payload.get("progress_basis") == "completed_file_bytes"
        and payload.get("downloaded_bytes") == 25
        and payload.get("progress") == pytest.approx(0.78)
        for _, payload in events
    )


@pytest.mark.asyncio
@pytest.mark.skip(reason="protocol retry/resume is now curl's responsibility")
async def test_http_retry_resumes_partial_file_across_task_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    InterruptingRangeHandler.interrupted = False
    InterruptingRangeHandler.ranges = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), InterruptingRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    source_uri = f"http://127.0.0.1:{server.server_port}/corpus.txt"
    task = {"payload": {"source_type": "http", "source_uri": source_uri, "source_config": {}}}
    monkeypatch.setattr(data_import_module, "DOWNLOAD_RETRY_ATTEMPTS", 1)
    try:
        with pytest.raises(RuntimeError, match="已尝试 1 次"):
            await DataImportAdapter().run(
                {**task, "runtime_context": {"run_dir": str(tmp_path / "task" / "attempt-1")}},
                emit,
            )
        partial = tmp_path / "task" / "dataset" / "corpus.txt.part"
        assert partial.exists()
        partial_size = partial.stat().st_size
        assert 0 < partial_size < len(InterruptingRangeHandler.payload)

        monkeypatch.setattr(data_import_module, "DOWNLOAD_RETRY_ATTEMPTS", 2)
        monkeypatch.setattr(data_import_module, "DOWNLOAD_RETRY_BASE_SECONDS", 0)
        result = await DataImportAdapter().run(
            {**task, "runtime_context": {"run_dir": str(tmp_path / "task" / "attempt-2")}},
            emit,
        )
    finally:
        await asyncio.to_thread(server.shutdown)
        thread.join(timeout=2)
        server.server_close()

    output = Path(result["outputs"][0]["path"])
    assert output == (tmp_path / "task" / "dataset").resolve()
    assert (output / "corpus.txt").read_bytes() == InterruptingRangeHandler.payload
    assert not (output / "corpus.txt.part").exists()
    assert InterruptingRangeHandler.ranges == [None, f"bytes={partial_size}-"]
    assert any(payload.get("resumed_from_bytes") == partial_size for _, payload in events)


@pytest.mark.asyncio
async def test_analysis_only_reads_file_uri_in_place_and_reuses_artifact(tmp_path: Path) -> None:
    source = tmp_path / "existing"
    source.mkdir()
    (source / "train.jsonl").write_text('{"text":"已有真实样本"}\n', encoding="utf-8")
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result = await DataImportAdapter().run(
        {
            "payload": {
                "source_type": "local",
                "source_uri": source.as_uri(),
                "source_config": {},
                "analysis_only": True,
                "reuse_artifact_id": "artifact-existing",
            },
            "runtime_context": {"run_dir": str(tmp_path / "run")},
        },
        emit,
    )

    assert result["document_count"] == 1
    assert result["sample_preview"][0]["raw_text"] == "已有真实样本"
    assert result["output_artifact_ids"] == ["artifact-existing"]
    assert result["outputs"][0]["kind"] == "dataset_record_manifest"
    assert Path(result["outputs"][0]["path"]).is_file()
    assert result["source_snapshot"]["analysis_only"] is True
    assert any(payload.get("phase") == "reading_existing" for _, payload in events)


@pytest.mark.asyncio
async def test_analysis_only_rejects_snapshot_drift_even_when_file_size_is_unchanged(
    tmp_path: Path,
) -> None:
    source = tmp_path / "existing-drift"
    source.mkdir()
    path = source / "train.txt"
    old_payload = b"old\n"
    path.write_bytes(b"new\n")

    async def emit(event_type: str, payload: dict) -> str:
        del event_type, payload
        return "running"

    with pytest.raises(SnapshotDriftError, match="snapshot_drift"):
        await DataImportAdapter().run(
            {
                "payload": {
                    "source_type": "local",
                    "source_uri": source.as_uri(),
                    "source_config": {},
                    "analysis_only": True,
                    "reuse_artifact_id": "artifact-existing",
                    "known_files": [],
                    "expected_snapshot_files": [
                        {
                            "path": "train.txt",
                            "size_bytes": len(old_payload),
                            "sha256": hashlib.sha256(old_payload).hexdigest(),
                        }
                    ],
                },
                "runtime_context": {"run_dir": str(tmp_path / "run-drift")},
            },
            emit,
        )


@pytest.mark.asyncio
async def test_analysis_only_reuses_external_huggingface_asset(tmp_path: Path) -> None:
    source = tmp_path / "existing-hf-asset"
    source.mkdir()
    (source / "train.jsonl").write_text('{"text":"cached"}\n', encoding="utf-8")

    async def emit(event_type: str, payload: dict) -> str:
        del event_type, payload
        return "running"

    result = await DataImportAdapter().run(
        {
            "payload": {
                "source_type": "huggingface",
                "source_uri": source.as_uri(),
                "source_config": {"commit_hash": "a" * 40},
                "analysis_only": True,
                "reuse_artifact_id": "artifact-hf",
            },
            "runtime_context": {"run_dir": str(tmp_path / "run-hf")},
        },
        emit,
    )

    assert result["document_count"] == 1
    assert result["output_artifact_ids"] == ["artifact-hf"]
    assert result["outputs"][0]["kind"] == "dataset_record_manifest"
    assert Path(result["outputs"][0]["path"]).is_file()


def test_unified_local_import_registers_one_public_directory(tmp_path, monkeypatch):
    from uuid import uuid4
    root = tmp_path / "shared"
    monkeypatch.setenv("LLM_PLATFORM_STORAGE_ROOT", str(root))
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"a shared dataset"}\n', encoding="utf-8")
    version = str(uuid4())
    async def emit(*_):
        return "running"
    result = asyncio.run(DataImportAdapter().run({
        "id": str(uuid4()), "payload": {"source_type": "local", "source_uri": str(source), "dataset_version_id": version},
        "runtime_context": {"run_dir": str(tmp_path / "run")}}, emit))
    output = next(item for item in result["outputs"] if item["kind"] == "dataset_snapshot")
    assert output["storage_type"] == "external"
    destination = root / "dataset" / "objects" / "external" / "imports" / version
    assert output["uri"] == destination.as_uri()
    assert (destination / source.name).read_bytes() == source.read_bytes()
    assert not (tmp_path / "dataset").exists()
    assert result["source_snapshot"]["storage_type"] == "external"
