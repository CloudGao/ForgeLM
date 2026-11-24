from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import zipfile
from pathlib import Path

import httpx
import pytest
from llm_platform.services.model_specifications import resolve_model_specification
from llm_platform_core.checkpoint_compatibility import build_checkpoint_compatibility
from llm_platform_worker.main import AgentConfig, WorkerAgent


def test_model_archive_extracts_the_directory_containing_config(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("tiny-model/config.json", "{}")
        archive.writestr("tiny-model/model.safetensors", b"weights")

    result = WorkerAgent._extract_model_archive(archive_path, tmp_path / "extracted")

    assert result == tmp_path / "extracted" / "tiny-model"
    assert (result / "model.safetensors").read_bytes() == b"weights"


def test_model_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside/config.json", "{}")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        WorkerAgent._extract_model_archive(archive_path, tmp_path / "extracted")

    assert not (tmp_path / "outside").exists()


def _model_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tiny/config.json", "{}")
        archive.writestr("tiny/model.safetensors", b"weights")
    return buffer.getvalue()


def _dataset_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dataset/train.jsonl", '{"text":"hello tokenizer"}\n')
    return buffer.getvalue()


def _tokenizer_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("microlm-tokenizer/tokenizer.json", "{}")
        archive.writestr("microlm-tokenizer/tokenizer_config.json", "{}")
    return buffer.getvalue()


def _checkpoint_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "checkpoint-final/model/config.json",
            '{"hidden_size":960,"use_cache":false}',
        )
        archive.writestr("checkpoint-final/model/model.safetensors", b"weights")
        archive.writestr("checkpoint-final/training-state.pt", b"state")
    return buffer.getvalue()


def test_worker_materializes_managed_single_dataset_input(tmp_path: Path) -> None:
    dataset_payload = _dataset_zip()
    dataset_hash = hashlib.sha256(dataset_payload).hexdigest()
    manifest_payload = (
        b'{"type":"header","format":"llm-dataset-records-v1","dataset_version_id":"version-1"}\n'
    )
    manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
    manifest_downloads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal manifest_downloads
        artifact_id = request.url.path.split("/")[
            -2 if request.url.path.endswith("/payload") else -1
        ]
        if request.url.path.endswith("/payload"):
            if artifact_id == "manifest-1":
                manifest_downloads += 1
            return httpx.Response(
                200,
                content=manifest_payload if artifact_id == "manifest-1" else dataset_payload,
            )
        if artifact_id == "manifest-1":
            return httpx.Response(
                200,
                json={
                    "id": artifact_id,
                    "kind": "dataset_record_manifest",
                    "content_hash": manifest_hash,
                    "locations": [{"storage_type": "managed", "uri": "file:///records.jsonl"}],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": artifact_id,
                "kind": "dataset_snapshot",
                "content_hash": dataset_hash,
                "locations": [{"storage_type": "managed", "uri": "file:///dataset.zip"}],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker.worker_id = "worker-1"
    task = {
        "payload": {
            "command": {"env": {}},
            "data_input": {
                "format": "llm-dataset-input-v1",
                "dataset_version_id": "version-1",
                "dataset_artifact_id": "dataset-1",
                "dataset_artifact_hash": dataset_hash,
                "record_manifest_artifact_id": "manifest-1",
                "record_manifest_hash": manifest_hash,
            },
        }
    }

    emitted: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        emitted.append((event_type, payload))
        return "running"

    asyncio.run(worker.materialize_data_inputs(task, tmp_path / "run", emit))

    data_input = task["payload"]["data_input"]
    assert Path(data_input["runtime_path"]).read_bytes() == manifest_payload
    source = Path(data_input["runtime_sources"][0]["runtime_path"])
    assert (source / "train.jsonl").is_file()
    assert (
        task["payload"]["command"]["env"]["LLM_PLATFORM_DATA_MANIFEST"]
        == data_input["runtime_path"]
    )

    asyncio.run(worker.materialize_data_inputs(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())
    assert manifest_downloads == 1
    messages = [payload["message"] for event_type, payload in emitted if event_type == "log"]
    assert any("正在准备数据输入 1/1" in message for message in messages)
    assert any("下载并校验完成" in message for message in messages)
    assert any("数据输入 1/1 已就绪" in message for message in messages)


def test_worker_materializes_custom_evaluation_inputs(tmp_path: Path) -> None:
    evaluation_payload = b'{"prompt":"2+2?","answer":"4"}\n'
    evaluation_hash = hashlib.sha256(evaluation_payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=evaluation_payload)
        return httpx.Response(
            200,
            json={
                "id": "eval-artifact-1",
                "kind": "dataset_snapshot",
                "content_hash": evaluation_hash,
                "size_bytes": len(evaluation_payload),
                "locations": [
                    {"storage_type": "managed", "uri": "file:///private-eval.jsonl"}
                ],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker.worker_id = "worker-1"
    task = {
        "project_id": "project-1",
        "payload": {
            "command": {"env": {}},
            "evaluation_inputs": [
                {
                    "eval_dataset_version_id": "eval-version-1",
                    "name": "Private regression",
                    "artifact_id": "eval-artifact-1",
                    "artifact_hash": evaluation_hash,
                    "schema_config": {
                        "prompt_field": "prompt",
                        "answer_field": "answer",
                    },
                }
            ],
        },
    }

    asyncio.run(worker.materialize_evaluation_inputs(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    frozen = task["payload"]["evaluation_inputs"][0]
    assert Path(frozen["runtime_path"]).read_bytes() == evaluation_payload
    environment = task["payload"]["command"]["env"]
    exported = json.loads(environment["LLM_PLATFORM_EVAL_DATASETS"])
    assert exported[0]["runtime_path"] == frozen["runtime_path"]
    assert exported[0]["schema_config"]["answer_field"] == "answer"


def test_worker_resolves_control_plane_local_windows_location_inside_wsl(
    tmp_path: Path, monkeypatch
) -> None:
    worker = WorkerAgent(
        AgentConfig(
            "http://test",
            "token",
            "worker",
            labels={"location": "local", "environment": "wsl", "distro": "Ubuntu"},
        )
    )

    class Result:
        stdout = str(tmp_path)

    calls: list[list[str]] = []

    def run(command, **_kwargs):
        calls.append(command)
        return Result()

    monkeypatch.setattr("llm_platform_worker.main.subprocess.run", run)

    path = worker._local_artifact_location(
        {
            "uri": "file:///F:/datasets/corpus",
            "storage_type": "managed",
        }
    )

    asyncio.run(worker.client.aclose())
    assert path == tmp_path.resolve()
    assert calls == [["wslpath", "-u", "F:/datasets/corpus"]]


def test_worker_cleanup_removes_only_reproducible_data_scratch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    for name in (
        "inputs",
        "mixture-shards",
        "corpus-index",
        "validation-corpus-index",
    ):
        directory = run_dir / name
        directory.mkdir(parents=True)
        (directory / "large.tmp").write_bytes(b"scratch")
    (run_dir / "data-mixture-manifest.jsonl.gz").write_text("scratch", encoding="utf-8")
    checkpoint = run_dir / "checkpoint-final" / "model.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"keep")

    WorkerAgent._cleanup_attempt_scratch(run_dir)

    assert checkpoint.read_bytes() == b"keep"
    assert not (run_dir / "inputs").exists()
    assert not (run_dir / "mixture-shards").exists()
    assert not (run_dir / "corpus-index").exists()
    assert not (run_dir / "validation-corpus-index").exists()
    assert not (run_dir / "data-mixture-manifest.jsonl.gz").exists()


def test_worker_materializes_frozen_training_tokenizer(tmp_path: Path) -> None:
    payload = _tokenizer_zip()
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=payload)
        return httpx.Response(
            200,
            json={
                "id": "tokenizer-artifact-1",
                "kind": "tokenizer",
                "content_hash": digest,
                "locations": [
                    {
                        "storage_type": "managed",
                        "uri": "file:///managed/tokenizer.zip",
                    }
                ],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    task = {
        "project_id": "project-1",
        "payload": {
            "command": {"env": {}},
            "tokenizer": {
                "version_id": "tokenizer-version-1",
                "artifact_id": "tokenizer-artifact-1",
                "content_hash": digest,
            },
        },
    }

    asyncio.run(worker.materialize_training_tokenizer(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    tokenizer_root = Path(task["payload"]["tokenizer"]["runtime_path"])
    assert (tokenizer_root / "tokenizer.json").is_file()
    assert (
        task["payload"]["command"]["env"]["LLM_PLATFORM_TOKENIZER_DIR"]
        == str(tokenizer_root)
    )


def test_worker_materializes_frozen_resume_checkpoint(tmp_path: Path) -> None:
    payload = _checkpoint_zip()
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=payload)
        return httpx.Response(
            200,
            json={
                "id": "checkpoint-artifact-1",
                "kind": "checkpoint",
                "content_hash": digest,
                "locations": [
                    {
                        "storage_type": "managed",
                        "uri": "file:///managed/checkpoint.zip",
                    }
                ],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    caller_thread = threading.get_ident()
    extraction_threads: list[int] = []
    extract_archive = worker._extract_data_archive

    def record_extraction_thread(payload_path: Path, destination: Path) -> None:
        extraction_threads.append(threading.get_ident())
        extract_archive(payload_path, destination)

    worker._extract_data_archive = record_extraction_thread  # type: ignore[method-assign]
    task = {
        "project_id": "project-1",
        "payload": {
            "command": {"env": {}},
            "checkpoint_compatibility": build_checkpoint_compatibility(
                {"hidden_size": 960, "use_cache": True},
                contract="test-model",
                runtime_only_fields=["use_cache"],
            ),
            "resume_checkpoint": {
                "checkpoint_id": "checkpoint-1",
                "artifact_id": "checkpoint-artifact-1",
                "content_hash": digest,
                "model_compatibility": build_checkpoint_compatibility(
                    {"hidden_size": 960, "use_cache": True},
                    contract="test-model",
                    runtime_only_fields=["use_cache"],
                ),
            },
        },
    }

    asyncio.run(worker.materialize_training_checkpoint(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    checkpoint_root = Path(task["payload"]["resume_checkpoint"]["runtime_path"])
    assert (checkpoint_root / "model" / "config.json").is_file()
    assert (
        task["payload"]["command"]["env"]["LLM_PLATFORM_RESUME_CHECKPOINT_DIR"]
        == str(checkpoint_root)
    )
    materialized_contract = task["payload"]["resume_checkpoint"]["model_compatibility"]
    assert materialized_contract["configuration"]["use_cache"] is False
    assert materialized_contract["provenance"] == "worker_model_config"
    assert extraction_threads and extraction_threads[0] != caller_thread
    assert not list((tmp_path / "run" / "inputs" / "resume-checkpoint").rglob("*.zip"))


def test_worker_materializes_frozen_training_validation_set(tmp_path: Path) -> None:
    payload = b'{"text":"stable validation sample"}\n'
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=payload)
        return httpx.Response(
            200,
            json={
                "id": "validation-artifact-1",
                "kind": "training_validation",
                "content_hash": digest,
                "locations": [
                    {
                        "storage_type": "managed",
                        "uri": "file:///managed/validation.jsonl",
                    }
                ],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    task = {
        "project_id": "project-1",
        "payload": {
            "command": {
                "args": ["train.py", "--validation-file", "${VALIDATION_FILE}"],
                "env": {},
            },
            "validation_set": {
                "validation_set_id": "validation-set-1",
                "artifact_id": "validation-artifact-1",
                "content_hash": digest,
                "filename": "validation.jsonl",
            },
        },
    }

    asyncio.run(worker.materialize_training_validation(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    runtime_path = Path(task["payload"]["validation_set"]["runtime_path"])
    assert runtime_path.read_bytes() == payload
    assert task["payload"]["command"]["args"][-1] == str(runtime_path)
    assert (
        task["payload"]["command"]["env"]["LLM_PLATFORM_VALIDATION_FILE"]
        == str(runtime_path)
    )


def test_worker_materializes_validation_data_mixture_without_overwriting_training_input(
    tmp_path: Path,
) -> None:
    manifest_payload = b'{"format":"llm-mixture-manifest-v1","sources":[]}\n'
    digest = hashlib.sha256(manifest_payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=manifest_payload)
        return httpx.Response(
            200,
            json={
                "id": "validation-mixture-manifest-1",
                "kind": "data_mixture_manifest",
                "content_hash": digest,
                "locations": [
                    {"storage_type": "managed", "uri": "file:///validation.jsonl"}
                ],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    task = {
        "project_id": "project-1",
        "payload": {
            "command": {
                "env": {
                    "LLM_PLATFORM_DATA_MANIFEST": "/runtime/train.jsonl",
                    "LLM_PLATFORM_DATA_MANIFESTS": '["/runtime/train.jsonl"]',
                }
            },
            "data_inputs": [{"runtime_path": "/runtime/train.jsonl"}],
            "validation_data_input": {
                "format": "llm-mixture-manifest-v1",
                "manifest_artifact_id": "validation-mixture-manifest-1",
                "manifest_hash": digest,
            },
        },
    }

    asyncio.run(worker.materialize_training_validation_data_input(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    validation_input = task["payload"]["validation_data_input"]
    assert Path(validation_input["runtime_path"]).read_bytes() == manifest_payload
    assert task["payload"]["data_inputs"] == [{"runtime_path": "/runtime/train.jsonl"}]
    command_env = task["payload"]["command"]["env"]
    assert command_env["LLM_PLATFORM_DATA_MANIFEST"] == "/runtime/train.jsonl"
    assert command_env["LLM_PLATFORM_VALIDATION_MANIFEST"] == validation_input["runtime_path"]


def test_worker_reuses_validation_holdout_from_frozen_training_view(
    tmp_path: Path,
) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    task = {
        "payload": {
            "command": {"env": {}},
            "validation_data_input": {
                "format": "forgelm-validation-token-holdout-v1",
                "training_data_view_version_id": "view-v1",
                "training_data_view_identity_hash": "a" * 64,
            },
            "training_data_view_inputs": [
                {"version_id": "view-v1", "identity_hash": "a" * 64}
            ],
        }
    }

    asyncio.run(worker.materialize_training_validation_data_input(task, tmp_path / "run"))

    assert "runtime_path" not in task["payload"]["validation_data_input"]
    assert task["payload"]["command"]["env"] == {}


def test_worker_downloads_verifies_and_materializes_a_managed_model(tmp_path: Path) -> None:
    payload = _model_zip()
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=payload)
        return httpx.Response(
            200,
            json={
                "id": "artifact-1",
                "uri": "file:///managed/model.zip",
                "content_hash": digest,
                "locations": [
                    {
                        "storage_type": "managed",
                        "uri": "file:///managed/model.zip",
                        "availability_state": "available",
                    }
                ],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    worker.worker_id = "worker-1"
    task = {"payload": {"model": {"artifact_id": "artifact-1"}}}

    asyncio.run(worker.materialize_inference_model(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    source = Path(task["payload"]["model"]["runtime_source"])
    assert (source / "config.json").is_file()
    assert (source / "model.safetensors").read_bytes() == b"weights"


def test_model_archive_extraction_does_not_block_the_event_loop(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _model_zip()
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=payload)
        return httpx.Response(
            200,
            json={
                "id": "artifact-slow",
                "uri": "file:///managed/model.zip",
                "content_hash": digest,
                "locations": [
                    {
                        "storage_type": "managed",
                        "uri": "file:///managed/model.zip",
                        "availability_state": "available",
                    }
                ],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    original_extract = worker._extract_model_archive

    def slow_extract(payload_path: Path, destination: Path) -> Path | None:
        import time

        time.sleep(0.1)
        return original_extract(payload_path, destination)

    monkeypatch.setattr(worker, "_extract_model_archive", slow_extract)
    task = {"payload": {"model": {"artifact_id": "artifact-slow"}}}

    async def materialize_with_ticker() -> int:
        ticks = 0
        operation = asyncio.create_task(
            worker.materialize_inference_model(task, tmp_path / "run")
        )
        while not operation.done():
            await asyncio.sleep(0.01)
            ticks += 1
        await operation
        return ticks

    ticks = asyncio.run(materialize_with_ticker())
    asyncio.run(worker.client.aclose())

    assert ticks >= 3


def test_worker_rejects_a_managed_model_with_the_wrong_hash(tmp_path: Path) -> None:
    payload = _model_zip()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=payload)
        return httpx.Response(
            200,
            json={
                "id": "artifact-1",
                "uri": "file:///managed/model.zip",
                "content_hash": "0" * 64,
                "locations": [{"storage_type": "managed", "uri": "file:///managed/model.zip"}],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    task = {"payload": {"model": {"artifact_id": "artifact-1"}}}

    with pytest.raises(ValueError, match="SHA-256"):
        asyncio.run(worker.materialize_inference_model(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())


def _code_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("example-code/train.py", "print('train')\n")
        archive.writestr("example-code/configs/train.json", "{}")
    return buffer.getvalue()


def test_worker_materializes_bound_code_and_sets_command_context(tmp_path: Path) -> None:
    payload = _code_zip()
    digest = hashlib.sha256(payload).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payload"):
            return httpx.Response(200, content=payload)
        return httpx.Response(
            200,
            json={
                "id": "code-artifact-1",
                "kind": "source_code",
                "uri": "file:///managed/source.zip",
                "content_hash": digest,
                "locations": [{"storage_type": "managed", "uri": "file:///managed/source.zip"}],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    task = {
        "payload": {
            "code": {
                "version_id": "code-version-1",
                "artifact_id": "code-artifact-1",
                "content_hash": digest,
                "entrypoints": {"training": "train.py"},
            },
            "command": {"executable": "python", "args": ["train.py"], "cwd": None, "env": {}},
        }
    }

    asyncio.run(worker.materialize_code_version(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    code_root = Path(task["payload"]["command"]["cwd"])
    assert (code_root / "train.py").is_file()
    assert task["payload"]["command"]["env"]["LLM_PLATFORM_CODE_DIR"] == str(code_root)
    assert task["payload"]["command"]["env"]["LLM_PLATFORM_CODE_VERSION_ID"] == "code-version-1"


def test_worker_rejects_code_when_frozen_hash_differs_from_artifact(tmp_path: Path) -> None:
    payload = _code_zip()
    digest = hashlib.sha256(payload).hexdigest()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "code-artifact-1",
                "kind": "source_code",
                "uri": "file:///managed/source.zip",
                "content_hash": digest,
                "locations": [],
            },
        )

    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    asyncio.run(worker.client.aclose())
    worker.client = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    )
    task = {
        "payload": {
            "code": {
                "version_id": "code-version-1",
                "artifact_id": "code-artifact-1",
                "content_hash": "0" * 64,
            },
            "command": {"executable": "python", "env": {}},
        }
    }

    with pytest.raises(ValueError, match="content hash"):
        asyncio.run(worker.materialize_code_version(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())


def test_worker_materializes_real_transformers_model_config(tmp_path: Path) -> None:
    pytest.importorskip("transformers", reason="Optional model runtime")
    resolved = resolve_model_specification(
        source_type="standard",
        architecture_family_id="llama_dense",
        preset_id=None,
        config={
            "vocab_size": 256,
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
        initialization={"type": "random"},
    )
    binding = {
        "revision_id": "model-revision-1",
        "source_type": "standard",
        "schema_version": "forgelm.model-spec.v2",
        "runtime_requirements": resolved["runtime_requirements"],
        "generated_json": resolved["generated_json"],
        "config_signature": resolved["config_signature"],
    }
    task = {
        "payload": {
            "model_specification": binding,
            "command": {"executable": "python", "env": {}},
        }
    }
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))

    asyncio.run(worker.materialize_model_specification(task, tmp_path / "run"))
    asyncio.run(worker.client.aclose())

    config_dir = Path(task["payload"]["command"]["env"]["LLM_PLATFORM_MODEL_CONFIG_DIR"])
    config = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "llama"
    assert config["num_hidden_layers"] == 2
    assert Path(
        task["payload"]["command"]["env"]["LLM_PLATFORM_MODEL_SPEC_PATH"]
    ).is_file()
