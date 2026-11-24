from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def test_real_api_worker_and_external_command_engine() -> None:
    root = Path(__file__).resolve().parents[2]
    state_root = Path(tempfile.mkdtemp(prefix="forge-lm-e2e-"))
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "LLM_PLATFORM_DATABASE_URL": f"sqlite:///{(state_root / 'platform.db').as_posix()}",
        }
    )
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "llm_platform.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    worker_process: subprocess.Popen | None = None
    try:
        with httpx.Client(base_url=base_url, timeout=2) as client:
            for _ in range(60):
                try:
                    if client.get("/api/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.1)
            else:
                raise AssertionError("API did not become ready")

            project_workspace = state_root / "project-workspace"
            project_workspace.mkdir()
            project_response = client.post(
                "/api/projects",
                json={
                    "name": "E2E-Project",
                    "target_model_name": "E2E-LM",
                    "project_type": "pretrain",
                    "target_parameters": "42M",
                    "primary_languages": ["zh"],
                    "target_capabilities": ["test"],
                    "workspace_uri": str(project_workspace),
                },
            )
            assert project_response.status_code == 201
            project = project_response.json()
            source_file = state_root / "e2e-corpus.jsonl"
            source_file.write_text(
                '{"text":"第一条端到端真实数据"}\n{"text":"second real e2e document"}\n',
                encoding="utf-8",
            )
            dataset_response = client.post(
                f"/api/projects/{project['id']}/datasets",
                json={
                    "name": "E2E Corpus",
                    "source_type": "local",
                    "source_uri": str(source_file),
                    "source_config": {"copy_mode": "managed"},
                    "license": "internal-test",
                    "language": "zh",
                    "initial_version": {"tier": "bronze"},
                    "start_import": True,
                    "adapter_id": "builtin.data-import",
                    "resource_request": {"gpus": 0},
                },
            )
            assert dataset_response.status_code == 201
            dataset_version = dataset_response.json()["current_version"]
            assert dataset_version["task_id"]

            worker_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "llm_platform_worker.main",
                    "--api-url",
                    base_url,
                    "--name",
                    "e2e-worker",
                    "--poll-seconds",
                    "0.05",
                ],
                cwd=root,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
            data_terminal = None
            for _ in range(600):
                task = client.get(f"/api/tasks/{dataset_version['task_id']}").json()
                if task["observed_state"] in {"succeeded", "failed", "cancelled", "lost"}:
                    data_terminal = task
                    break
                time.sleep(0.1)
            assert data_terminal is not None
            assert data_terminal["observed_state"] == "succeeded"
            assert data_terminal["result"]["output_artifact_ids"]
            refreshed_dataset = client.get(f"/api/datasets/{dataset_response.json()['id']}").json()
            dataset_version = refreshed_dataset["current_version"]
            assert dataset_version["document_count"] == 2
            assert dataset_version["sample_preview"]
            assert dataset_version["quality_report"]["analyzed_document_count"] == 2
            assert dataset_version["train_ready"] is True
            tokenizer_response = client.post(
                f"/api/projects/{project['id']}/tokenizers",
                json={
                    "name": "E2E External Tokenizer",
                    "algorithm": "BPE",
                    "vocab_size": 4096,
                        "data_ref_type": "dataset_version",
                        "data_ref_id": dataset_version["id"],
                        "config": {
                            "gate_policy": {
                                "version": "tokenizer-quality-gate-v1",
                                "rules": {
                                    "round_trip": {"enabled": False},
                                    "unk_rate": {"enabled": False},
                                    "zh_chars_per_token": {"enabled": False},
                                    "en_chars_per_token": {"enabled": False},
                                    "code_chars_per_token": {"enabled": False},
                                },
                            }
                        },
                        "adapter_id": "builtin.command.tokenizer",
                    "resource_request": {"gpus": 0},
                    "command": {
                        "executable": sys.executable,
                        "args": [str(root / "examples" / "command_engine.py")],
                    },
                },
            )
            assert tokenizer_response.status_code == 201
            tokenizer = tokenizer_response.json()
            command_terminal = None
            for _ in range(180):
                task = client.get(f"/api/tasks/{tokenizer['task_id']}").json()
                if task["observed_state"] in {"succeeded", "failed", "cancelled", "lost"}:
                    command_terminal = task
                    break
                time.sleep(0.1)
            assert command_terminal is not None
            assert command_terminal["observed_state"] == "succeeded", command_terminal
            assert command_terminal["result"]["output_artifact_ids"]
            refreshed = client.get(f"/api/projects/{project['id']}/tokenizers").json()
            external = next(item for item in refreshed if item["id"] == tokenizer["id"])
            assert external["status"] == "passed"
            assert external["artifact_id"] == command_terminal["result"]["artifact_id"]
    finally:
        if worker_process is not None:
            _stop(worker_process)
        _stop(api_process)
        shutil.rmtree(state_root, ignore_errors=True)
