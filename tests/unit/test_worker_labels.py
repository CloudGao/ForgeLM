from __future__ import annotations

import asyncio

from llm_platform_worker.main import AgentConfig, WorkerAgent


def test_worker_reports_wsl_and_configured_routing_labels(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WSL_DISTRO_NAME", "Test-Ubuntu")
    agent = WorkerAgent(
        AgentConfig(
            api_url="http://127.0.0.1:1",
            token="test",
            name="wsl-test",
            labels={"project": "example-model"},
            extra_capabilities=["project.example-inference"],
            workspace_root=str(tmp_path / "worker-root"),
        )
    )
    try:
        assert agent.labels()["environment"] == "wsl"
        assert agent.labels()["project"] == "example-model"
        assert agent.labels()["workspace_mode"] == "worker-local"
        assert "project.example-inference" in agent.capabilities()
        assert "builtin.command.training" in agent.capabilities()
        assert "builtin.training-resource-planner.v1" in agent.capabilities()
        assert agent.resources()["workspace"]["root"] == str(
            (tmp_path / "worker-root").resolve()
        )
    finally:
        asyncio.run(agent.close())


def test_data_worker_limits_resources_and_capabilities(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WSL_DISTRO_NAME", "MicroLM-Ubuntu")
    monkeypatch.setattr(
        WorkerAgent,
        "_discover_gpus",
        staticmethod(lambda: [{"index": 0, "name": "must-not-be-advertised"}]),
    )
    monkeypatch.setattr(
        "llm_platform_worker.resource_capacity.effective_capacity",
        lambda: {"cpu_count": 16, "host_cpu_count": 16,
                 "memory_total_bytes": 8 * 1024**3, "memory_available_bytes": 4 * 1024**3},
    )
    agent = WorkerAgent(
        AgentConfig(
            api_url="http://127.0.0.1:1",
            token="test",
            name="data-wsl-test",
            extra_capabilities=["builtin.command.training"],
            workspace_root=str(tmp_path / "data-worker"),
            profile="data",
            cpu_limit=4,
        )
    )
    try:
        resources = agent.resources()
        capabilities = agent.capabilities()
        assert resources["cpu_count"] == 4
        assert resources["gpus"] == []
        assert agent.labels()["worker_role"] == "data"
        assert "builtin.data-import" in capabilities
        assert "builtin.command.tokenizer" in capabilities
        assert "builtin.command.training" not in capabilities
        assert "builtin.command.inference" not in capabilities
    finally:
        asyncio.run(agent.close())


def test_general_worker_advertises_detected_microlm_inference(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm_platform_worker.main.microlm_inference_available", lambda: True
    )
    agent = WorkerAgent(
        AgentConfig(
            api_url="http://127.0.0.1:1",
            token="test",
            name="microlm-inference-test",
        )
    )
    try:
        capabilities = agent.capabilities()
        assert "builtin.microlm-inference" in capabilities
        assert "builtin.model-runtime.microlm" in capabilities
    finally:
        asyncio.run(agent.close())
