from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import llm_platform_worker.main as worker_main
import pytest
from llm_platform_worker.main import AgentConfig, WorkerAgent


class RecordingClient:
    def __init__(self) -> None:
        self.active_events = 0
        self.max_active_events = 0
        self.sequences: list[int] = []
        self.posts: list[tuple[str, dict[str, Any]]] = []

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        payload = json or dict(kwargs.get("data") or {})
        self.posts.append((url, payload))
        request = httpx.Request("POST", f"http://test{url}")
        if url.endswith("/start"):
            return httpx.Response(200, request=request, json={"desired_state": "running"})
        if url.endswith("/events"):
            self.active_events += 1
            self.max_active_events = max(self.max_active_events, self.active_events)
            await asyncio.sleep(0.01)
            self.sequences.append(int(payload["worker_sequence"]))
            self.active_events -= 1
            return httpx.Response(200, request=request, json={"desired_state": "running"})
        if url.endswith("/complete"):
            return httpx.Response(200, request=request, json={})
        if url.endswith("/artifacts"):
            return httpx.Response(200, request=request, json={"id": "runtime-metrics-id"})
        raise AssertionError(f"Unexpected request: {url}")


class FlakyDeliveryClient(RecordingClient):
    def __init__(
        self,
        *,
        event_failures: int = 0,
        fail_all_events: bool = False,
        completion_failures: int = 0,
    ) -> None:
        super().__init__()
        self.event_failures = event_failures
        self.fail_all_events = fail_all_events
        self.completion_failures = completion_failures
        self.event_attempts = 0
        self.completion_attempts = 0

    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        request = httpx.Request("POST", f"http://test{url}")
        if url.endswith("/events"):
            self.event_attempts += 1
            if self.fail_all_events or self.event_failures > 0:
                self.event_failures -= 1
                raise httpx.RemoteProtocolError(
                    "server disconnected before response",
                    request=request,
                )
        if url.endswith("/complete"):
            self.completion_attempts += 1
            if self.completion_failures > 0:
                self.completion_failures -= 1
                raise httpx.RemoteProtocolError(
                    "completion response was lost",
                    request=request,
                )
        return await super().post(url, json=json, **kwargs)


class StaleCompletionClient(RecordingClient):
    async def post(
        self,
        url: str,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if url.endswith("/complete"):
            request = httpx.Request("POST", f"http://test{url}")
            return httpx.Response(
                409,
                request=request,
                json={"detail": {"code": "attempt_already_completed"}},
            )
        return await super().post(url, json=json, **kwargs)


@pytest.mark.asyncio
async def test_worker_serializes_concurrent_task_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    client = RecordingClient()
    worker.client = client  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    monkeypatch.setattr(worker, "resources", lambda: {})

    async def run(_task: dict, emit, _desired_state) -> dict:
        attempt_dir = Path(_task["runtime_context"]["attempt_dir"])
        for filename in ("command-output.jsonl.gz", "generation-runtime.json", "generation-metrics.jsonl", "generation-validation.jsonl"):
            (attempt_dir / filename).write_bytes(b"test-observability-payload")
        await asyncio.gather(
            emit("progress", {"progress": 0.5}),
            emit("log", {"message": "token=supersecret"}),
        )
        return {}

    monkeypatch.setattr(worker.data_import, "run", run)
    monkeypatch.setattr(worker, "upload_outputs", lambda *_args: asyncio.sleep(0, result=[]))

    await worker.execute(
        {
            "task": {
                "id": "task-id",
                "project_id": "project-id",
                "project_workspace_uri": str(tmp_path),
                "adapter_id": "builtin.data-import",
                "payload": {},
            },
            "attempt": {"id": "attempt-id"},
            "lease": {"allocation": {}},
        }
    )

    assert client.max_active_events == 1
    assert client.sequences == list(range(1, len(client.sequences) + 1))
    event_types = [
        payload["event_type"]
        for url, payload in client.posts
        if url.endswith("/events")
    ]
    assert "progress" in event_types
    assert "log" in event_types
    assert event_types.count("resource.sample") == 2
    assert "resource.summary" in event_types
    complete = next(payload for url, payload in client.posts if url.endswith("/complete"))
    assert complete["runtime_metrics_artifact_id"] == "runtime-metrics-id"
    assert complete["resource_summary"]["sample_count"] == 2
    for kind in ("task_log", "generation_runtime", "generation_metrics", "generation_validation"):
        assert complete["resource_summary"][f"{kind}_artifact_id"] == "runtime-metrics-id"
    log_payload = next(
        payload["payload"]
        for url, payload in client.posts
        if url.endswith("/events") and payload["event_type"] == "log"
        and payload["payload"].get("message") == "token=[REDACTED]"
    )
    assert log_payload["message"] == "token=[REDACTED]"
    assert complete["log_summary"]["platform_lines"] == 4


@pytest.mark.asyncio
async def test_worker_retries_transient_event_delivery_without_failing_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    client = FlakyDeliveryClient(event_failures=1)
    worker.client = client  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    monkeypatch.setattr(worker, "resources", lambda: {})
    monkeypatch.setattr(worker_main, "_EVENT_DELIVERY_RETRY_DELAYS", (0.0, 0.0, 0.0))

    async def run(_task: dict, emit, _desired_state) -> dict:
        assert await emit("progress", {"progress": 0.5}) == "running"
        return {}

    monkeypatch.setattr(worker.data_import, "run", run)
    monkeypatch.setattr(worker, "upload_outputs", lambda *_args: asyncio.sleep(0, result=[]))

    await worker.execute(
        {
            "task": {
                "id": "task-flaky-event",
                "project_id": "project-id",
                "project_workspace_uri": str(tmp_path),
                "adapter_id": "builtin.data-import",
                "payload": {},
            },
            "attempt": {"id": "attempt-id"},
            "lease": {"allocation": {}},
        }
    )

    complete = next(payload for url, payload in client.posts if url.endswith("/complete"))
    assert complete["status"] == "succeeded"
    assert client.event_attempts > len(client.sequences)
    assert complete["log_summary"]["event_delivery_failures"] == 0


@pytest.mark.asyncio
async def test_worker_drops_unavailable_observability_events_but_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    client = FlakyDeliveryClient(fail_all_events=True, completion_failures=1)
    worker.client = client  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    monkeypatch.setattr(worker, "resources", lambda: {})
    monkeypatch.setattr(worker_main, "_EVENT_DELIVERY_RETRY_DELAYS", (0.0,))

    async def run(_task: dict, emit, _desired_state) -> dict:
        assert await emit("progress", {"progress": 0.5}) == "running"
        return {}

    monkeypatch.setattr(worker.data_import, "run", run)
    monkeypatch.setattr(worker, "upload_outputs", lambda *_args: asyncio.sleep(0, result=[]))

    await worker.execute(
        {
            "task": {
                "id": "task-dropped-event",
                "project_id": "project-id",
                "project_workspace_uri": str(tmp_path),
                "adapter_id": "builtin.data-import",
                "payload": {},
            },
            "attempt": {"id": "attempt-id"},
            "lease": {"allocation": {}},
        }
    )

    complete = next(payload for url, payload in client.posts if url.endswith("/complete"))
    assert complete["status"] == "succeeded"
    assert complete["log_summary"]["event_delivery_failures"] >= 1
    assert client.completion_attempts == 2


@pytest.mark.asyncio
async def test_output_archiving_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    marker = asyncio.Event()

    class Client(RecordingClient):
        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            if url.endswith("/artifacts"):
                assert marker.is_set()
                request = httpx.Request("POST", f"http://test{url}")
                return httpx.Response(200, request=request, json={"id": "artifact-id"})
            return await super().post(url, **kwargs)

    def slow_archive(_source: Path, destination: Path) -> None:
        time.sleep(0.05)
        destination.write_bytes(b"archive")

    worker.client = Client()  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    monkeypatch.setattr(worker_main, "_archive_output_directory", slow_archive)
    output = tmp_path / "output"
    output.mkdir()
    (output / "payload.bin").write_bytes(b"payload")
    marker_task = asyncio.create_task(asyncio.sleep(0.01))
    marker_task.add_done_callback(lambda _task: marker.set())

    uploaded = await worker.upload_outputs(
        {
            "id": "task-id",
            "payload": {"command": {"cwd": str(tmp_path), "outputs": []}},
        },
        "attempt-id",
        {"outputs": [{"path": str(output), "kind": "checkpoint"}]},
    )

    await marker_task
    assert uploaded == [
        {
            "ref": "output-0",
            "artifact_id": "artifact-id",
            "kind": "checkpoint",
            "storage_type": "managed",
        }
    ]


@pytest.mark.asyncio
async def test_stale_completion_does_not_crash_worker() -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    worker.client = StaleCompletionClient()  # type: ignore[assignment]

    await worker._post_completion("task-id", {"status": "failed"})


@pytest.mark.asyncio
async def test_worker_failure_keeps_redacted_traceback_and_observability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    client = RecordingClient()
    worker.client = client  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    monkeypatch.setattr(worker, "resources", lambda: {})

    async def run(_task: dict, _emit, _desired_state) -> dict:
        raise RuntimeError("password=hunter2")

    monkeypatch.setattr(worker.data_import, "run", run)

    await worker.execute(
        {
            "task": {
                "id": "task-failed",
                "project_id": "project-id",
                "project_workspace_uri": str(tmp_path),
                "adapter_id": "builtin.data-import",
                "payload": {},
            },
            "attempt": {"id": "attempt-failed"},
            "lease": {"allocation": {}},
        }
    )

    complete = next(payload for url, payload in client.posts if url.endswith("/complete"))
    assert complete["status"] == "failed"
    assert complete["error"]["message"] == "password=[REDACTED]"
    assert complete["runtime_metrics_artifact_id"] == "runtime-metrics-id"
    platform_error = next(
        payload["payload"]
        for url, payload in client.posts
        if url.endswith("/events")
        and payload["event_type"] == "log"
        and payload["payload"].get("level") == "error"
    )
    assert "hunter2" not in platform_error["traceback"]
    assert "password=[REDACTED]" in platform_error["traceback"]


@pytest.mark.asyncio
async def test_worker_reports_inaccessible_project_workspace_as_failed(tmp_path: Path) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    client = RecordingClient()
    worker.client = client  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    worker.resources = lambda: {}  # type: ignore[method-assign]

    await worker.execute(
        {
            "task": {
                "id": "task-missing-workspace",
                "project_id": "project-id",
                "project_workspace_uri": str(tmp_path / "missing"),
                "adapter_id": "builtin.data-import",
                "payload": {},
            },
            "attempt": {"id": "attempt-id"},
            "lease": {"allocation": {}},
        }
    )

    assert [url.rsplit("/", 1)[-1] for url, _ in client.posts] == ["start", "complete"]
    assert client.posts[-1][1]["status"] == "failed"
    assert "not accessible" in client.posts[-1][1]["error"]["message"]


@pytest.mark.asyncio
async def test_worker_local_root_rewrites_execution_workspace(tmp_path: Path) -> None:
    worker_root = tmp_path / "worker-root"
    worker = WorkerAgent(
        AgentConfig(
            "http://test",
            "token",
            "worker",
            workspace_root=str(worker_root),
        )
    )
    await worker.client.aclose()
    client = RecordingClient()
    worker.client = client  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    worker.resources = lambda: {}  # type: ignore[method-assign]
    observed: dict[str, Any] = {}

    async def run(task: dict[str, Any], _emit, _desired_state) -> dict[str, Any]:
        observed.update(task)
        return {}

    worker.data_import.run = run  # type: ignore[method-assign]
    worker.upload_outputs = lambda *_args: asyncio.sleep(0, result=[])  # type: ignore[method-assign]
    await worker.execute(
        {
            "task": {
                "id": "task-id",
                "project_id": "project-id",
                "project_workspace_uri": "Z:/control-plane-only/project",
                "adapter_id": "builtin.data-import",
                "payload": {
                    "project_workspace_uri": "Z:/control-plane-only/project",
                    "source_config": {
                        "project_workspace_uri": "Z:/control-plane-only/project"
                    },
                },
            },
            "attempt": {"id": "attempt-id"},
            "lease": {"allocation": {}},
        }
    )

    expected_workspace = (worker_root / "projects" / "project-id").resolve()
    assert observed["payload"]["project_workspace_uri"] == str(expected_workspace)
    assert observed["payload"]["source_config"]["project_workspace_uri"] == str(
        expected_workspace
    )
    assert observed["runtime_context"]["project_workspace"] == str(expected_workspace)
    assert observed["runtime_context"]["control_plane_workspace_uri"] == (
        "Z:/control-plane-only/project"
    )
    assert Path(observed["runtime_context"]["run_dir"]).is_relative_to(expected_workspace)


@pytest.mark.asyncio
async def test_training_attempts_share_stable_training_run_directory(tmp_path: Path) -> None:
    worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
    await worker.client.aclose()
    client = RecordingClient()
    worker.client = client  # type: ignore[assignment]
    worker.worker_id = "worker-id"
    worker.resources = lambda: {}  # type: ignore[method-assign]
    observed_tasks: list[dict[str, Any]] = []

    async def run(task: dict[str, Any], _emit, _desired_state) -> dict[str, Any]:
        observed_tasks.append(task)
        return {}

    worker.command.run = run  # type: ignore[method-assign]
    worker.upload_outputs = lambda *_args: asyncio.sleep(0, result=[])  # type: ignore[method-assign]
    task = {
        "id": "task-id",
        "kind": "train",
        "project_id": "project-id",
        "project_workspace_uri": str(tmp_path),
        "adapter_id": "builtin.command.training",
        "payload": {
            "training_run_id": "training-run-id",
            "command": {"executable": "python", "args": [], "env": {}, "outputs": []},
        },
    }

    await worker.execute({"task": task, "attempt": {"id": "attempt-1"}, "lease": {}})
    await worker.execute({"task": task, "attempt": {"id": "attempt-2"}, "lease": {}})

    expected_run_dir = (tmp_path / ".llm-platform" / "training-runs" / "training-run-id").resolve()
    assert [Path(item["runtime_context"]["run_dir"]) for item in observed_tasks] == [
        expected_run_dir,
        expected_run_dir,
    ]
    assert [Path(item["runtime_context"]["attempt_dir"]) for item in observed_tasks] == [
        expected_run_dir / "attempts" / "attempt-1",
        expected_run_dir / "attempts" / "attempt-2",
    ]
    for item in observed_tasks:
        command = item["payload"]["command"]
        assert Path(command["cwd"]) == expected_run_dir
        assert command["env"]["LLM_PLATFORM_RUN_DIR"] == str(expected_run_dir)
        assert command["env"]["LLM_PLATFORM_TRAINING_RUN_ID"] == "training-run-id"
    assert (expected_run_dir / "attempts" / "attempt-1" / "task-spec.json").is_file()
    assert (expected_run_dir / "attempts" / "attempt-2" / "task-spec.json").is_file()
    start_payloads = [payload for url, payload in client.posts if url.endswith("/start")]
    assert [Path(payload["runtime_handle"]["run_dir"]) for payload in start_payloads] == [
        expected_run_dir,
        expected_run_dir,
    ]
