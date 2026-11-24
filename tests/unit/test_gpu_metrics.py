import asyncio
import threading
from types import SimpleNamespace

import pytest
from llm_platform_worker.gpu_metrics import GpuCollector, GpuSamplingError, NvmlReader
from llm_platform_worker.main import WorkerAgent
from llm_platform_worker.runtime_metrics import AttemptResourceMonitor


def test_nvml_session_reused_and_unsupported_telemetry_keeps_device(monkeypatch):
    calls = []

    class Unsupported(Exception):
        pass

    def unavailable(*args):
        raise Unsupported()

    nvml = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"), NVMLError=Unsupported,
        nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetName=lambda h: b"test GPU",
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=8*1024**3, free=2*1024**3, used=6*1024**3),
        nvmlDeviceGetPowerUsage=unavailable, nvmlDeviceGetUtilizationRates=unavailable,
        nvmlDeviceGetTemperature=lambda h, sensor: 60, NVML_TEMPERATURE_GPU=0,
    )
    monkeypatch.setattr("llm_platform_worker.gpu_metrics.importlib.import_module", lambda _: nvml)
    reader = NvmlReader()
    assert reader() == reader()
    assert calls == ["init"]
    row = reader()[0]
    assert row["memory_free_mb"] == 2048
    assert row["power_watts"] is None and row["utilization_percent"] is None
    assert row["temperature_c"] == 60 and row["name"] == "test GPU"


def test_cached_rows_are_isolated_and_capacity_can_force_refresh():
    calls = []

    def read():
        calls.append(1)
        return [{"memory_free_mb": len(calls)}]

    collector = GpuCollector(read, max_age=60)
    collector.snapshot()[0]["memory_free_mb"] = -1
    assert collector.snapshot() == [{"memory_free_mb": 1}]
    assert collector.snapshot(fresh=True) == [{"memory_free_mb": 2}]


def test_hung_driver_has_bounded_wait_no_stale_result_or_extra_queries():
    release = threading.Event()
    calls = []

    def read():
        calls.append(1)
        if len(calls) > 1:
            release.wait(2)
        return [{"memory_free_mb": 100}]

    collector = GpuCollector(read, max_age=0)
    assert collector.snapshot()
    try:
        for _ in range(3):
            with pytest.raises(GpuSamplingError, match="timed out"):
                collector.snapshot(timeout=0.01)
        assert len(calls) == 2
    finally:
        release.set()
    assert collector.snapshot(timeout=1)


def test_sampling_errors_are_cached_without_subprocess_fallback():
    calls = []

    def read():
        calls.append(1)
        raise RuntimeError("driver unavailable")

    collector = GpuCollector(read, max_age=60)
    for _ in range(3):
        with pytest.raises(GpuSamplingError, match="driver unavailable"):
            collector.snapshot()
    assert calls == [1]


def test_resource_monitor_filters_allocation_and_reports_failure(monkeypatch, tmp_path):
    async def emit(*args):
        return "running"

    monitor = AttemptResourceMonitor(tmp_path / "metrics.gz", {"gpu_indexes": [1]}, emit)
    row = dict(index=1, memory_used_mb=2, memory_total_mb=3, utilization_percent=None,
               temperature_c=60, power_watts=None)
    monkeypatch.setattr("llm_platform_worker.runtime_metrics.gpu_snapshot",
                        lambda: [{**row, "index": 0}, row])
    assert monitor._collect_gpus() == [row]

    def fail():
        raise GpuSamplingError("GPU sampling timed out")

    monkeypatch.setattr("llm_platform_worker.runtime_metrics.gpu_snapshot", fail)
    assert monitor._collect_gpus() == []
    assert monitor._warnings == ["GPU sampling timed out"]


@pytest.mark.asyncio
async def test_heartbeat_does_not_block_other_agent_coroutines():
    worker = WorkerAgent.__new__(WorkerAgent)
    worker.worker_id = "test"
    worker.config = SimpleNamespace(inference_idle_seconds=600)
    worker.microlm_inference = SimpleNamespace(_last_used=None)
    worker.transformers_inference = SimpleNamespace(_last_used=None)
    worker.capabilities = lambda: []
    started, release = threading.Event(), threading.Event()

    def resources():
        started.set()
        release.wait(1)
        return {}

    async def post(*args, **kwargs):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"controls": []})

    worker.resources = resources
    worker.client = SimpleNamespace(post=post)
    task = asyncio.create_task(worker.heartbeat())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        assert not task.done(), "blocked resource query prevented event-loop progress"
    finally:
        release.set()
        await task


def test_capacity_snapshot_does_not_use_old_free_memory_when_refresh_fails(monkeypatch):
    worker = WorkerAgent.__new__(WorkerAgent)
    worker.resources = lambda: {"gpus": [{"memory_free_mb": 9999}]}
    worker.labels = lambda: {}

    def fail(**kwargs):
        raise GpuSamplingError("timeout")

    monkeypatch.setattr("llm_platform_worker.main.gpu_snapshot", fail)
    assert worker.environment_snapshot({"gpu_indexes": [0]})["gpus"] == []
