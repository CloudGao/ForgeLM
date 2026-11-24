from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from llm_platform_worker.runtime_metrics import AttemptResourceMonitor


@pytest.mark.asyncio
async def test_attempt_resource_monitor_writes_samples_and_summary(tmp_path: Path) -> None:
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    monitor = AttemptResourceMonitor(
        tmp_path / "runtime-metrics.jsonl.gz",
        {"cpus": 2, "gpu_indexes": [0]},
        emit,
        sample_interval_seconds=60,
    )
    samples = iter(
        [
            {
                "timestamp": "2026-08-08T00:00:00+00:00",
                "elapsed_seconds": 0,
                "process": {"cpu_percent": 25.0, "rss_bytes": 100},
                "system": {"cpu_percent": 30.0, "memory_percent": 40.0},
                "gpus": [{"index": 0, "utilization_percent": 50.0, "memory_used_mb": 200.0, "memory_total_mb": 1000.0, "temperature_c": 60.0, "power_watts": 80.0}],
            },
            {
                "timestamp": "2026-08-08T00:00:01+00:00",
                "elapsed_seconds": 1,
                "process": {"cpu_percent": 75.0, "rss_bytes": 300},
                "system": {"cpu_percent": 70.0, "memory_percent": 60.0},
                "gpus": [{"index": 0, "utilization_percent": 90.0, "memory_used_mb": 800.0, "memory_total_mb": 1000.0, "temperature_c": 70.0, "power_watts": 100.0}],
            },
        ]
    )
    monitor._collect_sample = lambda _now: next(samples)  # type: ignore[method-assign]

    await monitor.start()
    summary = await monitor.stop()

    assert summary["sample_count"] == 2
    assert summary["process"] == {
        "average_cpu_percent": 50.0,
        "peak_cpu_percent": 75.0,
        "peak_rss_bytes": 300,
    }
    assert summary["gpus"][0]["peak_memory_used_mb"] == 800.0
    assert [event_type for event_type, _ in events] == [
        "resource.sample",
        "resource.sample",
        "resource.summary",
    ]
    with gzip.open(monitor.output_path, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert [record["record_type"] for record in records] == [
        "metadata",
        "sample",
        "sample",
        "summary",
    ]


@pytest.mark.asyncio
async def test_attempt_resource_monitor_keeps_sampling_failure_as_warning(tmp_path: Path) -> None:
    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    monitor = AttemptResourceMonitor(
        tmp_path / "runtime-metrics.jsonl.gz",
        {},
        emit,
        sample_interval_seconds=60,
    )
    monitor._collect_sample = lambda _now: (_ for _ in ()).throw(RuntimeError("denied"))  # type: ignore[method-assign]

    await monitor.start()
    summary = await monitor.stop()

    assert summary["sample_count"] == 0
    assert summary["warnings"] == ["resource sample failed: RuntimeError: denied"]
