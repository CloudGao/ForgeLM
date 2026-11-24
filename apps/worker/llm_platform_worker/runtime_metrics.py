from __future__ import annotations

import asyncio
import gzip
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import psutil

from .gpu_metrics import GpuSamplingError, gpu_snapshot

Emit = Callable[[str, dict[str, Any]], Awaitable[str]]


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AttemptResourceMonitor:
    """Collect bounded control-plane samples and a complete attempt-local time series."""

    schema_version = "attempt_resources.v1"

    def __init__(
        self,
        output_path: Path,
        allocation: dict[str, Any],
        emit: Emit,
        *,
        sample_interval_seconds: float = 5.0,
        event_interval_seconds: float = 15.0,
        process_id: int | None = None,
    ) -> None:
        self.output_path = output_path
        self.allocation = allocation
        self.emit = emit
        self.sample_interval_seconds = sample_interval_seconds
        self.event_interval_seconds = event_interval_seconds
        self.process = psutil.Process(process_id or os.getpid())
        self.started_at = _utc_timestamp()
        self._started_monotonic: float | None = None
        self._last_process_cpu_seconds: float | None = None
        self._last_sample_monotonic: float | None = None
        self._last_event_monotonic: float | None = None
        self._sample_count = 0
        self._process_cpu_sum = 0.0
        self._process_cpu_peak: float | None = None
        self._process_rss_peak: int | None = None
        self._system_cpu_sum = 0.0
        self._system_cpu_peak: float | None = None
        self._system_memory_peak: float | None = None
        self._gpu_summary: dict[int, dict[str, Any]] = {}
        self._warnings: list[str] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._stream: TextIO | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._started_monotonic = loop.time()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = gzip.open(self.output_path, "wt", encoding="utf-8")
        self._write_record(
            {
                "record_type": "metadata",
                "schema_version": self.schema_version,
                "started_at": self.started_at,
                "allocation": self.allocation,
                "sample_interval_seconds": self.sample_interval_seconds,
            }
        )
        await self._sample_and_record(force_event=True)
        self._task = asyncio.create_task(self._run(), name="attempt-resource-monitor")

    async def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._task is not None:
            await self._task
        try:
            await self._sample_and_record(force_event=True)
            summary = self.summary()
            self._write_record({"record_type": "summary", **summary})
        finally:
            self.close()
        try:
            await self.emit("resource.summary", summary)
        except Exception as exc:  # resource collection must not replace the task result
            self._warn(f"resource summary event failed: {type(exc).__name__}: {exc}")
        return summary

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=max(0.1, self.sample_interval_seconds)
                )
                return
            except TimeoutError:
                await self._sample_and_record()

    async def _sample_and_record(self, *, force_event: bool = False) -> None:
        loop = asyncio.get_running_loop()
        try:
            sample = await asyncio.to_thread(self._collect_sample, loop.time())
        except Exception as exc:  # psutil can race with short-lived child processes
            self._warn(f"resource sample failed: {type(exc).__name__}: {exc}")
            return
        self._update_summary(sample)
        self._write_record({"record_type": "sample", **sample})
        now = loop.time()
        due = (
            self._last_event_monotonic is None
            or now - self._last_event_monotonic >= self.event_interval_seconds
        )
        if force_event or due:
            try:
                await self.emit("resource.sample", sample)
                self._last_event_monotonic = now
            except Exception as exc:
                self._warn(f"resource sample event failed: {type(exc).__name__}: {exc}")

    def _collect_sample(self, monotonic_now: float) -> dict[str, Any]:
        processes: list[psutil.Process] = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        cpu_seconds = 0.0
        rss_bytes = 0
        thread_count = 0
        read_bytes = 0
        write_bytes = 0
        alive_processes = 0
        for process in processes:
            try:
                with process.oneshot():
                    times = process.cpu_times()
                    memory = process.memory_info()
                    cpu_seconds += float(times.user + times.system)
                    rss_bytes += int(memory.rss)
                    thread_count += int(process.num_threads())
                    alive_processes += 1
                    try:
                        io = process.io_counters()
                        read_bytes += int(io.read_bytes)
                        write_bytes += int(io.write_bytes)
                    except (AttributeError, psutil.AccessDenied, psutil.NotImplementedError):
                        pass
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue

        process_cpu_percent = 0.0
        if self._last_process_cpu_seconds is not None and self._last_sample_monotonic is not None:
            elapsed = max(1e-6, monotonic_now - self._last_sample_monotonic)
            process_cpu_percent = max(
                0.0, (cpu_seconds - self._last_process_cpu_seconds) / elapsed * 100
            )
        self._last_process_cpu_seconds = cpu_seconds
        self._last_sample_monotonic = monotonic_now

        memory = psutil.virtual_memory()
        from .resource_capacity import effective_capacity

        capacity = effective_capacity()
        allocated_cpus = int(self.allocation.get("cpus") or capacity["cpu_count"])
        allocated_cpus = max(1, min(allocated_cpus, capacity["cpu_count"]))
        elapsed_seconds = max(0.0, monotonic_now - (self._started_monotonic or monotonic_now))
        return {
            "timestamp": _utc_timestamp(),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "process": {
                "cpu_percent": round(process_cpu_percent, 2),
                "cpu_cores_used": round(process_cpu_percent / 100, 3),
                "allocated_cpus": allocated_cpus,
                "cpu_allocation_percent": round(process_cpu_percent / allocated_cpus, 2),
                "rss_bytes": rss_bytes,
                "process_count": alive_processes,
                "child_process_count": max(0, alive_processes - 1),
                "thread_count": thread_count,
                "io_read_bytes": read_bytes,
                "io_write_bytes": write_bytes,
            },
            "system": {
                "cpu_percent": round(float(psutil.cpu_percent(interval=None)), 2),
                "memory_total_bytes": int(memory.total),
                "memory_available_bytes": int(memory.available),
                "memory_used_bytes": int(memory.used),
                "memory_percent": round(float(memory.percent), 2),
                "effective_cpu_count": capacity["cpu_count"],
                "effective_memory_total_bytes": capacity["memory_total_bytes"],
                "effective_memory_available_bytes": capacity["memory_available_bytes"],
            },
            "gpus": self._collect_gpus(),
        }

    def _collect_gpus(self) -> list[dict[str, Any]]:
        allocated = {int(index) for index in self.allocation.get("gpu_indexes") or []}
        if not allocated:
            return []
        try:
            rows = gpu_snapshot()
        except GpuSamplingError as exc:
            self._warn(str(exc))
            return []
        keys = ("index", "utilization_percent", "memory_used_mb", "memory_total_mb",
                "temperature_c", "power_watts")
        return [{key: row[key] for key in keys} for row in rows if row["index"] in allocated]

    def summary(self) -> dict[str, Any]:
        gpus = []
        for index, stats in sorted(self._gpu_summary.items()):
            gpus.append(
                {
                    "index": index,
                    "average_utilization_percent": (
                        round(stats["utilization_sum"] / stats["utilization_count"], 2)
                        if stats["utilization_count"]
                        else None
                    ),
                    "peak_utilization_percent": stats["utilization_peak"],
                    "peak_memory_used_mb": stats["memory_peak"],
                    "memory_total_mb": stats["memory_total_mb"],
                    "peak_temperature_c": stats["temperature_peak"],
                    "average_power_watts": (
                        round(stats["power_sum"] / stats["power_count"], 2)
                        if stats["power_count"]
                        else None
                    ),
                }
            )
        return {
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "ended_at": _utc_timestamp(),
            "sample_count": self._sample_count,
            "sample_interval_seconds": self.sample_interval_seconds,
            "allocation": self.allocation,
            "process": {
                "average_cpu_percent": (
                    round(self._process_cpu_sum / self._sample_count, 2)
                    if self._sample_count
                    else None
                ),
                "peak_cpu_percent": self._process_cpu_peak,
                "peak_rss_bytes": self._process_rss_peak,
            },
            "system": {
                "average_cpu_percent": (
                    round(self._system_cpu_sum / self._sample_count, 2)
                    if self._sample_count
                    else None
                ),
                "peak_cpu_percent": self._system_cpu_peak,
                "peak_memory_percent": self._system_memory_peak,
            },
            "gpus": gpus,
            "warnings": list(self._warnings),
        }

    def _update_summary(self, sample: dict[str, Any]) -> None:
        self._sample_count += 1
        process_cpu = float(sample["process"]["cpu_percent"])
        process_rss = int(sample["process"]["rss_bytes"])
        system_cpu = float(sample["system"]["cpu_percent"])
        system_memory = float(sample["system"]["memory_percent"])
        self._process_cpu_sum += process_cpu
        self._process_cpu_peak = max(self._process_cpu_peak or process_cpu, process_cpu)
        self._process_rss_peak = max(self._process_rss_peak or process_rss, process_rss)
        self._system_cpu_sum += system_cpu
        self._system_cpu_peak = max(self._system_cpu_peak or system_cpu, system_cpu)
        self._system_memory_peak = max(self._system_memory_peak or system_memory, system_memory)
        for gpu in sample["gpus"]:
            index = int(gpu["index"])
            stats = self._gpu_summary.setdefault(
                index,
                {
                    "utilization_sum": 0.0,
                    "utilization_count": 0,
                    "utilization_peak": None,
                    "memory_peak": None,
                    "memory_total_mb": None,
                    "temperature_peak": None,
                    "power_sum": 0.0,
                    "power_count": 0,
                },
            )
            utilization = gpu.get("utilization_percent")
            if utilization is not None:
                utilization = float(utilization)
                stats["utilization_sum"] += utilization
                stats["utilization_count"] += 1
                stats["utilization_peak"] = max(
                    stats["utilization_peak"] if stats["utilization_peak"] is not None else utilization,
                    utilization,
                )
            memory_used = gpu.get("memory_used_mb")
            if memory_used is not None:
                memory_used = float(memory_used)
                stats["memory_peak"] = max(
                    stats["memory_peak"] if stats["memory_peak"] is not None else memory_used,
                    memory_used,
                )
            stats["memory_total_mb"] = gpu.get("memory_total_mb")
            temperature = gpu.get("temperature_c")
            if temperature is not None:
                temperature = float(temperature)
                stats["temperature_peak"] = max(
                    stats["temperature_peak"] if stats["temperature_peak"] is not None else temperature,
                    temperature,
                )
            power = gpu.get("power_watts")
            if power is not None:
                stats["power_sum"] += float(power)
                stats["power_count"] += 1

    def _write_record(self, record: dict[str, Any]) -> None:
        if self._stream is None:
            return
        self._stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        self._stream.write("\n")
        self._stream.flush()

    def _warn(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)
