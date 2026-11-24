"""Bounded GPU snapshots from a persistent NVML session, shared within a Worker."""
from __future__ import annotations

import copy
import importlib
import threading
import time
from collections.abc import Callable
from typing import Any


class GpuSamplingError(RuntimeError):
    pass


class NvmlReader:
    def __init__(self) -> None:
        self._nvml = None

    def __call__(self) -> list[dict[str, Any]]:
        if self._nvml is None:
            nvml = importlib.import_module("pynvml")
            nvml.nvmlInit()
            self._nvml = nvml
        nvml = self._nvml

        def optional(call: Callable[[], Any]) -> Any:
            try:
                return call()
            except (nvml.NVMLError, AttributeError):
                return None

        rows = []
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            name = nvml.nvmlDeviceGetName(handle)
            uuid = optional(lambda handle=handle: nvml.nvmlDeviceGetUUID(handle))
            power = optional(lambda handle=handle: nvml.nvmlDeviceGetPowerUsage(handle))
            rows.append({
                "index": index,
                "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                "name": name.decode() if isinstance(name, bytes) else name,
                "memory_total_mb": int(memory.total // (1024 * 1024)),
                "memory_free_mb": int(memory.free // (1024 * 1024)),
                "memory_used_mb": memory.used / (1024 * 1024),
                "utilization_percent": optional(
                    lambda handle=handle: nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                ),
                "temperature_c": optional(
                    lambda handle=handle: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                ),
                "power_watts": None if power is None else power / 1000,
            })
        return rows


class GpuCollector:
    """At most one native query in flight, even if the driver never returns.

    A daemon owns the NVML session for its lifetime. Callers time out without
    creating more threads, returning stale free memory, or launching nvidia-smi.
    """

    def __init__(self, reader: Callable[[], list[dict[str, Any]]], *, max_age: float = 5.0):
        self._reader = reader
        self._max_age = max_age
        self._condition = threading.Condition()
        self._request = threading.Event()
        self._inflight = False
        self._completed_at = float("-inf")
        self._generation = 0
        self._rows: list[dict[str, Any]] = []
        self._error: str | None = None
        threading.Thread(target=self._run, name="worker-nvml", daemon=True).start()

    def _run(self) -> None:
        while True:
            self._request.wait()
            self._request.clear()
            rows, error = [], None
            try:
                rows = self._reader()
            except Exception as exc:
                error = f"GPU sampling unavailable: {type(exc).__name__}: {exc}"
            with self._condition:
                self._rows, self._error = rows, error
                self._completed_at = time.monotonic()
                self._generation += 1
                self._inflight = False
                self._condition.notify_all()

    def snapshot(self, *, timeout: float = 3.0, fresh: bool = False) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        with self._condition:
            # A forced refresh must start after any older query has completed.
            if fresh and self._inflight:
                while self._inflight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GpuSamplingError("GPU sampling timed out")
                    self._condition.wait(remaining)
            if fresh or time.monotonic() - self._completed_at >= self._max_age:
                generation = self._generation
                if not self._inflight:
                    self._inflight = True
                    self._request.set()
                while self._generation == generation:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise GpuSamplingError("GPU sampling timed out")
                    self._condition.wait(remaining)
            if self._error:
                raise GpuSamplingError(self._error)
            return copy.deepcopy(self._rows)


_collector: GpuCollector | None = None
_collector_lock = threading.Lock()


def gpu_snapshot(*, fresh: bool = False) -> list[dict[str, Any]]:
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = GpuCollector(NvmlReader())
        collector = _collector
    return collector.snapshot(fresh=fresh)
