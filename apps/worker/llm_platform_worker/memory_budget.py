from __future__ import annotations

import psutil

MIB = 1024 * 1024
GIB = 1024 * MIB

WORKER_MEMORY_RESERVE_BYTES = 2 * GIB
PARQUET_PROCESS_MEMORY_BYTES = 512 * MIB
PARQUET_AGGREGATE_MEMORY_CAP_BYTES = 4 * GIB
MIXTURE_DUCKDB_MEMORY_CAP_BYTES = 4 * GIB


def available_memory_bytes(override: int | None = None) -> int:
    if override is not None:
        return max(int(override), 0)
    return max(int(psutil.virtual_memory().available), 0)


def parquet_memory_worker_limit(available: int | None = None) -> int:
    current = available_memory_bytes(available)
    usable = max(current - WORKER_MEMORY_RESERVE_BYTES, PARQUET_PROCESS_MEMORY_BYTES)
    aggregate_limit = PARQUET_AGGREGATE_MEMORY_CAP_BYTES // PARQUET_PROCESS_MEMORY_BYTES
    return max(1, min(usable // PARQUET_PROCESS_MEMORY_BYTES, aggregate_limit))


def duckdb_memory_budget_bytes(available: int | None = None) -> int:
    current = available_memory_bytes(available)
    usable = max(current - WORKER_MEMORY_RESERVE_BYTES, PARQUET_PROCESS_MEMORY_BYTES)
    return max(
        PARQUET_PROCESS_MEMORY_BYTES,
        min(usable // 2, MIXTURE_DUCKDB_MEMORY_CAP_BYTES),
    )


def duckdb_memory_limit(value: int) -> str:
    return f"{max(int(value) // MIB, 1)}MB"
