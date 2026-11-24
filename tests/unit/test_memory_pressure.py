from __future__ import annotations

import os
from pathlib import Path

from llm_platform_worker.adapters.parquet_analysis import recommended_processes
from llm_platform_worker.file_cache import release_file_cache
from llm_platform_worker.memory_budget import GIB, MIB, duckdb_memory_budget_bytes


def test_parquet_process_count_is_bounded_by_available_memory(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 32)

    assert recommended_processes(100, configured=18, available_memory=2 * GIB) == 1
    assert recommended_processes(100, configured=18, available_memory=6 * GIB) == 8
    assert recommended_processes(3, configured=18, available_memory=32 * GIB) == 3


def test_duckdb_memory_budget_reserves_worker_memory_and_has_a_hard_cap() -> None:
    assert duckdb_memory_budget_bytes(2 * GIB) == 512 * MIB
    assert duckdb_memory_budget_bytes(4 * GIB) == GIB
    assert duckdb_memory_budget_bytes(32 * GIB) == 4 * GIB


def test_file_cache_release_is_best_effort(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[int, int, int, int]] = []
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(
        os,
        "posix_fadvise",
        lambda descriptor, offset, length, advice: calls.append(
            (descriptor, offset, length, advice)
        ),
        raising=False,
    )
    path = tmp_path / "large-manifest.jsonl"
    path.write_text("record\n", encoding="utf-8")

    assert release_file_cache(path) is True
    assert len(calls) == 1
    assert calls[0][1:] == (0, 0, 4)
