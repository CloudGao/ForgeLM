from types import SimpleNamespace

from llm_platform_worker import resource_capacity as module


def test_container_capacity_respects_cpu_and_memory_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(module.os, "cpu_count", lambda: 208)
    monkeypatch.setattr(module.psutil, "Process", lambda: SimpleNamespace(cpu_affinity=lambda: list(range(32))))
    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: SimpleNamespace(total=800 * 1024**3, available=600 * 1024**3))
    (tmp_path / "cpu.max").write_text("800000 100000")
    (tmp_path / "memory.max").write_text(str(64 * 1024**3))
    (tmp_path / "memory.current").write_text(str(16 * 1024**3))
    capacity = module.effective_capacity(tmp_path)
    assert capacity["cpu_count"] == 8
    assert capacity["memory_total_bytes"] == 64 * 1024**3
    assert capacity["memory_available_bytes"] == 48 * 1024**3
    (tmp_path / "memory.current").write_text(str(63 * 1024**3))
    (tmp_path / "memory.stat").write_text(f"inactive_file {40 * 1024**3}\n")
    assert module.effective_capacity(tmp_path)["memory_available_bytes"] == 41 * 1024**3


def test_parallelism_uses_allocation_and_memory_without_overriding_manual_cap():
    capacity = {"cpu_count": 32, "memory_available_bytes": 64 * 1024**3}
    assert module.processing_parallelism("auto", {"cpus": 8}, capacity) == 8
    assert module.processing_parallelism(2, {"cpus": 8}, capacity) == 2
    assert module.processing_parallelism(32, {"cpus": 8}, capacity) == 8
    capacity["memory_available_bytes"] = 4 * 1024**3
    assert module.processing_parallelism("auto", {"cpus": 8}, capacity) == 2


def test_cold_loading_reserves_more_memory_than_streaming_operators():
    capacity = {"cpu_count": 25, "memory_available_bytes": 84 * 1024**3}
    assert module.processing_parallelism("auto", {"cpus": 25}, capacity) == 25
    assert module.dataset_loading_parallelism(25, capacity) == 8
    assert module.dataset_loading_parallelism(2, capacity) == 2
    capacity["memory_available_bytes"] = 12 * 1024**3
    assert module.dataset_loading_parallelism(25, capacity) == 1
