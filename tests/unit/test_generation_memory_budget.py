from types import SimpleNamespace

import pytest
from llm_platform_worker.data_generation_runtime import vllm_memory_budget


def device(free, total=12):
    return SimpleNamespace(cuda=SimpleNamespace(
        mem_get_info=lambda: (free * 1024**3, total * 1024**3)))


def test_wsl_uses_available_memory_without_extra_reserve(capsys):
    assert vllm_memory_budget(.88, device(10), "microsoft-standard-WSL2") == 10 / 12
    assert 'wsl_available_memory_v2' in capsys.readouterr().out


def test_explicit_lower_budget_is_preserved():
    assert vllm_memory_budget(.5, device(11), "microsoft-standard-WSL2") == .5


def test_native_linux_does_not_query_or_change_budget():
    assert vllm_memory_budget(.88, None, "generic-linux") == .88


def test_insufficient_wsl_memory_fails_before_engine_start():
    with pytest.raises(RuntimeError, match="Insufficient"):
        vllm_memory_budget(.88, device(0), "microsoft-standard-WSL2")
