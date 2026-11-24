from types import SimpleNamespace
import weakref

import pytest

from llm_platform_worker.adapters.microlm_inference import MicroLMInferenceAdapter
from llm_platform_worker.adapters.transformers_inference import TransformersInferenceAdapter
from llm_platform_worker.main import WorkerAgent
from llm_platform_worker.adapters.inference_cache import release_expired


def test_default_ten_minute_idle_unload_resets_and_never_interrupts_generation():
    calls = []
    owner = SimpleNamespace(_last_used=100.0, _active=False, release_runtime=lambda: calls.append(1) or True)
    assert not release_expired(owner, now=699.99)
    owner._active = True
    assert not release_expired(owner, now=800)
    owner._active = False
    owner._last_used = 800.0  # Another request finished, restarting the idle period.
    assert not release_expired(owner, now=1399.99)
    assert release_expired(owner, now=1400)
    assert calls == [1]


class Runtime:
    pass


@pytest.mark.parametrize("adapter_type", [MicroLMInferenceAdapter, TransformersInferenceAdapter])
def test_release_drops_model_before_emptying_cuda_cache(adapter_type):
    adapter = adapter_type()
    runtime = Runtime()
    reference = weakref.ref(runtime)
    calls = []

    def empty_cache():
        assert adapter._runtime is None
        assert reference() is None
        calls.append("released")

    runtime.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=empty_cache))
    runtime.device = "cuda:0"
    adapter._runtime = runtime
    del runtime
    assert adapter.release_runtime()
    assert calls == ["released"]
    assert not adapter.release_runtime()


def test_worker_releases_other_gpu_engines_but_keeps_same_engine_and_cpu_cache():
    worker = WorkerAgent.__new__(WorkerAgent)
    calls = []
    worker.microlm_inference = SimpleNamespace(release_runtime=lambda: calls.append("micro") or True)
    worker.transformers_inference = SimpleNamespace(release_runtime=lambda: calls.append("hf") or True)
    assert worker.release_idle_inference_runtimes({"adapter_id": "builtin.command.training"}, {"gpu_indexes": [0]}) == [
        "builtin.microlm-inference", "builtin.transformers-inference"]
    assert calls == ["micro", "hf"]
    calls.clear()
    assert worker.release_idle_inference_runtimes({"adapter_id": "builtin.microlm-inference"}, {"gpus": 1}) == [
        "builtin.transformers-inference"]
    assert calls == ["hf"]
    calls.clear()
    assert worker.release_idle_inference_runtimes({"adapter_id": "builtin.data-import"}, {"gpus": 0}) == []
    assert calls == []
