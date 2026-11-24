import asyncio
import pytest
from types import SimpleNamespace

from llm_platform_worker.adapters.inference_stream import run_inference


def test_slow_transport_does_not_block_decoding_and_output_precedes_finalization():
    events, generated = [], []
    class Adapter:
        _prefix = None
        def _generate_sync(self, task):
            for i in range(1000):
                self._publish({"phase": "generating", "output": str(i)})
            generated.append(True)
            return {"output": "999", "output_tokens": 1000}
    async def emit(kind, payload):
        if kind == "inference.snapshot":
            assert generated
            await asyncio.sleep(0.03)
        events.append((kind, payload))
    adapter = Adapter()
    asyncio.run(run_inference(adapter, {}, emit, lambda: "running"))
    assert events[-2][0] == "inference.output"
    assert events[-1][1]["phase"] == "finalizing_outputs"
    assert len(events) <= 4
    assert not adapter._active


def test_cancellation_discards_prefix_and_never_publishes_final_output():
    events = []
    state = ["running"]
    class Adapter:
        _prefix = object()
        def _generate_sync(self, task):
            state[0] = "cancelled"
            return {"output": "partial", "output_tokens": 1}
    async def emit(kind, payload): events.append(kind)
    adapter = Adapter()
    result = asyncio.run(run_inference(adapter, {}, emit, lambda: state[0]))
    assert result == {"controlled_state": "cancelled"}
    assert adapter._prefix is None
    assert "inference.output" not in events


def test_transformers_cache_identity_survives_changed_materialization_path(monkeypatch):
    transformers = pytest.importorskip("transformers", reason="Optional inference runtime")
    from llm_platform_worker.adapters.transformers_inference import TransformersInferenceAdapter
    calls = []
    model = SimpleNamespace(to=lambda *a: None, eval=lambda: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: object())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **kw: calls.append(a[0]) or model)
    adapter = TransformersInferenceAdapter()
    first = adapter._load_runtime("/attempt1", "cpu", "float32", None, "artifact:sha")
    assert adapter._load_runtime("/attempt2", "cpu", "float32", None, "artifact:sha") is first
    assert calls == ["/attempt1"]
    assert adapter._load_runtime("/attempt3", "cpu", "float32", None, "artifact:new-sha") is not first
    assert len(calls) == 2


def test_transformers_real_generate_reports_first_token_and_stream(monkeypatch):
    torch = pytest.importorskip("torch", reason="Optional inference runtime")
    from transformers import GPT2Config, GPT2LMHeadModel
    from llm_platform_worker.adapters.transformers_inference import TransformersInferenceAdapter, _LoadedRuntime
    class Batch(dict):
        def to(self, device): return self
    class Tokenizer:
        chat_template = None
        pad_token_id, eos_token_id = 0, 31
        def __call__(self, prompt, **kwargs):
            return Batch(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
        def decode(self, ids, **kwargs): return " ".join(str(x) for x in ids)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_head=2, n_layer=1,
                                      eos_token_id=31, pad_token_id=0)).eval()
    runtime = _LoadedRuntime(("a", "cpu", "float32", None), torch, Tokenizer(), model, "cpu")
    adapter = TransformersInferenceAdapter()
    monkeypatch.setattr(adapter, "_load_runtime", lambda *args: runtime)
    events = []
    async def emit(kind, payload): events.append((kind, payload))
    result = asyncio.run(adapter.run({"payload": {"model": {"runtime_source": "local"},
        "prompt": "test", "config": {"device": "cpu", "prompt_mode": "completion", "max_tokens": 4, "temperature": 0}}}, emit, lambda: "running"))
    assert result["performance"]["ttft_ms"] is not None
    assert any(kind == "inference.snapshot" and payload.get("output") for kind, payload in events)
