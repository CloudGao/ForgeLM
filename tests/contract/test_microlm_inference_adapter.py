from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
pytest.importorskip("torch", reason="Optional inference runtime")
import torch
from llm_platform_worker.adapters.microlm_inference import (
    MicroLMInferenceAdapter,
    _LoadedRuntime,
)


class FakeTokenizer:
    chat_template = None
    eos_token_id = 2

    def __call__(self, prompt, *, return_tensors, add_special_tokens):
        assert prompt == "你好"
        assert return_tensors == "pt"
        assert add_special_tokens is True
        return {"input_ids": torch.tensor([[1, 7, 8]])}

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert token_ids in ([5], [5, 2])
        assert skip_special_tokens is True
        return "测试输出"


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=64)

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, input_ids, *, past_key_values, use_cache):
        self.calls.append((input_ids.detach().cpu().tolist(), past_key_values, use_cache))
        logits = torch.full((1, input_ids.shape[-1], 16), -100.0)
        logits[:, -1, 5 if len(self.calls) == 1 else 2] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=(f"cache-{len(self.calls)}",))


def test_microlm_adapter_runs_cached_generation_and_returns_playground_contract(
    monkeypatch,
) -> None:
    adapter = MicroLMInferenceAdapter()
    model = FakeModel()
    runtime = _LoadedRuntime(
        ("artifact-1", "cpu", "float32"),
        torch,
        FakeTokenizer(),
        model,
        "cpu",
    )
    monkeypatch.setattr(adapter, "_load_runtime", lambda *args: runtime)
    events = []

    async def emit(event_type, payload):
        events.append((event_type, payload))
        return "running"

    task = {
        "payload": {
            "model": {
                "artifact_id": "artifact-1",
                "runtime_source": "/models/microlm",
                "kind": "base",
            },
            "messages": [
                {"role": "system", "content": "精确回答"},
                {"role": "user", "content": "你好"},
            ],
            "prompt": "你好",
            "config": {
                "device": "cpu",
                "dtype": "float32",
                "temperature": 0,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.05,
                "max_tokens": 8,
                "prompt_mode": "auto",
            },
        }
    }

    result = asyncio.run(adapter.run(task, emit, lambda: "running"))

    assert result["output"] == "测试输出"
    assert result["rendered_prompt"] == "你好"
    assert result["prompt_mode"] == "completion"
    assert result["token_ids"] == [5, 2]
    assert result["input_tokens"] == 3
    assert result["output_tokens"] == 2
    assert result["stop_reason"] == "eos"
    assert result["performance"]["runtime"] == "microlm-native"
    assert model.calls == [
        ([[1, 7, 8]], None, True),
        ([[5]], ("cache-1",), True),
    ]
    assert events[0][1]["phase"] == "loading_model"
    assert events[-1][1]["phase"] == "finalizing_outputs"
    assert any(event == "inference.output" for event, _ in events)
