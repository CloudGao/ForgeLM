from __future__ import annotations

import asyncio
from contextlib import nullcontext
import pytest
from types import SimpleNamespace

from llm_platform_worker.adapters.transformers_inference import (
    TransformersInferenceAdapter,
    _LoadedRuntime,
)


class FakeGenerated:
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [10, 11]


class FakeOutput:
    def __getitem__(self, key):
        assert key == (0, slice(3, None, None))
        return FakeGenerated()


class FakeEncoded(dict):
    def __init__(self):
        super().__init__(input_ids=SimpleNamespace(shape=(1, 3)))

    def to(self, device: str):
        assert device == "cpu"
        return self


class FakeTokenizer:
    chat_template = "available"
    pad_token_id = 0
    eos_token_id = 11

    def __init__(self) -> None:
        self.messages = None
        self.add_special_tokens = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "rendered-chat-prompt"

    def __call__(self, prompt, *, return_tensors, add_special_tokens):
        assert prompt == "rendered-chat-prompt"
        assert return_tensors == "pt"
        self.add_special_tokens = add_special_tokens
        return FakeEncoded()

    @staticmethod
    def decode(token_ids, *, skip_special_tokens):
        assert token_ids == [10, 11]
        assert skip_special_tokens is True
        return "真实输出"


class FakeModel:
    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 32
        assert kwargs["temperature"] == 0.8
        assert kwargs["do_sample"] is True
        return FakeOutput()


def test_transformers_adapter_uses_chat_template_and_real_generation_result(monkeypatch) -> None:
    pytest.importorskip("transformers", reason="Optional inference runtime")
    tokenizer = FakeTokenizer()
    torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    adapter = TransformersInferenceAdapter()
    runtime = _LoadedRuntime(
        ("org/model", "cpu", "float32", None), torch, tokenizer, FakeModel(), "cpu"
    )
    monkeypatch.setattr(adapter, "_load_runtime", lambda *args: runtime)
    events = []

    async def emit(event_type, payload):
        events.append((event_type, payload))
        return "running"

    task = {
        "payload": {
            "model": {"runtime_source": "org/model"},
            "messages": [
                {"role": "system", "content": "精确回答"},
                {"role": "user", "content": "你好"},
            ],
            "config": {
                "device": "cpu",
                "dtype": "float32",
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40,
                "repetition_penalty": 1.05,
                "max_tokens": 32,
            },
        }
    }

    result = asyncio.run(adapter.run(task, emit, lambda: "running"))

    assert tokenizer.messages == task["payload"]["messages"]
    assert tokenizer.add_special_tokens is False
    assert result["rendered_prompt"] == "rendered-chat-prompt"
    assert result["output"] == "真实输出"
    assert result["token_ids"] == [10, 11]
    assert result["input_tokens"] == 3
    assert result["output_tokens"] == 2
    assert result["stop_reason"] == "eos"
    assert result["performance"]["tokens_per_second"] > 0
    assert events[0][1]["phase"] == "loading_model"
    assert events[-1][1]["phase"] == "finalizing_outputs"
    assert any(event == "inference.output" for event, _ in events)


def test_transformers_adapter_has_an_honest_plaintext_fallback() -> None:
    tokenizer = SimpleNamespace(chat_template=None)
    rendered, used_template = TransformersInferenceAdapter._render_prompt(
        tokenizer,
        [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "user", "content": "继续"},
        ],
    )

    assert used_template is False
    assert rendered == "System: 规则\nUser: 问题\nAssistant: 旧回答\nUser: 继续\nAssistant:"


def test_transformers_adapter_auto_uses_raw_completion_without_a_chat_template() -> None:
    tokenizer = SimpleNamespace(chat_template=None)
    payload = {
        "prompt": "人工智能的发展",
        "system_prompt": "不要注入",
        "messages": [
            {"role": "system", "content": "不要注入"},
            {"role": "user", "content": "人工智能的发展"},
        ],
        "config": {"prompt_mode": "auto"},
    }

    rendered, used_template, mode = TransformersInferenceAdapter._render_inference_prompt(
        tokenizer, payload
    )

    assert rendered == "人工智能的发展"
    assert used_template is False
    assert mode == "completion"


def test_transformers_adapter_explicit_chat_keeps_plaintext_role_fallback() -> None:
    tokenizer = SimpleNamespace(chat_template=None)
    payload = {
        "prompt": "问题",
        "messages": [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "问题"},
        ],
        "config": {"prompt_mode": "chat"},
    }

    rendered, used_template, mode = TransformersInferenceAdapter._render_inference_prompt(
        tokenizer, payload
    )

    assert rendered == "System: 规则\nUser: 问题\nAssistant:"
    assert used_template is False
    assert mode == "chat"


def test_transformers_adapter_legacy_task_keeps_old_prompt_semantics() -> None:
    tokenizer = SimpleNamespace(chat_template=None)
    payload = {
        "prompt": "问题",
        "messages": [{"role": "user", "content": "问题"}],
        "config": {},
    }

    rendered, used_template, mode = TransformersInferenceAdapter._render_inference_prompt(
        tokenizer, payload
    )

    assert rendered == "User: 问题\nAssistant:"
    assert used_template is False
    assert mode == "chat"
