from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

Emit = Callable[[str, dict[str, Any]], Awaitable[str]]


def transformers_inference_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


@dataclass
class _LoadedRuntime:
    key: tuple[str, str, str, str | None]
    torch: Any
    tokenizer: Any
    model: Any
    device: str


class TransformersInferenceAdapter:
    """Run a real, non-streaming causal-language-model generation."""

    def __init__(self) -> None:
        self._runtime: _LoadedRuntime | None = None
        self._prefix = None
        self._active = False
        self._publish = lambda payload: None
        self._desired_state = lambda: "running"
        self.resident = None

    def release_runtime(self) -> bool:
        from .inference_cache import release_runtime

        return release_runtime(self)

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        desired_state: Callable[[], str],
    ) -> dict[str, Any]:
        from .inference_stream import run_inference
        return await run_inference(self, task, emit, desired_state)

    def _generate_sync(self, task: dict[str, Any]) -> dict[str, Any]:
        payload = task.get("payload") or {}
        model_snapshot = payload.get("model") or {}
        source = str(model_snapshot.get("runtime_source") or "").strip()
        if not source:
            raise ValueError("Model runtime source was not materialized")

        config = payload.get("config") or {}
        requested_device = str(config.get("device") or "auto")
        if requested_device == "auto":
            allocation = (task.get("runtime_context") or {}).get("allocation") or {}
            requested_device = "cuda" if allocation.get("gpu_indexes") else "cpu"
        indexes = ((task.get("runtime_context") or {}).get("allocation") or {}).get("gpu_indexes", [])
        if requested_device.startswith("cuda") and indexes and int(indexes[0]) != 0:
            requested_device = f"cuda:{int(indexes[0])}"
        requested_dtype = str(config.get("dtype") or "auto")
        revision = model_snapshot.get("revision")
        load_started = time.perf_counter()
        previous_runtime = self._runtime
        runtime = self._load_runtime(source, requested_device, requested_dtype, revision, model_snapshot.get("runtime_identity") or model_snapshot.get("artifact_id"))
        load_ms = round((time.perf_counter() - load_started) * 1000, 2)
        cache_hit = previous_runtime is runtime
        self.resident = {"artifact_id": model_snapshot.get("artifact_id"),
            "model_id": model_snapshot.get("id"), "device": runtime.device,
            "requested_dtype": requested_dtype, "dtype": runtime.key[2],
            "gpu_indexes": ((task.get("runtime_context") or {}).get("allocation") or {}).get("gpu_indexes", [])}
        self._publish({"phase": "prefilling", "performance": {"model_cache_hit": cache_hit, "load_ms": load_ms}})
        torch = runtime.torch
        tokenizer = runtime.tokenizer
        model = runtime.model

        rendered_prompt, used_chat_template, prompt_mode = self._render_inference_prompt(
            tokenizer, payload
        )
        encoded = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=not used_chat_template,
        )
        encoded = encoded.to(runtime.device)
        input_tokens = int(encoded["input_ids"].shape[-1])

        temperature = float(config.get("temperature", 0.7))
        generation_args: dict[str, Any] = {
            "max_new_tokens": int(config.get("max_tokens", config.get("max_new_tokens", 512))),
            "do_sample": temperature > 0,
            "repetition_penalty": float(config.get("repetition_penalty", 1.0)),
        }
        if temperature > 0:
            generation_args.update(
                temperature=temperature,
                top_p=float(config.get("top_p", 1.0)),
                top_k=int(config.get("top_k", 50)),
            )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is not None:
            generation_args["pad_token_id"] = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            generation_args["eos_token_id"] = tokenizer.eos_token_id

        if runtime.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(runtime.device)
            torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()
        from .inference_stream import TokenStream
        stream = TokenStream(self, tokenizer, started, input_tokens, payload.get("request_received_at") or task.get("created_at"))
        from transformers import StoppingCriteria, StoppingCriteriaList
        adapter = self
        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return adapter._desired_state() in {"cancelled", "paused"}
        generation_args["streamer"] = stream
        generation_args["stopping_criteria"] = StoppingCriteriaList([Cancelled()])
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_args)
        if runtime.device.startswith("cuda"):
            torch.cuda.synchronize(runtime.device)
        elapsed_seconds = max(time.perf_counter() - started, 1e-9)

        generated = output_ids[0, input_tokens:]
        token_ids = [int(value) for value in generated.detach().cpu().tolist()]
        output = tokenizer.decode(token_ids, skip_special_tokens=True)
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        stop_reason = "eos" if token_ids and token_ids[-1] in (eos_ids or []) else "length"
        performance: dict[str, Any] = {
            "model_cache_hit": cache_hit,
            "file_cache_hit": model_snapshot.get("file_cache_hit"),
            "preparation_ms": model_snapshot.get("preparation_ms", 0),
            "load_ms": load_ms,
            "ttft_ms": stream.first_ms,
            "request_ttft_ms": stream.request_first_ms,
            "generation_ms": round(elapsed_seconds * 1000, 2),
            "tokens_per_second": round(len(token_ids) / elapsed_seconds, 2),
            "device": runtime.device,
        }
        if runtime.device.startswith("cuda"):
            performance["peak_memory_mb"] = round(
                torch.cuda.max_memory_allocated(runtime.device) / 1024 / 1024, 2
            )
        return {
            "rendered_prompt": rendered_prompt,
            "prompt_mode": prompt_mode,
            "output": output,
            "token_ids": token_ids,
            "input_tokens": input_tokens,
            "output_tokens": len(token_ids),
            "stop_reason": stop_reason,
            "performance": performance,
        }

    def _load_runtime(
        self, source: str, requested_device: str, requested_dtype: str, revision: str | None, runtime_identity: str | None = None
    ) -> _LoadedRuntime:
        if not transformers_inference_available():
            raise RuntimeError(
                "Transformers inference dependencies are missing; install with "
                '`pip install -e ".[inference-hf]"`'
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = (
            "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        )
        if device == "auto":
            device = "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this Worker")
        dtype_name = requested_dtype
        if dtype_name == "auto":
            dtype_name = "float16" if device.startswith("cuda") else "float32"
        dtype = getattr(torch, dtype_name, None)
        if dtype is None or dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported inference dtype: {requested_dtype}")
        key = (runtime_identity or source, device, dtype_name, revision)
        if self._runtime and self._runtime.key == key:
            return self._runtime

        self.release_runtime()
        tokenizer = AutoTokenizer.from_pretrained(
            source, revision=revision, trust_remote_code=False
        )
        load_args = {"revision": revision, "trust_remote_code": False, "dtype": dtype}
        try:
            model = AutoModelForCausalLM.from_pretrained(source, **load_args)
        except TypeError:
            load_args["torch_dtype"] = load_args.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(source, **load_args)
        model.to(device)
        model.eval()
        previous = self._runtime
        self._runtime = _LoadedRuntime(key, torch, tokenizer, model, device)
        if previous and previous.device.startswith("cuda"):
            del previous
            torch.cuda.empty_cache()
        return self._runtime

    @staticmethod
    def _default_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if payload.get("system_prompt"):
            messages.append({"role": "system", "content": str(payload["system_prompt"])})
        messages.append({"role": "user", "content": str(payload.get("prompt") or "")})
        return messages

    @staticmethod
    def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, bool]:
        if getattr(tokenizer, "chat_template", None):
            return (
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                True,
            )
        labels = {"system": "System", "user": "User", "assistant": "Assistant"}
        lines = [
            f"{labels.get(str(item.get('role')), str(item.get('role')).title())}: "
            f"{item.get('content', '')}"
            for item in messages
        ]
        lines.append("Assistant:")
        return "\n".join(lines), False

    @classmethod
    def _render_inference_prompt(
        cls, tokenizer: Any, payload: dict[str, Any]
    ) -> tuple[str, bool, str]:
        config = payload.get("config") or {}
        configured_mode = config.get("prompt_mode")
        # Tasks created before prompt_mode was added must retain their frozen
        # plaintext-role behavior when retried.
        if configured_mode is None:
            messages = payload.get("messages") or cls._default_messages(payload)
            rendered, used_chat_template = cls._render_prompt(tokenizer, messages)
            return rendered, used_chat_template, "chat"

        mode = str(configured_mode)
        if mode not in {"auto", "completion", "chat"}:
            raise ValueError(f"Unsupported prompt mode: {mode}")
        if mode == "completion" or (
            mode == "auto" and not getattr(tokenizer, "chat_template", None)
        ):
            return str(payload.get("prompt") or ""), False, "completion"

        messages = payload.get("messages") or cls._default_messages(payload)
        rendered, used_chat_template = cls._render_prompt(tokenizer, messages)
        return rendered, used_chat_template, "chat"
