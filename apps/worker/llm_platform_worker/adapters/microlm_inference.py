from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transformers_inference import TransformersInferenceAdapter

Emit = Callable[[str, dict[str, Any]], Awaitable[str]]



def _microlm_source_available() -> bool:
    try:
        if importlib.util.find_spec("microlm_306m_v3") is not None:
            return True
    except (ImportError, ValueError):
        pass
    return False


def microlm_inference_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("safetensors") is not None
        and _microlm_source_available()
    )


def _load_model_class() -> Any:
    if importlib.util.find_spec("microlm_306m_v3") is None:
        if not _microlm_source_available():
            raise RuntimeError(
                "MicroLM inference runtime is missing from this Worker installation"
            )
    from microlm_306m_v3 import MicroLMForCausalLM

    return MicroLMForCausalLM


@dataclass
class _LoadedRuntime:
    key: tuple[str, str, str]
    torch: Any
    tokenizer: Any
    model: Any
    device: str


class MicroLMInferenceAdapter:
    """Run cached autoregressive generation for ForgeLM's trusted MicroLM format."""

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
        runtime_identity = str(model_snapshot.get("runtime_identity") or model_snapshot.get("artifact_id") or source)
        load_started = time.perf_counter()
        previous_runtime = self._runtime
        runtime = self._load_runtime(
            source,
            runtime_identity,
            requested_device,
            requested_dtype,
        )
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

        rendered_prompt, used_chat_template, prompt_mode = (
            TransformersInferenceAdapter._render_inference_prompt(tokenizer, payload)
        )
        encoded = tokenizer(
            rendered_prompt,
            return_tensors="pt",
            add_special_tokens=not used_chat_template,
        )
        input_ids = encoded["input_ids"].to(runtime.device)
        input_tokens = int(input_ids.shape[-1])
        max_positions = int(getattr(model.config, "max_position_embeddings", 8192))
        if input_tokens >= max_positions:
            raise ValueError(
                f"Rendered prompt contains {input_tokens} tokens, exceeding the "
                f"MicroLM context limit of {max_positions}"
            )
        max_new_tokens = min(
            int(config.get("max_tokens", config.get("max_new_tokens", 512))),
            max_positions - input_tokens,
        )

        temperature = float(config.get("temperature", 0.7))
        top_p = float(config.get("top_p", 1.0))
        top_k = int(config.get("top_k", 50))
        repetition_penalty = float(config.get("repetition_penalty", 1.0))
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_set = {int(value) for value in (eos_ids or [])}

        if runtime.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(runtime.device)
            torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()
        from .inference_stream import TokenStream
        stream = TokenStream(self, tokenizer, started, input_tokens, payload.get("request_received_at") or task.get("created_at"))
        generated: list[int] = []
        seen = [int(value) for value in input_ids[0].detach().cpu().tolist()]
        past_key_values = None
        current_ids = input_ids
        prefix_tokens = 0
        scope = (task.get("project_id"), payload.get("session_id"), runtime.key)
        prefix = self._prefix
        self._prefix = None
        if (scope[0] and scope[1] and prefix and prefix[0] == scope
                and len(prefix[1]) < len(seen) and seen[:len(prefix[1])] == prefix[1]):
            prefix_tokens = len(prefix[1])
            past_key_values = prefix[2]
            current_ids = input_ids[:, prefix_tokens:]
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if self._desired_state() in {"cancelled", "paused"}:
                    break
                output = model(
                    current_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                logits = output.logits[:, -1, :].float()
                self._apply_repetition_penalty(
                    logits, seen, repetition_penalty, torch=torch
                )
                next_token = self._select_token(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    torch=torch,
                )
                token_id = int(next_token.item())
                generated.append(token_id)
                seen.append(token_id)
                stream.snapshot(generated)
                if token_id in eos_set:
                    break
                current_ids = next_token.view(1, 1)
        if runtime.device.startswith("cuda"):
            torch.cuda.synchronize(runtime.device)
        elapsed_seconds = max(time.perf_counter() - started, 1e-9)

        if generated and scope[0] and scope[1] and self._desired_state() == "running":
            self._prefix = (scope, seen[:-1], past_key_values)
        output_text = tokenizer.decode(generated, skip_special_tokens=True)
        stop_reason = "eos" if generated and generated[-1] in eos_set else "length"
        performance: dict[str, Any] = {
            "model_cache_hit": cache_hit,
            "file_cache_hit": model_snapshot.get("file_cache_hit"),
            "preparation_ms": model_snapshot.get("preparation_ms", 0),
            "load_ms": load_ms,
            "ttft_ms": stream.first_ms,
            "request_ttft_ms": stream.request_first_ms,
            "generation_ms": round(elapsed_seconds * 1000, 2),
            "tokens_per_second": round(len(generated) / elapsed_seconds, 2),
            "device": runtime.device,
            "runtime": "microlm-native",
            "prefix_tokens": prefix_tokens,
        }
        if runtime.device.startswith("cuda"):
            performance["peak_memory_mb"] = round(
                torch.cuda.max_memory_allocated(runtime.device) / 1024 / 1024, 2
            )
        return {
            "rendered_prompt": rendered_prompt,
            "prompt_mode": prompt_mode,
            "output": output_text,
            "token_ids": generated,
            "input_tokens": input_tokens,
            "output_tokens": len(generated),
            "stop_reason": stop_reason,
            "performance": performance,
        }

    def _load_runtime(
        self,
        source: str,
        runtime_identity: str,
        requested_device: str,
        requested_dtype: str,
    ) -> _LoadedRuntime:
        if not microlm_inference_available():
            raise RuntimeError(
                "MicroLM inference dependencies are missing; install the ForgeLM "
                "MicroLM Worker runtime and restart this Worker"
            )
        import torch
        from transformers import PreTrainedTokenizerFast

        device = requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this Worker")
        dtype_name = requested_dtype
        if dtype_name == "auto":
            dtype_name = (
                "bfloat16"
                if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else "float16"
                if device.startswith("cuda")
                else "float32"
            )
        dtype = getattr(torch, dtype_name, None)
        if dtype is None or dtype_name not in {"float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported inference dtype: {requested_dtype}")
        if device == "cpu" and dtype_name == "float16":
            raise ValueError("MicroLM float16 inference requires CUDA")

        key = (runtime_identity, device, dtype_name)
        if self._runtime and self._runtime.key == key:
            return self._runtime

        self.release_runtime()

        # Loading through AutoTokenizer also probes config.json and emits an alarming
        # unknown-model warning for this intentionally non-Transformers architecture.
        # The training artifact always ships tokenizer.json, so use the concrete fast
        # tokenizer loader and leave model configuration to the trusted MicroLM runtime.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(source)
        model_class = _load_model_class()
        model = model_class.from_pretrained(source, map_location="cpu")
        model.to(device=device, dtype=dtype)
        model.eval()
        model.config.use_cache = True
        self._runtime = _LoadedRuntime(key, torch, tokenizer, model, device)
        return self._runtime

    @staticmethod
    def _apply_repetition_penalty(
        logits: Any,
        token_ids: list[int],
        penalty: float,
        *,
        torch: Any,
    ) -> None:
        if penalty == 1.0 or not token_ids:
            return
        unique = torch.tensor(sorted(set(token_ids)), device=logits.device, dtype=torch.long)
        selected = logits.index_select(-1, unique)
        adjusted = torch.where(selected < 0, selected * penalty, selected / penalty)
        logits.index_copy_(-1, unique, adjusted)

    @staticmethod
    def _select_token(
        logits: Any,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        torch: Any,
    ) -> Any:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        scores = logits / temperature
        if top_k > 0:
            threshold = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1).values[
                :, -1:
            ]
            scores = scores.masked_fill(scores < threshold, float("-inf"))
        if top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf")).scatter(
                -1, sorted_indices, sorted_scores
            )
        probabilities = torch.softmax(scores, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)
