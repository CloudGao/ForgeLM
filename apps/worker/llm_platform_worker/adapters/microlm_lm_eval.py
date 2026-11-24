from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .lm_eval import builtin_lm_eval_task_path
from .microlm_inference import _load_model_class, microlm_inference_available

ProgressCallback = Callable[[int, int, str], None]


def _progress_tqdm(base_tqdm: Any, callback: ProgressCallback | None) -> Any:
    """Wrap the HFLM request progress bars without copying lm-eval internals."""

    class ProgressTqdm(base_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            description = str(kwargs.get("desc") or "")
            self._forgelm_operation = description
            self._forgelm_total = int(kwargs.get("total") or 0)
            self._forgelm_completed = 0
            self._forgelm_tracked = description.startswith("Running ")
            super().__init__(*args, **kwargs)

        def update(self, n: int | None = 1) -> Any:
            increment = int(n or 0)
            result = super().update(n)
            if self._forgelm_tracked and self._forgelm_total > 0 and callback is not None:
                self._forgelm_completed = min(
                    self._forgelm_total, self._forgelm_completed + increment
                )
                callback(
                    self._forgelm_completed,
                    self._forgelm_total,
                    self._forgelm_operation,
                )
            return result

    return ProgressTqdm


def _model_path(model_args: str) -> Path:
    values = {}
    for part in model_args.split(","):
        key, separator, value = part.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    pretrained = values.get("pretrained")
    if not pretrained:
        raise ValueError("microlm-native model_args must contain pretrained")
    path = Path(pretrained).expanduser().resolve()
    if not path.is_dir():
        raise ValueError("microlm-native pretrained path is not a materialized model directory")
    return path


def load_harness(
    model_args: str,
    *,
    batch_size: int | str,
    device: str,
    dtype_name: str = "auto",
) -> Any:
    if not microlm_inference_available():
        raise RuntimeError("MicroLM inference dependencies are missing on this Worker")
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import PreTrainedTokenizerFast

    class MicroLMHarness(HFLM):
        def _detect_batch_size(self, requests: Any = None, pos: int = 0) -> int:
            detected = int(super()._detect_batch_size(requests, pos))
            sizes = getattr(self, "forgelm_detected_batch_sizes", set())
            sizes.add(detected)
            self.forgelm_detected_batch_sizes = sizes
            return detected

        def _model_generate(
            self,
            context: Any,
            max_length: int,
            stop: list[str],
            **generation_kwargs: Any,
        ) -> Any:
            del stop
            do_sample = bool(generation_kwargs.get("do_sample", False))
            temperature = float(generation_kwargs.get("temperature", 0.0) or 0.0)
            generated = context
            current = context
            past_key_values = None
            eos_ids = generation_kwargs.get("eos_token_id", self.tokenizer.eos_token_id)
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            eos_set = {int(token_id) for token_id in (eos_ids or [])}
            with torch.inference_mode():
                while generated.shape[1] < max_length:
                    output = self.model(
                        current,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
                    logits = output.logits[:, -1, :].float()
                    if do_sample and temperature > 0:
                        next_token = torch.multinomial(
                            torch.softmax(logits / temperature, dim=-1), num_samples=1
                        )
                    else:
                        next_token = logits.argmax(dim=-1, keepdim=True)
                    generated = torch.cat((generated, next_token), dim=-1)
                    if all(
                        int(token_id) in eos_set
                        for token_id in next_token.view(-1).tolist()
                    ):
                        break
                    current = next_token
            return generated

    model_path = _model_path(model_args)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this Worker")
    if dtype_name == "auto":
        dtype_name = (
            "bfloat16"
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else "float16"
            if device == "cuda"
            else "float32"
        )
    dtype = getattr(torch, dtype_name, None)
    if dtype is None or dtype_name not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported MicroLM evaluation dtype: {dtype_name}")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    model_class = _load_model_class()
    model = model_class.from_pretrained(model_path, map_location="cpu")
    model.to(device=torch.device(device), dtype=dtype)
    model.eval()
    # Full-sequence log-likelihood scoring does not consume KV cache. Keeping it
    # enabled here makes every layer allocate a cache that HFLM immediately drops.
    # Generation explicitly enables cache in _model_generate above.
    model.config.use_cache = False
    model.device = torch.device(device)
    model.name_or_path = str(model_path)
    return MicroLMHarness(
        pretrained=model,
        backend="causal",
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=int(model.config.max_position_embeddings),
        add_bos_token=False,
    )


def evaluate_task(
    harness: Any,
    benchmark: dict[str, Any],
    inference_config: dict[str, Any],
    scoring_config: dict[str, Any],
    output_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[int]:
    from lm_eval import simple_evaluate
    from lm_eval.models import huggingface as huggingface_module
    from lm_eval.tasks import TaskManager

    task_id = str(benchmark.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("benchmark task_id is required")
    execution = benchmark.get("execution") or {}
    seed_values = [0, 1234, 1234, 1234]
    raw_seed = inference_config.get("seed")
    if raw_seed not in (None, ""):
        parsed = [int(value.strip()) for value in str(raw_seed).split(",") if value.strip()]
        if len(parsed) == 1:
            seed_values = [parsed[0]] * 4
        elif len(parsed) == 4:
            seed_values = parsed
    original_tqdm = huggingface_module.tqdm
    huggingface_module.tqdm = _progress_tqdm(original_tqdm, progress_callback)
    try:
        result = simple_evaluate(
            model=harness,
            tasks=[task_id],
            task_manager=TaskManager(include_path=builtin_lm_eval_task_path()),
            num_fewshot=execution.get("num_fewshot"),
            batch_size=inference_config.get("batch_size"),
            limit=(
                None
                if inference_config.get("sampling")
                else inference_config.get("limit") or None
            ),
            samples=inference_config.get("_sample_plan"),
            log_samples=bool(scoring_config.get("save_samples", True)),
            random_seed=seed_values[0],
            numpy_random_seed=seed_values[1],
            torch_random_seed=seed_values[2],
            fewshot_random_seed=seed_values[3],
        )
    finally:
        huggingface_module.tqdm = original_tqdm
    if result is None:
        raise RuntimeError(f"lm-eval returned no result for {task_id}")
    document = {
        key: result.get(key, {})
        for key in ("results", "groups", "group_subtasks", "configs", "versions", "n-shot")
        if key in result
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if scoring_config.get("save_samples", True):
        samples = result.get("samples") or {}
        with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for rows in samples.values():
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return sorted(
        int(value) for value in getattr(harness, "forgelm_detected_batch_sizes", set())
    )
