from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def _load_spec() -> tuple[dict[str, Any], Path]:
    spec_path = Path(os.environ["LLM_PLATFORM_TASK_SPEC"]).resolve()
    value = json.loads(spec_path.read_text(encoding="utf-8"))
    task = value.get("task") or {}
    attempt_dir = Path(os.environ["LLM_PLATFORM_ATTEMPT_DIR"]).resolve()
    if not attempt_dir.is_dir():
        raise ValueError("Worker attempt directory does not exist")
    return task, attempt_dir


def _recipe(config: dict[str, Any], quantization_modifier: Any, gptq_modifier: Any) -> Any:
    common = {
        "targets": config.get("targets") or ["Linear"],
        "scheme": config["scheme"],
        "ignore": config.get("ignore") or ["lm_head"],
    }
    if config["algorithm"] == "RTN":
        if config["scheme"] != "FP8_DYNAMIC":
            raise ValueError("RTN v1 only supports FP8_DYNAMIC")
        return quantization_modifier(**common)
    if config["algorithm"] == "GPTQ":
        if config["scheme"] != "W4A16":
            raise ValueError("GPTQ v1 only supports W4A16")
        return gptq_modifier(
            **common,
            block_size=int(config.get("block_size", 128)),
            dampening_frac=float(config.get("dampening_frac", 0.01)),
        )
    raise ValueError("Unsupported quantization algorithm")


def main() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    task, attempt_dir = _load_spec()
    payload = task.get("payload") or {}
    model_snapshot = payload.get("model") or {}
    metadata = model_snapshot.get("metadata") or {}
    inference = metadata.get("inference") if isinstance(metadata.get("inference"), dict) else {}
    if str(inference.get("format") or "").startswith("microlm-"):
        raise ValueError("LLM Compressor does not support the MicroLM native format")
    source = Path(str(model_snapshot.get("runtime_source") or "")).resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise ValueError("LLM Compressor requires a materialized local Transformers directory")
    config = payload.get("config") or {}
    calibration = payload.get("calibration_data_ref") or {}
    output_dir = (attempt_dir / "quantized-model").resolve()
    output_dir.relative_to(attempt_dir)
    if output_dir.exists():
        raise FileExistsError("Quantized output directory already exists")

    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    _emit("progress", phase="loading_source_model", progress=0.05)
    model = AutoModelForCausalLM.from_pretrained(
        str(source), trust_remote_code=False, local_files_only=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), trust_remote_code=False, local_files_only=True
    )
    recipe = _recipe(config, QuantizationModifier, GPTQModifier)
    kwargs: dict[str, Any] = {
        "model": model,
        "recipe": recipe,
        "output_dir": str(output_dir),
    }
    if config.get("algorithm") == "GPTQ":
        samples = calibration.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("GPTQ requires frozen calibration text samples")
        max_sequence_length = int(config.get("max_sequence_length", 2048))
        encoded = tokenizer(
            samples,
            truncation=True,
            max_length=max_sequence_length,
            add_special_tokens=True,
        )
        kwargs.update(
            {
                "dataset": Dataset.from_dict(encoded),
                "num_calibration_samples": min(
                    len(samples), int(config.get("num_calibration_samples", len(samples)))
                ),
                "max_seq_length": max_sequence_length,
            }
        )
    _emit("progress", phase="quantizing", progress=0.15)
    oneshot(**kwargs)
    tokenizer.save_pretrained(output_dir)
    if not (output_dir / "config.json").is_file():
        raise RuntimeError("LLM Compressor did not produce a Transformers config.json")
    if not any(output_dir.glob("*.safetensors")):
        raise RuntimeError("LLM Compressor did not produce compressed SafeTensors weights")

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    _emit("progress", phase="smoke_loading_output", progress=0.9)
    AutoConfig.from_pretrained(str(output_dir), trust_remote_code=False, local_files_only=True)
    reloaded = AutoModelForCausalLM.from_pretrained(
        str(output_dir), trust_remote_code=False, local_files_only=True, device_map="auto"
    )
    inputs = tokenizer("Hello", return_tensors="pt")
    device = next(reloaded.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output = reloaded.generate(**inputs, max_new_tokens=1, do_sample=False)
    if output.shape[-1] <= inputs["input_ids"].shape[-1]:
        raise RuntimeError("Quantized model smoke generation produced no new token")
    elapsed = time.monotonic() - started
    _emit(
        "result",
        outputs=[
            {
                "path": str(output_dir),
                "kind": "quantized_model",
                "ref": "quantized_model",
                "media_type": "application/gzip",
                "metadata": {
                    "format": "compressed-transformers",
                    "algorithm": config.get("algorithm"),
                    "scheme": config.get("scheme"),
                },
            }
        ],
        metrics_before={"source_artifact_id": model_snapshot.get("artifact_id")},
        metrics_after={
            "algorithm": config.get("algorithm"),
            "scheme": config.get("scheme"),
            "smoke_generated_tokens": 1,
            "duration_seconds": round(elapsed, 3),
        },
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
