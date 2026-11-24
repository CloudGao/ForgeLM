"""Local subprocess bridge; uses the existing Worker command/event protocol."""

import json
from pathlib import Path

from .command import CommandAdapter


def generation_capabilities():
    from importlib.util import find_spec

    result = []
    if find_spec("torch") and find_spec("transformers"):
        result.append("builtin.data-generation.transformers")
        if find_spec("vllm"):
            result.append("builtin.data-generation.vllm")
    return result


def generation_environment(task):
    """Build the isolated generator environment with WSL-safe vLLM defaults."""
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONIOENCODING": "utf-8",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        # vLLM 0.29's V2 runner requires CUDA UVA during buffer setup. CUDA on
        # WSL can report a GPU while withholding UVA, so use the compatible V1
        # runner. This does not switch away from the vLLM engine.
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        # The lean Worker image intentionally ships CUDA runtime libraries but
        # not nvcc. FlashInfer's sampler JIT-compiles on first use, whereas the
        # native vLLM sampler works without a CUDA toolkit.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        **((task["payload"].get("command") or {}).get("env") or {}),
    }


class DataGenerationAdapter:
    async def run(self, task, emit, get_desired_state=None):
        context = task["runtime_context"]
        attempt = Path(context["attempt_dir"])
        attempt.mkdir(parents=True, exist_ok=True)
        config_path = attempt / "data-generation-task.json"
        config_path.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
        command = {
            "executable": "__worker_python__",
            "args": ["-m", "llm_platform_worker.data_generation_runtime", str(config_path)],
            # Dataset analysis returns bounded previews in the final JSON event;
            # 64 KiB is too small for several whole Chinese articles.
            "max_output_frame_bytes": 8 * 1024 * 1024,
            "env": generation_environment(task),
        }
        # Preserve the device selection injected into command.env by the lease.
        delegated = {**task, "payload": {**task["payload"], "command": command}}
        return await CommandAdapter().run(delegated, emit, get_desired_state)
