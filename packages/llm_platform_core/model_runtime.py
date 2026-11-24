"""Runtime contract for constructing a frozen ForgeLM model specification.

This module deliberately imports user model code only inside the training subprocess.
The control plane and Worker merely validate and materialize immutable inputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


def _canonical_signature(specification: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in specification.items() if key != "config_signature"}
    payload = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_frozen_model_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load and verify the immutable model contract materialized by a Worker."""

    raw_path = path or os.environ.get("LLM_PLATFORM_MODEL_SPEC_PATH")
    if not raw_path:
        raise RuntimeError("LLM_PLATFORM_MODEL_SPEC_PATH is not set")
    specification = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if specification.get("schema_version") != "forgelm.model-spec.v2":
        raise ValueError("build_frozen_model requires forgelm.model-spec.v2")
    expected = str(specification.get("config_signature") or "")
    actual = _canonical_signature(specification)
    if not expected or expected != actual:
        raise ValueError("Frozen model specification signature verification failed")
    return specification


def _load_custom_factory(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Custom model factory is missing: {path}")
    module_name = f"forgelm_model_factory_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot import custom model factory: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    factory = getattr(module, "build_model", None)
    if not callable(factory):
        raise AttributeError("model_factory entrypoint must export build_model(config, initialization)")
    return factory


def _build_standard_model(specification: dict[str, Any]):
    requirements = specification.get("runtime_requirements") or {}
    config_class_name = str(requirements.get("config_class") or "")
    if not config_class_name:
        raise ValueError("Standard model contract is missing config_class")
    try:
        import transformers
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "The Worker environment needs the 'inference-hf' dependencies to construct this model"
        ) from exc
    config_class = getattr(transformers, config_class_name, None)
    if config_class is None:
        raise RuntimeError(f"Installed Transformers does not provide {config_class_name}")
    config = config_class(**(specification.get("config") or {}))
    initialization = specification.get("initialization") or {}
    initialization_type = str(initialization.get("type") or "random")
    if initialization_type == "random":
        return AutoModelForCausalLM.from_config(config)
    if initialization_type == "pretrained":
        reference = str(initialization.get("reference") or "")
        if not reference:
            raise ValueError("Pretrained initialization requires a reference")
        return AutoModelForCausalLM.from_pretrained(
            reference,
            config=config,
            trust_remote_code=False,
        )
    runtime_reference = os.environ.get("LLM_PLATFORM_MODEL_INITIALIZATION_PATH")
    if not runtime_reference:
        raise RuntimeError(
            f"{initialization_type} initialization was not materialized by the Worker"
        )
    return AutoModelForCausalLM.from_pretrained(
        runtime_reference,
        config=config,
        trust_remote_code=False,
    )


def _build_custom_model(specification: dict[str, Any]):
    root_value = os.environ.get("LLM_PLATFORM_MODEL_IMPLEMENTATION_DIR")
    entrypoint_value = os.environ.get("LLM_PLATFORM_MODEL_FACTORY_ENTRYPOINT")
    if not root_value or not entrypoint_value:
        raise RuntimeError("Custom model implementation was not materialized by the Worker")
    root = Path(root_value).resolve()
    entrypoint = (root / entrypoint_value).resolve()
    try:
        entrypoint.relative_to(root)
    except ValueError as exc:
        raise ValueError("Custom model factory escapes its immutable code root") from exc
    factory = _load_custom_factory(entrypoint)
    return factory(
        dict(specification.get("config") or {}),
        dict(specification.get("initialization") or {}),
    )


def build_frozen_model(path: str | Path | None = None):
    """Construct the configured model and report the signature handshake on stdout.

    Returns ``(model, specification)`` so training harnesses can consume both the
    instantiated module and the immutable contract that created it.
    """

    specification = load_frozen_model_spec(path)
    source_type = str(specification.get("source_type") or "")
    model = (
        _build_custom_model(specification)
        if source_type == "custom_code"
        else _build_standard_model(specification)
    )
    event = {
        "event": "model.loaded",
        "config_signature": specification["config_signature"],
        "architecture_family_id": (
            (specification.get("architecture") or {}).get("family_id")
        ),
    }
    print(json.dumps(event, ensure_ascii=False), flush=True)
    return model, specification
