from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from llm_platform_core.model_runtime import build_frozen_model, load_frozen_model_spec


def _write_spec(path: Path, **overrides) -> dict:
    specification = {
        "schema_version": "forgelm.model-spec.v2",
        "source_type": "custom_code",
        "architecture": {"family_id": "custom_code", "family_version": "2"},
        "config": {"hidden_size": 16},
        "initialization": {"type": "random"},
        "parameters": {"total_parameters": 1},
        "runtime_requirements": {"contract_version": 2},
        **overrides,
    }
    payload = json.dumps(
        specification, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    specification["config_signature"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(specification), encoding="utf-8")
    return specification


def test_custom_model_factory_builds_and_emits_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    implementation = tmp_path / "implementation"
    implementation.mkdir()
    (implementation / "model_factory.py").write_text(
        "def build_model(config, initialization):\n"
        "    return {'config': config, 'initialization': initialization}\n",
        encoding="utf-8",
    )
    spec_path = tmp_path / "model-spec.json"
    specification = _write_spec(spec_path)
    monkeypatch.setenv("LLM_PLATFORM_MODEL_SPEC_PATH", str(spec_path))
    monkeypatch.setenv("LLM_PLATFORM_MODEL_IMPLEMENTATION_DIR", str(implementation))
    monkeypatch.setenv("LLM_PLATFORM_MODEL_FACTORY_ENTRYPOINT", "model_factory.py")

    model, loaded = build_frozen_model()

    assert model["config"] == {"hidden_size": 16}
    assert loaded == specification
    event = json.loads(capsys.readouterr().out)
    assert event == {
        "event": "model.loaded",
        "config_signature": specification["config_signature"],
        "architecture_family_id": "custom_code",
    }


def test_frozen_model_signature_tampering_is_rejected(tmp_path: Path) -> None:
    spec_path = tmp_path / "model-spec.json"
    specification = _write_spec(spec_path)
    specification["config"]["hidden_size"] = 32
    spec_path.write_text(json.dumps(specification), encoding="utf-8")

    with pytest.raises(ValueError, match="signature verification failed"):
        load_frozen_model_spec(spec_path)
