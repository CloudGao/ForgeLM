from __future__ import annotations

import json
from pathlib import Path

from llm_platform_worker.plugin_runners import (
    forgelm_native_export,
    local_directory_publish,
)


def _result(stdout: str) -> dict:
    events = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    return next(event for event in reversed(events) if event.get("event") == "result")


def test_native_exporter_builds_a_hashed_runtime_directory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "microlm_hybrid"}), encoding="utf-8"
    )
    (source / "model.safetensors").write_bytes(b"safe-weights")
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    spec = tmp_path / "task.json"
    spec.write_text(
        json.dumps(
            {
                "task": {
                    "payload": {
                        "model": {
                            "runtime_source": str(source),
                            "artifact_id": "source-artifact",
                            "artifact_content_hash": "a" * 64,
                            "format": "microlm-hybrid-safetensors",
                            "metadata": {
                                "inference": {"format": "microlm-hybrid-safetensors"}
                            },
                        },
                        "config": {"verify_smoke": False},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PLATFORM_TASK_SPEC", str(spec))
    monkeypatch.setenv("LLM_PLATFORM_ATTEMPT_DIR", str(attempt))
    forgelm_native_export.main()
    result = _result(capsys.readouterr().out)
    output = Path(result["outputs"][0]["path"])
    manifest = json.loads((output / "forgelm-runtime.json").read_text(encoding="utf-8"))
    assert result["runtime"]["format"] == "forgelm-native-model-directory-v1"
    assert len(manifest["payload_tree_hash"]) == 64
    assert manifest["smoke_test"]["status"] == "not_supported"
    assert (output / "model.safetensors").read_bytes() == b"safe-weights"


def test_local_publisher_is_root_bounded_and_idempotent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "config.json").write_text("{}", encoding="utf-8")
    (runtime / "model.safetensors").write_bytes(b"weights")
    (runtime / "forgelm-runtime.json").write_text(
        json.dumps({"schema": "forgelm-release-runtime-v1"}), encoding="utf-8"
    )
    release_root = tmp_path / "releases"
    release_root.mkdir()
    spec = tmp_path / "delivery.json"
    spec.write_text(
        json.dumps(
            {
                "task": {
                    "project_id": "project-1",
                    "payload": {
                        "model": {"runtime_source": str(runtime)},
                        "release": {
                            "id": "release-1",
                            "name": "MicroLM",
                            "version": "v1.0.0",
                        },
                        "artifact": {
                            "id": "artifact-1",
                            "content_hash": "b" * 64,
                        },
                        "config": {"target_subdirectory": "published-models"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PLATFORM_TASK_SPEC", str(spec))
    monkeypatch.setenv("LLM_PLATFORM_RELEASE_ROOT", str(release_root))
    local_directory_publish.main()
    first = _result(capsys.readouterr().out)["delivery"]
    destination = Path(first["uri"].removeprefix("file:///"))
    if not destination.is_dir():
        destination = release_root / "published-models" / "project-1" / "MicroLM" / "v1.0.0"
    assert destination.is_dir()
    assert first["idempotent"] is False
    assert Path(destination, "forgelm-delivery.json").is_file()

    local_directory_publish.main()
    second = _result(capsys.readouterr().out)["delivery"]
    assert second["idempotent"] is True
    assert second["immutable_locator"] == first["immutable_locator"]
