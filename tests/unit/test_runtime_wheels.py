from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock

import pytest
from llm_platform.services import runtime_wheels


def test_pinned_wheel_is_cached_and_requirements_use_remote_path(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    requirements = source / "requirements-cuda.txt"
    requirements.write_text(
        "torch==2.9.1\nflash-attn @ https://github.com/release/flash_attn-1%2Bcu13.whl\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_wheels, "credential_directory", lambda: tmp_path / "keys")
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("module.py", "# wheel payload")
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_bytes.return_value = [payload.getvalue()]
    stream = MagicMock(return_value=response)
    monkeypatch.setattr(runtime_wheels.httpx, "stream", stream)

    uploads = runtime_wheels.stage_runtime_wheels(requirements, tmp_path, "/tmp/staging", lambda _: None)
    assert uploads[0].read_text(encoding="utf-8") == (
        "torch==2.9.1\nflash-attn @ file:///tmp/staging/flash_attn-1+cu13.whl\n"
    )
    assert uploads[1].read_bytes() == payload.getvalue()
    runtime_wheels.stage_runtime_wheels(requirements, tmp_path, "/tmp/retry", lambda _: None)
    stream.assert_called_once()

    uploads[1].unlink()
    response.iter_bytes.side_effect = RuntimeError("download interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        runtime_wheels.stage_runtime_wheels(requirements, tmp_path, "/tmp/retry", lambda _: None)
    assert not list((tmp_path / "runtime-wheel-cache").rglob("*.partial"))


def test_rejects_encoded_filename_traversal(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_wheels, "credential_directory", lambda: tmp_path / "keys")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("bad @ https://github.com/%2Fescape.whl", encoding="utf-8")
    with pytest.raises(ValueError, match="filename"):
        runtime_wheels.stage_runtime_wheels(requirements, tmp_path, "/tmp/staging", lambda _: None)
