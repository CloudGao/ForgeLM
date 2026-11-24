from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from llm_platform_worker.adapters.code_import import CodeImportAdapter


@pytest.mark.asyncio
async def test_local_code_import_creates_filtered_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.py").write_text("print('train')\n", encoding="utf-8")
    (source / "configs").mkdir()
    (source / "configs" / "train.json").write_text("{}", encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "ignored.py").write_text("pass", encoding="utf-8")
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result = await CodeImportAdapter().run(
        {
            "id": "task-1",
            "runtime_context": {"run_dir": str(tmp_path / "run")},
            "payload": {
                "source_type": "local",
                "source_uri": str(source),
                "source_config": {},
                "entrypoints": {"training": "train.py"},
            },
        },
        emit,
    )

    snapshot = Path(result["outputs"][0]["path"])
    assert (snapshot / "train.py").is_file()
    assert (snapshot / "configs" / "train.json").is_file()
    assert not (snapshot / ".venv").exists()
    assert result["source_snapshot"]["file_count"] == 2
    assert any(payload.get("phase") == "creating_snapshot" for _, payload in events)


@pytest.mark.asyncio
async def test_code_import_rejects_archive_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.py", "pass")

    async def emit(_event_type: str, _payload: dict) -> str:
        return "running"

    with pytest.raises(ValueError, match="Unsafe archive member"):
        await CodeImportAdapter().run(
            {
                "id": "task-unsafe",
                "runtime_context": {"run_dir": str(tmp_path / "run")},
                "payload": {
                    "source_type": "archive",
                    "source_uri": str(archive),
                    "source_config": {},
                },
            },
            emit,
        )
