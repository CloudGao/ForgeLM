from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from llm_platform_worker.adapters import data_juicer as module
from llm_platform_worker.adapters.data_juicer import (
    MAX_PROCESS_LOG_CHARS,
    DataJuicerAdapter,
    _forward_process_stream,
    _validated_export_files,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("with_selection", [False, True])
async def test_data_juicer_adapter_generates_controlled_config_and_analyzes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_selection: bool
) -> None:
    workspace = tmp_path / "workspace"
    input_dir = tmp_path / "input"
    run_dir = workspace / ".llm-platform" / "runs" / "task" / "attempt"
    workspace.mkdir()
    input_dir.mkdir()
    run_dir.mkdir(parents=True)
    (input_dir / "bronze.jsonl").write_text(
        '{"text":"first"}\n{"text":"second"}\n', encoding="utf-8"
    )
    selected_source = input_dir / "bronze.jsonl"
    file_selection = {"schema": "forgelm-processing-file-selection-v1", "files": [{
        "path": selected_source.name, "size_bytes": selected_source.stat().st_size,
        "sha256": hashlib.sha256(selected_source.read_bytes()).hexdigest()}]}
    if with_selection:
        (input_dir / "excluded.jsonl").write_text('{"text":"excluded"}\n', encoding="utf-8")
    monkeypatch.setattr(module, "data_juicer_executable", lambda: "dj-process")
    monkeypatch.setattr(module, "data_juicer_version", lambda: "1.5.3")
    monkeypatch.setattr(
        "llm_platform_worker.resource_capacity.effective_capacity",
        lambda: {"cpu_count": 25, "memory_available_bytes": 84 * 1024**3},
    )
    captured: dict = {}

    async def fake_process(executable, config_path, cwd, emit, get_desired_state):
        del emit, get_desired_state
        captured["executable"] = executable
        captured["cwd"] = cwd
        captured["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        export_path = Path(captured["config"]["export_path"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"text": ["第一条清洗正文", "第二条清洗正文"]}), export_path)
        export_path.with_name("dataset_stats.jsonl").write_text(
            '{"text":"统计侧文件不得混入训练正文"}\n', encoding="utf-8"
        )
        return None

    monkeypatch.setattr(DataJuicerAdapter, "_run_process", staticmethod(fake_process))
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> str:
        events.append((event_type, payload))
        return "running"

    result = await DataJuicerAdapter().run(
        {
            "project_id": "project-1",
            "payload": {
                "processing_run_id": "run-1",
                "input_dataset_version_id": "version-1",
                "output_dataset_version_id": "version-2",
                "data_processing_input": {
                    "runtime_path": str(input_dir),
                    "artifact_hash": "a" * 64,
                    "document_count": 2,
                    "token_estimate": 12,
                    **({"file_selection": file_selection} if with_selection else {}),
                },
                "recipe": {
                    "recipe_version_id": "recipe-version-1",
                    "operators": [{"whitespace_normalization_mapper": {}}],
                    "execution_config": {"np": "auto"},
                },
            },
            "runtime_context": {
                "run_dir": str(run_dir),
                "attempt_dir": str(run_dir),
                "project_workspace": str(workspace),
                "worker_id": "worker-1",
            },
        },
        emit,
        lambda: "running",
    )

    assert captured["executable"] == "dj-process"
    if with_selection:
        assert captured["config"]["dataset_path"] == ""
        assert captured["config"]["dataset"] == {"configs": [{"type": "local", "path": str(selected_source)}]}
        assert result["source_snapshot"]["input_selection"] == file_selection
    else:
        assert captured["config"]["dataset_path"] == str(input_dir.resolve())
    assert captured["config"]["np"] == 8
    assert captured["config"]["export_shard_size"] == 0
    assert captured["config"]["export_in_parallel"] is False
    assert captured["config"]["export_extra_args"] == {"export_stats": False}
    assert captured["config"]["process"] == [{"whitespace_normalization_mapper": {"num_proc": 25}}]
    assert "text_keys" not in captured["config"]
    assert Path(captured["config"]["export_path"]).is_relative_to(
        run_dir
    )
    assert result["engine_version"] == "1.5.3"
    assert result["document_count"] == 2
    assert result["processing_summary"]["document_delta"] == 0
    assert result["processing_summary"]["filter_rate"] == 0
    assert [item["path"] for item in result["source_snapshot"]["files"]] == [
        "dataset.parquet"
    ]
    outputs = {item["ref"]: item for item in result["outputs"]}
    assert outputs["dataset"]["storage_type"] == "external"
    assert outputs["record_manifest"]["kind"] == "dataset_record_manifest"
    assert outputs["processing_recipe"]["kind"] == "data_processing_recipe"
    assert any(payload.get("phase") == "data_juicer_completed" for _, payload in events)


def test_export_validation_rejects_missing_and_truncated_shards(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    export_path = tmp_path / "dataset.parquet"
    first = tmp_path / "dataset-00-of-02.parquet"
    second = tmp_path / "dataset-01-of-02.parquet"
    pq.write_table(pa.table({"text": ["中文正文"]}), first)
    with pytest.raises(ValueError, match="missing or inconsistent"):
        _validated_export_files(export_path)
    second.write_bytes(b"PAR1incomplete")
    with pytest.raises(pa.ArrowInvalid):
        _validated_export_files(export_path)
    pq.write_table(pa.table({"text": ["后续正文"]}), second)
    assert _validated_export_files(export_path) == [first, second]


def test_data_juicer_capability_requires_module_cli_and_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(module, "data_juicer_executable", lambda: "dj-process")
    monkeypatch.setattr(module, "data_juicer_version", lambda: "1.5.3")
    assert module.data_juicer_available() is True
    monkeypatch.setattr(module, "data_juicer_executable", lambda: None)
    assert module.data_juicer_available() is False


@pytest.mark.asyncio
async def test_data_juicer_adapter_allows_first_mapper_to_create_conversation_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    input_dir = tmp_path / "input"
    run_dir = workspace / ".llm-platform" / "runs" / "task" / "attempt"
    workspace.mkdir()
    input_dir.mkdir()
    run_dir.mkdir(parents=True)
    (input_dir / "conversations.jsonl").write_text(
        '{"conversations":[{"role":"user","content":"hello"},'
        '{"role":"assistant","content":"hi","reasoning_content":"hidden"}]}\n'
        '{"conversations":[{"role":"system","content":"be helpful","tools":"x"},'
        '{"role":"user","content":"question"},'
        '{"role":"assistant","content":"answer","tool_calls":[{"id":"1"}]}]}\n'
        '{"conversations":[{"role":"user","content":"one"},'
        '{"role":"user","content":"two"}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "data_juicer_executable", lambda: "dj-process")
    monkeypatch.setattr(module, "data_juicer_version", lambda: "1.5.5")
    captured: dict = {}

    async def fake_process(executable, config_path, cwd, emit, get_desired_state):
        del executable, cwd, emit, get_desired_state
        captured["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        projected_path = Path(captured["config"]["dataset_path"]) / "dataset.jsonl"
        captured["projected"] = [
            json.loads(line)
            for line in projected_path.read_text(encoding="utf-8").splitlines()
        ]
        export_path = Path(captured["config"]["export_path"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            '{"conversations":[],"text":"User: hello\\n\\nAssistant: hi",'
            '"query":"hello","response":"hi"}\n',
            encoding="utf-8",
        )
        return None

    monkeypatch.setattr(DataJuicerAdapter, "_run_process", staticmethod(fake_process))

    async def emit(event_type: str, payload: dict) -> str:
        del event_type, payload
        return "running"

    result = await DataJuicerAdapter().run(
        {
            "id": "task-2",
            "project_id": "project-1",
            "payload": {
                "processing_run_id": "run-2",
                "input_dataset_version_id": "version-1",
                "output_dataset_version_id": "version-2",
                "data_processing_input": {
                    "runtime_path": str(input_dir),
                    "artifact_hash": "b" * 64,
                    "document_count": 3,
                    "token_estimate": 8,
                },
                "recipe": {
                    "recipe_version_id": "recipe-version-2",
                    "operators": [
                        {
                            "agent_dialog_normalize_mapper": {
                                "messages_key": "conversations"
                            }
                        },
                        {"document_minhash_deduplicator": {"tokenization": "character"}},
                    ],
                    "execution_config": {"np": 1},
                },
            },
            "runtime_context": {
                "run_dir": str(run_dir),
                "attempt_dir": str(run_dir),
                "project_workspace": str(workspace),
                "worker_id": "worker-1",
            },
        },
        emit,
        lambda: "running",
    )

    assert captured["config"]["text_keys"] == ["conversations"]
    assert captured["config"]["dataset_path"] != str(input_dir.resolve())
    assert captured["projected"] == [
        {
            "__forgelm_source": {
                "logical_path": "conversations.jsonl",
                "record_index": 0,
            },
            "conversations": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        },
        {
            "__forgelm_source": {
                "logical_path": "conversations.jsonl",
                "record_index": 1,
            },
            "conversations": [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        },
    ]
    assert captured["config"]["process"][0] == {
        "agent_dialog_normalize_mapper": {
            "messages_key": "conversations",
            "text_key": "text",
            "turbo": True,
            "num_proc": 1,
        }
    }
    assert captured["config"]["process"][1]["document_minhash_deduplicator"] == {
        "tokenization": "character",
        "text_key": "text",
        "num_proc": 1,
    }
    assert result["document_count"] == 1
    assert result["processing_summary"]["conversation_projection"] == {
        "version": "role-content-v1",
        "accepted_record_count": 2,
        "rejected_record_count": 1,
        "rejection_reasons": {"assistant_missing": 1},
    }


def test_data_juicer_executable_prefers_unresolved_virtualenv_bin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    executable_name = "dj-process.exe" if module.os.name == "nt" else "dj-process"
    executable = venv_bin / executable_name
    python.write_text("", encoding="utf-8")
    executable.write_text("", encoding="utf-8")
    monkeypatch.setattr(module.os.sys, "executable", str(python))
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    assert module.data_juicer_executable() == str(executable)


@pytest.mark.asyncio
async def test_process_stream_coalesces_carriage_returns_and_bounds_long_lines() -> None:
    reader = module.asyncio.StreamReader(limit=64)
    reader.feed_data(("progress 1\rprogress 2\r" + "x" * (MAX_PROCESS_LOG_CHARS + 9)).encode())
    reader.feed_data(b"\ndone\n")
    reader.feed_eof()
    events: list[tuple[str, dict]] = []

    async def emit(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    await _forward_process_stream(reader, "stderr", emit)

    messages = [payload for event_type, payload in events if event_type == "log"]
    assert messages[-1]["message"] == "done"
    assert all(len(payload["message"]) <= MAX_PROCESS_LOG_CHARS for payload in messages)
    assert any(payload.get("continued") is True for payload in messages)
