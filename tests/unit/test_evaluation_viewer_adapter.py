import json
from types import SimpleNamespace

import pytest
from llm_platform_worker.adapters import evaluation_viewer as viewer
from llm_platform_worker.adapters.dataset_viewer import execute_dataset_query


def test_search_all_records_nested_values_and_literal_quotes(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [{"id": i, "messages": [{"content": "中文 O'Reilly 100%_" if i >= 105 else "other"}]} for i in range(120)]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    def query(search, offset=0):
        return execute_dataset_query(input_path=path, sql="SELECT * FROM dataset ORDER BY id", search=search, split="", offset=offset, limit=10, timeout_seconds=10, temp_dir=tmp_path / "temp")
    first = query("中文 o'reilly 100%_")
    assert first["total_rows"] == 15
    assert first["rows"][0][0] == 105
    assert first["has_more"]
    assert query("中文 o'reilly 100%_", 10)["rows"][0][0] == 115
    assert query("' OR 1=1 --")["total_rows"] == 0


@pytest.mark.asyncio
async def test_export_and_reuse_full_benchmark_snapshot(tmp_path, monkeypatch):
    import sys
    tasks = {name: SimpleNamespace(config=SimpleNamespace(test_split="test", validation_split=None), eval_docs=[{"question": f"{name}-{i}", "answer": i} for i in range(120)], VERSION=1) for name in ["alpha", "beta"]}
    class Manager:
        def __init__(self, **kwargs):
            pass
        def load(self, names):
            assert names == ["group"]
            return {"tasks": tasks}
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", SimpleNamespace(TaskManager=Manager))
    monkeypatch.setattr(viewer.metadata, "version", lambda _: "test-version")
    snapshot = tmp_path / "snapshot"
    viewer.export_benchmark_documents("group", snapshot, "identity")
    selection, manifest = viewer.snapshot_selection(snapshot, "identity", "beta")
    assert len(selection["path"].read_text().splitlines()) == 120
    assert len(manifest["tasks"]) == 2
    async def emit(*args):
        pass
    task = {"id": "page-2", "runtime_context": {"attempt_dir": str(tmp_path / "attempt")}, "payload": {"evaluation_snapshot": True, "evaluation_viewer_identity": "identity", "source_task_id": "root-query", "dataset_viewer_input": {"runtime_path": str(snapshot)}, "configuration": "beta", "split": "test", "sql": "SELECT * FROM dataset", "offset": 110, "limit": 10}}
    result = await viewer.EvaluationViewerAdapter().run(task, emit)
    assert result["total_rows"] == 120
    assert result["rows"][0] == ["beta-110", 110]
    assert result["source_task_id"] == "root-query"
    assert result["configurations"] == ["alpha", "beta"]
    assert result["split"] == "test"
    assert not result["labelable"]
    with pytest.raises(ValueError, match="不匹配"):
        viewer.snapshot_selection(snapshot, "different", "beta")
    with pytest.raises(ValueError, match="未知"):
        viewer.snapshot_selection(snapshot, "identity", "missing")
    selection["path"].write_text("changed")
    with pytest.raises(ValueError, match="校验"):
        viewer.snapshot_selection(snapshot, "identity", "beta")


@pytest.mark.asyncio
async def test_benchmark_loading_can_be_cancelled(tmp_path, monkeypatch):
    import asyncio
    class Process:
        returncode = None
        stdout = asyncio.StreamReader()
        def terminate(self):
            self.returncode = -15
            self.stdout.feed_eof()
        async def wait(self):
            return self.returncode
    process = Process()
    async def create(*args, **kwargs):
        return process
    async def emit(*args):
        pass
    monkeypatch.setattr(viewer.asyncio, "create_subprocess_exec", create)
    assert await viewer.load_benchmark_snapshot("task", tmp_path, "identity", emit, lambda: "cancelled") == "cancelled"
    assert process.returncode == -15
