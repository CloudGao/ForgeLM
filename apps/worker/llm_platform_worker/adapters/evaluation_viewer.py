from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any

from .dataset_viewer import DatasetViewerAdapter, Emit, GetDesiredState
from .lm_eval import builtin_lm_eval_task_path

MANIFEST = "evaluation-viewer.json"
SNAPSHOT_FORMAT = "evaluation-viewer-v1"
LOAD_TIMEOUT_SECONDS = 600


def export_benchmark_documents(task_id: str, destination: Path, identity: str) -> None:
    """Export actual evaluation documents, using the same trusted task loader as evaluation."""
    from lm_eval.tasks import TaskManager

    loaded = TaskManager(include_path=builtin_lm_eval_task_path()).load([task_id])
    tasks = loaded.get("tasks") or {}
    if not tasks:
        raise ValueError(f"Benchmark {task_id} has no evaluation tasks")
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, task in sorted(tasks.items()):
        split = str(task.config.test_split or task.config.validation_split or "evaluation")
        filename = hashlib.sha256(name.encode()).hexdigest() + ".jsonl"
        documents = task.eval_docs
        count = 0
        digest = hashlib.sha256()
        with (destination / filename).open("wb") as output:
            for document in documents:
                if not isinstance(document, dict):
                    document = {"text": document}
                encoded = (json.dumps(document, ensure_ascii=False, default=str) + "\n").encode("utf-8")
                output.write(encoded)
                digest.update(encoded)
                count += 1
        if count:
            entries.append({"name": name, "split": split, "file": filename, "count": count, "sha256": digest.hexdigest(), "task_version": getattr(task, "VERSION", None)})
    if not entries:
        raise ValueError("Benchmark has no evaluation documents")
    (destination / MANIFEST).write_text(json.dumps({
        "format": SNAPSHOT_FORMAT, "identity": identity, "task_id": task_id,
        "framework_version": metadata.version("lm_eval"), "tasks": entries,
    }, ensure_ascii=False), encoding="utf-8")


async def load_benchmark_snapshot(task_id: str, destination: Path, identity: str, emit: Emit, get_desired_state: GetDesiredState | None) -> str | None:
    destination.mkdir(parents=True, exist_ok=True)
    await emit("progress", {"progress": 0.05, "phase": "loading_benchmark_documents"})
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "llm_platform_worker.adapters.evaluation_viewer", task_id, str(destination), identity,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    tail = bytearray()

    async def drain() -> None:
        assert process.stdout is not None
        while chunk := await process.stdout.read(4096):
            tail.extend(chunk)
            if len(tail) > 16000:
                del tail[:-16000]

    reader = asyncio.create_task(drain())
    deadline = time.monotonic() + LOAD_TIMEOUT_SECONDS
    try:
        while process.returncode is None:
            desired = get_desired_state() if get_desired_state else "running"
            if desired in {"paused", "cancelled"}:
                return desired
            if time.monotonic() > deadline:
                raise TimeoutError("加载 Benchmark 题目超过 10 分钟，请检查 Worker 的题库下载连接")
            try:
                await asyncio.wait_for(process.wait(), timeout=0.25)
            except TimeoutError:
                pass
        await asyncio.wait_for(asyncio.shield(reader), timeout=5)
        if process.returncode:
            raise RuntimeError("加载 Benchmark 题目失败：" + tail.decode("utf-8", errors="replace"))
        return None
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
        try:
            await asyncio.wait_for(reader, timeout=5)
        except TimeoutError:
            reader.cancel()


def snapshot_selection(root: Path, identity: str, configuration: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = [root / MANIFEST, *root.glob(f"*/{MANIFEST}")]
    manifest_path = next((path for path in manifests if path.is_file()), None)
    if manifest_path is None:
        raise ValueError("题目查询快照缺少清单")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != SNAPSHOT_FORMAT or manifest.get("identity") != identity:
        raise ValueError("题目查询快照与当前评测定义不匹配")
    entries = manifest["tasks"]
    selected = next((entry for entry in entries if entry["name"] == configuration), None)
    if selected is None:
        if configuration not in {"", "default"}:
            raise ValueError("未知 Benchmark 子任务")
        selected = entries[0]
    path = (manifest_path.parent / selected["file"]).resolve()
    if path.parent != manifest_path.parent.resolve() or not path.is_file():
        raise ValueError("题目查询快照文件不可用")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != selected["sha256"]:
            raise ValueError("题目查询快照内容校验失败")
    return {**selected, "path": path}, manifest


class EvaluationViewerAdapter:
    async def run(self, task: dict[str, Any], emit: Emit, get_desired_state: GetDesiredState | None = None) -> dict[str, Any]:
        task = deepcopy(task)
        payload = task.get("payload") or {}
        benchmark = payload.get("benchmark_viewer_input")
        fresh_snapshot = bool(benchmark)
        snapshot_root: Path | None = None
        identity = str(payload.get("evaluation_viewer_identity") or "")
        if fresh_snapshot:
            attempt_dir = Path(task["runtime_context"]["attempt_dir"]).resolve()
            snapshot_root = attempt_dir / "inputs" / "evaluation-viewer-snapshot"
            controlled = await load_benchmark_snapshot(str(benchmark["task_id"]), snapshot_root, identity, emit, get_desired_state)
            if controlled:
                return {"controlled_state": controlled}
        elif payload.get("evaluation_snapshot"):
            snapshot_root = Path(payload["dataset_viewer_input"]["runtime_path"]).resolve()

        selection = manifest = None
        if snapshot_root is not None:
            selection, manifest = await asyncio.to_thread(snapshot_selection, snapshot_root, identity, str(payload.get("configuration") or "default"))
            if payload.get("split") and payload["split"] != selection["split"]:
                raise ValueError("未知评测 Split，请选择此子任务的实际评测划分")
            payload["dataset_viewer_input"] = {"runtime_path": str(selection["path"])}
            payload["split"] = ""  # The manifest, not a generated filename, identifies the split.
        elif not (payload.get("dataset_viewer_input") or {}).get("runtime_path"):
            raise ValueError("评测集没有物化题目快照")
        result = await DatasetViewerAdapter().run(task, emit, get_desired_state)
        if result.get("controlled_state"):
            return result
        result["labelable"] = False
        if selection and manifest:
            result.update({
                "configuration": selection["name"], "configurations": [entry["name"] for entry in manifest["tasks"]],
                "split": selection["split"], "splits": [selection["split"]],
                "source_task_id": task["id"] if fresh_snapshot else payload["source_task_id"],
                "snapshot_metadata": {"task_id": manifest["task_id"], "framework_version": manifest["framework_version"], "sha256": selection["sha256"]},
            })
        if fresh_snapshot:
            result["outputs"] = [{"ref": "evaluation_viewer_snapshot", "kind": "dataset_snapshot", "path": str(snapshot_root), "metadata": {"purpose": "evaluation_viewer", "identity": identity}}]
        return result


if __name__ == "__main__":
    export_benchmark_documents(sys.argv[1], Path(sys.argv[2]), sys.argv[3])
