from __future__ import annotations

from copy import deepcopy
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models import Artifact, DatasetVersion, EvalDatasetVersion, Task
from ..schemas import EvaluationDatasetViewerQuery
from .artifacts import artifact_storage
from .benchmarks import canonical_hash
from .tasks import create_task


def create_evaluation_query(
    session: Session, *, project_id: str, name: str, payload: EvaluationDatasetViewerQuery,
    version: EvalDatasetVersion | None = None, benchmark: dict[str, Any] | None = None,
) -> Task:
    benchmark = deepcopy(benchmark or (version.benchmark_config if version else {}) or {})
    identity = canonical_hash({"version_id": version.id if version else None, "benchmark": benchmark})
    inputs: dict[str, Any] = {}
    snapshot = session.get(Artifact, version.artifact_id) if version and version.artifact_id else None
    source_task_id = ""
    if payload.source_task_id:
        source = session.get(Task, payload.source_task_id)
        if (source is None or source.project_id != project_id or source.observed_state != "succeeded"
                or source.adapter_id != "builtin.evaluation-viewer"
                or (source.payload or {}).get("evaluation_viewer_identity") != identity):
            raise HTTPException(409, detail={"code": "evaluation_query_source_mismatch", "message": "查询快照不属于当前评测集版本，或尚未成功"})
        artifacts = (source.result or {}).get("output_artifacts") or []
        output = next((item for item in artifacts if item.get("ref") == "evaluation_viewer_snapshot"), None)
        if not output:
            raise HTTPException(409, detail={"code": "evaluation_query_snapshot_missing", "message": "查询没有可复用的题目快照"})
        snapshot = session.get(Artifact, output.get("artifact_id"))
        if snapshot is None:
            raise HTTPException(409, detail={"code": "evaluation_query_snapshot_missing", "message": "查询题目快照不可用"})
        source_task_id = source.id
        inputs["evaluation_snapshot"] = True

    selector: dict[str, Any] = {"capabilities": ["builtin.evaluation-viewer"]}
    if snapshot is not None:
        if snapshot.kind != "dataset_snapshot" or snapshot.availability_state != "available":
            raise HTTPException(409, detail={"code": "snapshot_unavailable", "message": "评测题目快照当前不可用"})
        access = artifact_storage.inference_access(session, snapshot, project_id)
        if access["type"] == "unavailable":
            raise HTTPException(409, detail={"code": "snapshot_location_unavailable", "message": "没有 Worker 可以访问该评测题目快照"})
        if access["type"] == "external" and not access["globally_accessible"]:
            if not access["worker_ids"]:
                raise HTTPException(409, detail={"code": "snapshot_worker_unavailable", "message": "保存题目快照的 Worker 当前不可选"})
            selector["worker_ids"] = list(access["worker_ids"])
        logical_files: list[str] = []
        if version and not source_task_id:
            source_id = (version.source_config or {}).get("dataset_version_id")
            source_version = session.get(DatasetVersion, source_id) if source_id else None
            if source_version and source_version.artifact_id == snapshot.id:
                logical_files = [str(item["path"]) for item in (source_version.source_snapshot or {}).get("files", []) if isinstance(item, dict) and item.get("path")]
        inputs["dataset_viewer_input"] = {"artifact_id": snapshot.id, "artifact_hash": snapshot.content_hash, "logical_files": logical_files}
    elif benchmark:
        if version and version.artifact_id:
            raise HTTPException(409, detail={"code": "snapshot_unavailable", "message": "冻结题目快照不存在"})
        selector["capabilities"].append("builtin.lm-eval")
        inputs["benchmark_viewer_input"] = benchmark
    else:
        raise HTTPException(409, detail={"code": "evaluation_snapshot_required", "message": "此历史评测集没有题目文件，请导入完整题目后创建新版本"})

    return create_task(
        session, project_id=project_id, kind="data_query", title=f"查看评测集：{name}",
        adapter_id="builtin.evaluation-viewer", launcher_id="builtin.local-process",
        payload={
            **payload.model_dump(exclude={"source_task_id"}), **inputs,
            "eval_dataset_version_id": version.id if version else None,
            "evaluation_viewer_identity": identity, "source_task_id": source_task_id,
            "resource_request": {"gpus": 0, "cpus": 1, "worker_selector": selector}, "command": {},
        },
    )
