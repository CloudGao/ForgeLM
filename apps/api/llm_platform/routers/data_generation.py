from copy import deepcopy

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.data_generation import DataGenerationCreate, FactTranslationCreate
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import models as m
from ..database import get_session
from ..services.artifacts import artifact_storage
from ..services.dataset_materialization import freeze_derived_materialization
from ..services.global_data import ensure_global_data_project, project_has_asset_access
from ..services.tasks import create_task
from ..services.transactions import commit_or_flush
from .common import add_audit, get_or_404

router = APIRouter(tags=["data-generation"])


def _continuation_binding(session, project_id, payload, mode):
    if not payload.checkpoint_source_task_id:
        return None
    original = get_or_404(session, m.Task, payload.checkpoint_source_task_id, "接续任务")
    attempt = get_or_404(session, m.TaskAttempt, payload.checkpoint_source_attempt_id, "接续尝试")
    if (mode != "fact_translation" or original.project_id != project_id
            or original.kind != "data_generate" or original.payload.get("generation", {}).get("mode") != mode
            or original.observed_state != "paused" or original.desired_state != "paused"
            or attempt.task_id != original.id or attempt.status != "paused" or not attempt.worker_id):
        raise HTTPException(409, "接续只接受同项目已安全暂停的三元组翻译任务与尝试")
    old = FactTranslationCreate.model_validate({k: v for k, v in original.payload["generation"].items() if k != "mode"})
    ignored = {"name", "record_range", "checkpoint_source_task_id", "checkpoint_source_attempt_id", "resource_request"}
    if old.model_dump(exclude=ignored) != payload.model_dump(exclude=ignored):
        raise HTTPException(409, "接续来源的输入、模型或生成协议不一致")
    if old.record_range or old.checkpoint_source_task_id:
        raise HTTPException(409, "已有分批任务请使用原任务恢复入口")
    return {"task": {"id": original.id, "payload": deepcopy(original.payload)},
            "attempt_id": attempt.id, "worker_id": attempt.worker_id}


def _resources(session, project_id, artifacts, requested, engine):
    request = deepcopy(requested)
    selector = dict(request.get("worker_selector") or {})
    eligible = None
    for artifact in artifacts:
        actor = (
            ensure_global_data_project(session).id
            if artifact.kind == "model_weights"
            else project_id
        )
        access = artifact_storage.inference_access(session, artifact, actor)
        if access["type"] == "unavailable":
            raise HTTPException(409, "生成输入没有可用的本地数据/模型位置")
        if access["type"] == "external" and not access["globally_accessible"]:
            workers = set(access["worker_ids"])
            eligible = workers if eligible is None else eligible & workers
    if eligible is not None:
        if selector.get("worker_id"):
            eligible &= {selector["worker_id"]}
        if selector.get("worker_ids"):
            eligible &= set(selector["worker_ids"])
        if not eligible:
            raise HTTPException(409, "没有 Worker 能同时读取所选数据和模型权重")
        selector["worker_ids"] = sorted(eligible)
    selector["capabilities"] = list(
        dict.fromkeys(
            [
                *(selector.get("capabilities") or []),
                f"builtin.data-generation.{engine}",
            ]
        )
    )
    request.update(worker_selector=selector, gpus=1)
    request.setdefault("cpus", 2)
    return request


def _create(project_id, payload, session, mode):
    get_or_404(session, m.Project, project_id, "项目")
    source = get_or_404(session, m.DatasetVersion, payload.input_dataset_version_id, "数据版本")
    if not project_has_asset_access(session, project_id, "dataset_version", source.id):
        raise HTTPException(422, "请选择项目已拥有或绑定的数据版本")
    dataset = session.get(m.Dataset, source.dataset_id)
    artifact = session.get(m.Artifact, source.artifact_id) if source.artifact_id else None
    if (
        source.state != "ready"
        or (not source.train_ready and not (source.source_snapshot.get("generation_empty") is True
                and source.document_count == 0 and mode == "knowledge"))
        or not artifact
        or artifact.kind != "dataset_snapshot"
    ):
        raise HTTPException(409, "输入必须为已就绪的可训练文本快照，请先转换原始文档")
    weight = get_or_404(
        session, m.ModelWeightVersion, payload.model_weight_version_id, "本地权重版本"
    )
    weight_artifact = session.get(m.Artifact, weight.artifact_id) if weight.artifact_id else None
    if weight.state != "ready" or not weight_artifact or weight_artifact.kind != "model_weights":
        raise HTTPException(409, "请先在权重中心准备本地模型")
    continuation = _continuation_binding(session, project_id, payload, mode)
    requested = deepcopy(payload.resource_request)
    if continuation:
        selector = requested.setdefault("worker_selector", {})
        if selector.get("worker_id") and selector["worker_id"] != continuation["worker_id"]:
            raise HTTPException(409, "断点接续必须使用来源 Worker")
        selector["worker_id"] = continuation["worker_id"]
    if payload.record_range and payload.record_range.logical_file not in [f["path"] for f in source.source_snapshot.get("files", [])]:
        raise HTTPException(422, "记录范围文件不在输入版本快照内")
    resources = _resources(session, project_id, [artifact, weight_artifact], requested, payload.engine)
    frozen = {**payload.model_dump(), "mode": mode}
    # Keep original input and create a new globally discoverable derived asset.
    output_dataset = session.scalar(
        select(m.Dataset).where(m.Dataset.project_id == project_id, m.Dataset.name == payload.name)
    )
    if output_dataset is not None and output_dataset.source_type != "generated":
        raise HTTPException(409, "输出名称已被其他数据集使用，请更换名称")
    if output_dataset is None:
        output_dataset = m.Dataset(
            project_id=project_id,
            name=payload.name,
            source_type="generated",
            language="zh",
            license=dataset.license,
            status="queued",
            source_config={"generation": frozen},
        )
        session.add(output_dataset)
        session.flush()
    version_number = (
        int(
            session.scalar(
                select(func.coalesce(func.max(m.DatasetVersion.version), 0)).where(
                    m.DatasetVersion.dataset_id == output_dataset.id
                )
            )
        )
        + 1
    )
    output = m.DatasetVersion(
        dataset_id=output_dataset.id,
        version=version_number,
        tier="bronze",
        state="queued",
        parent_version_id=source.id,
        train_ready=False,
        source_snapshot={"generation": frozen},
    )
    session.add(output)
    session.flush()
    task = create_task(
        session,
        project_id=project_id,
        kind="data_generate",
        title=f"{ {'translation': '数据翻译', 'fact_translation': '三元组翻译', 'knowledge': '知识数据生成'}[mode]}：{payload.name}",
        adapter_id=f"builtin.data-generation.{payload.engine}",
        launcher_id="builtin.local-process",
        payload={
            "dataset_id": output_dataset.id,
            "dataset_version_id": output.id,
            "output_dataset_version_id": output.id,
            "input_dataset_version_id": source.id,
            "model_weight_version_id": weight.id,
            "generation": frozen,
            **({"generation_continuation": continuation} if continuation else {}),
            "generation_input": {
                "artifact_id": artifact.id,
                "artifact_hash": artifact.content_hash,
                "dataset_version_id": source.id,
                "logical_files": [f["path"] for f in (source.source_snapshot.get("files") or [])],
            },
            "generation_model": {
                "artifact_id": weight_artifact.id,
                "content_hash": weight_artifact.content_hash,
                "identity_hash": weight.identity_hash,
                "model_weight_version_id": weight.id,
            },
            "resource_request": resources,
            "command": {},
        },
    )
    output.task_id = task.id
    if not output_dataset.current_version_id:
        output_dataset.current_version_id = output.id
    freeze_derived_materialization(output, task)
    # Stochastic model output cannot be rebuilt byte-for-byte. Retain/copy it.
    output.materialization_spec = {
        "schema_version": 1,
        "kind": "derived",
        "source_task_id": task.id,
        "parent_dataset_version_id": source.id,
        "generation": frozen,
    }
    add_audit(
        session,
        project_id=project_id,
        action="data_generation.create",
        subject_type="dataset_version",
        subject_id=output.id,
        payload=task.payload,
    )
    commit_or_flush(session)
    return {
        "output_dataset_version_id": output.id,
        "task_id": task.id,
        "dataset_id": output_dataset.id,
    }


@router.post("/projects/{project_id}/data-translation-runs", status_code=202)
def create_data_translation(
    project_id: str, payload: DataGenerationCreate, session: Session = Depends(get_session)
):
    return _create(project_id, payload, session, "translation")


@router.post("/projects/{project_id}/fact-translation-runs", status_code=202)
def create_fact_translation(
    project_id: str, payload: FactTranslationCreate, session: Session = Depends(get_session)
):
    return _create(project_id, payload, session, "fact_translation")


@router.post("/projects/{project_id}/knowledge-generation-runs", status_code=202)
def create_knowledge_generation(
    project_id: str, payload: DataGenerationCreate, session: Session = Depends(get_session)
):
    return _create(project_id, payload, session, "knowledge")


def create_generation_collection(project_id, payload, session):
    """Collect committed batches using the existing immutable mixture protocol."""
    from ..schemas import MixtureCreate
    from .data import create_mixture

    entries, empty = [], []
    for entry in payload.entries:
        version = get_or_404(session, m.DatasetVersion, entry.get("dataset_version_id"), "批次版本")
        if not project_has_asset_access(session, project_id, "dataset_version", version.id):
            raise HTTPException(422, "批次版本不属于项目或尚未绑定")
        producer = session.get(m.Task, version.task_id) if version.task_id else None
        if version.state != "ready" or not producer or producer.kind != "data_generate":
            raise HTTPException(409, "只可汇集已提交的生成批次")
        if version.document_count == 0 and version.source_snapshot.get("generation_empty") is True:
            empty.append(version.id)
        else:
            entries.append({"dataset_version_id": version.id, "weight": max(version.token_estimate or 0, 1)})
    if not entries:
        return {"empty": True, "empty_dataset_version_ids": empty}
    result = create_mixture(project_id, MixtureCreate(
        name=payload.name, description=payload.description, entries=entries, target_tokens=0, seed=payload.seed,
    ), session)
    return {**result, "empty_dataset_version_ids": empty}
