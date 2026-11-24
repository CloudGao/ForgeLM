from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    DataProcessingRecipe,
    DataProcessingRecipeVersion,
    DataProcessingRun,
    Dataset,
    DatasetVersion,
    Project,
    Task,
)
from ..schemas import (
    DataProcessingRecipeCreate,
    DataProcessingRecipeVersionCreate,
    DataProcessingRunCreate,
)
from ..services.artifacts import artifact_storage
from ..services.dataset_analyses import effective_supports_capability
from ..services.dataset_materialization import freeze_derived_materialization
from ..services.global_data import project_can_manage_asset
from ..services.tasks import create_task
from .common import add_audit, get_or_404

router = APIRouter(tags=["data-processing"])
TIER_ORDER = {"bronze": 0, "silver": 1, "gold": 2}


def _execution_config(value: dict) -> dict:
    unexpected = sorted(set(value) - {"np"})
    if unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_data_processing_config",
                "message": "首版 Data-Juicer 运行配置只接受 np",
                "fields": unexpected,
            },
        )
    if value.get("np", "auto") == "auto":
        return {"np": "auto"}
    try:
        processes = int(value["np"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_data_processing_np", "message": "np 必须是整数"},
        ) from exc
    if not 1 <= processes <= 256:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_data_processing_np", "message": "np 必须在 1–256 之间"},
        )
    return {"np": processes}


def _recipe_view(session: Session, recipe: DataProcessingRecipe) -> dict:
    versions = session.scalars(
        select(DataProcessingRecipeVersion)
        .where(DataProcessingRecipeVersion.recipe_id == recipe.id)
        .order_by(DataProcessingRecipeVersion.version.desc())
    ).all()
    current = (
        session.get(DataProcessingRecipeVersion, recipe.current_version_id)
        if recipe.current_version_id
        else None
    )
    return {
        **recipe.to_dict(),
        "current_version": current.to_dict() if current else None,
        "versions": [version.to_dict() for version in versions],
    }


def _run_view(session: Session, run: DataProcessingRun) -> dict:
    input_version = session.get(DatasetVersion, run.input_dataset_version_id)
    output_version = (
        session.get(DatasetVersion, run.output_dataset_version_id)
        if run.output_dataset_version_id
        else None
    )
    dataset = session.get(Dataset, input_version.dataset_id) if input_version else None
    recipe_version = session.get(DataProcessingRecipeVersion, run.recipe_version_id)
    recipe = (
        session.get(DataProcessingRecipe, recipe_version.recipe_id) if recipe_version else None
    )
    task = session.get(Task, run.task_id) if run.task_id else None
    return {
        **run.to_dict(),
        "dataset_name": dataset.name if dataset else None,
        "recipe_name": recipe.name if recipe else None,
        "recipe_version": recipe_version.to_dict() if recipe_version else None,
        "input_version": input_version.to_dict() if input_version else None,
        "output_version": output_version.to_dict() if output_version else None,
        "task": task.to_dict() if task else None,
    }


def _route_snapshot_request(
    session: Session,
    project_id: str,
    artifact: Artifact,
    resource_request: dict,
) -> dict:
    request = {**resource_request}
    access = artifact_storage.inference_access(session, artifact, project_id)
    if access["type"] == "unavailable":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_snapshot_unavailable",
                "message": "输入数据快照没有托管副本或可用的 Worker 位置",
            },
        )
    selector = dict(request.get("worker_selector") or {})
    capabilities = [str(value) for value in selector.get("capabilities") or []]
    selector["capabilities"] = list(
        dict.fromkeys([*capabilities, "builtin.data-juicer"])
    )
    if access["type"] == "external" and not access["globally_accessible"]:
        eligible = {str(value) for value in access["worker_ids"]}
        selected_worker_id = str(selector.get("worker_id") or "")
        if selected_worker_id:
            eligible &= {selected_worker_id}
        selected_worker_ids = {str(value) for value in selector.get("worker_ids") or []}
        if selected_worker_ids:
            eligible &= selected_worker_ids
        if not eligible:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "dataset_worker_location_mismatch",
                    "message": "所选 Worker 无法访问输入数据快照",
                },
            )
        if len(eligible) == 1:
            selector.pop("worker_ids", None)
            selector["worker_id"] = next(iter(eligible))
        else:
            selector.pop("worker_id", None)
            selector["worker_ids"] = sorted(eligible)
    request["worker_selector"] = selector
    request.setdefault("gpus", 0)
    return request


@router.get("/projects/{project_id}/data-processing-recipes")
def list_data_processing_recipes(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    recipes = session.scalars(
        select(DataProcessingRecipe)
        .where(DataProcessingRecipe.project_id == project_id)
        .order_by(DataProcessingRecipe.updated_at.desc())
    ).all()
    return [_recipe_view(session, recipe) for recipe in recipes]


@router.post("/projects/{project_id}/data-processing-recipes", status_code=201)
def create_data_processing_recipe(
    project_id: str,
    payload: DataProcessingRecipeCreate,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    recipe = DataProcessingRecipe(
        project_id=project_id,
        name=payload.name.strip(),
        description=payload.description.strip(),
    )
    session.add(recipe)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_data_processing_recipe", "message": "处理配方名称已存在"},
        ) from exc
    version = DataProcessingRecipeVersion(
        recipe_id=recipe.id,
        version=1,
        engine_adapter_id=payload.engine_adapter_id,
        operators=payload.operators,
        execution_config=_execution_config(payload.execution_config),
    )
    session.add(version)
    session.flush()
    recipe.current_version_id = version.id
    add_audit(
        session,
        project_id=project_id,
        action="data_processing_recipe.create",
        subject_type="data_processing_recipe",
        subject_id=recipe.id,
        payload={"version_id": version.id},
    )
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return _recipe_view(session, recipe)


@router.post("/data-processing-recipes/{recipe_id}/versions", status_code=201)
def create_data_processing_recipe_version(
    recipe_id: str,
    payload: DataProcessingRecipeVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    recipe = get_or_404(session, DataProcessingRecipe, recipe_id, "处理配方")
    version_number = int(
        session.scalar(
            select(func.coalesce(func.max(DataProcessingRecipeVersion.version), 0)).where(
                DataProcessingRecipeVersion.recipe_id == recipe.id
            )
        )
        or 0
    ) + 1
    version = DataProcessingRecipeVersion(
        recipe_id=recipe.id,
        version=version_number,
        engine_adapter_id=payload.engine_adapter_id,
        operators=payload.operators,
        execution_config=_execution_config(payload.execution_config),
    )
    session.add(version)
    session.flush()
    recipe.current_version_id = version.id
    add_audit(
        session,
        project_id=recipe.project_id,
        action="data_processing_recipe.version.create",
        subject_type="data_processing_recipe_version",
        subject_id=version.id,
        payload={"recipe_id": recipe.id, "version": version.version},
    )
    session.commit()
    return _recipe_view(session, recipe)


@router.get("/projects/{project_id}/data-processing-runs")
def list_data_processing_runs(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    runs = session.scalars(
        select(DataProcessingRun)
        .where(DataProcessingRun.project_id == project_id)
        .order_by(DataProcessingRun.created_at.desc())
    ).all()
    return [_run_view(session, run) for run in runs]


@router.post("/projects/{project_id}/data-processing-runs", status_code=202)
def create_data_processing_run(
    project_id: str,
    payload: DataProcessingRunCreate,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    input_version = get_or_404(
        session, DatasetVersion, payload.input_dataset_version_id, "输入数据版本"
    )
    from ..services.global_data import ensure_global_data_project, project_has_asset_access
    from ..services.processing_selection import freeze_processing_selection

    consumer_project_id = project_id
    if payload.asset_scope == "global" and not project_has_asset_access(
        session, consumer_project_id, "dataset_version", input_version.id
    ):
        raise HTTPException(422, "全局加工前，消费项目必须绑定来源数据版本")
    selection = freeze_processing_selection(session, consumer_project_id, input_version, payload.input_mixture_version_id)
    if payload.asset_scope == "global":
        project_id = ensure_global_data_project(session).id
    dataset = get_or_404(session, Dataset, input_version.dataset_id, "数据集")
    if not project_can_manage_asset(
        session, project_id, "dataset_version", input_version.id
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_processing_input",
                "message": "输入版本只能由全局数据中心或其原始项目继续处理",
            },
        )
    if input_version.state != "ready" or not effective_supports_capability(
        session, input_version, "text_pretraining"
    ):
        raise HTTPException(
            status_code=409,
            detail={"code": "processing_input_not_ready", "message": "输入数据版本尚不可处理"},
        )
    input_tier_order = TIER_ORDER.get(str(input_version.tier), -1)
    if TIER_ORDER[payload.output_tier] <= input_tier_order:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_processing_output_tier",
                "message": "输出层级必须高于输入数据版本",
            },
        )
    recipe_version = get_or_404(
        session,
        DataProcessingRecipeVersion,
        payload.recipe_version_id,
        "处理配方版本",
    )
    recipe = get_or_404(session, DataProcessingRecipe, recipe_version.recipe_id, "处理配方")
    if recipe.project_id != project_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_processing_recipe", "message": "处理配方不属于当前项目"},
        )
    if recipe_version.engine_adapter_id != "builtin.data-juicer":
        raise HTTPException(
            status_code=422,
            detail={"code": "unsupported_processing_engine", "message": "当前只支持 Data-Juicer"},
        )
    artifact = (
        session.get(Artifact, input_version.artifact_id) if input_version.artifact_id else None
    )
    if artifact is None or artifact.kind != "dataset_snapshot":
        raise HTTPException(
            status_code=409,
            detail={"code": "processing_input_snapshot_missing", "message": "输入数据快照不存在"},
        )
    resource_request = _route_snapshot_request(
        session, project_id, artifact, payload.resource_request
    )
    if selection:
        # A subset-derived output must never replace the full source Dataset's head.
        output_name = f"{dataset.name[:110]} selection {payload.input_mixture_version_id}"
        output_dataset = session.scalar(select(Dataset).where(
            Dataset.project_id == project_id, Dataset.name == output_name,
        ))
        source_config = {"parent_dataset_version_id": input_version.id,
                         "input_mixture_version_id": payload.input_mixture_version_id}
        if output_dataset is None:
            output_dataset = Dataset(project_id=project_id, name=output_name, source_type="derived",
                                     source_config=source_config, language=dataset.language,
                                     license=dataset.license, status="registered")
            session.add(output_dataset)
            session.flush()
        elif output_dataset.source_config != source_config:
            raise HTTPException(409, "选择输出的数据集名称已被不同来源占用")
        dataset = output_dataset
    next_version = int(
        session.scalar(
            select(func.coalesce(func.max(DatasetVersion.version), 0)).where(
                DatasetVersion.dataset_id == dataset.id
            )
        )
        or 0
    ) + 1
    config_snapshot = {
        "engine_adapter_id": recipe_version.engine_adapter_id,
        "recipe_id": recipe.id,
        "recipe_version_id": recipe_version.id,
        "recipe_version": recipe_version.version,
        "operators": recipe_version.operators,
        "execution_config": recipe_version.execution_config,
        "input_dataset_version_id": input_version.id,
        "input_dataset_version": input_version.version,
        "input_artifact_id": artifact.id,
        "input_artifact_hash": artifact.content_hash,
        "output_tier": payload.output_tier,
        "input_mixture_version_id": payload.input_mixture_version_id,
        "input_selection": selection,
        "consumer_project_id": consumer_project_id,
        "asset_scope": payload.asset_scope,
    }
    run = DataProcessingRun(
        project_id=project_id,
        recipe_version_id=recipe_version.id,
        input_dataset_version_id=input_version.id,
        output_tier=payload.output_tier,
        status="queued",
        config_snapshot=config_snapshot,
    )
    session.add(run)
    session.flush()
    output_version = DatasetVersion(
        dataset_id=dataset.id,
        version=next_version,
        tier=payload.output_tier,
        state="queued",
        parent_version_id=input_version.id,
        processing_recipe_version_id=recipe_version.id,
        processing_run_id=run.id,
        source_snapshot={
            "processing_input_version_id": input_version.id,
            "processing_recipe_version_id": recipe_version.id,
            "input_selection": selection,
        },
        train_ready=False,
    )
    session.add(output_version)
    session.flush()
    task = create_task(
        session,
        project_id=project_id,
        kind="data_process",
        title=f"处理数据：{dataset.name} V{input_version.version} → V{next_version}",
        adapter_id="builtin.data-juicer",
        launcher_id="builtin.local-process",
        payload={
            "processing_run_id": run.id,
            "dataset_id": dataset.id,
            "input_dataset_version_id": input_version.id,
            "input_mixture_version_id": payload.input_mixture_version_id,
            "consumer_project_id": consumer_project_id,
            "output_dataset_version_id": output_version.id,
            "data_processing_input": {
                "artifact_id": artifact.id,
                "artifact_hash": artifact.content_hash,
                "dataset_version_id": input_version.id,
                "document_count": selection["document_count"] if selection else input_version.document_count,
                "token_estimate": selection["token_estimate"] if selection else input_version.token_estimate,
                "file_selection": selection,
            },
            "recipe": config_snapshot,
            "resource_request": resource_request,
            "command": {},
        },
    )
    run.output_dataset_version_id = output_version.id
    run.task_id = task.id
    output_version.task_id = task.id
    freeze_derived_materialization(output_version, task)
    if consumer_project_id != project_id:
        from ..schemas import ProjectDataBindingCreate
        from ..services.transactions import domain_transaction
        from .catalog import create_project_data_binding

        with domain_transaction(session):
            create_project_data_binding(consumer_project_id, ProjectDataBindingCreate(
                asset_type="dataset_version", asset_version_id=output_version.id, purpose="training",
                alias=f"{dataset.name} 清洗输出",
            ), session)
    add_audit(
        session,
        project_id=project_id,
        action="data_processing_run.create",
        subject_type="data_processing_run",
        subject_id=run.id,
        payload={
            "input_dataset_version_id": input_version.id,
            "output_dataset_version_id": output_version.id,
            "recipe_version_id": recipe_version.id,
        },
    )
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return _run_view(session, run)
