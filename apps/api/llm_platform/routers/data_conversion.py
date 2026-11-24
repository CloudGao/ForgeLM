from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from llm_platform_core.conversion import (
    CONVERSION_ENGINE_VERSION,
    CONVERSION_OPERATOR_CATALOG,
)
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    DataConversionRecipe,
    DataConversionRecipeVersion,
    DataConversionRun,
    Dataset,
    DatasetVersion,
    Project,
    Task,
)
from ..schemas import (
    DataConversionRecipeCreate,
    DataConversionRecipeVersionCreate,
    DataConversionRunCreate,
)
from ..services.artifacts import artifact_storage
from ..services.dataset_materialization import freeze_derived_materialization
from ..services.global_data import project_can_manage_asset
from ..services.tasks import create_task
from .common import add_audit, get_or_404

router = APIRouter(tags=["data-conversion"])


def _execution_config(value: dict) -> dict:
    if value:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_data_conversion_config",
                "message": "首版数据转换没有配方级执行参数；请在算子参数中配置加工逻辑",
                "fields": sorted(value),
            },
        )
    return {}


def _recipe_view(session: Session, recipe: DataConversionRecipe) -> dict:
    versions = session.scalars(
        select(DataConversionRecipeVersion)
        .where(DataConversionRecipeVersion.recipe_id == recipe.id)
        .order_by(DataConversionRecipeVersion.version.desc())
    ).all()
    current = (
        session.get(DataConversionRecipeVersion, recipe.current_version_id)
        if recipe.current_version_id
        else None
    )
    return {
        **recipe.to_dict(),
        "current_version": current.to_dict() if current else None,
        "versions": [version.to_dict() for version in versions],
    }


def _run_view(session: Session, run: DataConversionRun) -> dict:
    input_version = session.get(DatasetVersion, run.input_dataset_version_id)
    output_version = (
        session.get(DatasetVersion, run.output_dataset_version_id)
        if run.output_dataset_version_id
        else None
    )
    dataset = session.get(Dataset, input_version.dataset_id) if input_version else None
    recipe_version = session.get(DataConversionRecipeVersion, run.recipe_version_id)
    recipe = session.get(DataConversionRecipe, recipe_version.recipe_id) if recipe_version else None
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
                "code": "conversion_input_snapshot_unavailable",
                "message": "输入数据快照没有托管副本或可用的 Worker 位置",
            },
        )
    selector = dict(request.get("worker_selector") or {})
    capabilities = [str(value) for value in selector.get("capabilities") or []]
    selector["capabilities"] = list(dict.fromkeys([*capabilities, "builtin.data-conversion"]))
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
                    "code": "conversion_worker_location_mismatch",
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


@router.get("/data-conversion/operators")
def list_data_conversion_operators() -> dict:
    return {
        "engine_adapter_id": "builtin.data-conversion",
        "engine_version": CONVERSION_ENGINE_VERSION,
        "operators": CONVERSION_OPERATOR_CATALOG,
    }


@router.get("/projects/{project_id}/data-conversion-recipes")
def list_data_conversion_recipes(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    recipes = session.scalars(
        select(DataConversionRecipe)
        .where(DataConversionRecipe.project_id == project_id)
        .order_by(DataConversionRecipe.updated_at.desc())
    ).all()
    return [_recipe_view(session, recipe) for recipe in recipes]


@router.post("/projects/{project_id}/data-conversion-recipes", status_code=201)
def create_data_conversion_recipe(
    project_id: str,
    payload: DataConversionRecipeCreate,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    recipe = DataConversionRecipe(
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
            detail={"code": "duplicate_data_conversion_recipe", "message": "转换配方名称已存在"},
        ) from exc
    version = DataConversionRecipeVersion(
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
        action="data_conversion_recipe.create",
        subject_type="data_conversion_recipe",
        subject_id=recipe.id,
        payload={"version_id": version.id},
    )
    session.commit()
    return _recipe_view(session, recipe)


@router.post("/data-conversion-recipes/{recipe_id}/versions", status_code=201)
def create_data_conversion_recipe_version(
    recipe_id: str,
    payload: DataConversionRecipeVersionCreate,
    session: Session = Depends(get_session),
) -> dict:
    recipe = get_or_404(session, DataConversionRecipe, recipe_id, "转换配方")
    version_number = (
        int(
            session.scalar(
                select(func.coalesce(func.max(DataConversionRecipeVersion.version), 0)).where(
                    DataConversionRecipeVersion.recipe_id == recipe.id
                )
            )
            or 0
        )
        + 1
    )
    version = DataConversionRecipeVersion(
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
        action="data_conversion_recipe.version.create",
        subject_type="data_conversion_recipe_version",
        subject_id=version.id,
        payload={"recipe_id": recipe.id, "version": version.version},
    )
    session.commit()
    return _recipe_view(session, recipe)


@router.get("/projects/{project_id}/data-conversion-runs")
def list_data_conversion_runs(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    get_or_404(session, Project, project_id, "项目")
    runs = session.scalars(
        select(DataConversionRun)
        .where(DataConversionRun.project_id == project_id)
        .order_by(DataConversionRun.created_at.desc())
    ).all()
    return [_run_view(session, run) for run in runs]


@router.post("/projects/{project_id}/data-conversion-runs", status_code=202)
def create_data_conversion_run(
    project_id: str,
    payload: DataConversionRunCreate,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    input_version = get_or_404(
        session, DatasetVersion, payload.input_dataset_version_id, "输入数据版本"
    )
    dataset = get_or_404(session, Dataset, input_version.dataset_id, "数据集")
    if not project_can_manage_asset(
        session, project_id, "dataset_version", input_version.id
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_conversion_input",
                "message": "输入版本只能由全局数据中心或其原始项目继续转换",
            },
        )
    if input_version.state != "ready":
        raise HTTPException(
            status_code=409,
            detail={"code": "conversion_input_not_ready", "message": "输入数据版本尚未完成导入"},
        )
    recipe_version = get_or_404(
        session,
        DataConversionRecipeVersion,
        payload.recipe_version_id,
        "转换配方版本",
    )
    recipe = get_or_404(session, DataConversionRecipe, recipe_version.recipe_id, "转换配方")
    if recipe.project_id != project_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_conversion_recipe", "message": "转换配方不属于当前项目"},
        )
    if recipe_version.engine_adapter_id != "builtin.data-conversion":
        raise HTTPException(
            status_code=422,
            detail={"code": "unsupported_conversion_engine", "message": "当前只支持内置数据转换器"},
        )
    artifact = (
        session.get(Artifact, input_version.artifact_id) if input_version.artifact_id else None
    )
    if artifact is None or artifact.kind != "dataset_snapshot":
        raise HTTPException(
            status_code=409,
            detail={"code": "conversion_input_snapshot_missing", "message": "输入数据快照不存在"},
        )
    resource_request = _route_snapshot_request(
        session, project_id, artifact, payload.resource_request
    )
    next_version = (
        int(
            session.scalar(
                select(func.coalesce(func.max(DatasetVersion.version), 0)).where(
                    DatasetVersion.dataset_id == dataset.id
                )
            )
            or 0
        )
        + 1
    )
    config_snapshot = {
        "engine_adapter_id": recipe_version.engine_adapter_id,
        "engine_version": CONVERSION_ENGINE_VERSION,
        "recipe_id": recipe.id,
        "recipe_version_id": recipe_version.id,
        "recipe_version": recipe_version.version,
        "operators": recipe_version.operators,
        "execution_config": recipe_version.execution_config,
        "input_dataset_version_id": input_version.id,
        "input_dataset_version": input_version.version,
        "input_artifact_id": artifact.id,
        "input_artifact_hash": artifact.content_hash,
        "output_tier": "bronze",
    }
    run = DataConversionRun(
        project_id=project_id,
        recipe_version_id=recipe_version.id,
        input_dataset_version_id=input_version.id,
        status="queued",
        config_snapshot=config_snapshot,
    )
    session.add(run)
    session.flush()
    output_version = DatasetVersion(
        dataset_id=dataset.id,
        version=next_version,
        tier="bronze",
        state="queued",
        parent_version_id=input_version.id,
        conversion_recipe_version_id=recipe_version.id,
        conversion_run_id=run.id,
        source_snapshot={
            "conversion_input_version_id": input_version.id,
            "conversion_recipe_version_id": recipe_version.id,
        },
        train_ready=False,
    )
    session.add(output_version)
    session.flush()
    task = create_task(
        session,
        project_id=project_id,
        kind="data_convert",
        title=f"转换数据：{dataset.name} V{input_version.version} → V{next_version}",
        adapter_id="builtin.data-conversion",
        launcher_id="builtin.local-process",
        payload={
            "conversion_run_id": run.id,
            "dataset_id": dataset.id,
            "input_dataset_version_id": input_version.id,
            "output_dataset_version_id": output_version.id,
            "data_conversion_input": {
                "artifact_id": artifact.id,
                "artifact_hash": artifact.content_hash,
                "dataset_version_id": input_version.id,
                "document_count": input_version.document_count,
                "token_estimate": input_version.token_estimate,
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
    add_audit(
        session,
        project_id=project_id,
        action="data_conversion_run.create",
        subject_type="data_conversion_run",
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
