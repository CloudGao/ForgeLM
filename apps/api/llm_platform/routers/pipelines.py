from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Project,
    ProjectPipeline,
    ProjectPipelineInstance,
    ProjectPipelineRun,
)
from ..schemas import (
    ProjectPipelineCreate,
    ProjectPipelineDefinitionCreate,
    ProjectPipelineDraftUpdate,
    ProjectPipelineInstanceCreate,
    ProjectPipelineInstanceDraftUpdate,
    ProjectPipelineRunCreate,
)
from ..services.lineage import asset_lineage_view, run_lineage_view
from ..services.pipeline_dags import (
    PipelineDagError,
    activate_pipeline_instance,
    cancel_pipeline_run,
    create_imported_pipeline_execution,
    create_pipeline_definition,
    create_pipeline_instance,
    list_pipeline_definitions,
    list_pipeline_instances,
    list_pipeline_runs,
    pipeline_definition_view,
    pipeline_instance_view,
    pipeline_node_catalog,
    pipeline_run_view,
    publish_pipeline_definition,
    save_pipeline_draft,
    save_pipeline_instance_draft,
    start_pipeline_run,
    validate_pipeline_definition,
    validate_pipeline_instance,
)
from ..services.project_pipelines import (
    ProjectPipelineError,
    create_project_pipeline,
    list_project_pipeline_views,
    project_pipeline_options,
    project_pipeline_view,
)
from .common import add_audit, get_or_404

router = APIRouter(tags=["pipelines"])


def _user_project(session: Session, project_id: str) -> Project:
    project = get_or_404(session, Project, project_id, "项目")
    if project.system_kind:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _project_pipeline(
    session: Session, project_id: str, pipeline_id: str
) -> ProjectPipeline:
    pipeline = get_or_404(session, ProjectPipeline, pipeline_id, "Pipeline")
    if pipeline.project_id != project_id or pipeline.archived:
        raise HTTPException(status_code=404, detail="Pipeline 不存在")
    return pipeline


def _project_pipeline_run(
    session: Session, project_id: str, run_id: str
) -> ProjectPipelineRun:
    run = get_or_404(session, ProjectPipelineRun, run_id, "Pipeline 运行")
    if run.project_id != project_id:
        raise HTTPException(status_code=404, detail="Pipeline 运行不存在")
    return run


def _project_pipeline_instance(
    session: Session, project_id: str, instance_id: str
) -> ProjectPipelineInstance:
    instance = get_or_404(session, ProjectPipelineInstance, instance_id, "DAG 实例")
    if instance.project_id != project_id or instance.status == "archived":
        raise HTTPException(status_code=404, detail="DAG 实例不存在")
    return instance


def _instance_pipeline_run(
    session: Session,
    project_id: str,
    instance_id: str,
    run_id: str,
) -> ProjectPipelineRun:
    run = _project_pipeline_run(session, project_id, run_id)
    if run.instance_id != instance_id:
        raise HTTPException(status_code=404, detail="DAG 实例运行不存在")
    return run


def _raise_dag_error(exc: PipelineDagError) -> None:
    raise HTTPException(
        status_code=exc.status_code,
        detail={"code": exc.code, "message": exc.message},
    ) from exc


@router.get("/projects/{project_id}/pipelines")
def list_project_pipelines(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    _user_project(session, project_id)
    return list_project_pipeline_views(session, project_id)


@router.get("/projects/{project_id}/pipeline-options")
def get_project_pipeline_options(
    project_id: str, session: Session = Depends(get_session)
) -> dict:
    _user_project(session, project_id)
    return project_pipeline_options(session, project_id)


@router.post("/projects/{project_id}/pipelines", status_code=201)
def create_pipeline(
    project_id: str,
    payload: ProjectPipelineCreate,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    try:
        pipeline = create_project_pipeline(session, project_id, payload)
        imported_instance = create_imported_pipeline_execution(session, pipeline)
    except ProjectPipelineError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    add_audit(
        session,
        project_id=project_id,
        action="project_pipeline.create",
        subject_type="project_pipeline",
        subject_id=pipeline.id,
        payload={
            "name": pipeline.name,
            "objective": pipeline.objective,
            "current_revision_id": pipeline.current_revision_id,
        },
    )
    if imported_instance is not None:
        add_audit(
            session,
            project_id=project_id,
            action="pipeline_instance.import_existing_run",
            subject_type="project_pipeline_instance",
            subject_id=imported_instance.id,
            payload={
                "pipeline_id": pipeline.id,
                "definition_revision_id": imported_instance.definition_revision_id,
                "execution_owner": "domain_task",
            },
        )
    session.commit()
    return project_pipeline_view(session, pipeline)


@router.get("/projects/{project_id}/pipeline-node-catalog")
def get_pipeline_node_catalog(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    _user_project(session, project_id)
    return pipeline_node_catalog()


@router.get("/projects/{project_id}/pipeline-definitions")
def get_pipeline_definitions(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    _user_project(session, project_id)
    return list_pipeline_definitions(session, project_id)


@router.post("/projects/{project_id}/pipeline-definitions", status_code=201)
def create_definition(
    project_id: str,
    payload: ProjectPipelineDefinitionCreate,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    try:
        pipeline = create_pipeline_definition(session, project_id, payload)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_definition.create",
        subject_type="project_pipeline",
        subject_id=pipeline.id,
        payload={"name": pipeline.name, "draft_revision_id": pipeline.draft_revision_id},
    )
    session.commit()
    return pipeline_definition_view(session, pipeline)


@router.get("/projects/{project_id}/pipeline-definitions/{pipeline_id}")
def get_pipeline_definition(
    project_id: str, pipeline_id: str, session: Session = Depends(get_session)
) -> dict:
    _user_project(session, project_id)
    return pipeline_definition_view(
        session, _project_pipeline(session, project_id, pipeline_id)
    )


@router.put("/projects/{project_id}/pipeline-definitions/{pipeline_id}/draft")
def update_pipeline_draft(
    project_id: str,
    pipeline_id: str,
    payload: ProjectPipelineDraftUpdate,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    pipeline = _project_pipeline(session, project_id, pipeline_id)
    try:
        draft = save_pipeline_draft(session, pipeline, payload)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_definition.draft_saved",
        subject_type="project_pipeline_revision",
        subject_id=draft.id,
        payload={
            "pipeline_id": pipeline.id,
            "revision": draft.revision,
            "node_count": len(draft.nodes or []),
            "edge_count": len(draft.edges or []),
        },
    )
    session.commit()
    return pipeline_definition_view(session, pipeline)


@router.post("/projects/{project_id}/pipeline-definitions/{pipeline_id}/validate")
def validate_definition(
    project_id: str, pipeline_id: str, session: Session = Depends(get_session)
) -> dict:
    _user_project(session, project_id)
    pipeline = _project_pipeline(session, project_id, pipeline_id)
    try:
        report = validate_pipeline_definition(session, pipeline)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    session.commit()
    return report


@router.post("/projects/{project_id}/pipeline-definitions/{pipeline_id}/publish")
def publish_definition(
    project_id: str, pipeline_id: str, session: Session = Depends(get_session)
) -> dict:
    _user_project(session, project_id)
    pipeline = _project_pipeline(session, project_id, pipeline_id)
    try:
        revision = publish_pipeline_definition(session, pipeline)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_definition.publish",
        subject_type="project_pipeline_revision",
        subject_id=revision.id,
        payload={
            "pipeline_id": pipeline.id,
            "revision": revision.revision,
            "content_hash": revision.content_hash,
        },
    )
    session.commit()
    return pipeline_definition_view(session, pipeline)


@router.post(
    "/projects/{project_id}/pipeline-definitions/{pipeline_id}/runs",
    status_code=201,
)
def run_definition(
    project_id: str,
    pipeline_id: str,
    payload: ProjectPipelineRunCreate,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    _project_pipeline(session, project_id, pipeline_id)
    _raise_dag_error(
        PipelineDagError(
            "pipeline_instance_required",
            "Pipeline 定义不能直接运行；请先从已发布修订创建并激活 DAG 实例",
            status_code=409,
        )
    )


@router.get("/projects/{project_id}/pipeline-instances")
def get_pipeline_instances(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    _user_project(session, project_id)
    return list_pipeline_instances(session, project_id)


@router.post("/projects/{project_id}/pipeline-instances", status_code=201)
def create_instance(
    project_id: str,
    payload: ProjectPipelineInstanceCreate,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    try:
        instance = create_pipeline_instance(session, project_id, payload)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_instance.create",
        subject_type="project_pipeline_instance",
        subject_id=instance.id,
        payload={
            "pipeline_id": instance.pipeline_id,
            "definition_revision_id": instance.definition_revision_id,
            "name": instance.name,
        },
    )
    session.commit()
    return pipeline_instance_view(session, instance)


@router.get("/projects/{project_id}/pipeline-instances/{instance_id}")
def get_pipeline_instance(
    project_id: str,
    instance_id: str,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    return pipeline_instance_view(
        session, _project_pipeline_instance(session, project_id, instance_id)
    )


@router.put("/projects/{project_id}/pipeline-instances/{instance_id}/draft")
def update_pipeline_instance_draft(
    project_id: str,
    instance_id: str,
    payload: ProjectPipelineInstanceDraftUpdate,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    instance = _project_pipeline_instance(session, project_id, instance_id)
    try:
        save_pipeline_instance_draft(session, instance, payload)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_instance.draft_saved",
        subject_type="project_pipeline_instance",
        subject_id=instance.id,
        payload={
            "node_count": len(instance.nodes or []),
            "edge_count": len(instance.edges or []),
        },
    )
    session.commit()
    return pipeline_instance_view(session, instance)


@router.post("/projects/{project_id}/pipeline-instances/{instance_id}/validate")
def validate_instance(
    project_id: str,
    instance_id: str,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    instance = _project_pipeline_instance(session, project_id, instance_id)
    try:
        report = validate_pipeline_instance(session, instance)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    session.commit()
    return report


@router.post("/projects/{project_id}/pipeline-instances/{instance_id}/activate")
def activate_instance(
    project_id: str,
    instance_id: str,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    instance = _project_pipeline_instance(session, project_id, instance_id)
    try:
        activate_pipeline_instance(session, instance)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_instance.activate",
        subject_type="project_pipeline_instance",
        subject_id=instance.id,
        payload={"content_hash": instance.content_hash},
    )
    session.commit()
    return pipeline_instance_view(session, instance)


@router.get("/projects/{project_id}/pipeline-instances/{instance_id}/runs")
def get_instance_runs(
    project_id: str,
    instance_id: str,
    session: Session = Depends(get_session),
) -> list[dict]:
    _user_project(session, project_id)
    _project_pipeline_instance(session, project_id, instance_id)
    return list_pipeline_runs(session, project_id, instance_id=instance_id)


@router.post(
    "/projects/{project_id}/pipeline-instances/{instance_id}/runs",
    status_code=201,
)
def run_instance(
    project_id: str,
    instance_id: str,
    payload: ProjectPipelineRunCreate,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    instance = _project_pipeline_instance(session, project_id, instance_id)
    try:
        run = start_pipeline_run(session, instance, payload.inputs)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_run.create",
        subject_type="project_pipeline_run",
        subject_id=run.id,
        payload={
            "pipeline_id": run.pipeline_id,
            "instance_id": run.instance_id,
            "revision_id": run.revision_id,
        },
    )
    session.commit()
    return pipeline_run_view(session, run)


@router.get(
    "/projects/{project_id}/pipeline-instances/{instance_id}/runs/{run_id}"
)
def get_instance_run(
    project_id: str,
    instance_id: str,
    run_id: str,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    _project_pipeline_instance(session, project_id, instance_id)
    return pipeline_run_view(
        session,
        _instance_pipeline_run(session, project_id, instance_id, run_id),
    )


@router.post(
    "/projects/{project_id}/pipeline-instances/{instance_id}/runs/{run_id}/cancel"
)
def cancel_instance_run(
    project_id: str,
    instance_id: str,
    run_id: str,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    _project_pipeline_instance(session, project_id, instance_id)
    run = _instance_pipeline_run(session, project_id, instance_id, run_id)
    try:
        cancel_pipeline_run(session, run)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_run.cancel",
        subject_type="project_pipeline_run",
        subject_id=run.id,
        payload={"instance_id": instance_id},
    )
    session.commit()
    return pipeline_run_view(session, run)


@router.get("/projects/{project_id}/pipeline-runs")
def get_pipeline_runs(
    project_id: str, session: Session = Depends(get_session)
) -> list[dict]:
    _user_project(session, project_id)
    return list_pipeline_runs(session, project_id)


@router.get("/projects/{project_id}/pipeline-runs/{run_id}/lineage")
def get_pipeline_run_lineage(
    project_id: str,
    run_id: str,
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    return run_lineage_view(session, _project_pipeline_run(session, project_id, run_id))


@router.get("/projects/{project_id}/lineage/assets/{asset_type}/{asset_id}")
def get_asset_lineage(
    project_id: str,
    asset_type: str,
    asset_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    session: Session = Depends(get_session),
) -> dict:
    _user_project(session, project_id)
    return asset_lineage_view(
        session,
        project_id,
        asset_type,
        asset_id,
        limit=limit,
    )


@router.get("/projects/{project_id}/pipeline-runs/{run_id}")
def get_pipeline_run(
    project_id: str, run_id: str, session: Session = Depends(get_session)
) -> dict:
    _user_project(session, project_id)
    return pipeline_run_view(
        session, _project_pipeline_run(session, project_id, run_id)
    )


@router.post("/projects/{project_id}/pipeline-runs/{run_id}/cancel")
def cancel_run(
    project_id: str, run_id: str, session: Session = Depends(get_session)
) -> dict:
    _user_project(session, project_id)
    run = _project_pipeline_run(session, project_id, run_id)
    try:
        cancel_pipeline_run(session, run)
    except PipelineDagError as exc:
        _raise_dag_error(exc)
    add_audit(
        session,
        project_id=project_id,
        action="pipeline_run.cancel",
        subject_type="project_pipeline_run",
        subject_id=run.id,
    )
    session.commit()
    return pipeline_run_view(session, run)
