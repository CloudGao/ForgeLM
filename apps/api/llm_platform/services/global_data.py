from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import (
    DataMixture,
    DataMixtureVersion,
    Dataset,
    DatasetAnalysisRevision,
    DatasetVersion,
    DatasetVersionIndex,
    EvalDatasetVersion,
    EvalSuiteVersion,
    ModelVersion,
    ModelWeightVersion,
    PlanRevision,
    Project,
    ProjectDataBinding,
    SupervisionCorpusVersion,
    SupervisionDataset,
    SupervisionDatasetVersion,
    SupervisionMixture,
    SupervisionMixtureVersion,
    Task,
    TokenizedCorpusVersion,
    TokenizedDatasetVersion,
    TokenizerVersion,
    TrainingDataViewVersion,
    TrainingPlan,
    TrainingRun,
)

GLOBAL_DATA_PROJECT_KIND = "global_data_catalog"
GLOBAL_DATA_PROJECT_NAME = "__forgelm_global_data__"
ASSET_TYPES = {
    "dataset_version",
    "mixture_version",
    "supervision_dataset_version",
    "supervision_mixture_version",
    "eval_dataset_version",
}


def _default_global_workspace() -> Path:
    settings = get_settings()
    from llm_platform_core.storage_layout import unified_storage_root
    unified = unified_storage_root(settings.storage_root)
    if unified:
        return unified / "platform" / "global-data-jobs"
    configured = str(settings.global_assets_workspace_uri or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if settings.database_url.startswith("sqlite:///"):
        raw = settings.database_url.removeprefix("sqlite:///")
        if raw != ":memory:":
            return (Path(raw).expanduser().resolve().parent / "global-assets").resolve()
    return (Path.cwd() / ".llm-platform" / "global-assets").resolve()


def _available_global_workspace(
    session: Session, preferred: Path, *, exclude_project_id: str | None = None
) -> Path:
    claimed = False
    for project_id, workspace_uri in session.execute(
        select(Project.id, Project.workspace_uri).where(Project.workspace_uri != "")
    ):
        if project_id == exclude_project_id:
            continue
        try:
            claimed = Path(str(workspace_uri)).expanduser().resolve() == preferred
        except (OSError, RuntimeError):
            claimed = str(workspace_uri).strip() == str(preferred)
        if claimed:
            break
    if not claimed:
        return preferred
    return preferred.with_name(f"{preferred.name}-{uuid4().hex[:8]}")


def ensure_global_data_project(session: Session) -> Project:
    project = session.scalar(
        select(Project)
        .where(Project.system_kind == GLOBAL_DATA_PROJECT_KIND)
        .order_by(Project.created_at)
        .limit(1)
    )
    workspace = _default_global_workspace()
    if project is not None:
        if not str(project.workspace_uri or "").strip():
            workspace = _available_global_workspace(
                session, workspace, exclude_project_id=project.id
            )
            workspace.mkdir(parents=True, exist_ok=True)
            project.workspace_uri = str(workspace)
        return project
    workspace = _available_global_workspace(session, workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    reserved_name = GLOBAL_DATA_PROJECT_NAME
    if session.scalar(select(Project.id).where(Project.name == reserved_name)) is not None:
        reserved_name = f"{GLOBAL_DATA_PROJECT_NAME}-{uuid4().hex[:8]}"
    project = Project(
        name=reserved_name,
        target_model_name="global-data-catalog",
        workspace_uri=str(workspace),
        project_type="pretrain",
        target_parameters="n/a",
        lifecycle_stage="data_preparation",
        system_kind=GLOBAL_DATA_PROJECT_KIND,
    )
    session.add(project)
    session.flush()
    return project


def is_global_data_project(project: Project | None) -> bool:
    return bool(project and project.system_kind == GLOBAL_DATA_PROJECT_KIND)


def asset_record(session: Session, asset_type: str, asset_version_id: str) -> Any | None:
    model = {
        "dataset_version": DatasetVersion,
        "mixture_version": DataMixtureVersion,
        "supervision_dataset_version": SupervisionDatasetVersion,
        "supervision_mixture_version": SupervisionMixtureVersion,
        "eval_dataset_version": EvalDatasetVersion,
    }.get(asset_type)
    return session.get(model, asset_version_id) if model is not None else None


def asset_origin_project_id(session: Session, asset_type: str, asset_version_id: str) -> str | None:
    record = asset_record(session, asset_type, asset_version_id)
    if record is None:
        return None
    if asset_type == "dataset_version":
        dataset = session.get(Dataset, record.dataset_id)
        return dataset.project_id if dataset else None
    if asset_type == "mixture_version":
        mixture = session.get(DataMixture, record.mixture_id)
        return mixture.project_id if mixture else None
    if asset_type == "supervision_dataset_version":
        dataset = session.get(SupervisionDataset, record.supervision_dataset_id)
        return dataset.project_id if dataset else None
    if asset_type == "supervision_mixture_version":
        mixture = session.get(SupervisionMixture, record.mixture_id)
        return mixture.project_id if mixture else None
    return record.project_id


def project_has_asset_access(
    session: Session,
    project_id: str,
    asset_type: str,
    asset_version_id: str,
) -> bool:
    if asset_type not in ASSET_TYPES or asset_record(session, asset_type, asset_version_id) is None:
        return False
    project = session.get(Project, project_id)
    if is_global_data_project(project):
        return True
    if asset_origin_project_id(session, asset_type, asset_version_id) == project_id:
        return True
    return (
        session.scalar(
            select(ProjectDataBinding.id).where(
                ProjectDataBinding.project_id == project_id,
                ProjectDataBinding.asset_type == asset_type,
                ProjectDataBinding.asset_version_id == asset_version_id,
            )
        )
        is not None
    )


def project_can_manage_asset(
    session: Session,
    project_id: str,
    asset_type: str,
    asset_version_id: str,
) -> bool:
    """Return whether a project may create a new lineage version from an asset.

    Bindings grant read/consume access only. Global catalog maintenance and the
    legacy origin project retain lineage mutation rights.
    """
    if asset_type not in ASSET_TYPES or asset_record(session, asset_type, asset_version_id) is None:
        return False
    project = session.get(Project, project_id)
    return is_global_data_project(project) or (
        asset_origin_project_id(session, asset_type, asset_version_id) == project_id
    )


def _contains_reference(value: object, references: set[str]) -> bool:
    if isinstance(value, str):
        return value in references
    if isinstance(value, dict):
        return any(_contains_reference(item, references) for item in value.values())
    if isinstance(value, list):
        return any(_contains_reference(item, references) for item in value)
    return False


def binding_usage_blockers(session: Session, binding: ProjectDataBinding) -> list[dict[str, Any]]:
    """Find durable project records that would lose a required data reference."""
    project_id = binding.project_id
    version_id = binding.asset_version_id
    references = {version_id, *asset_artifact_ids(session, binding.asset_type, version_id)}
    blockers: list[dict[str, Any]] = []

    def add(kind: str, record_id: str, label: str) -> None:
        blockers.append({"kind": kind, "id": record_id, "label": label})

    for task in session.scalars(select(Task).where(Task.project_id == project_id)).all():
        if any(
            _contains_reference(value, references)
            for value in (task.payload, task.result, task.error)
        ):
            add("task", task.id, task.title)

    for tokenizer in session.scalars(
        select(TokenizerVersion).where(TokenizerVersion.project_id == project_id)
    ).all():
        if tokenizer.data_ref_id == version_id:
            add("tokenizer_version", tokenizer.id, tokenizer.name)

    owned_mixture_versions = session.scalars(
        select(DataMixtureVersion)
        .join(DataMixture, DataMixture.id == DataMixtureVersion.mixture_id)
        .where(DataMixture.project_id == project_id)
    ).all()
    for version in owned_mixture_versions:
        if _contains_reference(version.entries, references):
            add("mixture_version", version.id, f"mixture_version:{version.version}")

    owned_supervision_versions = session.scalars(
        select(SupervisionDatasetVersion)
        .join(
            SupervisionDataset,
            SupervisionDataset.id == SupervisionDatasetVersion.supervision_dataset_id,
        )
        .where(SupervisionDataset.project_id == project_id)
    ).all()
    for version in owned_supervision_versions:
        lookup_version_ids = {
            str(lookup.get("dataset_version_id"))
            for lookup in (version.mapping_config or {}).get("_lookups") or []
            if lookup.get("dataset_version_id")
        }
        if version.dataset_version_id == version_id or version_id in lookup_version_ids:
            add(
                "supervision_dataset_version",
                version.id,
                f"supervision_dataset_version:{version.version}",
            )

    owned_supervision_mixtures = session.scalars(
        select(SupervisionMixtureVersion)
        .join(SupervisionMixture, SupervisionMixture.id == SupervisionMixtureVersion.mixture_id)
        .where(SupervisionMixture.project_id == project_id)
    ).all()
    for version in owned_supervision_mixtures:
        if _contains_reference(version.entries, references):
            add(
                "supervision_mixture_version",
                version.id,
                f"supervision_mixture_version:{version.version}",
            )

    direct_models = (
        (TokenizedCorpusVersion, "dataset_version_id", "tokenized_corpus_version"),
        (TokenizedDatasetVersion, "mixture_version_id", "tokenized_dataset_version"),
        (TrainingDataViewVersion, "mixture_version_id", "training_data_view_version"),
        (SupervisionCorpusVersion, "mixture_version_id", "supervision_corpus_version"),
    )
    for model, column_name, kind in direct_models:
        rows = session.scalars(
            select(model).where(
                model.project_id == project_id,
                getattr(model, column_name) == version_id,
            )
        ).all()
        for row in rows:
            add(kind, row.id, f"{kind}:{row.id}")

    plans = session.scalars(
        select(PlanRevision)
        .join(TrainingPlan, TrainingPlan.id == PlanRevision.plan_id)
        .where(TrainingPlan.project_id == project_id)
    ).all()
    for revision in plans:
        if revision.validation_mixture_version_id == version_id or any(
            _contains_reference(value, references)
            for value in (revision.stages, revision.effective_config)
        ):
            add("plan_revision", revision.id, f"plan_revision:{revision.revision}")

    for run in session.scalars(
        select(TrainingRun).where(TrainingRun.project_id == project_id)
    ).all():
        if _contains_reference(run.config_snapshot, references):
            add("training_run", run.id, f"training_run:{run.id}")

    for suite in session.scalars(
        select(EvalSuiteVersion).where(EvalSuiteVersion.project_id == project_id)
    ).all():
        if _contains_reference(suite.datasets, references):
            add("eval_suite_version", suite.id, suite.name)

    for dataset in session.scalars(
        select(EvalDatasetVersion).where(EvalDatasetVersion.project_id == project_id)
    ).all():
        if any(
            _contains_reference(value, references)
            for value in (dataset.source_config, dataset.schema_config)
        ):
            add("eval_dataset_version", dataset.id, dataset.name)

    return blockers


def _entry_ids(entries: list | None, key: str) -> set[str]:
    return {
        str(entry.get(key))
        for entry in entries or []
        if isinstance(entry, dict) and entry.get(key)
    }


def asset_artifact_ids(
    session: Session,
    asset_type: str,
    asset_version_id: str,
    *,
    _seen: set[tuple[str, str]] | None = None,
) -> set[str]:
    seen = _seen if _seen is not None else set()
    key = (asset_type, asset_version_id)
    if key in seen:
        return set()
    seen.add(key)
    record = asset_record(session, asset_type, asset_version_id)
    if record is None:
        return set()
    artifact_ids: set[str] = set()
    if asset_type == "dataset_version":
        artifact_ids.update(
            value
            for value in (record.artifact_id, record.record_manifest_artifact_id)
            if value
        )
        artifact_ids.update(
            value
            for value in session.scalars(
                select(DatasetAnalysisRevision.record_manifest_artifact_id).where(
                    DatasetAnalysisRevision.dataset_version_id == record.id,
                    DatasetAnalysisRevision.record_manifest_artifact_id.is_not(None),
                )
            ).all()
            if value
        )
        artifact_ids.update(
            session.scalars(
                select(DatasetVersionIndex.artifact_id).where(
                    DatasetVersionIndex.dataset_version_id == record.id
                )
            ).all()
        )
    elif asset_type == "mixture_version":
        if record.manifest_artifact_id:
            artifact_ids.add(record.manifest_artifact_id)
        for child_id in _entry_ids(record.entries, "dataset_version_id"):
            artifact_ids.update(
                asset_artifact_ids(
                    session, "dataset_version", child_id, _seen=seen
                )
            )
    elif asset_type == "supervision_dataset_version":
        if record.artifact_id:
            artifact_ids.add(record.artifact_id)
        artifact_ids.update(
            asset_artifact_ids(
                session, "dataset_version", record.dataset_version_id, _seen=seen
            )
        )
        for lookup in (record.mapping_config or {}).get("_lookups") or []:
            if lookup.get("dataset_version_id"):
                artifact_ids.update(
                    asset_artifact_ids(
                        session,
                        "dataset_version",
                        str(lookup["dataset_version_id"]),
                        _seen=seen,
                    )
                )
    elif asset_type == "supervision_mixture_version":
        for child_id in _entry_ids(record.entries, "supervision_dataset_version_id"):
            artifact_ids.update(
                asset_artifact_ids(
                    session, "supervision_dataset_version", child_id, _seen=seen
                )
            )
    elif asset_type == "eval_dataset_version" and record.artifact_id:
        artifact_ids.add(record.artifact_id)
    return artifact_ids


def project_can_access_artifact(session: Session, project_id: str, artifact_id: str) -> bool:
    project = session.get(Project, project_id)
    if project is None:
        return False
    # CPT generation consumes a globally registered weight through its explicit,
    # frozen task binding, without creating an unrelated project model record.
    if session.scalar(select(Task.id).where(
        Task.project_id == project_id, Task.kind == "data_generate",
        Task.payload["generation_model"]["artifact_id"].as_string() == artifact_id,
    ).limit(1)):
        return True
    if is_global_data_project(project):
        return any(
            value is not None
            for value in (
                session.scalar(
                    select(DatasetVersion.id)
                    .where(
                        (DatasetVersion.artifact_id == artifact_id)
                        | (DatasetVersion.record_manifest_artifact_id == artifact_id)
                    )
                    .limit(1)
                ),
                session.scalar(
                    select(DatasetAnalysisRevision.id)
                    .where(
                        (DatasetAnalysisRevision.snapshot_artifact_id == artifact_id)
                        | (DatasetAnalysisRevision.record_manifest_artifact_id == artifact_id)
                    )
                    .limit(1)
                ),
                session.scalar(
                    select(DatasetVersionIndex.id)
                    .where(DatasetVersionIndex.artifact_id == artifact_id)
                    .limit(1)
                ),
                session.scalar(
                    select(DataMixtureVersion.id)
                    .where(DataMixtureVersion.manifest_artifact_id == artifact_id)
                    .limit(1)
                ),
                session.scalar(
                    select(SupervisionDatasetVersion.id)
                    .where(SupervisionDatasetVersion.artifact_id == artifact_id)
                    .limit(1)
                ),
                session.scalar(
                    select(EvalDatasetVersion.id)
                    .where(EvalDatasetVersion.artifact_id == artifact_id)
                    .limit(1)
                ),
                session.scalar(
                    select(ModelWeightVersion.id)
                    .where(ModelWeightVersion.artifact_id == artifact_id)
                    .limit(1)
                ),
            )
        )
    owns_dataset_artifact = session.scalar(
        select(DatasetVersion.id)
        .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
        .where(
            Dataset.project_id == project_id,
            (DatasetVersion.artifact_id == artifact_id)
            | (DatasetVersion.record_manifest_artifact_id == artifact_id),
        )
        .limit(1)
    )
    owns_dataset_index_artifact = session.scalar(
        select(DatasetVersionIndex.id)
        .join(
            DatasetVersion,
            DatasetVersion.id == DatasetVersionIndex.dataset_version_id,
        )
        .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
        .where(
            Dataset.project_id == project_id,
            DatasetVersionIndex.artifact_id == artifact_id,
        )
        .limit(1)
    )
    owns_dataset_analysis_artifact = session.scalar(
        select(DatasetAnalysisRevision.id)
        .join(
            DatasetVersion,
            DatasetVersion.id == DatasetAnalysisRevision.dataset_version_id,
        )
        .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
        .where(
            Dataset.project_id == project_id,
            (DatasetAnalysisRevision.snapshot_artifact_id == artifact_id)
            | (DatasetAnalysisRevision.record_manifest_artifact_id == artifact_id),
        )
        .limit(1)
    )
    owns_mixture_artifact = session.scalar(
        select(DataMixtureVersion.id)
        .join(DataMixture, DataMixture.id == DataMixtureVersion.mixture_id)
        .where(
            DataMixture.project_id == project_id,
            DataMixtureVersion.manifest_artifact_id == artifact_id,
        )
        .limit(1)
    )
    owns_supervision_artifact = session.scalar(
        select(SupervisionDatasetVersion.id)
        .join(
            SupervisionDataset,
            SupervisionDataset.id == SupervisionDatasetVersion.supervision_dataset_id,
        )
        .where(
            SupervisionDataset.project_id == project_id,
            SupervisionDatasetVersion.artifact_id == artifact_id,
        )
        .limit(1)
    )
    owns_eval_artifact = session.scalar(
        select(EvalDatasetVersion.id)
        .where(
            EvalDatasetVersion.project_id == project_id,
            EvalDatasetVersion.artifact_id == artifact_id,
        )
        .limit(1)
    )
    owns_model_weight_artifact = session.scalar(
        select(ModelVersion.id)
        .where(
            ModelVersion.project_id == project_id,
            ModelVersion.artifact_id == artifact_id,
            ModelVersion.model_weight_version_id.is_not(None),
        )
        .limit(1)
    )
    if any(
        value is not None
        for value in (
            owns_dataset_artifact,
            owns_dataset_analysis_artifact,
            owns_dataset_index_artifact,
            owns_mixture_artifact,
            owns_supervision_artifact,
            owns_eval_artifact,
            owns_model_weight_artifact,
        )
    ):
        return True
    bindings = session.scalars(
        select(ProjectDataBinding).where(ProjectDataBinding.project_id == project_id)
    ).all()
    return any(
        artifact_id
        in asset_artifact_ids(session, binding.asset_type, binding.asset_version_id)
        for binding in bindings
    )


def asset_descriptor(session: Session, asset_type: str, asset_version_id: str) -> dict[str, Any] | None:
    record = asset_record(session, asset_type, asset_version_id)
    if record is None:
        return None
    origin_project_id = asset_origin_project_id(session, asset_type, asset_version_id)
    origin_project = session.get(Project, origin_project_id) if origin_project_id else None
    logical_name = ""
    tags: list[str] = []
    if asset_type == "dataset_version":
        logical = session.get(Dataset, record.dataset_id)
        logical_name = logical.name if logical else ""
        tags = list(logical.tags or []) if logical else []
    elif asset_type == "mixture_version":
        logical = session.get(DataMixture, record.mixture_id)
        logical_name = logical.name if logical else ""
    elif asset_type == "supervision_dataset_version":
        logical = session.get(SupervisionDataset, record.supervision_dataset_id)
        logical_name = logical.name if logical else ""
    elif asset_type == "supervision_mixture_version":
        logical = session.get(SupervisionMixture, record.mixture_id)
        logical_name = logical.name if logical else ""
    else:
        logical_name = record.name
    return {
        "asset_type": asset_type,
        "asset_version_id": record.id,
        "name": logical_name,
        "version": record.version,
        "status": getattr(record, "status", getattr(record, "state", "ready")),
        "tags": tags,
        "origin_project_id": origin_project_id,
        "origin_project_name": origin_project.name if origin_project else "",
        "global_managed": is_global_data_project(origin_project),
        "asset": record.to_dict(),
    }
