from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..models import (
    AuditEvent,
    Checkpoint,
    Dataset,
    DatasetVersion,
    EvalSuiteVersion,
    EvaluationRun,
    GateResult,
    HistoryImport,
    HistoryImportEntity,
    ModelVersion,
    PlanRevision,
    Project,
    Task,
    TaskEvent,
    TokenizerVersion,
    TrainingPlan,
    TrainingRun,
)
from ..schemas import HistoryImportCreate
from .artifacts import artifact_storage


def history_import_view(row: HistoryImport, *, replayed: bool = False) -> dict[str, Any]:
    return {**row.to_dict(), "replayed": replayed}


def _manifest_hash(payload: HistoryImportCreate) -> str:
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _mapping(
    session: Session,
    *,
    project_id: str,
    source_key: str,
    entity_type: str,
    external_id: str,
) -> HistoryImportEntity | None:
    return session.scalar(
        select(HistoryImportEntity).where(
            HistoryImportEntity.project_id == project_id,
            HistoryImportEntity.source_key == source_key,
            HistoryImportEntity.entity_type == entity_type,
            HistoryImportEntity.external_id == external_id,
        )
    )


def _record_mapping(
    session: Session,
    *,
    project_id: str,
    source_key: str,
    entity_type: str,
    external_id: str,
    local_id: str,
    metadata: dict[str, Any] | None = None,
) -> HistoryImportEntity:
    mapping = HistoryImportEntity(
        project_id=project_id,
        source_key=source_key,
        entity_type=entity_type,
        external_id=external_id,
        local_id=local_id,
        metadata_json=metadata or {},
    )
    session.add(mapping)
    session.flush()
    return mapping


def _mapped_id(
    session: Session,
    *,
    project_id: str,
    source_key: str,
    entity_type: str,
    external_id: str | None,
    required: bool = False,
) -> str | None:
    if not external_id:
        return None
    mapping = _mapping(
        session,
        project_id=project_id,
        source_key=source_key,
        entity_type=entity_type,
        external_id=external_id,
    )
    if mapping:
        return mapping.local_id
    if required:
        raise ValueError(f"unknown {entity_type} external_id: {external_id}")
    return None


def _artifact_id(
    session: Session,
    project_id: str,
    source_key: str,
    artifact_ref: str | None,
) -> str | None:
    return _mapped_id(
        session,
        project_id=project_id,
        source_key=source_key,
        entity_type="artifact",
        external_id=artifact_ref,
        required=bool(artifact_ref),
    )


def _imported_task(
    session: Session,
    *,
    project_id: str,
    kind: str,
    title: str,
    status: str,
    source_key: str,
    external_id: str,
    adapter_id: str = "imported.history",
    launcher_id: str = "imported.history",
    payload: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    started_at=None,
    ended_at=None,
) -> Task:
    task = Task(
        project_id=project_id,
        kind=kind,
        title=title,
        adapter_id=adapter_id,
        launcher_id=launcher_id,
        desired_state="running",
        observed_state=status,
        progress=1 if status == "succeeded" else 0,
        payload={
            **(payload or {}),
            "history_import": {"source_key": source_key, "external_id": external_id},
        },
        result=result or {},
        started_at=started_at,
        ended_at=ended_at,
    )
    session.add(task)
    session.flush()
    session.add(
        TaskEvent(
            task_id=task.id,
            sequence=1,
            event_type="task.imported",
            payload={"source_key": source_key, "external_id": external_id, "status": status},
        )
    )
    return task


def import_history(
    session: Session,
    *,
    project: Project,
    payload: HistoryImportCreate,
) -> dict[str, Any]:
    project_id = project.id
    source_key = payload.source_key
    manifest_hash = _manifest_hash(payload)
    existing_import = session.scalar(
        select(HistoryImport).where(
            HistoryImport.project_id == project_id,
            HistoryImport.source_key == source_key,
            HistoryImport.manifest_hash == manifest_hash,
        )
    )
    if existing_import:
        return history_import_view(existing_import, replayed=True)

    counts: dict[str, dict[str, int]] = {
        "created": defaultdict(int),
        "reused": defaultdict(int),
    }

    def counted(entity_type: str, reused: bool) -> None:
        counts["reused" if reused else "created"][entity_type] += 1

    history_import = HistoryImport(
        project_id=project_id,
        source_key=source_key,
        manifest_hash=manifest_hash,
        status="importing",
        summary={},
    )
    session.add(history_import)
    session.flush()

    for spec in payload.artifacts:
        artifact, _location = artifact_storage.register_external(
            session,
            project_id=project_id,
            kind=spec.kind,
            uri=spec.uri,
            content_hash=spec.content_hash,
            size_bytes=spec.size_bytes,
            media_type=spec.media_type,
            availability_state=spec.availability_state,
            worker_id=spec.worker_id,
            metadata={
                **spec.metadata,
                "project_id": project_id,
                "history_source": source_key,
                "external_id": spec.ref,
            },
        )
        mapping = _mapping(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="artifact",
            external_id=spec.ref,
        )
        if mapping and mapping.local_id != artifact.id:
            raise ValueError(f"artifact ref changed immutable content: {spec.ref}")
        if mapping is None:
            _record_mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="artifact",
                external_id=spec.ref,
                local_id=artifact.id,
                metadata={"kind": spec.kind, "content_hash": spec.content_hash.lower()},
            )
        counted("artifact", mapping is not None)

    for spec in payload.datasets:
        mapping = _mapping(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="dataset",
            external_id=spec.external_id,
        )
        if mapping:
            dataset = session.get(Dataset, mapping.local_id)
            if dataset is None or dataset.project_id != project_id:
                raise ValueError(f"invalid dataset mapping: {spec.external_id}")
        else:
            dataset = Dataset(
                project_id=project_id,
                name=spec.name,
                source_type=spec.source_type,
                source_uri=spec.source_uri,
                license=spec.license,
                language=spec.language,
                tags=spec.tags,
                status=spec.status,
            )
            session.add(dataset)
            session.flush()
            _record_mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="dataset",
                external_id=spec.external_id,
                local_id=dataset.id,
            )
        counted("dataset", mapping is not None)
        version_ids: dict[str, str] = {}
        for version_spec in spec.versions:
            version_mapping = _mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="dataset_version",
                external_id=version_spec.external_id,
            )
            if version_mapping:
                version = session.get(DatasetVersion, version_mapping.local_id)
                if version is None or version.dataset_id != dataset.id:
                    raise ValueError(f"invalid dataset version mapping: {version_spec.external_id}")
            else:
                version = DatasetVersion(
                    dataset_id=dataset.id,
                    version=version_spec.version,
                    tier=version_spec.tier,
                    state=version_spec.state,
                    raw_size_bytes=version_spec.raw_size_bytes,
                    document_count=version_spec.document_count,
                    token_estimate=version_spec.token_estimate,
                    automatic_quality_score=version_spec.automatic_quality_score,
                    human_score=version_spec.human_score,
                    quality_report=version_spec.quality_report,
                    source_snapshot=version_spec.source_snapshot,
                    artifact_id=_artifact_id(
                        session, project_id, source_key, version_spec.artifact_ref
                    ),
                    train_ready=version_spec.train_ready,
                )
                session.add(version)
                session.flush()
                _record_mapping(
                    session,
                    project_id=project_id,
                    source_key=source_key,
                    entity_type="dataset_version",
                    external_id=version_spec.external_id,
                    local_id=version.id,
                )
            version_ids[version_spec.external_id] = version.id
            counted("dataset_version", version_mapping is not None)
        current_external_id = spec.current_version_external_id
        if current_external_id:
            current_id = version_ids.get(current_external_id) or _mapped_id(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="dataset_version",
                external_id=current_external_id,
                required=True,
            )
            current = session.get(DatasetVersion, current_id)
            if current is None or current.dataset_id != dataset.id:
                raise ValueError(f"dataset current version is not part of dataset: {spec.name}")
            dataset.current_version_id = current_id
        elif version_ids and not dataset.current_version_id:
            dataset.current_version_id = list(version_ids.values())[-1]

    for spec in payload.tokenizers:
        mapping = _mapping(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="tokenizer",
            external_id=spec.external_id,
        )
        if mapping:
            tokenizer = session.get(TokenizerVersion, mapping.local_id)
            if tokenizer is None or tokenizer.project_id != project_id:
                raise ValueError(f"invalid tokenizer mapping: {spec.external_id}")
        else:
            tokenizer = TokenizerVersion(
                project_id=project_id,
                name=spec.name,
                version=spec.version,
                algorithm=spec.algorithm,
                vocab_size=spec.vocab_size,
                data_ref_type=spec.data_ref_type,
                data_ref_id=spec.data_ref_id,
                config=spec.config,
                metrics=spec.metrics,
                status=spec.status,
                locked=spec.locked,
                is_default=spec.is_default,
                artifact_id=_artifact_id(session, project_id, source_key, spec.artifact_ref),
            )
            session.add(tokenizer)
            session.flush()
            _record_mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="tokenizer",
                external_id=spec.external_id,
                local_id=tokenizer.id,
            )
        counted("tokenizer", mapping is not None)
        if spec.is_default:
            session.execute(
                update(TokenizerVersion)
                .where(
                    TokenizerVersion.project_id == project_id,
                    TokenizerVersion.id != tokenizer.id,
                )
                .values(is_default=False)
            )
            tokenizer.is_default = True

    deferred_models: list[tuple[ModelVersion, Any]] = []
    deferred_checkpoint_parents: list[tuple[Checkpoint, str]] = []
    for spec in payload.training_runs:
        run_mapping = _mapping(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="training_run",
            external_id=spec.external_id,
        )
        if run_mapping:
            run = session.get(TrainingRun, run_mapping.local_id)
            if run is None or run.project_id != project_id:
                raise ValueError(f"invalid training run mapping: {spec.external_id}")
        else:
            task = _imported_task(
                session,
                project_id=project_id,
                kind="train",
                title=f"历史训练：{spec.name}",
                status=spec.status,
                source_key=source_key,
                external_id=spec.external_id,
                adapter_id=spec.engine_plugin_id,
                launcher_id=spec.launcher_plugin_id,
                payload={"resource_request": spec.resource_request},
                started_at=spec.started_at,
                ended_at=spec.ended_at,
            )
            plan = TrainingPlan(
                project_id=project_id,
                name=f"历史计划：{spec.name}",
                description=f"由 {source_key} 导入",
                status=spec.status,
            )
            session.add(plan)
            session.flush()
            revision = PlanRevision(
                plan_id=plan.id,
                revision=1,
                stages=spec.stages,
                engine_plugin_id=spec.engine_plugin_id,
                launcher_plugin_id=spec.launcher_plugin_id,
                resource_request=spec.resource_request,
                effective_config=spec.config_snapshot,
                locked=True,
            )
            session.add(revision)
            session.flush()
            plan.current_revision_id = revision.id
            run = TrainingRun(
                project_id=project_id,
                plan_revision_id=revision.id,
                task_id=task.id,
                status=spec.status,
                current_stage=spec.current_stage,
                current_step=spec.current_step,
                current_tokens=spec.current_tokens,
                latest_metrics=spec.latest_metrics,
                config_snapshot=spec.config_snapshot,
                started_at=spec.started_at,
                ended_at=spec.ended_at,
            )
            session.add(run)
            session.flush()
            _record_mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="training_run",
                external_id=spec.external_id,
                local_id=run.id,
                metadata={"task_id": task.id, "plan_revision_id": revision.id},
            )
        counted("training_run", run_mapping is not None)

        for checkpoint_spec in spec.checkpoints:
            checkpoint_mapping = _mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="checkpoint",
                external_id=checkpoint_spec.external_id,
            )
            if checkpoint_mapping:
                checkpoint = session.get(Checkpoint, checkpoint_mapping.local_id)
                if checkpoint is None or checkpoint.training_run_id != run.id:
                    raise ValueError(f"invalid checkpoint mapping: {checkpoint_spec.external_id}")
            else:
                checkpoint = Checkpoint(
                    project_id=project_id,
                    training_run_id=run.id,
                    step=checkpoint_spec.step,
                    token_count=checkpoint_spec.token_count,
                    artifact_id=_artifact_id(
                        session, project_id, source_key, checkpoint_spec.artifact_ref
                    ),
                    metrics=checkpoint_spec.metrics,
                    status=checkpoint_spec.status,
                )
                session.add(checkpoint)
                session.flush()
                _record_mapping(
                    session,
                    project_id=project_id,
                    source_key=source_key,
                    entity_type="checkpoint",
                    external_id=checkpoint_spec.external_id,
                    local_id=checkpoint.id,
                    metadata={"training_run_external_id": spec.external_id},
                )
            counted("checkpoint", checkpoint_mapping is not None)
            if checkpoint_spec.parent_model_external_id:
                deferred_checkpoint_parents.append(
                    (checkpoint, checkpoint_spec.parent_model_external_id)
                )

        for model_spec in spec.models:
            model_mapping = _mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="model",
                external_id=model_spec.external_id,
            )
            if model_mapping:
                model = session.get(ModelVersion, model_mapping.local_id)
                if model is None or model.project_id != project_id:
                    raise ValueError(f"invalid model mapping: {model_spec.external_id}")
                if model.name != model_spec.name or model.version != model_spec.version:
                    raise ValueError(f"model identity changed: {model_spec.external_id}")
                model.metadata_json = {
                    **(model.metadata_json or {}),
                    **model_spec.metadata,
                    "history_source": source_key,
                    "external_id": model_spec.external_id,
                }
            else:
                checkpoint_id = _mapped_id(
                    session,
                    project_id=project_id,
                    source_key=source_key,
                    entity_type="checkpoint",
                    external_id=model_spec.source_checkpoint_external_id,
                    required=bool(model_spec.source_checkpoint_external_id),
                )
                tokenizer_id = _mapped_id(
                    session,
                    project_id=project_id,
                    source_key=source_key,
                    entity_type="tokenizer",
                    external_id=model_spec.tokenizer_external_id,
                    required=bool(model_spec.tokenizer_external_id),
                )
                model = ModelVersion(
                    project_id=project_id,
                    name=model_spec.name,
                    version=model_spec.version,
                    kind=model_spec.kind,
                    source_checkpoint_id=checkpoint_id,
                    artifact_id=_artifact_id(
                        session, project_id, source_key, model_spec.artifact_ref
                    ),
                    tokenizer_version_id=tokenizer_id,
                    parameter_count=model_spec.parameter_count,
                    status=model_spec.status,
                    is_best=model_spec.is_best,
                    is_candidate=model_spec.is_candidate,
                    is_official=model_spec.is_official,
                    metadata_json={
                        **model_spec.metadata,
                        "history_source": source_key,
                        "external_id": model_spec.external_id,
                    },
                )
                session.add(model)
                session.flush()
                _record_mapping(
                    session,
                    project_id=project_id,
                    source_key=source_key,
                    entity_type="model",
                    external_id=model_spec.external_id,
                    local_id=model.id,
                )
            counted("model", model_mapping is not None)
            deferred_models.append((model, model_spec))

    for model, spec in deferred_models:
        if spec.parent_model_external_id:
            model.parent_model_version_id = _mapped_id(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="model",
                external_id=spec.parent_model_external_id,
                required=True,
            )
    for checkpoint, parent_external_id in deferred_checkpoint_parents:
        checkpoint.parent_model_version_id = _mapped_id(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="model",
            external_id=parent_external_id,
            required=True,
        )
    official_models = [model for model, spec in deferred_models if spec.is_official]
    if official_models:
        official = official_models[-1]
        session.execute(
            update(ModelVersion)
            .where(ModelVersion.project_id == project_id, ModelVersion.id != official.id)
            .values(is_official=False)
        )
        official.is_official = True
        official.is_candidate = False
        project.current_model_version_id = official.id

    for spec in payload.evaluation_suites:
        mapping = _mapping(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="evaluation_suite",
            external_id=spec.external_id,
        )
        if mapping:
            suite = session.get(EvalSuiteVersion, mapping.local_id)
            if suite is None or suite.project_id != project_id:
                raise ValueError(f"invalid evaluation suite mapping: {spec.external_id}")
        else:
            suite = EvalSuiteVersion(
                project_id=project_id,
                name=spec.name,
                version=spec.version,
                datasets=spec.datasets,
                scoring_config=spec.scoring_config,
                release_gate=spec.release_gate,
            )
            session.add(suite)
            session.flush()
            _record_mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="evaluation_suite",
                external_id=spec.external_id,
                local_id=suite.id,
            )
        counted("evaluation_suite", mapping is not None)

    for spec in payload.evaluations:
        mapping = _mapping(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="evaluation",
            external_id=spec.external_id,
        )
        if mapping:
            evaluation = session.get(EvaluationRun, mapping.local_id)
            if evaluation is None or evaluation.project_id != project_id:
                raise ValueError(f"invalid evaluation mapping: {spec.external_id}")
        else:
            model_ids = [
                _mapped_id(
                    session,
                    project_id=project_id,
                    source_key=source_key,
                    entity_type="model",
                    external_id=external_id,
                    required=True,
                )
                for external_id in spec.model_external_ids
            ]
            suite_ids = [
                _mapped_id(
                    session,
                    project_id=project_id,
                    source_key=source_key,
                    entity_type="evaluation_suite",
                    external_id=external_id,
                    required=True,
                )
                for external_id in spec.suite_external_ids
            ]
            report_artifact_id = _artifact_id(session, project_id, source_key, spec.artifact_ref)
            task = _imported_task(
                session,
                project_id=project_id,
                kind="evaluate",
                title=f"历史评测：{spec.name}",
                status=spec.status,
                source_key=source_key,
                external_id=spec.external_id,
                result={
                    "artifact_id": report_artifact_id,
                    "summary": spec.summary,
                },
                started_at=spec.started_at,
                ended_at=spec.ended_at,
            )
            evaluation = EvaluationRun(
                project_id=project_id,
                task_id=task.id,
                model_version_ids=model_ids,
                suite_version_ids=suite_ids,
                inference_config=spec.inference_config,
                scoring_config=spec.scoring_config,
                status=spec.status,
                summary=spec.summary,
                sample_results=spec.sample_results,
            )
            session.add(evaluation)
            session.flush()
            _record_mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="evaluation",
                external_id=spec.external_id,
                local_id=evaluation.id,
                metadata={"task_id": task.id, "artifact_id": report_artifact_id},
            )
        counted("evaluation", mapping is not None)

    for spec in payload.gates:
        mapping = _mapping(
            session,
            project_id=project_id,
            source_key=source_key,
            entity_type="gate",
            external_id=spec.external_id,
        )
        if mapping:
            gate = session.get(GateResult, mapping.local_id)
            if gate is None or gate.project_id != project_id:
                raise ValueError(f"invalid gate mapping: {spec.external_id}")
        else:
            subject_id = _mapped_id(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type=spec.subject_entity_type,
                external_id=spec.subject_external_id,
                required=True,
            )
            gate = GateResult(
                project_id=project_id,
                subject_type=spec.subject_entity_type,
                subject_id=subject_id,
                gate_type=spec.gate_type,
                rule_version=spec.rule_version,
                status=spec.status,
                issues=spec.issues,
                input_snapshot=spec.input_snapshot,
                waiver=spec.waiver,
            )
            session.add(gate)
            session.flush()
            _record_mapping(
                session,
                project_id=project_id,
                source_key=source_key,
                entity_type="gate",
                external_id=spec.external_id,
                local_id=gate.id,
            )
        counted("gate", mapping is not None)

    summary = {
        "created": dict(counts["created"]),
        "reused": dict(counts["reused"]),
        "manifest_hash": manifest_hash,
    }
    history_import.status = "succeeded"
    history_import.summary = summary
    session.add(
        AuditEvent(
            project_id=project_id,
            action="history.import",
            subject_type="history_import",
            subject_id=history_import.id,
            payload={"source_key": source_key, **summary},
        )
    )
    session.flush()
    return history_import_view(history_import)
