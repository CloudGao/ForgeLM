"""Freeze code and verified parent snapshots for command-based dataset creation."""

from fastapi import HTTPException

from ..models import Artifact, CodeSource, CodeVersion, Dataset, DatasetVersion
from .dataset_analyses import effective_record_manifest, latest_successful_analysis
from .global_data import project_has_asset_access


def freeze_command_data_inputs(session, project_id, payload):
    if not payload.code_version_id and not payload.input_dataset_version_ids:
        return {}
    if not payload.start_import or not (payload.adapter_id or "").startswith("builtin.command."):
        raise HTTPException(
            422,
            detail={
                "code": "command_data_binding_required",
                "message": "代码与数据输入绑定仅用于立即执行的命令导入",
            },
        )
    frozen = {}
    if payload.code_version_id:
        frozen["code"] = freeze_command_code(session, project_id, payload.code_version_id)
    inputs = []
    for identity in dict.fromkeys(payload.input_dataset_version_ids):
        if not project_has_asset_access(session, project_id, "dataset_version", identity):
            raise HTTPException(
                422,
                detail={
                    "code": "invalid_data_input",
                    "message": "输入数据版本不存在或未绑定当前项目",
                },
            )
        version = session.get(DatasetVersion, identity)
        dataset = session.get(Dataset, version.dataset_id)
        snapshot = session.get(Artifact, version.artifact_id) if version.artifact_id else None
        manifest = effective_record_manifest(session, version)
        analysis = latest_successful_analysis(session, identity)
        # A command may implement the parser for a structured source archive.
        # Import readiness is distinct from the built-in text parser's train_ready.
        if (
            version.state != "ready"
            or not snapshot
            or snapshot.kind != "dataset_snapshot"
            or not snapshot.content_hash
            or not manifest
            or manifest.kind != "dataset_record_manifest"
            or not manifest.content_hash
        ):
            raise HTTPException(
                409,
                detail={
                    "code": "data_input_not_ready",
                    "message": "输入数据版本或记录索引尚未准备完成",
                },
            )
        inputs.append(
            {
                "format": "llm-dataset-input-v1",
                "dataset_version_id": identity,
                "dataset_artifact_id": snapshot.id,
                "dataset_artifact_hash": snapshot.content_hash,
                "record_manifest_artifact_id": manifest.id,
                "record_manifest_hash": manifest.content_hash,
                "document_count": analysis.document_count if analysis else version.document_count,
                "token_estimate": analysis.token_estimate if analysis else version.token_estimate,
                "source_config": dataset.source_config or {},
                **(
                    {"analysis_revision_id": analysis.id, "analysis_revision": analysis.revision}
                    if analysis
                    else {}
                ),
            }
        )
    if inputs:
        frozen["data_inputs"] = inputs
    return frozen


def freeze_command_code(session, project_id, code_version_id):
    code = session.get(CodeVersion, code_version_id)
    source = session.get(CodeSource, code.source_id) if code else None
    artifact = session.get(Artifact, code.artifact_id) if code and code.artifact_id else None
    if not source or source.project_id != project_id:
        raise HTTPException(
            422,
            detail={
                "code": "invalid_code_version",
                "message": "代码版本不存在或不属于当前项目",
            },
        )
    if code.state != "ready" or not artifact or not artifact.content_hash:
        raise HTTPException(
            409, detail={"code": "code_version_not_ready", "message": "代码版本尚未准备完成"}
        )
    return {
        "version_id": code.id,
        "version": code.version,
        "artifact_id": artifact.id,
        "content_hash": artifact.content_hash,
        "git_commit": code.git_commit,
        "entrypoints": code.entrypoints,
        "dependency_spec": code.dependency_spec,
    }
