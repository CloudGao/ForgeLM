"""Freeze file-only mixture selections for the existing data processing pipeline."""
import fnmatch
from copy import deepcopy
from pathlib import PurePosixPath

from fastapi import HTTPException

from ..models import Artifact, DataMixtureVersion
from .global_data import project_has_asset_access


def freeze_processing_selection(session, project_id, version, mixture_id):
    if not mixture_id:
        return None
    mixture = session.get(DataMixtureVersion, mixture_id)
    if not mixture or not project_has_asset_access(session, project_id, "mixture_version", mixture_id):
        raise HTTPException(422, "项目尚未绑定所选混合版本")
    artifact = session.get(Artifact, mixture.manifest_artifact_id) if mixture.manifest_artifact_id else None
    if mixture.status != "ready" or not mixture.locked or not artifact or artifact.availability_state != "available" or artifact.content_hash != mixture.manifest_hash:
        raise HTTPException(409, "清洗选择必须是解析成功且已锁定的混合版本")
    entries = mixture.entries or []
    if len(entries) != 1 or entries[0].get("dataset_version_id") != version.id:
        raise HTTPException(422, "清洗选择必须只引用指定的一个来源数据版本")
    entry = entries[0]
    source = session.get(Artifact, version.artifact_id)
    if not source or entry.get("dataset_artifact_id") != source.id or entry.get("dataset_artifact_hash") != source.content_hash:
        raise HTTPException(409, "混合冻结的来源与输入快照不一致")
    rule = entry.get("selection") or {}
    if set(rule) - {"include_paths", "exclude_paths", "seed", "sample_ratio", "max_tokens", "min_length", "max_length"} or (
        rule.get("sample_ratio", 1) != 1 or any(rule.get(k, 0) for k in ("max_tokens", "min_length", "max_length"))
    ):
        raise HTTPException(422, "清洗暂只支持整文件路径选择，不支持记录过滤或抽样")
    selected = []
    for item in (version.source_snapshot or {}).get("files", []):
        path = str(item.get("path") or "")
        if (rule.get("include_paths") and not any(fnmatch.fnmatchcase(path, pat) for pat in rule["include_paths"])) or any(
            fnmatch.fnmatchcase(path, pat) for pat in rule.get("exclude_paths", [])
        ):
            continue
        rel = PurePosixPath(path)
        if not path or rel.is_absolute() or ".." in rel.parts or "\\" in path or ":" in path:
            raise HTTPException(422, "选择中含不安全的来源路径")
        if not item.get("sha256") or not isinstance(item.get("size_bytes"), int):
            raise HTTPException(409, "选择缺少已冻结的文件大小或 SHA-256")
        selected.append(deepcopy(item))
    if not selected:
        raise HTTPException(422, "选择没有匹配来源文件")
    return {"schema": "forgelm-processing-file-selection-v1", "mixture_version_id": mixture.id,
            "mixture_artifact_id": artifact.id, "mixture_artifact_hash": artifact.content_hash,
            "dataset_version_id": version.id, "selection": deepcopy(rule), "files": selected,
            "document_count": mixture.resolved_document_count, "token_estimate": mixture.resolved_token_estimate}
