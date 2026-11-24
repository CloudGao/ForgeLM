from __future__ import annotations

import hashlib
import json

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import PluginVersion, WorkerNode, WorkerPluginInstallation

SUITE_KIND = "plugin_suite"
SUITE_SOURCE = "builtin-suite"


def _component(key: str, label: str, node: str, plugin_id: str, mode: str = "bundled", note: str = "") -> dict:
    return {"key": key, "label": label, "node_kind": node, "plugin_id": plugin_id,
            "version": "0.1.0", "deployment": mode, "note": note}


def register_builtin_suites(session: Session) -> None:
    shared = [
        _component("evaluation", "Benchmark 评测", "evaluation", "builtin.lm-eval", note="模型格式与评测后端需在具体任务中校验"),
        _component("export", "Native 导出", "export", "plugin.forgelm.native-exporter", "installable"),
        _component("delivery", "本地目录交付", "delivery", "plugin.local-directory.publisher", "installable", "需 Worker 配置交付根目录"),
    ]
    specs = [
        ("suite.transformers", "Transformers", ["transformers"], "标准 Transformers 模型的获取、推理、评测与交付；通用微调尚未接入", [
            _component("weights", "Hugging Face 权重获取", "model_import", "builtin.model-weight"),
            _component("inference", "Transformers 推理", "inference", "builtin.transformers-inference", note="具体模型兼容性需验证，不自动运行远程模型代码"),
            _component("quantization", "LLM Compressor 量化", "quantization", "plugin.llm-compressor.quantization", "installable", "不支持 MicroLM 原生格式"),
            *shared,
        ]),
        ("suite.data-preparation", "数据准备", ["architecture_independent"], "跨架构共享的数据与 Tokenizer 准备能力", [
            _component("import", "数据导入", "data_import", "builtin.data-import"),
            _component("convert", "文档转换", "transform", "builtin.data-conversion"),
            _component("clean", "数据清洗", "clean", "builtin.data-juicer"),
            _component("tokenizer", "Tokenizer 训练", "tokenizer", "builtin.hf-tokenizer"),
            _component("supervision", "监督数据编码", "preparation", "builtin.supervision-corpus"),
        ]),
    ]
    for suite_id, name, architectures, description, members in specs:
        if session.scalar(select(PluginVersion).where(PluginVersion.plugin_id == suite_id, PluginVersion.version == "0.1.0")):
            continue
        resolved = []
        for member in members:
            plugin = session.scalar(select(PluginVersion).where(PluginVersion.plugin_id == member["plugin_id"], PluginVersion.version == member["version"]))
            if plugin is None:
                raise RuntimeError(f"Suite component missing: {member['plugin_id']}@{member['version']}")
            resolved.append({**member, "plugin_version_id": plugin.id})
        manifest = {"schema": "forgelm.plugin-suite.v1", "suite_id": suite_id, "version": "0.1.0", "name": name, "architectures": architectures, "description": description, "components": resolved}
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        session.add(PluginVersion(plugin_id=suite_id, version="0.1.0", kind=SUITE_KIND, entrypoint="", source={"type": SUITE_SOURCE}, capabilities={"manifest": manifest, "manifest_hash": digest}, config_schema={"type": "object", "additionalProperties": False}, support_level="experimental"))
    session.commit()


def get_suite(session: Session, version_id: str) -> PluginVersion:
    suite = session.get(PluginVersion, version_id)
    if not suite or suite.kind != SUITE_KIND:
        raise HTTPException(404, detail={"code": "suite_not_found", "message": "套件版本不存在"})
    return suite


def suite_view(session: Session, suite: PluginVersion) -> dict:
    manifest = suite.capabilities["manifest"]
    components = []
    for member in manifest["components"]:
        plugin = session.get(PluginVersion, member["plugin_version_id"])
        components.append({**member, "plugin": plugin.to_dict() if plugin else None})
    return {"id": suite.id, "suite_id": suite.plugin_id, "version": suite.version, "enabled": suite.enabled, "manifest_hash": suite.capabilities["manifest_hash"], **{key: manifest[key] for key in ["name", "architectures", "description"]}, "components": components}


def deployment_view(session: Session, suite: PluginVersion, worker: WorkerNode) -> dict:
    installations = {row.plugin_id: row for row in session.scalars(select(WorkerPluginInstallation).where(WorkerPluginInstallation.worker_id == worker.id)).all()}
    components = []
    for member in suite_view(session, suite)["components"]:
        plugin = member["plugin"]
        row = installations.get(member["plugin_id"])
        state = "missing"
        if not plugin:
            state = "unavailable"
        elif member["deployment"] == "installable":
            if row and any(item.get("plugin_version_id") == member["plugin_version_id"] for item in row.installed_versions or []):
                state = "installed"
            elif row and row.target_plugin_version_id == member["plugin_version_id"]:
                state = "installing" if row.state in {"installing", "updating"} else "failed" if row.state == "failed" else "missing"
        elif member["plugin_id"] in (worker.capabilities or []):
            state = "project_required" if member["deployment"] == "project_code" else "provided"
        components.append({"key": member["key"], "state": state, "task_id": row.task_id if row else None, "last_verified_at": row.last_verified_at.isoformat() if row and row.last_verified_at else None})
    ready = sum(item["state"] in {"installed", "provided", "project_required"} for item in components)
    return {"suite_version_id": suite.id, "worker_id": worker.id, "worker_name": worker.name, "worker_status": worker.status, "state": "ready" if ready == len(components) else "partial" if ready else "missing", "ready_count": ready, "total_count": len(components), "project_configuration_required": any(item["state"] == "project_required" for item in components), "components": components}
