from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .command import Emit, GetDesiredState
from .data_import import DataImportAdapter
from .data_transfer import ProcessInterrupted, ScriptTransferRunner

MODEL_WEIGHT_FORMAT = "forgelm-model-weight-v1"
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


class ModelWeightAdapter:
    """Downloads immutable Hugging Face model snapshots into Worker-local storage."""

    @staticmethod
    def _destination(asset_key: str, project_workspace: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{64}", asset_key):
            raise ValueError("A server-generated model-weight identity hash is required")
        workspace = DataImportAdapter._local_path(project_workspace)
        if not workspace.exists() or not workspace.is_dir():
            raise FileNotFoundError(
                f"Global asset workspace is not a directory on this Worker: {workspace}"
            )
        from llm_platform_core.storage_layout import unified_storage_root
        unified = unified_storage_root()
        root = (unified / "weights" / "external" if unified else workspace / ".llm-platform" / "model-weights").resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = (root / asset_key).resolve()
        destination.relative_to(root)
        return destination

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _build_manifest(
        cls,
        destination: Path,
        source_config: dict[str, Any],
        identity_hash: str,
    ) -> tuple[dict[str, Any], bytes]:
        selected = [str(value) for value in source_config.get("selected_paths") or []]
        files: list[dict[str, Any]] = []
        for relative in selected:
            path = (destination / relative).resolve()
            path.relative_to(destination.resolve())
            if not path.is_file():
                raise FileNotFoundError(f"Downloaded model file is missing: {relative}")
            files.append(
                {
                    "path": relative.replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": cls._sha256(path),
                }
            )
        names = {Path(item["path"]).name for item in files}
        suffixes = {Path(item["path"]).suffix.lower() for item in files}
        config_path = next(
            (
                destination / item["path"]
                for item in files
                if Path(item["path"]).name == "config.json"
            ),
            None,
        )
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
        requires_remote_code = bool(config.get("auto_map")) if isinstance(config, dict) else False
        transformers_ready = (
            "config.json" in names
            and bool(suffixes & _WEIGHT_SUFFIXES)
            and not requires_remote_code
        )
        tokenizer_ready = bool(names & _TOKENIZER_FILES)
        if "config.json" not in names or not bool(suffixes & _WEIGHT_SUFFIXES):
            raise ValueError(
                "Selected files do not form a Transformers model snapshot "
                "(config.json and at least one weight file are required)"
            )
        selected_set = {item["path"] for item in files}
        for item in files:
            if not item["path"].endswith(".index.json"):
                continue
            index = json.loads((destination / item["path"]).read_text(encoding="utf-8"))
            referenced = set((index.get("weight_map") or {}).values())
            missing = sorted(str(path) for path in referenced - selected_set)
            if missing:
                raise ValueError(f"Selected model weight index is missing shards: {missing[:10]}")
        capabilities: list[str] = []
        if transformers_ready:
            capabilities.append("transformers")
        if transformers_ready and tokenizer_ready:
            capabilities.append("lm_eval_hf")
            capabilities.extend(["generate", "loglikelihood", "loglikelihood_rolling"])
        manifest = {
            "format": MODEL_WEIGHT_FORMAT,
            "identity_hash": identity_hash,
            "source": {
                "provider": "huggingface",
                "repo_type": "model",
                "repository_id": source_config.get("repo_id"),
                "requested_revision": source_config.get("requested_revision"),
                "resolved_commit": source_config.get("commit_hash"),
                "endpoint": source_config.get("hf_endpoint"),
            },
            "files": sorted(files, key=lambda item: item["path"]),
            "format_summary": {
                "transformers": transformers_ready,
                "tokenizer": tokenizer_ready,
                "weight_suffixes": sorted(suffixes & _WEIGHT_SUFFIXES),
                "requires_remote_code": requires_remote_code,
            },
            "runtime_capabilities": capabilities,
        }
        encoded = json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return manifest, encoded

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        operation = str(payload.get("operation") or "")
        source_config = dict(payload.get("source_config") or {})
        source_config["repo_type"] = "model"
        repo_id = str(payload.get("source_uri") or source_config.get("repo_id") or "")
        run_dir = Path((task.get("runtime_context") or {}).get("run_dir") or ".").resolve()
        worker_id = str((task.get("runtime_context") or {}).get("worker_id") or "") or None
        project_workspace = str(
            (task.get("runtime_context") or {}).get("project_workspace")
            or payload.get("project_workspace_uri")
            or ""
        )

        if operation == "hf_scan":
            await emit("progress", {"progress": 0.1, "phase": "resolving_revision"})
            try:
                result = await DataImportAdapter._run_hf_metadata_operation(
                    "scan", repo_id, source_config, run_dir, get_desired_state
                )
            except ProcessInterrupted as exc:
                return {"controlled_state": exc.state, "operation": operation}
            await emit("progress", {"progress": 1.0, "phase": "repository_resolved"})
            return {**result, "worker_id": worker_id, "operation": operation}

        identity_hash = str(payload.get("identity_hash") or "")
        destination = self._destination(identity_hash, project_workspace)
        if operation == "hf_plan":
            await emit("progress", {"progress": 0.1, "phase": "checking_cache"})
            try:
                result = await DataImportAdapter._run_hf_metadata_operation(
                    "plan",
                    repo_id,
                    source_config,
                    run_dir,
                    get_desired_state,
                    destination=destination,
                )
            except ProcessInterrupted as exc:
                return {"controlled_state": exc.state, "operation": operation}
            await emit("progress", {"progress": 1.0, "phase": "download_planned"})
            return {
                **result,
                "identity_hash": identity_hash,
                "project_workspace_uri": project_workspace,
                "worker_id": worker_id,
                "operation": operation,
            }

        if operation == "evict":
            expected_uri = str(payload.get("location_uri") or "")
            if destination.as_uri() != expected_uri:
                raise ValueError("Model-weight location does not match its immutable identity")
            if destination.is_symlink():
                raise ValueError("Refusing to evict a symbolic-link model-weight directory")
            manifest_path = destination / "manifest.json"
            raw_manifest = manifest_path.read_bytes() if manifest_path.is_file() else b""
            if hashlib.sha256(raw_manifest).hexdigest() != str(payload.get("content_hash") or ""):
                raise ValueError("Model-weight manifest does not match its registered content hash")
            manifest = json.loads(raw_manifest)
            if (
                manifest.get("format") != MODEL_WEIGHT_FORMAT
                or manifest.get("identity_hash") != identity_hash
            ):
                raise ValueError("Model-weight manifest identity is invalid")
            await emit("progress", {"progress": 0.2, "phase": "evicting"})
            if destination.exists():
                trash_root = (destination.parent / ".trash").resolve()
                trash_root.mkdir(parents=True, exist_ok=True)
                trash = (trash_root / f"{identity_hash}-{task.get('id')}").resolve()
                trash.relative_to(trash_root)
                destination.replace(trash)
                await asyncio.to_thread(shutil.rmtree, trash)
            await emit("progress", {"progress": 1.0, "phase": "evicted"})
            return {
                "operation": operation,
                "location_id": payload.get("location_id"),
                "identity_hash": identity_hash,
            }

        if operation != "hf_download":
            raise ValueError(f"Unsupported model-weight operation: {operation}")
        destination.mkdir(parents=True, exist_ok=True)
        transfer_runner = ScriptTransferRunner(run_dir)
        try:
            transfer = await DataImportAdapter._download_planned_huggingface(
                transfer_runner,
                repo_id,
                source_config,
                destination,
                run_dir,
                emit,
                get_desired_state,
            )
        except ProcessInterrupted as exc:
            return {"controlled_state": exc.state, "operation": operation}
        await emit("progress", {"progress": 0.82, "phase": "verifying"})
        manifest, encoded = await asyncio.to_thread(
            self._build_manifest, destination, source_config, identity_hash
        )
        manifest_path = destination / "manifest.json"
        manifest_path.write_bytes(encoded)
        manifest_hash = hashlib.sha256(encoded).hexdigest()
        size_bytes = sum(int(item["size_bytes"]) for item in manifest["files"])
        await emit("progress", {"progress": 1.0, "phase": "ready"})
        return {
            "operation": operation,
            "identity_hash": identity_hash,
            "source_manifest": manifest["source"],
            "format_summary": manifest["format_summary"],
            "runtime_capabilities": manifest["runtime_capabilities"],
            "transfer": transfer,
            "outputs": [
                {
                    "ref": "model_weight",
                    "kind": "model_weights",
                    "storage_type": "external",
                    "uri": destination.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": size_bytes,
                    "media_type": "application/vnd.forgelm.model-weight",
                    "availability_state": "available",
                    "worker_id": worker_id,
                    "metadata": {
                        "format": MODEL_WEIGHT_FORMAT,
                        "identity_hash": identity_hash,
                        "repository_id": repo_id,
                        "resolved_commit": source_config.get("commit_hash"),
                        "manifest_file": manifest_path.name,
                    },
                }
            ],
        }
