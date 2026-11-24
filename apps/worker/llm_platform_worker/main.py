from __future__ import annotations

import argparse
import asyncio
import time
import copy
import gzip
import hashlib
import json
import mimetypes
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from llm_platform_core.checkpoint_compatibility import (
    build_checkpoint_compatibility,
    compare_checkpoint_compatibility,
)
from llm_platform_core.mixtures import MIXTURE_INDEX_FORMAT, open_text_manifest
from llm_platform_core.training_resources import resolve_resource_batch_plan

from .adapters.code_import import CodeImportAdapter
from .adapters.command import CommandAdapter
from .adapters.data_conversion import DataConversionAdapter
from .adapters.data_generation import DataGenerationAdapter, generation_capabilities
from .adapters.data_import import AnalysisInterrupted, DataImportAdapter
from .adapters.data_juicer import DataJuicerAdapter, data_juicer_available
from .adapters.data_mixture import DataMixtureAdapter
from .adapters.dataset_labeler import DatasetLabelerAdapter
from .adapters.dataset_materializer import DatasetMaterializerAdapter
from .adapters.dataset_viewer import DatasetViewerAdapter
from .adapters.disk_bpe_tokenizer import DiskBpeTokenizerAdapter
from .adapters.evaluation_viewer import EvaluationViewerAdapter
from .adapters.hf_tokenizer import HfTokenizerAdapter, hf_tokenizer_available
from .adapters.lm_eval import LmEvalAdapter, lm_eval_available
from .adapters.microlm_inference import (
    MicroLMInferenceAdapter,
    microlm_inference_available,
)
from .adapters.model_weight import ModelWeightAdapter
from .adapters.pretokenize import PretokenizeAdapter, pretokenize_available
from .adapters.storage_maintenance import StorageMaintenanceAdapter
from .adapters.supervision import (
    SupervisionCorpusAdapter,
    SupervisionScanAdapter,
    SupervisionTrainingIndexAdapter,
    supervision_available,
)
from .adapters.tokenized_corpus import (
    TokenizedCorpusAdapter,
    tokenize_corpus_available,
)
from .adapters.training_data_index import TrainingDataIndexAdapter
from .adapters.transformers_inference import (
    TransformersInferenceAdapter,
    transformers_inference_available,
)
from .file_cache import advise_sequential, release_file_cache_descriptor
from .gpu_metrics import GpuSamplingError, gpu_snapshot
from .hf_connector import huggingface_connector_available
from .log_redaction import MAX_LOG_VALUE_CHARS as _MAX_LOG_VALUE_CHARS
from .log_redaction import redact_log_value as _redact_log_value
from .plugin_runtime import (
    InstalledPluginAdapter,
    PluginManagerAdapter,
    discover_plugin_capabilities,
    plugin_root_path,
)
from .project_storage import (
    project_data_asset_root,
    project_run_root,
    project_training_run_root,
    scoped_runtime_path,
    worker_project_workspace,
)
from .runtime_metrics import AttemptResourceMonitor
from .suite_runtime import SuiteRuntime


@dataclass
class AgentConfig:
    api_url: str
    token: str
    name: str
    poll_seconds: float = 1.0
    labels: dict[str, str] = field(default_factory=dict)
    extra_capabilities: list[str] = field(default_factory=list)
    workspace_root: str | None = None
    profile: str = "general"
    disable_gpu: bool = False
    cpu_limit: int | None = None
    plugin_root: str | None = None
    release_root: str | None = None
    enrollment_environment_id: str | None = None
    token_file: str | None = None
    inference_idle_seconds: float = 600.0


DEFAULT_CAPABILITIES = [
    "worker.direct-dispatch.v1",
    "builtin.storage-maintenance",
    "storage.model-deletion.v1",
    "builtin.training-resource-planner.v1",
    "builtin.plugin-manager",
    "builtin.code-import",
    "builtin.data-conversion",
    "builtin.data-import",
    "builtin.data-mixture",
    "builtin.dataset-materializer",
    "builtin.dataset-viewer",
    "builtin.evaluation-viewer",
    "builtin.dataset-labeler",
    "builtin.command.data",
    "builtin.command.tokenizer",
    "builtin.command.training",
    "builtin.command.evaluation",
    "builtin.command.quantization",
    "builtin.command.inference",
    "builtin.model-runtime.custom",
    "builtin.local-process",
]

DATA_WORKER_CAPABILITIES = {
    "builtin.storage-maintenance",
    "storage.model-deletion.v1",
    "builtin.data-conversion",
    "builtin.data-import",
    "builtin.model-weight",
    "builtin.data-mixture",
    "builtin.dataset-materializer",
    "builtin.dataset-viewer",
    "builtin.evaluation-viewer",
    "builtin.dataset-labeler",
    "builtin.command.data",
    "builtin.command.tokenizer",
    "builtin.local-process",
    "builtin.hf-asset",
    "builtin.data-juicer",
    "builtin.hf-tokenizer",
    "builtin.disk-bpe-tokenizer",
    "builtin.pretokenize",
    "builtin.tokenize-corpus",
    "builtin.build-training-index",
    "builtin.supervision-scan",
    "builtin.build-supervision-index",
    "builtin.supervision-corpus",
}

_EVENT_DELIVERY_RETRY_DELAYS = (0.0, 0.25, 1.0)
_RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


def _archive_output_directory(
    source: Path, destination: Path, *, deterministic: bool = False
) -> None:
    if not deterministic:
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(source, arcname=source.name)
        return

    with destination.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                paths = [source, *sorted(source.rglob("*"), key=lambda path: path.as_posix())]
                for path in paths:
                    if path.is_symlink():
                        raise ValueError(
                            f"Dataset snapshot contains a symbolic link: {path}"
                        )
                    relative = Path(source.name) / path.relative_to(source)
                    info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o755 if path.is_dir() else 0o644
                    if info.isreg():
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)


def _response_error_code(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail") or {}
    except (ValueError, AttributeError):
        return ""
    return str(detail.get("code") or "") if isinstance(detail, dict) else ""


def _format_bytes(value: int) -> str:
    amount = float(max(value, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{int(value)} B"




class WorkerAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.worker_id: str | None = None
        token = config.token
        self.identity_path = (
            Path(config.token_file).expanduser().resolve()
            if str(config.token_file or "").strip()
            else None
        )
        if self.identity_path is not None and self.identity_path.is_file():
            try:
                identity = json.loads(self.identity_path.read_text(encoding="utf-8"))
                self.worker_id = str(identity["worker_id"])
                token = str(identity["agent_token"])
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise RuntimeError("Worker identity file is invalid") from exc
        headers = {"X-Agent-Token": token}
        if self.worker_id:
            headers["X-Worker-ID"] = self.worker_id
        self.client = httpx.AsyncClient(
            base_url=config.api_url.rstrip("/"),
            headers=headers,
            timeout=30,
        )
        self.command = CommandAdapter()
        self.code_import = CodeImportAdapter()
        self.data_conversion = DataConversionAdapter()
        self.data_generation = DataGenerationAdapter()
        self.data_import = DataImportAdapter()
        self.data_juicer = DataJuicerAdapter()
        self.storage_maintenance = StorageMaintenanceAdapter()
        self.data_mixture = DataMixtureAdapter(self.client)
        self.dataset_materializer = DatasetMaterializerAdapter()
        self.dataset_viewer = DatasetViewerAdapter()
        self.evaluation_viewer = EvaluationViewerAdapter()
        self.dataset_labeler = DatasetLabelerAdapter()
        self.disk_bpe_tokenizer = DiskBpeTokenizerAdapter()
        self.hf_tokenizer = HfTokenizerAdapter()
        self.lm_eval = LmEvalAdapter()
        self.model_weight = ModelWeightAdapter()
        self.microlm_inference = MicroLMInferenceAdapter()
        self.pretokenize = PretokenizeAdapter()
        self.tokenized_corpus = TokenizedCorpusAdapter()
        self.training_data_index = TrainingDataIndexAdapter()
        self.supervision_scan = SupervisionScanAdapter()
        self.supervision_corpus = SupervisionCorpusAdapter()
        self.supervision_training_index = SupervisionTrainingIndexAdapter()
        self.transformers_inference = TransformersInferenceAdapter()
        self.plugin_root = plugin_root_path(config.plugin_root, config.workspace_root)
        self.plugin_manager = PluginManagerAdapter(self.plugin_root)
        self.suite_runtime = SuiteRuntime(self.plugin_root, self.client)
        self.release_root = (
            Path(config.release_root).expanduser().resolve()
            if str(config.release_root or "").strip()
            else None
        )
        if self.release_root is not None:
            self.release_root.mkdir(parents=True, exist_ok=True)
        self.installed_plugin = InstalledPluginAdapter(self.plugin_root, self.release_root)

    def local_workspace_root(self) -> Path | None:
        from llm_platform_core.storage_layout import unified_storage_root
        unified = unified_storage_root()
        if unified:
            return unified
        value = str(self.config.workspace_root or "").strip()
        if not value:
            return None
        root = Path(value).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def resources(self) -> dict[str, Any]:
        from .resource_capacity import effective_capacity

        capacity = effective_capacity()
        cpu_count = capacity["cpu_count"]
        if self.config.cpu_limit is not None:
            cpu_count = min(cpu_count, self.config.cpu_limit)
        resources: dict[str, Any] = {
            "cpu_count": cpu_count,
            "cpu_model": platform.processor() or "unknown",
            "memory_total_bytes": capacity["memory_total_bytes"],
            "memory_available_bytes": capacity["memory_available_bytes"],
            "host_cpu_count": capacity["host_cpu_count"],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "suite_runtime": {"os": platform.system().lower(), "arch": platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64"), "python": f"{sys.version_info.major}.{sys.version_info.minor}", "plugin_host": "1.1.0"},
            "gpus": (
                []
                if self.config.disable_gpu or self.config.profile == "data"
                else self._discover_gpus()
            ),
        }
        workspace_root = self.local_workspace_root()
        if workspace_root is not None:
            from llm_platform_core.storage_layout import local_storage_layout

            disk = shutil.disk_usage(workspace_root)
            resources["storage_layout"] = local_storage_layout(str(workspace_root))
            resources["workspace"] = {
                "root": str(workspace_root),
                "total_bytes": int(disk.total),
                "free_bytes": int(disk.free),
            }
        resources["release_delivery"] = {
            "local_directory": self.release_root is not None
        }
        resources["inference_runtimes"] = [
            {**adapter.resident, "adapter_id": adapter_id, "idle_timeout_seconds": self.config.inference_idle_seconds}
            for adapter_id, adapter in (("builtin.microlm-inference", self.microlm_inference),
                                        ("builtin.transformers-inference", self.transformers_inference))
            if getattr(adapter, "resident", None)
        ]
        return resources

    @staticmethod
    def _cleanup_attempt_scratch(run_dir: Path) -> None:
        """Remove reproducible data-plane scratch while retaining checkpoints/results."""
        root = run_dir.resolve()
        for name in (
            "inputs",
            "mixture-shards",
            "corpus-index",
            "selection-inputs",
            "validation-corpus-index",
        ):
            target = (root / name).resolve()
            target.relative_to(root)
            shutil.rmtree(target, ignore_errors=True)
        for name in ("data-mixture-manifest.jsonl", "data-mixture-manifest.jsonl.gz"):
            target = (root / name).resolve()
            target.relative_to(root)
            target.unlink(missing_ok=True)

    def environment_snapshot(self, allocation: dict[str, Any]) -> dict[str, Any]:
        resources = self.resources()
        # Capacity selection must not reuse pre-release free-memory observations.
        if allocation.get("gpu_indexes"):
            try:
                resources["gpus"] = [
                    {key: row.get(key) for key in ("index", "uuid", "name", "memory_total_mb", "memory_free_mb")}
                    for row in gpu_snapshot(fresh=True)
                ]
            except GpuSamplingError:
                resources["gpus"] = []
        return {
            **resources,
            "hostname": socket.gethostname(),
            "labels": self.labels(),
            "allocation": allocation,
        }

    def labels(self) -> dict[str, str]:
        is_wsl = bool(os.getenv("WSL_DISTRO_NAME")) or "microsoft" in platform.release().lower()
        labels = {
            "location": "local",
            "os": platform.system().lower(),
            "environment": "wsl" if is_wsl else "native",
            "workspace_mode": (
                "worker-local"
                if str(self.config.workspace_root or "").strip()
                else "project-shared"
            ),
        }
        if self.config.profile != "general":
            labels["worker_role"] = self.config.profile
        return {**labels, **self.config.labels}

    def capabilities(self) -> list[str]:
        detected = []
        if self.config.profile != "data":
            detected.extend(generation_capabilities())
            if lm_eval_available():
                detected.append("builtin.lm-eval")
            if transformers_inference_available():
                detected.extend(
                    ["builtin.transformers-inference", "builtin.model-runtime.transformers"]
                )
            if microlm_inference_available():
                detected.extend(
                    ["builtin.microlm-inference", "builtin.model-runtime.microlm"]
                )
        if huggingface_connector_available():
            detected.extend(["builtin.hf-asset", "builtin.model-weight"])
        if data_juicer_available():
            detected.append("builtin.data-juicer")
        if hf_tokenizer_available():
            detected.extend(["builtin.hf-tokenizer", "builtin.disk-bpe-tokenizer"])
        if pretokenize_available():
            detected.append("builtin.pretokenize")
        if tokenize_corpus_available():
            detected.extend(["builtin.tokenize-corpus", "builtin.build-training-index"])
        detected.extend(["builtin.supervision-scan", "builtin.build-supervision-index"])
        if supervision_available():
            detected.append("builtin.supervision-corpus")
        capabilities = list(
            dict.fromkeys(
                [
                    *DEFAULT_CAPABILITIES,
                    *detected,
                    *discover_plugin_capabilities(self.plugin_root),
                    *self.suite_runtime.capabilities(),
                    *self.config.extra_capabilities,
                ]
            )
        )
        if self.release_root is None:
            capabilities = [
                capability
                for capability in capabilities
                if not capability.startswith("plugin.local-directory.publisher")
            ]
        if self.config.profile == "data":
            return [
                capability for capability in capabilities if capability in DATA_WORKER_CAPABILITIES
            ]
        return capabilities

    @staticmethod
    def _discover_gpus() -> list[dict[str, Any]]:
        try:
            return [
                {key: row.get(key) for key in ("index", "uuid", "name", "memory_total_mb", "memory_free_mb")}
                for row in gpu_snapshot()
            ]
        except GpuSamplingError:
            return []

    def _persist_identity(self, worker_id: str, agent_token: str) -> None:
        if self.identity_path is None:
            raise RuntimeError("enrolled Worker requires LLM_PLATFORM_AGENT_TOKEN_FILE")
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.identity_path.with_name(
            f".{self.identity_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(
                {"worker_id": worker_id, "agent_token": agent_token},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        if os.name != "nt":
            temporary.chmod(0o600)
        temporary.replace(self.identity_path)

    async def register(self) -> None:
        registration = {
            "name": self.config.name,
            "hostname": socket.gethostname(),
            "labels": self.labels(),
            "resources": await asyncio.to_thread(self.resources),
            "capabilities": self.capabilities(),
        }
        if self.worker_id is None and self.config.enrollment_environment_id:
            response = await self.client.post(
                "/api/agent/enroll",
                json={
                    **registration,
                    "environment_id": self.config.enrollment_environment_id,
                },
            )
            response.raise_for_status()
            result = response.json()
            worker_id = str(result["worker"]["id"])
            agent_token = str(result["agent_token"])
            self._persist_identity(worker_id, agent_token)
            self.worker_id = worker_id
            self.client.headers["X-Agent-Token"] = agent_token
            self.client.headers["X-Worker-ID"] = worker_id
            return
        response = await self.client.post("/api/agent/register", json=registration)
        response.raise_for_status()
        self.worker_id = response.json()["id"]

    async def heartbeat(self, running_attempt_ids: list[str] | None = None) -> list[dict]:
        assert self.worker_id
        from .adapters.inference_cache import release_expired
        for adapter in (self.microlm_inference, self.transformers_inference):
            await asyncio.to_thread(release_expired, adapter, self.config.inference_idle_seconds)
        response = await self.client.post(
            f"/api/agent/workers/{self.worker_id}/heartbeat",
            json={
                "resources": await asyncio.to_thread(self.resources),
                "running_attempt_ids": running_attempt_ids or [],
                "capabilities": self.capabilities(),
            },
        )
        response.raise_for_status()
        controls = response.json().get("controls", [])
        if not running_attempt_ids and any(c.get("action") == "release_idle_inference" for c in controls):
            for adapter in (self.microlm_inference, self.transformers_inference):
                if not getattr(adapter, "_active", False):
                    await asyncio.to_thread(adapter.release_runtime)
            # Acknowledge with a fresh GPU snapshot before attempting a claim.
            await self.client.post(f"/api/agent/workers/{self.worker_id}/heartbeat", json={
                "resources": await asyncio.to_thread(self.resources), "running_attempt_ids": [],
                "capabilities": self.capabilities()})
        return [c for c in controls if c.get("task_id")]

    async def claim(self) -> dict | None:
        assert self.worker_id
        response = await self.client.post(
            "/api/agent/tasks/claim", json={"worker_id": self.worker_id}
        )
        response.raise_for_status()
        return response.json() if response.content and response.text != "null" else None

    async def receive_dispatch(self) -> dict | None:
        """Wait for server delivery while maintaining idle resource housekeeping."""
        assert self.worker_id
        pending = asyncio.create_task(self.client.get(
            f"/api/agent/workers/{self.worker_id}/dispatch", timeout=35.0))
        try:
            while not pending.done():
                done, _ = await asyncio.wait({pending}, timeout=5.0)
                if not done:
                    # Complete idle cleanup before handing a delivered assignment to execute.
                    await self.heartbeat()
            response = await pending
            if response.status_code in {404, 405}:
                # Rolling upgrades can still connect to an older control plane.
                claimed = await self.claim()
                if not claimed:
                    await asyncio.sleep(self.config.poll_seconds)
                return claimed
            response.raise_for_status()
            return response.json() if response.content else None
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def _post_completion(self, task_id: str, payload: dict[str, Any]) -> None:
        delay = 0.25
        while True:
            try:
                response = await self.client.post(
                    f"/api/agent/tasks/{task_id}/complete",
                    json=payload,
                )
            except httpx.TransportError:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
                continue
            if response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 5.0)
                continue
            if response.status_code == 409 and _response_error_code(response) in {
                "attempt_already_completed",
                "attempt_mismatch",
            }:
                return
            response.raise_for_status()
            return

    async def upload_outputs(
        self, task: dict, attempt_id: str, result: dict[str, Any]
    ) -> list[dict[str, str]]:
        command = (task.get("payload") or {}).get("command") or {}
        output_specs = [*(command.get("outputs") or []), *(result.get("outputs") or [])]
        cwd = Path(command.get("cwd") or ".").resolve()
        uploaded: list[dict[str, str]] = []
        for index, raw_spec in enumerate(output_specs):
            spec = {"path": raw_spec} if isinstance(raw_spec, str) else raw_spec
            if not isinstance(spec, dict):
                continue
            storage_type = spec.get("storage_type") or spec.get("storage_mode") or "managed"
            if storage_type == "external":
                continue
            if not spec.get("path"):
                continue
            output_path = Path(str(spec["path"]))
            if not output_path.is_absolute():
                output_path = (cwd / output_path).resolve()
            if not output_path.exists():
                raise FileNotFoundError(f"Configured output does not exist: {output_path}")

            upload_path = output_path
            temporary_archive: Path | None = None
            if output_path.is_dir():
                archive_handle = tempfile.NamedTemporaryFile(
                    prefix="llm-platform-output-", suffix=".tar.gz", delete=False
                )
                archive_handle.close()
                temporary_archive = Path(archive_handle.name)
                # A multi-GiB checkpoint can take longer than the task lease to
                # gzip. Keep compression off the event loop so attempt heartbeats
                # and resource monitoring continue while the archive is built.
                if str(spec.get("kind") or "") == "dataset_snapshot":
                    await asyncio.to_thread(
                        _archive_output_directory,
                        output_path,
                        temporary_archive,
                        deterministic=True,
                    )
                else:
                    await asyncio.to_thread(
                        _archive_output_directory, output_path, temporary_archive
                    )
                upload_path = temporary_archive
            try:
                if (
                    task.get("kind") == "dataset_materialize"
                    and str(spec.get("ref") or "") == "dataset"
                    and str(spec.get("kind") or "") == "dataset_snapshot"
                ):
                    upload_path = await asyncio.to_thread(
                        self._persist_materialized_payload,
                        task,
                        output_path,
                        upload_path,
                    )
                    result["materialized_uri"] = upload_path.as_uri()
                media_type = str(
                    spec.get("media_type")
                    or mimetypes.guess_type(upload_path.name)[0]
                    or "application/octet-stream"
                )
                with upload_path.open("rb") as upload_file:
                    response = await self.client.post(
                        f"/api/agent/tasks/{task['id']}/artifacts",
                        data={
                            "worker_id": self.worker_id,
                            "attempt_id": attempt_id,
                            "kind": str(spec.get("kind") or "task_output"),
                            "metadata_json": json.dumps(
                                {
                                    "source_path": str(output_path),
                                    "archived_directory": output_path.is_dir(),
                                    **(spec.get("metadata") or {}),
                                }
                            ),
                        },
                        files={"file": (upload_path.name, upload_file, media_type)},
                        timeout=None,
                    )
                response.raise_for_status()
                artifact_id = str(response.json()["id"])
                uploaded.append(
                    {
                        "ref": str(spec.get("ref") or f"output-{index}"),
                        "artifact_id": artifact_id,
                        "kind": str(spec.get("kind") or "task_output"),
                        "storage_type": "managed",
                    }
                )
            finally:
                if temporary_archive:
                    temporary_archive.unlink(missing_ok=True)
        return uploaded

    async def upload_attempt_artifact(
        self,
        task: dict[str, Any],
        attempt_id: str,
        path: Path,
        *,
        kind: str,
        media_type: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        with path.open("rb") as upload_file:
            response = await self.client.post(
                f"/api/agent/tasks/{task['id']}/artifacts",
                data={
                    "worker_id": self.worker_id,
                    "attempt_id": attempt_id,
                    "kind": kind,
                    "metadata_json": json.dumps(metadata),
                },
                files={"file": (path.name, upload_file, media_type)},
                timeout=None,
            )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _local_file_uri(uri: str) -> Path | None:
        from llm_platform_core.storage_layout import resolve_relocated_path
        uri = resolve_relocated_path(uri)
        parsed = urlparse(uri)
        if parsed.scheme not in {"", "file"}:
            return None
        raw_path = unquote(parsed.path) if parsed.scheme else uri
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        path = Path(raw_path).resolve()
        return path if path.exists() else None

    def _local_artifact_location(self, location: dict[str, Any]) -> Path | None:
        metadata = location.get("metadata_json") or {}
        uri = str(location.get("uri") or "")
        direct = self._local_file_uri(uri)
        if direct is not None:
            return direct
        if self.labels().get("environment") != "wsl":
            return None
        parsed = urlparse(uri)
        raw_path = unquote(parsed.path)
        if parsed.scheme != "file" or parsed.netloc:
            return None
        if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        if not re.fullmatch(r"[A-Za-z]:/.*", raw_path):
            return None
        try:
            result = subprocess.run(
                ["wslpath", "-u", raw_path],
                capture_output=True,
                check=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        path = Path(result.stdout.strip()).resolve()
        if not path.exists():
            return None
        # Explicit bridge metadata remains authoritative for external locations.
        # Managed control-plane files may also be visible through the same Windows
        # drive mount; probing existence lets a local WSL Worker reuse them in place.
        if metadata.get("access_mode") in {None, "local_windows_wsl"}:
            return path
        return None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            advise_sequential(source.fileno())
            released_offset = 0
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
                current_offset = source.tell()
                if current_offset - released_offset >= 64 * 1024 * 1024:
                    release_file_cache_descriptor(
                        source.fileno(), released_offset, current_offset - released_offset
                    )
                    released_offset = current_offset
            release_file_cache_descriptor(source.fileno(), released_offset, 0)
        return digest.hexdigest()

    def _persist_materialized_payload(
        self,
        task: dict[str, Any],
        output_path: Path,
        upload_path: Path,
    ) -> Path:
        """Keep the exact immutable bytes uploaded for a rebuilt/downloaded replica."""

        payload = task.get("payload") or {}
        expected_hash = str(payload.get("expected_content_hash") or "")
        version_id = str(payload.get("dataset_version_id") or "")
        materialization_id = str(payload.get("materialization_id") or "")
        workspace = str((task.get("runtime_context") or {}).get("project_workspace") or "")
        if not expected_hash or not version_id or not materialization_id or not workspace:
            raise ValueError("Dataset materialization output is missing its frozen identity")
        if self._sha256_file(upload_path) != expected_hash:
            raise ValueError(
                "Rebuilt dataset bytes do not match the immutable dataset version"
            )

        replica_root = scoped_runtime_path(
            project_data_asset_root(workspace), "dataset-replicas"
        )
        replica_root.mkdir(parents=True, exist_ok=True)
        version_root = scoped_runtime_path(replica_root, version_id)
        version_root.mkdir(parents=True, exist_ok=True)
        suffix = ".tar.gz" if output_path.is_dir() else "".join(output_path.suffixes[-2:])
        if not suffix or not re.fullmatch(r"(?:\.[A-Za-z0-9_-]{1,16}){1,2}", suffix):
            suffix = ".bin"
        target = scoped_runtime_path(version_root, f"{expected_hash}{suffix}")
        if target.exists():
            if target.is_file() and self._sha256_file(target) == expected_hash:
                return target
            if not target.is_file():
                raise ValueError("Dataset replica target is not a regular file")

        staging = scoped_runtime_path(
            version_root, f".{expected_hash}.{materialization_id}.payload-staging"
        )
        if staging.exists():
            if not staging.is_file():
                raise ValueError("Dataset replica staging target is not a regular file")
            staging.unlink()
        shutil.copy2(upload_path, staging)
        try:
            if self._sha256_file(staging) != expected_hash:
                raise ValueError("Persisted dataset replica failed content verification")
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
        return target

    @staticmethod
    def _safe_archive_target(root: Path, member_name: str) -> Path:
        normalized = member_name.replace("\\", "/")
        if normalized.startswith("/") or normalized.startswith("../"):
            raise ValueError(f"Unsafe archive member: {member_name}")
        target = (root / normalized).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Unsafe archive member: {member_name}") from exc
        return target

    @classmethod
    def _extract_model_archive(cls, payload_path: Path, destination: Path) -> Path | None:
        destination.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(payload_path):
            with zipfile.ZipFile(payload_path) as archive:
                for member in archive.infolist():
                    mode = member.external_attr >> 16
                    if mode & 0o170000 == 0o120000:
                        raise ValueError(
                            f"Model archive contains a symbolic link: {member.filename}"
                        )
                    target = cls._safe_archive_target(destination, member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open("wb") as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
        elif tarfile.is_tarfile(payload_path):
            with tarfile.open(payload_path) as archive:
                for member in archive.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Model archive contains a link: {member.name}")
                    target = cls._safe_archive_target(destination, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        with source, target.open("wb") as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
        else:
            return None
        configs = sorted(
            destination.rglob("config.json"),
            key=lambda path: len(path.relative_to(destination).parts),
        )
        if not configs:
            raise ValueError("Model archive does not contain config.json")
        return configs[0].parent

    @classmethod
    def _extract_code_archive(cls, payload_path: Path, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(payload_path):
            with zipfile.ZipFile(payload_path) as archive:
                for member in archive.infolist():
                    mode = member.external_attr >> 16
                    if mode & 0o170000 == 0o120000:
                        raise ValueError(
                            f"Code archive contains a symbolic link: {member.filename}"
                        )
                    target = cls._safe_archive_target(destination, member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open("wb") as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
        elif tarfile.is_tarfile(payload_path):
            with tarfile.open(payload_path) as archive:
                for member in archive.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Code archive contains a link: {member.name}")
                    target = cls._safe_archive_target(destination, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = archive.extractfile(member)
                        if source is not None:
                            with source, target.open("wb") as output:
                                while chunk := source.read(1024 * 1024):
                                    output.write(chunk)
        else:
            raise ValueError("Code artifact must be a zip or tar archive")
        children = [child for child in destination.iterdir() if child.name != "__MACOSX"]
        return children[0] if len(children) == 1 and children[0].is_dir() else destination

    async def materialize_code_version(self, task: dict, run_dir: Path) -> None:
        payload = task.get("payload") or {}
        code = payload.get("code") or {}
        artifact_id = code.get("artifact_id")
        if not artifact_id:
            raise ValueError("Bound code version has no source_code artifact")
        artifact_params = {"project_id": str(task.get("project_id") or "")}
        response = await self.client.get(
            f"/api/agent/artifacts/{artifact_id}", params=artifact_params
        )
        response.raise_for_status()
        artifact = response.json()
        if artifact.get("kind") != "source_code":
            raise ValueError("Bound code version does not reference a source_code artifact")
        if code.get("content_hash") and code.get("content_hash") != artifact.get("content_hash"):
            raise ValueError("Bound code version content hash does not match the artifact record")
        managed_uri = next(
            (
                str(location.get("uri"))
                for location in (artifact.get("locations") or [])
                if location.get("storage_type") == "managed"
            ),
            str(artifact.get("uri") or "source-code.tar.gz"),
        )
        filename = Path(unquote(urlparse(managed_uri).path)).name or "source-code.tar.gz"
        input_dir = run_dir / "inputs" / str(artifact_id)
        input_dir.mkdir(parents=True, exist_ok=True)
        archive_path = input_dir / filename
        digest = hashlib.sha256()
        async with self.client.stream(
            "GET",
            f"/api/agent/artifacts/{artifact_id}/payload",
            params=artifact_params,
            timeout=None,
        ) as download:
            download.raise_for_status()
            with archive_path.open("wb") as output:
                async for chunk in download.aiter_bytes(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != artifact.get("content_hash"):
            archive_path.unlink(missing_ok=True)
            raise ValueError("Downloaded code artifact failed SHA-256 verification")
        code_root = self._extract_code_archive(archive_path, input_dir / "code")
        entrypoints = code.get("entrypoints") or {}
        for kind, relative_path in entrypoints.items():
            entrypoint = (code_root / str(relative_path)).resolve()
            try:
                entrypoint.relative_to(code_root.resolve())
            except ValueError as exc:
                raise ValueError(f"Code entrypoint escapes code root: {relative_path}") from exc
            if not entrypoint.is_file():
                raise FileNotFoundError(f"Code entrypoint is missing: {kind}={relative_path}")
        command = payload.get("command") or {}
        configured_cwd = command.get("cwd")
        if not configured_cwd:
            command["cwd"] = str(code_root)
        elif not Path(str(configured_cwd)).is_absolute():
            resolved_cwd = (code_root / str(configured_cwd)).resolve()
            try:
                resolved_cwd.relative_to(code_root.resolve())
            except ValueError as exc:
                raise ValueError("Relative command cwd escapes the bound code version") from exc
            if not resolved_cwd.is_dir():
                raise FileNotFoundError(
                    f"Command cwd does not exist in code version: {configured_cwd}"
                )
            command["cwd"] = str(resolved_cwd)
        command_env = command.setdefault("env", {})
        command_env["LLM_PLATFORM_CODE_DIR"] = str(code_root)
        command_env["LLM_PLATFORM_CODE_VERSION_ID"] = str(code.get("version_id") or "")
        code["runtime_path"] = str(code_root)

    async def materialize_model_specification(self, task: dict, run_dir: Path) -> None:
        payload = task.get("payload") or {}
        binding = payload.get("model_specification") or {}
        if binding.get("schema_version") != "forgelm.model-spec.v2":
            return
        frozen = binding.get("generated_json") or {}
        expected_signature = str(binding.get("config_signature") or "")
        if not expected_signature or frozen.get("config_signature") != expected_signature:
            raise ValueError("Frozen model specification has an invalid config signature")

        model_dir = run_dir / "inputs" / "model-spec" / str(binding.get("revision_id"))
        model_dir.mkdir(parents=True, exist_ok=True)
        spec_path = model_dir / "model-spec.json"
        spec_path.write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        command = payload.get("command") or {}
        command_env = command.setdefault("env", {})
        command_env["LLM_PLATFORM_MODEL_SPEC_PATH"] = str(spec_path)
        command_env["LLM_PLATFORM_MODEL_CONFIG_SIGNATURE"] = expected_signature
        requirements = binding.get("runtime_requirements") or {}

        if binding.get("source_type") == "custom_code":
            code = requirements.get("code_version") or {}
            artifact_id = code.get("artifact_id")
            entrypoint = str(requirements.get("factory_entrypoint") or "")
            if not artifact_id or not entrypoint:
                raise ValueError("Custom model contract has no code artifact or model factory")
            artifact_params = {"project_id": str(task.get("project_id") or "")}
            response = await self.client.get(
                f"/api/agent/artifacts/{artifact_id}", params=artifact_params
            )
            response.raise_for_status()
            artifact = response.json()
            if artifact.get("kind") != "source_code":
                raise ValueError("Custom model code must reference a source_code artifact")
            if code.get("content_hash") and code.get("content_hash") != artifact.get(
                "content_hash"
            ):
                raise ValueError("Custom model code hash does not match the artifact record")
            managed_uri = next(
                (
                    str(location.get("uri"))
                    for location in (artifact.get("locations") or [])
                    if location.get("storage_type") == "managed"
                ),
                str(artifact.get("uri") or "model-code.tar.gz"),
            )
            filename = Path(unquote(urlparse(managed_uri).path)).name or "model-code.tar.gz"
            archive_path = model_dir / filename
            digest = hashlib.sha256()
            async with self.client.stream(
                "GET",
                f"/api/agent/artifacts/{artifact_id}/payload",
                params=artifact_params,
                timeout=None,
            ) as download:
                download.raise_for_status()
                with archive_path.open("wb") as output:
                    async for chunk in download.aiter_bytes(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != artifact.get("content_hash"):
                archive_path.unlink(missing_ok=True)
                raise ValueError("Downloaded custom model code failed SHA-256 verification")
            code_root = self._extract_code_archive(archive_path, model_dir / "implementation")
            factory_path = (code_root / entrypoint).resolve()
            try:
                factory_path.relative_to(code_root.resolve())
            except ValueError as exc:
                raise ValueError("Custom model factory escapes code root") from exc
            if not factory_path.is_file():
                raise FileNotFoundError(f"Custom model factory is missing: {entrypoint}")
            command_env["LLM_PLATFORM_MODEL_IMPLEMENTATION_DIR"] = str(code_root)
            command_env["LLM_PLATFORM_MODEL_FACTORY_ENTRYPOINT"] = entrypoint
            binding["implementation_runtime_path"] = str(code_root)
        else:
            config_class_name = str(requirements.get("config_class") or "")
            try:
                import transformers
            except ImportError as exc:
                raise RuntimeError(
                    "This Worker lacks Transformers model runtime dependencies"
                ) from exc
            config_class = getattr(transformers, config_class_name, None)
            if config_class is None:
                raise RuntimeError(
                    f"Installed Transformers does not provide {config_class_name}"
                )
            config = config_class(**(frozen.get("config") or {}))
            config_dir = model_dir / "transformers-config"
            config.save_pretrained(config_dir)
            command_env["LLM_PLATFORM_MODEL_CONFIG_DIR"] = str(config_dir)
            binding["config_runtime_path"] = str(config_dir)
        binding["spec_runtime_path"] = str(spec_path)

    async def materialize_data_inputs(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        raw_inputs = payload.get("data_inputs")
        inputs = list(raw_inputs) if isinstance(raw_inputs, list) else []
        if isinstance(payload.get("data_input"), dict):
            inputs.append(payload["data_input"])
        if not inputs:
            return

        async def report(message: str, **details: Any) -> None:
            if emit is not None:
                await emit(
                    "log",
                    {
                        "stream": "platform",
                        "level": "info",
                        "message": message,
                        **details,
                    },
                )

        runtime_paths: list[str] = []
        for index, data_input in enumerate(inputs):
            input_format = str(data_input.get("format") or "")
            if input_format in {"llm-mixture-manifest-v1", MIXTURE_INDEX_FORMAT}:
                artifact_id = str(data_input.get("manifest_artifact_id") or "")
                expected_hash = str(data_input.get("manifest_hash") or "")
                expected_kind = "data_mixture_manifest"
                target_name = "mixture-manifest.jsonl.gz"
            elif input_format == "llm-dataset-input-v1":
                artifact_id = str(data_input.get("record_manifest_artifact_id") or "")
                expected_hash = str(data_input.get("record_manifest_hash") or "")
                expected_kind = "dataset_record_manifest"
                target_name = "dataset-records.jsonl"
            else:
                raise ValueError(f"Resolved data input has an unsupported format: {input_format}")
            if not artifact_id or not expected_hash:
                raise ValueError("Resolved data input is missing its manifest identity")
            artifact_params = {"project_id": str(task.get("project_id") or "")}
            response = await self.client.get(
                f"/api/agent/artifacts/{artifact_id}",
                params={**artifact_params, "worker_id": self.worker_id},
            )
            response.raise_for_status()
            artifact = response.json()
            if artifact.get("kind") != expected_kind:
                raise ValueError("Resolved data input references the wrong manifest kind")
            if artifact.get("content_hash") != expected_hash:
                raise ValueError("Resolved data input manifest hash does not match")
            target_dir = run_dir / "inputs" / "data" / f"{index:03d}-{artifact_id}"
            target_dir.mkdir(parents=True, exist_ok=True)
            local_manifest = next(
                (
                    path
                    for location in (artifact.get("locations") or [])
                    if location.get("availability_state", "available") == "available"
                    and (path := self._local_artifact_location(location)) is not None
                    and path.is_file()
                ),
                None,
            )
            target = local_manifest or (target_dir / target_name)
            marker = (
                None
                if local_manifest is not None
                else target.with_name(f"{target.name}.verified.json")
            )
            expected_size = int(artifact.get("size_bytes") or 0)
            if local_manifest is None and expected_size:
                free_bytes = shutil.disk_usage(target_dir).free
                required_bytes = expected_size + 512 * 1024 * 1024
                if free_bytes < required_bytes:
                    raise OSError(
                        f"Data input {index + 1}/{len(inputs)} needs "
                        f"{_format_bytes(required_bytes)} free for its "
                        f"immutable manifest, but only {_format_bytes(free_bytes)} is available"
                    )
            input_label = f"数据输入 {index + 1}/{len(inputs)}"
            await report(
                f"正在准备{input_label}：{target_name}"
                + (f"（{_format_bytes(expected_size)}）" if expected_size else ""),
                phase="materializing_data_input",
                input_index=index,
                input_count=len(inputs),
                size_bytes=expected_size,
            )
            cached = local_manifest is not None
            if target.is_file() and marker is not None and marker.is_file():
                try:
                    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                    cached = (
                        marker_payload.get("content_hash") == expected_hash
                        and int(marker_payload.get("size_bytes") or 0)
                        == target.stat().st_size
                        and (not expected_size or target.stat().st_size == expected_size)
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    cached = False
            if target.is_file() and not cached:
                cached = await asyncio.to_thread(self._sha256_file, target) == expected_hash
            if not cached:
                partial = target.with_name(f".{target.name}.partial")
                partial.unlink(missing_ok=True)
                digest = hashlib.sha256()
                downloaded = 0
                next_report_percent = 5
                next_report_bytes = 1024 * 1024 * 1024
                async with self.client.stream(
                    "GET",
                    f"/api/agent/artifacts/{artifact_id}/payload",
                    params=artifact_params,
                    timeout=None,
                ) as download:
                    download.raise_for_status()
                    with partial.open("wb") as output:
                        async for chunk in download.aiter_bytes(1024 * 1024):
                            digest.update(chunk)
                            output.write(chunk)
                            downloaded += len(chunk)
                            if expected_size:
                                percent = min(100, int(downloaded * 100 / expected_size))
                                if percent >= next_report_percent:
                                    await report(
                                        f"{input_label} 下载中：{percent}%（{_format_bytes(downloaded)} / {_format_bytes(expected_size)}）",
                                        phase="downloading_data_input",
                                        input_index=index,
                                        bytes_downloaded=downloaded,
                                        bytes_total=expected_size,
                                        progress_percent=percent,
                                    )
                                    next_report_percent = (percent // 5 + 1) * 5
                            elif downloaded >= next_report_bytes:
                                await report(
                                    f"{input_label} 下载中：已接收 {_format_bytes(downloaded)}",
                                    phase="downloading_data_input",
                                    input_index=index,
                                    bytes_downloaded=downloaded,
                                )
                                next_report_bytes = downloaded + 1024 * 1024 * 1024
                if digest.hexdigest() != expected_hash:
                    partial.unlink(missing_ok=True)
                    raise ValueError(
                        "Downloaded data input manifest failed SHA-256 verification"
                    )
                partial.replace(target)
                await report(
                    f"{input_label} 下载并校验完成（{_format_bytes(target.stat().st_size)}）",
                    phase="verifying_data_input",
                    input_index=index,
                    size_bytes=target.stat().st_size,
                )
            else:
                await report(
                    f"{input_label} 已复用本地校验缓存",
                    phase="materializing_data_input",
                    input_index=index,
                    size_bytes=target.stat().st_size,
                )
            if marker is not None:
                marker_temp = marker.with_name(f".{marker.name}.tmp")
                marker_temp.write_text(
                    json.dumps(
                        {"content_hash": expected_hash, "size_bytes": target.stat().st_size}
                    ),
                    encoding="utf-8",
                )
                marker_temp.replace(marker)
            data_input["runtime_path"] = str(target)
            runtime_paths.append(str(target))
            if input_format in {"llm-mixture-manifest-v1", MIXTURE_INDEX_FORMAT}:
                with open_text_manifest(target) as stream:
                    header = json.loads(stream.readline())
                raw_sources = header.get("sources") or []
            else:
                raw_sources = [
                    {
                        "dataset_version_id": data_input.get("dataset_version_id"),
                        "dataset_artifact_id": data_input.get("dataset_artifact_id"),
                        "dataset_artifact_hash": data_input.get("dataset_artifact_hash"),
                    }
                ]
            runtime_sources: list[dict[str, Any]] = []
            for source_index, source in enumerate(raw_sources):
                dataset_artifact_id = str(source.get("dataset_artifact_id") or "")
                dataset_hash = str(source.get("dataset_artifact_hash") or "")
                if not dataset_artifact_id or not dataset_hash:
                    raise ValueError(
                        "Resolved data source is missing its dataset snapshot identity"
                    )
                runtime_source = await self._materialize_dataset_snapshot(
                    dataset_artifact_id,
                    dataset_hash,
                    target_dir / f"source-{source_index:05d}",
                    str(task.get("project_id") or ""),
                    emit=emit,
                    source_label=f"{input_label}的数据快照 {source_index + 1}/{len(raw_sources)}",
                )
                runtime_sources.append(
                    {
                        "source_index": source_index,
                        "dataset_version_id": source.get("dataset_version_id"),
                        "artifact_id": dataset_artifact_id,
                        "content_hash": dataset_hash,
                        "runtime_path": str(runtime_source),
                    }
                )
                if input_format == MIXTURE_INDEX_FORMAT:
                    record_artifact_id = str(
                        source.get("record_manifest_artifact_id") or ""
                    )
                    record_hash = str(source.get("record_manifest_hash") or "")
                    if not record_artifact_id or not record_hash:
                        raise ValueError(
                            "Mixture index source is missing its Record Manifest identity"
                        )
                    record_response = await self.client.get(
                        f"/api/agent/artifacts/{record_artifact_id}",
                        params={**artifact_params, "worker_id": self.worker_id},
                    )
                    record_response.raise_for_status()
                    record_artifact = record_response.json()
                    if (
                        record_artifact.get("kind") != "dataset_record_manifest"
                        or record_artifact.get("content_hash") != record_hash
                    ):
                        raise ValueError("Mixture index Record Manifest identity does not match")
                    record_path = next(
                        (
                            path
                            for location in (record_artifact.get("locations") or [])
                            if location.get("availability_state", "available") == "available"
                            and (path := self._local_artifact_location(location)) is not None
                            and path.is_file()
                        ),
                        None,
                    )
                    if record_path is None:
                        record_dir = target_dir / f"source-{source_index:05d}" / "record-index"
                        record_dir.mkdir(parents=True, exist_ok=True)
                        record_path = record_dir / "dataset-records.jsonl.gz"
                        partial = record_path.with_name(f".{record_path.name}.partial")
                        partial.unlink(missing_ok=True)
                        digest = hashlib.sha256()
                        async with self.client.stream(
                            "GET",
                            f"/api/agent/artifacts/{record_artifact_id}/payload",
                            params=artifact_params,
                            timeout=None,
                        ) as download:
                            download.raise_for_status()
                            with partial.open("wb") as output:
                                async for chunk in download.aiter_bytes(1024 * 1024):
                                    digest.update(chunk)
                                    output.write(chunk)
                        if digest.hexdigest() != record_hash:
                            partial.unlink(missing_ok=True)
                            raise ValueError(
                                "Downloaded mixture Record Manifest failed SHA-256 verification"
                            )
                        partial.replace(record_path)
                    runtime_sources[-1]["runtime_record_manifest_path"] = str(record_path)
            data_input["runtime_sources"] = runtime_sources
            await report(
                f"{input_label} 已就绪",
                phase="data_input_ready",
                input_index=index,
                source_count=len(runtime_sources),
            )
        command = payload.get("command") or {}
        command_env = command.setdefault("env", {})
        command_env["LLM_PLATFORM_DATA_MANIFESTS"] = json.dumps(runtime_paths)
        if len(runtime_paths) == 1:
            command_env["LLM_PLATFORM_DATA_MANIFEST"] = runtime_paths[0]

    async def materialize_evaluation_inputs(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        raw_inputs = payload.get("evaluation_inputs")
        inputs = list(raw_inputs) if isinstance(raw_inputs, list) else []
        if not inputs:
            return
        project_id = str(task.get("project_id") or "")
        for index, evaluation_input in enumerate(inputs):
            artifact_id = str(evaluation_input.get("artifact_id") or "")
            expected_hash = str(evaluation_input.get("artifact_hash") or "")
            if not artifact_id and not expected_hash:
                continue
            if not artifact_id or not expected_hash:
                raise ValueError("Frozen evaluation input is missing its artifact identity")
            runtime_path = await self._materialize_dataset_snapshot(
                artifact_id,
                expected_hash,
                run_dir / "inputs" / "evaluation" / f"{index:03d}-{artifact_id}",
                project_id,
                emit=emit,
                source_label=f"评测数据 {index + 1}/{len(inputs)}",
            )
            evaluation_input["runtime_path"] = str(runtime_path)
        command = payload.get("command") or {}
        command_env = command.setdefault("env", {})
        command_env["LLM_PLATFORM_EVAL_DATASETS"] = json.dumps(
            inputs, ensure_ascii=False, separators=(",", ":")
        )

    async def materialize_evaluation_model_weights(self, task: dict) -> None:
        payload = task.get("payload") or {}
        bindings = list(payload.get("model_weight_inputs") or [])
        if not bindings:
            return
        runtimes = (payload.get("inference_config") or {}).get("models") or {}
        for binding in bindings:
            runtime_root = await self._materialize_external_directory_binding(
                task,
                binding,
                expected_kind="model_weights",
                expected_formats={"forgelm-model-weight-v1"},
            )
            binding["runtime_path"] = str(runtime_root)
            model_id = str(binding.get("model_version_id") or "")
            runtime = runtimes.get(model_id)
            if not isinstance(runtime, dict):
                raise ValueError(f"Managed model weight {model_id} has no lm-eval runtime")
            model_args = str(runtime.get("model_args") or "")
            if "__FORGELM_MODEL_WEIGHT__" not in model_args:
                raise ValueError("Managed model runtime is missing its weight placeholder")
            runtime["model_args"] = model_args.replace(
                "__FORGELM_MODEL_WEIGHT__", str(runtime_root)
            )

    async def materialize_evaluation_models(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        bindings = list(payload.get("evaluation_model_inputs") or [])
        if not bindings:
            return
        runtimes = (payload.get("inference_config") or {}).get("models") or {}
        original_model = payload.get("model")
        try:
            for index, binding in enumerate(bindings):
                model = binding.get("model")
                if not isinstance(model, dict):
                    raise ValueError("Frozen evaluation model input is missing its model snapshot")
                model_id = str(binding.get("model_version_id") or model.get("id") or "")
                if emit is not None:
                    await emit(
                        "progress",
                        {
                            "progress": 0,
                            "phase": f"preparing:model:{index + 1}/{len(bindings)}",
                            "model_version_id": model_id,
                        },
                    )
                    await emit(
                        "log",
                        {
                            "stream": "platform",
                            "level": "info",
                            "message": (
                                f"正在准备评测模型 {index + 1}/{len(bindings)}"
                                f"（{model_id}）"
                            ),
                            "phase": "preparing_model",
                            "model_version_id": model_id,
                        },
                    )
                payload["model"] = model
                await self.materialize_inference_model(task, run_dir)
                runtime_source = str(model.get("runtime_source") or "").strip()
                if not runtime_source:
                    raise ValueError("Evaluation model runtime source was not materialized")
                binding["runtime_path"] = runtime_source
                runtime = runtimes.get(model_id)
                if not isinstance(runtime, dict):
                    raise ValueError(f"Evaluation model {model_id} has no frozen runtime")
                model_args = str(runtime.get("model_args") or "")
                if "__FORGELM_MODEL_ARTIFACT__" not in model_args:
                    raise ValueError("Evaluation model runtime is missing its artifact placeholder")
                runtime["model_args"] = model_args.replace(
                    "__FORGELM_MODEL_ARTIFACT__", runtime_source
                )
                if emit is not None:
                    await emit(
                        "log",
                        {
                            "stream": "platform",
                            "level": "info",
                            "message": f"评测模型 {index + 1}/{len(bindings)} 准备完成",
                            "phase": "preparing_model",
                            "model_version_id": model_id,
                        },
                    )
        finally:
            if original_model is None:
                payload.pop("model", None)
            else:
                payload["model"] = original_model

    async def materialize_data_processing_input(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        data_input = payload.get("data_processing_input") or {}
        artifact_id = str(data_input.get("artifact_id") or "")
        expected_hash = str(data_input.get("artifact_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Frozen data processing input is missing its artifact identity")
        runtime_path = await self._materialize_dataset_snapshot(
            artifact_id,
            expected_hash,
            run_dir / "inputs" / "data-processing" / artifact_id,
            str(task.get("project_id") or ""),
            emit=emit,
            source_label="数据处理输入",
        )
        data_input["runtime_path"] = str(runtime_path)

    async def materialize_dataset_analysis_input(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        analysis_input = payload.get("analysis_input") or {}
        artifact_id = str(analysis_input.get("artifact_id") or "")
        expected_hash = str(analysis_input.get("artifact_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Frozen dataset analysis input is missing its artifact identity")
        runtime_path = await self._materialize_dataset_snapshot(
            artifact_id,
            expected_hash,
            run_dir / "inputs" / "dataset-analysis" / artifact_id,
            str(task.get("project_id") or ""),
            emit=emit,
            source_label="数据分析输入",
        )
        analysis_input["runtime_path"] = str(runtime_path)
        payload["source_type"] = "local"
        payload["source_uri"] = runtime_path.resolve().as_uri()

    async def materialize_dataset_viewer_input(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        viewer_input = payload.get("dataset_viewer_input") or {}
        artifact_id = str(viewer_input.get("artifact_id") or "")
        expected_hash = str(viewer_input.get("artifact_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Frozen dataset viewer input is missing its artifact identity")
        runtime_path = await self._materialize_dataset_snapshot(
            artifact_id,
            expected_hash,
            run_dir / "inputs" / "dataset-viewer" / artifact_id,
            str(task.get("project_id") or ""),
            emit=emit,
            source_label="数据查看器输入",
        )
        viewer_input["runtime_path"] = str(runtime_path)
        for binding_key, expected_kind, label in (
            ("record_manifest_input", "dataset_record_manifest", "Record Manifest"),
            ("label_snapshot_input", "data_label_snapshot", "标签快照"),
        ):
            binding = payload.get(binding_key) or {}
            binding_artifact_id = str(binding.get("artifact_id") or "")
            binding_hash = str(binding.get("artifact_hash") or "")
            if not binding_artifact_id:
                continue
            if not binding_hash:
                raise ValueError(f"Frozen {label} input is missing its artifact hash")
            binding_path = await self._materialize_dataset_snapshot(
                binding_artifact_id,
                binding_hash,
                run_dir / "inputs" / "dataset-viewer" / binding_artifact_id,
                str(task.get("project_id") or ""),
                emit=emit,
                source_label=label,
                expected_kinds={expected_kind},
                extract_archives=False,
            )
            binding["runtime_path"] = str(binding_path)

    async def materialize_dataset_label_inputs(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        bindings = (
            ("dataset_snapshot_input", {"dataset_snapshot"}, True, "数据打标快照"),
            ("record_manifest_input", {"dataset_record_manifest"}, False, "Record Manifest"),
            ("label_snapshot_input", {"data_label_snapshot"}, False, "基础标签快照"),
        )
        for binding_key, expected_kinds, extract_archives, label in bindings:
            binding = payload.get(binding_key) or {}
            artifact_id = str(binding.get("artifact_id") or "")
            expected_hash = str(binding.get("artifact_hash") or "")
            if not artifact_id:
                if binding_key == "label_snapshot_input":
                    continue
                raise ValueError(f"Frozen {label} input is missing its artifact identity")
            runtime_path = await self._materialize_dataset_snapshot(
                artifact_id,
                expected_hash,
                run_dir / "inputs" / "dataset-labeler" / artifact_id,
                str(task.get("project_id") or ""),
                emit=emit,
                source_label=label,
                expected_kinds=expected_kinds,
                extract_archives=extract_archives,
            )
            binding["runtime_path"] = str(runtime_path)

    async def materialize_data_conversion_input(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        data_input = payload.get("data_conversion_input") or {}
        artifact_id = str(data_input.get("artifact_id") or "")
        expected_hash = str(data_input.get("artifact_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Frozen data conversion input is missing its artifact identity")
        runtime_path = await self._materialize_dataset_snapshot(
            artifact_id,
            expected_hash,
            run_dir / "inputs" / "data-conversion" / artifact_id,
            str(task.get("project_id") or ""),
            emit=emit,
            source_label="数据转换输入",
        )
        data_input["runtime_path"] = str(runtime_path)

    async def materialize_dataset_replica_input(
        self, task: dict, run_dir: Path, emit: Any | None = None
    ) -> None:
        payload = task.get("payload") or {}
        binding = payload.get("materialization_input") or {}
        artifact_id = str(binding.get("artifact_id") or "")
        expected_hash = str(binding.get("artifact_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Dataset materialization input is missing its artifact identity")
        runtime_path = await self._materialize_dataset_snapshot(
            artifact_id,
            expected_hash,
            run_dir / "inputs" / "dataset-materialization" / artifact_id,
            str(task.get("project_id") or ""),
            emit=emit,
            source_label="数据集副本源",
            extract_archives=False,
            preferred_location_id=str(binding.get("preferred_location_id") or "") or None,
        )
        binding["runtime_path"] = str(runtime_path)

    async def materialize_training_tokenizer(self, task: dict, run_dir: Path) -> None:
        tokenizer = (task.get("payload") or {}).get("tokenizer") or {}
        artifact_id = str(tokenizer.get("artifact_id") or "")
        expected_hash = str(tokenizer.get("content_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Frozen training tokenizer is missing its artifact identity")
        project_id = str(task.get("project_id") or "")
        response = await self.client.get(
            f"/api/agent/artifacts/{artifact_id}", params={"project_id": project_id}
        )
        response.raise_for_status()
        artifact = response.json()
        if artifact.get("kind") != "tokenizer":
            raise ValueError("Frozen training tokenizer references the wrong artifact kind")
        if artifact.get("content_hash") != expected_hash:
            raise ValueError("Frozen training tokenizer hash does not match")
        managed_uri = next(
            (
                str(location.get("uri"))
                for location in (artifact.get("locations") or [])
                if location.get("storage_type") == "managed"
            ),
            None,
        )
        if managed_uri is None:
            raise FileNotFoundError("Frozen training tokenizer has no managed payload")
        target_dir = run_dir / "inputs" / "tokenizer" / artifact_id
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(unquote(urlparse(managed_uri).path)).name or "tokenizer.tar.gz"
        payload_path = target_dir / filename
        digest = hashlib.sha256()
        async with self.client.stream(
            "GET",
            f"/api/agent/artifacts/{artifact_id}/payload",
            params={"project_id": project_id},
            timeout=None,
        ) as download:
            download.raise_for_status()
            with payload_path.open("wb") as output:
                async for chunk in download.aiter_bytes(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected_hash:
            payload_path.unlink(missing_ok=True)
            raise ValueError("Downloaded training tokenizer failed SHA-256 verification")
        if not (zipfile.is_zipfile(payload_path) or tarfile.is_tarfile(payload_path)):
            raise ValueError("Training tokenizer artifact must be a zip or tar archive")
        extracted = target_dir / "extracted"
        self._extract_data_archive(payload_path, extracted)
        tokenizer_files = sorted(
            extracted.rglob("tokenizer.json"),
            key=lambda path: len(path.relative_to(extracted).parts),
        )
        if not tokenizer_files:
            raise ValueError("Training tokenizer artifact does not contain tokenizer.json")
        tokenizer_root = tokenizer_files[0].parent
        tokenizer["runtime_path"] = str(tokenizer_root)
        command = (task.get("payload") or {}).get("command") or {}
        command.setdefault("env", {})["LLM_PLATFORM_TOKENIZER_DIR"] = str(tokenizer_root)

    async def materialize_tokenized_data_inputs(
        self, task: dict, run_dir: Path
    ) -> None:
        payload = task.get("payload") or {}
        inputs = list(payload.get("tokenized_data_inputs") or [])
        if not inputs:
            return
        project_id = str(task.get("project_id") or "")
        runtime_paths: list[str] = []
        for binding in inputs:
            artifact_id = str(binding.get("artifact_id") or "")
            expected_hash = str(binding.get("content_hash") or "")
            if not artifact_id or not expected_hash:
                raise ValueError("Frozen Tokenized Dataset is missing its artifact identity")
            response = await self.client.get(
                f"/api/agent/artifacts/{artifact_id}",
                params={"project_id": project_id, "worker_id": self.worker_id},
            )
            response.raise_for_status()
            artifact = response.json()
            if artifact.get("kind") != "tokenized_dataset":
                raise ValueError("Frozen Tokenized Dataset references the wrong artifact kind")
            if artifact.get("content_hash") != expected_hash:
                raise ValueError("Frozen Tokenized Dataset hash does not match")
            runtime_root = next(
                (
                    path
                    for location in (artifact.get("locations") or [])
                    if location.get("availability_state", "available") == "available"
                    and location.get("worker_id") in {None, self.worker_id}
                    and (path := self._local_artifact_location(location)) is not None
                    and path.is_dir()
                ),
                None,
            )
            if runtime_root is None:
                raise ValueError(
                    "Tokenized Dataset is a large external artifact and has no "
                    "accessible directory on this Worker"
                )
            manifest_path = runtime_root / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError("Tokenized Dataset directory has no manifest.json")
            manifest_bytes = manifest_path.read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
                raise ValueError("Tokenized Dataset manifest failed content verification")
            manifest = json.loads(manifest_bytes)
            if manifest.get("format") != "forgelm-packed-tokens-v1":
                raise ValueError("Tokenized Dataset format is unsupported")
            if manifest.get("identity_hash") != binding.get("identity_hash"):
                raise ValueError("Tokenized Dataset immutable identity does not match")
            if int(manifest.get("sequence_length") or 0) != int(
                binding.get("sequence_length") or 0
            ):
                raise ValueError("Tokenized Dataset sequence length does not match")
            if int(manifest.get("token_count") or 0) < int(
                binding.get("required_tokens") or 0
            ):
                raise ValueError("Tokenized Dataset does not cover the required Token range")
            for shard in manifest.get("shards") or []:
                for file_spec in (shard.get("files") or {}).values():
                    path = (runtime_root / str(file_spec.get("path") or "")).resolve()
                    if runtime_root.resolve() not in path.parents or not path.is_file():
                        raise ValueError("Tokenized Dataset shard path is missing or unsafe")
                    if path.stat().st_size != int(file_spec.get("size_bytes") or -1):
                        raise ValueError("Tokenized Dataset shard size does not match manifest")
            binding["runtime_path"] = str(runtime_root)
            runtime_paths.append(str(runtime_root))
        command = payload.get("command") or {}
        command.setdefault("env", {})["LLM_PLATFORM_TOKENIZED_DATASETS"] = json.dumps(
            runtime_paths, ensure_ascii=False, separators=(",", ":")
        )

    async def materialize_training_selections(self, task: dict, run_dir: Path, emit) -> None:
        bindings = (task.get("payload") or {}).get("training_selection_inputs") or []
        directory = run_dir / "selection-inputs"
        directory.mkdir(parents=True, exist_ok=True)
        downloaded = {}
        for binding in bindings:
            for id_key, hash_key, runtime_key, kind in (
                ("manifest_artifact_id", "manifest_hash", "runtime_path", "data_mixture_manifest"),
                ("record_manifest_artifact_id", "record_manifest_hash", "runtime_record_manifest_path", "dataset_record_manifest"),
            ):
                artifact_id = binding[id_key]
                expected_hash = binding[hash_key]
                if artifact_id not in downloaded:
                    response = await self.client.get(f"/api/agent/artifacts/{artifact_id}", params={"project_id": task["project_id"], "worker_id": self.worker_id})
                    response.raise_for_status()
                    artifact = response.json()
                    if artifact.get("kind") != kind or artifact.get("content_hash") != expected_hash:
                        raise ValueError("Training selection artifact identity does not match")
                    target = directory / f"{expected_hash}.download"
                    state = await self.data_mixture._download_manifest(artifact_id, expected_hash, task["project_id"], target, emit, None)
                    if state in {"paused", "cancelled"}:
                        raise AnalysisInterrupted(state)
                    with target.open("rb") as stream:
                        suffix = ".jsonl.gz" if stream.read(2) == b"\x1f\x8b" else ".jsonl"
                    target = target.replace(target.with_suffix(suffix))
                    downloaded[artifact_id] = (expected_hash, str(target))
                if downloaded[artifact_id][0] != expected_hash:
                    raise ValueError("Conflicting training selection artifact hashes")
                binding[runtime_key] = downloaded[artifact_id][1]

    async def materialize_tokenized_corpora(self, task: dict, run_dir: Path) -> None:
        payload = task.get("payload") or {}
        bindings = list(payload.get("tokenized_corpora") or [])
        if not bindings:
            return
        project_id = str(task.get("project_id") or "")
        runtime_paths: list[str] = []
        for binding in bindings:
            artifact_id = str(binding.get("artifact_id") or "")
            expected_hash = str(binding.get("content_hash") or "")
            response = await self.client.get(
                f"/api/agent/artifacts/{artifact_id}",
                params={"project_id": project_id, "worker_id": self.worker_id},
            )
            response.raise_for_status()
            artifact = response.json()
            if artifact.get("kind") != "tokenized_corpus":
                raise ValueError("Frozen Tokenized Corpus references the wrong artifact kind")
            if artifact.get("content_hash") != expected_hash:
                raise ValueError("Frozen Tokenized Corpus manifest hash does not match")
            runtime_root = next(
                (
                    path
                    for location in (artifact.get("locations") or [])
                    if location.get("availability_state", "available") == "available"
                    and location.get("worker_id") in {None, self.worker_id}
                    and (path := self._local_artifact_location(location)) is not None
                    and path.is_dir()
                ),
                None,
            )
            if runtime_root is None:
                raise ValueError("Tokenized Corpus has no accessible directory on this Worker")
            manifest_path = runtime_root / "manifest.json"
            manifest_bytes = manifest_path.read_bytes() if manifest_path.is_file() else b""
            if hashlib.sha256(manifest_bytes).hexdigest() != expected_hash:
                raise ValueError("Tokenized Corpus manifest failed content verification")
            manifest = json.loads(manifest_bytes)
            if manifest.get("format") != "forgelm-token-corpus-v2":
                raise ValueError("Tokenized Corpus format is unsupported")
            if manifest.get("identity_hash") != binding.get("identity_hash"):
                raise ValueError("Tokenized Corpus immutable identity does not match")
            for shard in manifest.get("shards") or []:
                for file_spec in (shard.get("files") or {}).values():
                    path = (runtime_root / str(file_spec.get("path") or "")).resolve()
                    if runtime_root.resolve() not in path.parents or not path.is_file():
                        raise ValueError("Tokenized Corpus shard path is missing or unsafe")
                    if path.stat().st_size != int(file_spec.get("size_bytes") or -1):
                        raise ValueError("Tokenized Corpus shard size does not match manifest")
            binding["runtime_path"] = str(runtime_root)
            runtime_paths.append(str(runtime_root))
        command = payload.get("command") or {}
        command.setdefault("env", {})["LLM_PLATFORM_TOKENIZED_CORPORA"] = json.dumps(
            runtime_paths, ensure_ascii=False, separators=(",", ":")
        )

    async def materialize_training_data_views(self, task: dict, run_dir: Path) -> None:
        payload = task.get("payload") or {}
        bindings = list(payload.get("training_data_view_inputs") or [])
        if not bindings:
            return
        project_id = str(task.get("project_id") or "")
        runtime_paths: list[str] = []
        for binding in bindings:
            response = await self.client.get(
                f"/api/agent/artifacts/{binding.get('artifact_id')}",
                params={"project_id": project_id, "worker_id": self.worker_id},
            )
            response.raise_for_status()
            artifact = response.json()
            if artifact.get("kind") != "training_data_index":
                raise ValueError("Training Data View references the wrong artifact kind")
            if artifact.get("content_hash") != binding.get("content_hash"):
                raise ValueError("Training Data View manifest hash does not match")
            runtime_root = next(
                (
                    path
                    for location in (artifact.get("locations") or [])
                    if location.get("availability_state", "available") == "available"
                    and location.get("worker_id") in {None, self.worker_id}
                    and (path := self._local_artifact_location(location)) is not None
                    and path.is_dir()
                ),
                None,
            )
            if runtime_root is None:
                raise ValueError("Training Data View has no accessible directory on this Worker")
            manifest_path = runtime_root / "manifest.json"
            manifest_bytes = manifest_path.read_bytes() if manifest_path.is_file() else b""
            if hashlib.sha256(manifest_bytes).hexdigest() != binding.get("content_hash"):
                raise ValueError("Training Data View manifest failed content verification")
            manifest = json.loads(manifest_bytes)
            if manifest.get("format") != "forgelm-training-index-v1":
                raise ValueError("Training Data View format is unsupported")
            if manifest.get("identity_hash") != binding.get("identity_hash"):
                raise ValueError("Training Data View immutable identity does not match")
            binding["runtime_path"] = str(runtime_root)
            runtime_paths.append(str(runtime_root))
        command = payload.get("command") or {}
        command.setdefault("env", {})["LLM_PLATFORM_TRAINING_DATA_VIEWS"] = json.dumps(
            runtime_paths, ensure_ascii=False, separators=(",", ":")
        )

    async def materialize_training_data_view_prefix(self, task: dict) -> None:
        payload = task.get("payload") or {}
        binding = payload.get("prefix_training_data_view") or {}
        if not binding:
            return
        runtime_root = await self._materialize_external_directory_binding(
            task,
            binding,
            expected_kind="training_data_index",
            expected_formats={"forgelm-training-index-v1"},
        )
        binding["runtime_path"] = str(runtime_root)

    async def _materialize_external_directory_binding(
        self,
        task: dict,
        binding: dict[str, Any],
        *,
        expected_kind: str,
        expected_formats: set[str],
    ) -> Path:
        artifact_id = str(binding.get("artifact_id") or "")
        expected_hash = str(binding.get("content_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError(f"Frozen {expected_kind} is missing its artifact identity")
        response = await self.client.get(
            f"/api/agent/artifacts/{artifact_id}",
            params={
                "project_id": str(task.get("project_id") or ""),
                "worker_id": self.worker_id,
            },
        )
        response.raise_for_status()
        artifact = response.json()
        if artifact.get("kind") != expected_kind:
            raise ValueError(f"Frozen {expected_kind} references the wrong artifact kind")
        if artifact.get("content_hash") != expected_hash:
            raise ValueError(f"Frozen {expected_kind} manifest hash does not match")
        runtime_root = next(
            (
                path
                for location in (artifact.get("locations") or [])
                if location.get("availability_state", "available") == "available"
                and location.get("worker_id") in {None, self.worker_id}
                and (path := self._local_artifact_location(location)) is not None
                and path.is_dir()
            ),
            None,
        )
        if runtime_root is None:
            raise ValueError(f"{expected_kind} has no accessible directory on this Worker")
        manifest_path = runtime_root / "manifest.json"
        raw = manifest_path.read_bytes() if manifest_path.is_file() else b""
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"{expected_kind} manifest failed content verification")
        manifest = json.loads(raw)
        if manifest.get("format") not in expected_formats:
            raise ValueError(f"{expected_kind} format is unsupported")
        if binding.get("identity_hash") and manifest.get("identity_hash") != binding.get(
            "identity_hash"
        ):
            raise ValueError(f"{expected_kind} immutable identity does not match")
        return runtime_root

    async def materialize_supervision_sources(self, task: dict) -> None:
        payload = task.get("payload") or {}
        sources = list(payload.get("supervision_sources") or [])
        for source in sources:
            binding = source.get("record_index") or {}
            runtime_root = await self._materialize_external_directory_binding(
                task,
                binding,
                expected_kind="supervision_record_index",
                expected_formats={"forgelm-supervision-record-index-v1"},
            )
            source["runtime_record_index_path"] = str(runtime_root)

    async def materialize_supervision_corpus(self, task: dict) -> None:
        payload = task.get("payload") or {}
        binding = payload.get("supervision_corpus") or {}
        runtime_root = await self._materialize_external_directory_binding(
            task,
            binding,
            expected_kind="supervision_corpus",
            expected_formats={"forgelm-sft-corpus-v1", "forgelm-preference-corpus-v1"},
        )
        binding["runtime_path"] = str(runtime_root)

    async def materialize_supervision_training_views(self, task: dict) -> None:
        payload = task.get("payload") or {}
        bindings = list(payload.get("supervision_training_view_inputs") or [])
        runtime_paths: list[str] = []
        for binding in bindings:
            runtime_root = await self._materialize_external_directory_binding(
                task,
                binding,
                expected_kind="supervision_training_index",
                expected_formats={
                    "forgelm-sft-training-index-v1",
                    "forgelm-preference-training-index-v1",
                },
            )
            binding["runtime_path"] = str(runtime_root)
            runtime_paths.append(str(runtime_root))
            corpus = binding.get("supervision_corpus") or {}
            corpus_root = await self._materialize_external_directory_binding(
                task,
                corpus,
                expected_kind="supervision_corpus",
                expected_formats={"forgelm-sft-corpus-v1", "forgelm-preference-corpus-v1"},
            )
            corpus["runtime_path"] = str(corpus_root)
        command = payload.get("command") or {}
        command.setdefault("env", {})["LLM_PLATFORM_SUPERVISION_TRAINING_VIEWS"] = (
            json.dumps(runtime_paths, ensure_ascii=False, separators=(",", ":"))
        )

    async def materialize_training_checkpoint(
        self,
        task: dict,
        run_dir: Path,
        *,
        binding_key: str = "resume_checkpoint",
        environment_key: str = "LLM_PLATFORM_RESUME_CHECKPOINT_DIR",
    ) -> None:
        checkpoint = (task.get("payload") or {}).get(binding_key) or {}
        artifact_id = str(checkpoint.get("artifact_id") or "")
        expected_hash = str(checkpoint.get("content_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Frozen resume checkpoint is missing its artifact identity")
        project_id = str(task.get("project_id") or "")
        response = await self.client.get(
            f"/api/agent/artifacts/{artifact_id}", params={"project_id": project_id}
        )
        response.raise_for_status()
        artifact = response.json()
        if artifact.get("kind") != "checkpoint":
            raise ValueError("Frozen resume checkpoint references the wrong artifact kind")
        if artifact.get("content_hash") != expected_hash:
            raise ValueError("Frozen resume checkpoint hash does not match")
        locations = artifact.get("locations") or []
        runtime_path = next(
            (
                path
                for location in locations
                if location.get("storage_type") == "external"
                and location.get("availability_state") == "available"
                and location.get("worker_id") in {None, self.worker_id}
                and (path := self._local_artifact_location(location)) is not None
            ),
            None,
        )
        target_dir = run_dir / "inputs" / binding_key.replace("_", "-") / artifact_id
        downloaded_payload_path: Path | None = None
        if runtime_path is None:
            managed_uri = next(
                (
                    str(location.get("uri"))
                    for location in locations
                    if location.get("storage_type") == "managed"
                ),
                None,
            )
            if managed_uri is None:
                raise FileNotFoundError(
                    "Frozen resume checkpoint has no location visible to this Worker"
                )
            target_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(unquote(urlparse(managed_uri).path)).name or "checkpoint.tar.gz"
            payload_path = target_dir / filename
            digest = hashlib.sha256()
            async with self.client.stream(
                "GET",
                f"/api/agent/artifacts/{artifact_id}/payload",
                params={"project_id": project_id},
                timeout=None,
            ) as download:
                download.raise_for_status()
                with payload_path.open("wb") as output:
                    async for chunk in download.aiter_bytes(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != expected_hash:
                payload_path.unlink(missing_ok=True)
                raise ValueError("Downloaded resume checkpoint failed SHA-256 verification")
            runtime_path = payload_path
            downloaded_payload_path = payload_path
        if runtime_path.is_file():
            if not (zipfile.is_zipfile(runtime_path) or tarfile.is_tarfile(runtime_path)):
                raise ValueError("Resume checkpoint artifact must be a directory or archive")
            extracted = target_dir / "extracted"
            # Multi-GiB checkpoints can take longer to extract than the attempt
            # lease. Keep archive I/O off the event loop so the heartbeat task can
            # renew the lease while materialization is in progress.
            await asyncio.to_thread(self._extract_data_archive, runtime_path, extracted)
            if downloaded_payload_path is not None:
                downloaded_payload_path.unlink(missing_ok=True)
            children = [child for child in extracted.iterdir() if child.name != "__MACOSX"]
            runtime_path = children[0] if len(children) == 1 and children[0].is_dir() else extracted
        if not runtime_path.is_dir():
            raise ValueError("Materialized resume checkpoint is not a directory")
        checkpoint["runtime_path"] = str(runtime_path)
        expected_compatibility = (
            (task.get("payload") or {}).get("checkpoint_compatibility")
            if binding_key == "resume_checkpoint"
            else None
        )
        recorded_compatibility = checkpoint.get("model_compatibility")
        if isinstance(expected_compatibility, dict):
            model_config_paths = (
                runtime_path / "model" / "config.json",
                runtime_path / "config.json",
            )
            model_config_path = next(
                (candidate for candidate in model_config_paths if candidate.is_file()),
                None,
            )
            if model_config_path is not None:
                saved_configuration = json.loads(
                    model_config_path.read_text(encoding="utf-8")
                )
                if not isinstance(saved_configuration, dict):
                    raise ValueError("Resume checkpoint model config.json must be an object")
                recorded_configuration = (
                    recorded_compatibility.get("configuration")
                    if isinstance(recorded_compatibility, dict)
                    else expected_compatibility.get("configuration")
                )
                if not isinstance(recorded_configuration, dict):
                    recorded_configuration = {}
                effective_configuration = dict(recorded_configuration)
                for field in (expected_compatibility.get("configuration") or {}):
                    if field in saved_configuration:
                        effective_configuration[field] = saved_configuration[field]
                checkpoint["model_compatibility"] = {
                    **build_checkpoint_compatibility(
                        effective_configuration,
                        contract=str(expected_compatibility.get("contract") or ""),
                        contract_version=int(
                            expected_compatibility.get("contract_version") or 1
                        ),
                        runtime_only_fields=expected_compatibility.get(
                            "runtime_only_fields"
                        )
                        or [],
                    ),
                    "provenance": "worker_model_config",
                }
        command = (task.get("payload") or {}).get("command") or {}
        command.setdefault("env", {})[environment_key] = str(runtime_path)

    async def validate_training_checkpoint_compatibility(
        self, task: dict, emit: Any
    ) -> None:
        payload = task.get("payload") or {}
        resume_checkpoint = payload.get("resume_checkpoint") or {}
        if not resume_checkpoint:
            return
        expected = payload.get("checkpoint_compatibility")
        actual = resume_checkpoint.get("model_compatibility")
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            await emit(
                "system.validation",
                {
                    "validation_id": "resume_checkpoint_compatible",
                    "status": "pending",
                    "phase": "worker",
                    "message": (
                        "Checkpoint 缺少可比较的模型兼容性描述；"
                        "平台未把该项记录为通过，训练入口仍须执行自身兼容性复核"
                    ),
                },
            )
            return
        report = compare_checkpoint_compatibility(expected, actual)
        await emit(
            "system.validation",
            {
                "validation_id": "resume_checkpoint_compatible",
                "status": "passed" if report["compatible"] else "blocked",
                "phase": "worker",
                "message": report.get("message"),
                "reason": report.get("reason"),
                "differences": report.get("differences") or [],
                "runtime_differences": report.get("runtime_differences") or [],
            },
        )
        if not report["compatible"]:
            fields = ", ".join(
                str(item.get("field")) for item in (report.get("differences") or [])[:8]
            )
            raise ValueError(
                "Resume checkpoint failed the frozen model compatibility contract"
                + (f": {fields}" if fields else "")
            )

    @staticmethod
    def validate_training_initialization_binding(task: dict) -> None:
        payload = task.get("payload") or {}
        if not payload.get("supervision_training_view_inputs"):
            return
        input_model_ref = payload.get("input_model_ref") or {}
        if str(input_model_ref.get("type") or "") != "checkpoint_finetune":
            return
        resume_checkpoint = payload.get("resume_checkpoint") or {}
        if not resume_checkpoint.get("checkpoint_id"):
            raise ValueError(
                "Native post-training declares checkpoint_finetune input but the frozen "
                "resume checkpoint is missing; refusing random initialization"
            )

    async def materialize_training_validation(self, task: dict, run_dir: Path) -> None:
        validation_set = (task.get("payload") or {}).get("validation_set") or {}
        artifact_id = str(validation_set.get("artifact_id") or "")
        expected_hash = str(validation_set.get("content_hash") or "")
        if not artifact_id or not expected_hash:
            raise ValueError("Frozen training validation set is missing its artifact identity")
        project_id = str(task.get("project_id") or "")
        response = await self.client.get(
            f"/api/agent/artifacts/{artifact_id}", params={"project_id": project_id}
        )
        response.raise_for_status()
        artifact = response.json()
        if artifact.get("kind") != "training_validation":
            raise ValueError("Frozen training validation set references the wrong artifact kind")
        if artifact.get("content_hash") != expected_hash:
            raise ValueError("Frozen training validation set hash does not match")
        if not any(
            location.get("storage_type") == "managed"
            for location in (artifact.get("locations") or [])
        ):
            raise FileNotFoundError("Frozen training validation set has no managed payload")

        target_dir = run_dir / "inputs" / "validation" / artifact_id
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(str(validation_set.get("filename") or "validation.jsonl")).name
        target = target_dir / filename
        digest = hashlib.sha256()
        async with self.client.stream(
            "GET",
            f"/api/agent/artifacts/{artifact_id}/payload",
            params={"project_id": project_id},
            timeout=None,
        ) as download:
            download.raise_for_status()
            with target.open("wb") as output:
                async for chunk in download.aiter_bytes(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected_hash:
            target.unlink(missing_ok=True)
            raise ValueError("Downloaded training validation set failed SHA-256 verification")

        runtime_path = str(target)
        validation_set["runtime_path"] = runtime_path
        command = (task.get("payload") or {}).get("command") or {}
        command.setdefault("env", {})["LLM_PLATFORM_VALIDATION_FILE"] = runtime_path
        replacements = {
            "__validation_file__": runtime_path,
            "${VALIDATION_FILE}": runtime_path,
            "${LLM_PLATFORM_VALIDATION_FILE}": runtime_path,
        }
        command["args"] = [
            self._replace_runtime_tokens(arg, replacements)
            for arg in (command.get("args") or [])
        ]

    async def materialize_training_validation_data_input(
        self, task: dict, run_dir: Path
    ) -> None:
        payload = task.get("payload") or {}
        validation_input = payload.get("validation_data_input")
        if not isinstance(validation_input, dict):
            raise ValueError("Frozen validation data input must be an object")
        if validation_input.get("format") == "forgelm-validation-token-holdout-v1":
            view_id = str(
                validation_input.get("training_data_view_version_id") or ""
            )
            matching_views = [
                item
                for item in (payload.get("training_data_view_inputs") or [])
                if str(item.get("version_id") or "") == view_id
                and item.get("identity_hash")
                == validation_input.get("training_data_view_identity_hash")
            ]
            if not view_id or len(matching_views) != 1:
                raise ValueError(
                    "Validation holdout does not match a frozen Training Data View"
                )
            return
        command = payload.get("command") or {}
        command_env = command.setdefault("env", {})
        data_env_keys = ("LLM_PLATFORM_DATA_MANIFEST", "LLM_PLATFORM_DATA_MANIFESTS")
        saved_env = {key: command_env.get(key) for key in data_env_keys}
        original_inputs = payload.get("data_inputs")
        try:
            payload["data_inputs"] = [validation_input]
            await self.materialize_data_inputs(task, run_dir)
        finally:
            payload["data_inputs"] = original_inputs
            for key, value in saved_env.items():
                if value is None:
                    command_env.pop(key, None)
                else:
                    command_env[key] = value
        runtime_path = str(validation_input.get("runtime_path") or "")
        if not runtime_path:
            raise ValueError("Validation data mixture was not materialized")
        command_env["LLM_PLATFORM_VALIDATION_MANIFEST"] = runtime_path

    @staticmethod
    def _replace_runtime_tokens(value: str, replacements: dict[str, str]) -> str:
        result = str(value)
        for token, replacement in replacements.items():
            result = result.replace(token, replacement)
        return result

    async def _materialize_dataset_snapshot(
        self,
        artifact_id: str,
        expected_hash: str,
        target_dir: Path,
        project_id: str,
        *,
        emit: Any | None = None,
        source_label: str = "数据快照",
        expected_kinds: set[str] | None = None,
        extract_archives: bool = True,
        preferred_location_id: str | None = None,
    ) -> Path:
        async def report(message: str, **details: Any) -> None:
            if emit is not None:
                await emit(
                    "log",
                    {
                        "stream": "platform",
                        "level": "info",
                        "message": message,
                        **details,
                    },
                )

        artifact_params = {"project_id": project_id}
        response = await self.client.get(
            f"/api/agent/artifacts/{artifact_id}",
            params={**artifact_params, "worker_id": self.worker_id},
        )
        response.raise_for_status()
        artifact = response.json()
        if artifact.get("kind") not in (expected_kinds or {"dataset_snapshot"}):
            raise ValueError(f"{source_label} references the wrong artifact kind")
        if artifact.get("content_hash") != expected_hash:
            raise ValueError("Dataset snapshot hash does not match")
        locations = artifact.get("locations") or []
        if preferred_location_id:
            preferred = next(
                (
                    location
                    for location in locations
                    if str(location.get("id") or "") == preferred_location_id
                    and location.get("availability_state") == "available"
                ),
                None,
            )
            if preferred is None:
                raise FileNotFoundError(
                    f"Selected dataset location {preferred_location_id} is unavailable"
                )
            locations = [preferred]
        external = next(
            (
                path
                for location in locations
                if location.get("storage_type") == "external"
                and location.get("availability_state") == "available"
                and location.get("worker_id") in {None, self.worker_id}
                and (path := self._local_artifact_location(location)) is not None
            ),
            None,
        )
        if external is not None:
            await report(
                f"{source_label}已使用 Worker 可访问的外部位置",
                phase="materializing_dataset_snapshot",
                artifact_id=artifact_id,
            )
            if extract_archives and external.is_file() and (
                zipfile.is_zipfile(external) or tarfile.is_tarfile(external)
            ):
                destination = target_dir / "extracted"
                self._extract_data_archive(external, destination)
                children = [
                    child for child in destination.iterdir() if child.name != "__MACOSX"
                ]
                return (
                    children[0]
                    if len(children) == 1 and children[0].is_dir()
                    else destination
                )
            return external

        managed_uri = next(
            (
                str(location.get("uri"))
                for location in locations
                if location.get("storage_type") == "managed"
            ),
            None,
        )
        if managed_uri is None:
            raise FileNotFoundError(
                f"Dataset snapshot {artifact_id} has no location visible to this Worker"
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(unquote(urlparse(managed_uri).path)).name or "dataset.bin"
        payload_path = target_dir / filename
        expected_size = int(artifact.get("size_bytes") or 0)
        if expected_size:
            free_bytes = shutil.disk_usage(target_dir).free
            required_bytes = expected_size + 1024 * 1024 * 1024
            if free_bytes < required_bytes:
                raise OSError(
                    f"{source_label} needs {_format_bytes(required_bytes)} free, but only "
                    f"{_format_bytes(free_bytes)} is available"
                )
        await report(
            f"正在下载{source_label}"
            + (f"（{_format_bytes(expected_size)}）" if expected_size else ""),
            phase="downloading_dataset_snapshot",
            artifact_id=artifact_id,
            size_bytes=expected_size,
        )
        digest = hashlib.sha256()
        downloaded = 0
        next_report_percent = 5
        next_report_bytes = 1024 * 1024 * 1024
        async with self.client.stream(
            "GET",
            f"/api/agent/artifacts/{artifact_id}/payload",
            params=artifact_params,
            timeout=None,
        ) as download:
            download.raise_for_status()
            with payload_path.open("wb") as output:
                async for chunk in download.aiter_bytes(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
                    downloaded += len(chunk)
                    if expected_size:
                        percent = min(100, int(downloaded * 100 / expected_size))
                        if percent >= next_report_percent:
                            await report(
                                f"{source_label}下载中：{percent}%（{_format_bytes(downloaded)} / {_format_bytes(expected_size)}）",
                                phase="downloading_dataset_snapshot",
                                artifact_id=artifact_id,
                                bytes_downloaded=downloaded,
                                bytes_total=expected_size,
                                progress_percent=percent,
                            )
                            next_report_percent = (percent // 5 + 1) * 5
                    elif downloaded >= next_report_bytes:
                        await report(
                            f"{source_label}下载中：已接收 {_format_bytes(downloaded)}",
                            phase="downloading_dataset_snapshot",
                            artifact_id=artifact_id,
                            bytes_downloaded=downloaded,
                        )
                        next_report_bytes = downloaded + 1024 * 1024 * 1024
        if digest.hexdigest() != expected_hash:
            payload_path.unlink(missing_ok=True)
            raise ValueError("Downloaded dataset snapshot failed SHA-256 verification")
        await report(
            f"{source_label}下载并校验完成（{_format_bytes(payload_path.stat().st_size)}）",
            phase="verifying_dataset_snapshot",
            artifact_id=artifact_id,
            size_bytes=payload_path.stat().st_size,
        )
        if extract_archives and (zipfile.is_zipfile(payload_path) or tarfile.is_tarfile(payload_path)):
            destination = target_dir / "extracted"
            self._extract_data_archive(payload_path, destination)
            children = [child for child in destination.iterdir() if child.name != "__MACOSX"]
            return children[0] if len(children) == 1 and children[0].is_dir() else destination
        return payload_path

    @classmethod
    def _extract_data_archive(cls, payload_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(payload_path):
            with zipfile.ZipFile(payload_path) as archive:
                for member in archive.infolist():
                    mode = member.external_attr >> 16
                    if mode & 0o170000 == 0o120000:
                        raise ValueError(
                            f"Dataset archive contains a symbolic link: {member.filename}"
                        )
                    target = cls._safe_archive_target(destination, member.filename)
                    if member.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, target.open("wb") as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
        else:
            with tarfile.open(payload_path) as archive:
                for member in archive.getmembers():
                    if member.issym() or member.islnk():
                        raise ValueError(f"Dataset archive contains a link: {member.name}")
                    target = cls._safe_archive_target(destination, member.name)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = archive.extractfile(member)
                        if source is not None:
                            with source, target.open("wb") as output:
                                while chunk := source.read(1024 * 1024):
                                    output.write(chunk)

    async def materialize_inference_model(self, task: dict, run_dir: Path, emit=None, desired_state=lambda: "running") -> None:
        model = (task.get("payload") or {}).get("model") or {}
        if model.get("runtime_source"):
            return
        artifact_id = model.get("artifact_id")
        if not artifact_id:
            raise ValueError(
                "Model version has neither inference.pretrained metadata nor an artifact"
            )
        artifact_params = {"project_id": str(task.get("project_id") or "")}
        response = await self.client.get(
            f"/api/agent/artifacts/{artifact_id}", params=artifact_params
        )
        response.raise_for_status()
        artifact = response.json()
        locations = artifact.get("locations") or []
        external_locations = [
            location
            for location in locations
            if location.get("storage_type") == "external"
            and location.get("availability_state") == "available"
            and location.get("worker_id") in {None, self.worker_id}
        ]
        source_path = next(
            (
                path
                for location in external_locations
                if (path := self._local_file_uri(str(location.get("uri") or ""))) is not None
            ),
            None,
        )
        if source_path is not None and source_path.is_dir():
            configs = sorted(
                source_path.rglob("config.json"),
                key=lambda path: len(path.relative_to(source_path).parts),
            )
            if not configs:
                raise ValueError("Model directory does not contain config.json")
            runtime_source = configs[0].parent
        else:
            from .inference_files import cached_model
            artifact.setdefault("id", str(artifact_id))
            runtime_source, hit = await cached_model(self, task, run_dir, artifact, source_path, emit, desired_state)
            model["file_cache_hit"] = hit
        model["runtime_identity"] = str(artifact_id) + ":" + str(artifact.get("content_hash") or "")
        model["runtime_source"] = str(runtime_source)

    def release_idle_inference_runtimes(self, task: dict, allocation: dict) -> list[str]:
        # loop() awaits each task: these caches cannot belong to another active task.
        if not allocation.get("gpu_indexes") and not allocation.get("gpus"):
            return []
        released = []
        for adapter_id, adapter in (
            ("builtin.microlm-inference", self.microlm_inference),
            ("builtin.transformers-inference", self.transformers_inference),
        ):
            if task.get("adapter_id") != adapter_id and adapter.release_runtime():
                released.append(adapter_id)
        return released

    async def execute(self, claimed: dict) -> None:
        assert self.worker_id
        from llm_platform_core.storage_layout import relocated_runtime_view
        task = copy.deepcopy(claimed["task"])
        attempt = claimed["attempt"]
        task_id = task["id"]
        attempt_id = attempt["id"]
        command = (task.get("payload") or {}).get("command")
        allocation = (claimed.get("lease") or {}).get("allocation") or {}
        environment_snapshot: dict[str, Any] | None = None
        try:
            task = relocated_runtime_view(task)
            command = (task.get("payload") or {}).get("command")
            control_plane_workspace_uri = str(task.get("project_workspace_uri") or "")
            project_workspace = worker_project_workspace(
                str(task.get("project_id") or ""),
                workspace_root=self.config.workspace_root,
                legacy_workspace_uri=control_plane_workspace_uri,
                project_name=str(task.get("project_name") or ""),
                system_kind=str(task.get("project_system_kind") or ""),
                project_storage_key=str(task.get("project_storage_key") or ""),
            )
            workspace_uri = str(project_workspace)
            payload = task.get("payload") or {}
            if isinstance(payload, dict):
                if "project_workspace_uri" in payload:
                    payload["project_workspace_uri"] = workspace_uri
                source_config = payload.get("source_config")
                if isinstance(source_config, dict) and "project_workspace_uri" in source_config:
                    source_config["project_workspace_uri"] = workspace_uri
            training_run_id = str((task.get("payload") or {}).get("training_run_id") or "")
            if task.get("kind") == "train" and training_run_id:
                run_dir = scoped_runtime_path(
                    project_training_run_root(workspace_uri), training_run_id
                )
                attempt_dir = scoped_runtime_path(run_dir, "attempts", attempt_id)
            else:
                run_dir = scoped_runtime_path(project_run_root(workspace_uri), task_id, attempt_id)
                attempt_dir = run_dir
            attempt_dir.mkdir(parents=True, exist_ok=True)
            released_runtimes = await asyncio.to_thread(
                self.release_idle_inference_runtimes, task, allocation
            )
            environment_snapshot = await asyncio.to_thread(self.environment_snapshot, allocation)
            if released_runtimes:
                environment_snapshot["released_inference_runtimes"] = released_runtimes
            resource_batch_policy = (task.get("payload") or {}).get(
                "resource_batch_policy"
            )
            if resource_batch_policy is not None:
                resolved_resource_batch_plan = resolve_resource_batch_plan(
                    resource_batch_policy, environment_snapshot, allocation
                )
                task["payload"]["resolved_resource_batch_plan"] = (
                    resolved_resource_batch_plan
                )
                environment_snapshot["resource_batch_plan"] = resolved_resource_batch_plan
        except Exception as exc:
            safe_message = str(_redact_log_value(str(exc)))
            started = await self.client.post(
                f"/api/agent/tasks/{task_id}/start",
                json={
                    "worker_id": self.worker_id,
                    "attempt_id": attempt_id,
                    "runtime_handle": {"type": "agent-coroutine", "task_id": task_id},
                    "environment_snapshot": environment_snapshot
                    or self.environment_snapshot(allocation),
                },
            )
            started.raise_for_status()
            completed = await self.client.post(
                f"/api/agent/tasks/{task_id}/complete",
                json={
                    "worker_id": self.worker_id,
                    "attempt_id": attempt_id,
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": safe_message},
                    "exit_code": 1,
                },
            )
            completed.raise_for_status()
            return
        task["runtime_context"] = {
            "worker_workspace": str(self.local_workspace_root() or ""),
            "run_dir": str(run_dir),
            "attempt_dir": str(attempt_dir),
            "project_workspace": workspace_uri,
            "control_plane_workspace_uri": control_plane_workspace_uri,
            "worker_id": self.worker_id,
            "allocation": allocation,
        }
        spec_path: Path | None = None
        if isinstance(command, dict):
            if (
                task.get("kind") == "train"
                and not (task.get("payload") or {}).get("code")
                and not command.get("cwd")
            ):
                command["cwd"] = str(run_dir)
            command_env = command.setdefault("env", {})
            spec_path = attempt_dir / "task-spec.json"
            spec_path.write_text(
                json.dumps(
                    {"task": task, "attempt": attempt, "allocation": allocation},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            gpu_indexes = allocation.get("gpu_indexes") or []
            if gpu_indexes:
                command_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in gpu_indexes)
            command_env["LLM_PLATFORM_TASK_ID"] = task_id
            command_env["LLM_PLATFORM_ATTEMPT_ID"] = attempt_id
            command_env["LLM_PLATFORM_TASK_SPEC"] = str(spec_path)
            command_env["LLM_PLATFORM_RUN_DIR"] = str(run_dir)
            command_env["LLM_PLATFORM_ATTEMPT_DIR"] = str(attempt_dir)
            command_env["LLM_PLATFORM_PROJECT_WORKSPACE"] = workspace_uri
            if training_run_id:
                command_env["LLM_PLATFORM_TRAINING_RUN_ID"] = training_run_id
            if task.get("project_id"):
                command_env["LLM_PLATFORM_PROJECT_ID"] = task["project_id"]
        response = await self.client.post(
            f"/api/agent/tasks/{task_id}/start",
            json={
                "worker_id": self.worker_id,
                "attempt_id": attempt_id,
                "runtime_handle": {
                    "type": "agent-coroutine",
                    "task_id": task_id,
                    "run_dir": str(run_dir),
                    "attempt_dir": str(attempt_dir),
                },
                "environment_snapshot": environment_snapshot
                or self.environment_snapshot(allocation),
            },
        )
        response.raise_for_status()
        started_task = response.json()
        worker_sequence = 0
        event_lock = asyncio.Lock()
        requested_state = str(started_task.get("desired_state", "running"))
        heartbeat_stop = asyncio.Event()
        log_summary: dict[str, Any] = {
            "line_count": 0,
            "stdout_lines": 0,
            "stderr_lines": 0,
            "platform_lines": 0,
            "truncated_values": 0,
            "collection": "task_events",
            "event_delivery_failures": 0,
        }

        async def keep_attempt_alive() -> None:
            nonlocal requested_state
            while not heartbeat_stop.is_set():
                try:
                    await asyncio.wait_for(heartbeat_stop.wait(), timeout=10)
                    return
                except TimeoutError:
                    try:
                        controls = await self.heartbeat([attempt_id])
                    except (httpx.HTTPError, httpx.TimeoutException):
                        continue
                    for control in controls:
                        if control.get("attempt_id") == attempt_id:
                            requested_state = str(control.get("desired_state", "running"))

        heartbeat_task = asyncio.create_task(keep_attempt_alive())

        async def emit(event_type: str, payload: dict[str, Any]) -> str:
            nonlocal requested_state, worker_sequence
            if event_type == "training.checkpoint.upload":
                uploaded = await self.upload_outputs({**task, "payload": {}}, attempt_id, {"outputs": [{"path": payload["path"], "kind": "training_state_checkpoint", "metadata": payload["metadata"]}]})
                return await emit("training.checkpoint", {"artifact_id": uploaded[0]["artifact_id"], "metadata": payload["metadata"]})
            # SuiteRuntime validates and persists these raw plugin announcements.
            if event_type == "training.checkpoint" and "artifact_id" not in payload:
                return requested_state
            if event_type == "log":
                original_strings = [
                    value for value in payload.values() if isinstance(value, str)
                ]
                payload = _redact_log_value(payload)
                log_summary["line_count"] += 1
                stream = str(payload.get("stream") or "platform")
                counter = f"{stream}_lines" if stream in {"stdout", "stderr"} else "platform_lines"
                log_summary[counter] += 1
                log_summary["truncated_values"] += sum(
                    len(value) > _MAX_LOG_VALUE_CHARS for value in original_strings
                )
            # stdout, stderr, and adapter-level progress monitors can emit at the
            # same time. Keep a task attempt's event requests ordered so the API's
            # task-local sequence allocation cannot race with itself.
            async with event_lock:
                worker_sequence += 1
                event_request = {
                    "worker_id": self.worker_id,
                    "attempt_id": attempt_id,
                    "event_type": event_type,
                    "payload": payload,
                    "worker_sequence": worker_sequence,
                }
                last_error: Exception | None = None
                for delay in _EVENT_DELIVERY_RETRY_DELAYS:
                    if delay:
                        await asyncio.sleep(delay)
                    try:
                        event_response = await self.client.post(
                            f"/api/agent/tasks/{task_id}/events",
                            json=event_request,
                        )
                    except httpx.TransportError as exc:
                        last_error = exc
                        continue
                    if event_response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                        last_error = httpx.HTTPStatusError(
                            f"retryable event response {event_response.status_code}",
                            request=event_response.request,
                            response=event_response,
                        )
                        continue
                    event_response.raise_for_status()
                    requested_state = str(
                        event_response.json().get("desired_state", "running")
                    )
                    return requested_state
                log_summary["event_delivery_failures"] += 1
                if event_type in {"execution.resolved", "training.checkpoint", "execution.interval"}:
                    raise RuntimeError(f"Required training record could not be persisted: {event_type}") from last_error
                log_summary["last_event_delivery_error"] = str(
                    _redact_log_value(str(last_error or "unknown transport error"))
                )
                return requested_state

        monitor = AttemptResourceMonitor(
            attempt_dir / "runtime-metrics.jsonl.gz",
            allocation,
            emit,
        )
        monitor_started = False
        observability_result: tuple[dict[str, Any], str | None] | None = None

        async def finish_observability() -> tuple[dict[str, Any], str | None]:
            nonlocal observability_result
            if observability_result is not None:
                return observability_result
            resource_summary: dict[str, Any]
            if monitor_started:
                try:
                    resource_summary = await monitor.stop()
                except Exception as exc:
                    monitor.close()
                    resource_summary = monitor.summary()
                    resource_summary.setdefault("warnings", []).append(
                        f"resource monitor finalization failed: {type(exc).__name__}: {exc}"
                    )
                    try:
                        await emit(
                            "log",
                            {
                                "stream": "platform",
                                "level": "warning",
                                "message": "资源监控结束失败，任务结果不受影响",
                                "error_type": type(exc).__name__,
                            },
                        )
                    except Exception:
                        pass
            else:
                monitor.close()
                resource_summary = {
                    "schema_version": monitor.schema_version,
                    "sample_count": 0,
                    "allocation": allocation,
                    "warnings": ["resource monitor did not start"],
                }
            metrics_artifact_id: str | None = None
            if monitor.output_path.is_file():
                try:
                    artifact = await self.upload_attempt_artifact(
                        task,
                        attempt_id,
                        monitor.output_path,
                        kind="runtime_metrics",
                        media_type="application/gzip",
                        metadata={
                            "schema_version": monitor.schema_version,
                            "sample_count": resource_summary.get("sample_count", 0),
                            "attempt_id": attempt_id,
                        },
                    )
                    metrics_artifact_id = str(artifact["id"])
                except Exception as exc:
                    resource_summary.setdefault("warnings", []).append(
                        f"runtime metrics upload failed: {type(exc).__name__}"
                    )
                    try:
                        await emit(
                            "log",
                            {
                                "stream": "platform",
                                "level": "warning",
                                "message": "完整资源时序制品上传失败",
                                "error_type": type(exc).__name__,
                            },
                        )
                    except Exception:
                        pass
            for filename, kind, media_type in (
                ("command-output.jsonl.gz", "task_log", "application/gzip"),
                ("generation-runtime.json", "generation_runtime", "application/json"),
                ("generation-metrics.jsonl", "generation_metrics", "application/x-ndjson"),
                ("generation-validation.jsonl", "generation_validation", "application/x-ndjson"),
            ):
                path = attempt_dir / filename
                if not path.is_file():
                    continue
                try:
                    artifact = await self.upload_attempt_artifact(
                        task, attempt_id, path, kind=kind, media_type=media_type,
                        metadata={"attempt_id": attempt_id},
                    )
                    resource_summary[f"{kind}_artifact_id"] = str(artifact["id"])
                except Exception as exc:
                    resource_summary.setdefault("warnings", []).append(
                        f"{kind} upload failed: {type(exc).__name__}"
                    )
            observability_result = (resource_summary, metrics_artifact_id)
            return observability_result

        active_training_validation = "task_input_materialization"
        try:
            try:
                await monitor.start()
                monitor_started = True
            except Exception as exc:
                await emit(
                    "log",
                    {
                        "stream": "platform",
                        "level": "warning",
                        "message": "资源监控启动失败，任务将继续执行",
                        "error_type": type(exc).__name__,
                    },
                )
            await emit(
                "log",
                {
                    "stream": "platform",
                    "level": "info",
                    "message": f"Worker 已开始准备任务（{task['adapter_id']}）",
                    "phase": "preparing_task",
                    "adapter_id": task["adapter_id"],
                },
            )
            if environment_snapshot.get("released_inference_runtimes"):
                await emit("resource.inference_cache.released", {
                    "adapters": environment_snapshot["released_inference_runtimes"],
                    "message": "切换GPU任务前已释放空闲推理缓存，批量适配使用释放后的显存容量",
                })
            resolved_resource_batch_plan = (task.get("payload") or {}).get(
                "resolved_resource_batch_plan"
            )
            if isinstance(resolved_resource_batch_plan, dict):
                await emit("resource.batch_plan.resolved", resolved_resource_batch_plan)
            materialized_task_spec = False
            if task.get("kind") == "train":
                self.validate_training_initialization_binding(task)
            if isinstance(command, dict) and (
                (task.get("payload") or {}).get("data_inputs")
                or (task.get("payload") or {}).get("data_input")
            ):
                await self.materialize_data_inputs(task, run_dir, emit)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "evaluation_inputs"
            ):
                await self.materialize_evaluation_inputs(task, run_dir, emit)
                materialized_task_spec = True
            if (task.get("payload") or {}).get("generation_input"):
                binding = task["payload"]["generation_input"]
                binding["runtime_path"] = str(await self._materialize_dataset_snapshot(
                    binding["artifact_id"], binding["artifact_hash"],
                    run_dir / "inputs" / "generation", str(task.get("project_id") or ""),
                    emit=emit, source_label="生成输入",
                ))
                model = task["payload"]["generation_model"]
                model["runtime_path"] = str(await self._materialize_external_directory_binding(
                    task, model, expected_kind="model_weights", expected_formats={"forgelm-model-weight-v1"},
                ))
            if (task.get("payload") or {}).get("model_weight_inputs"):
                await self.materialize_evaluation_model_weights(task)
                materialized_task_spec = True
            if (task.get("payload") or {}).get("evaluation_model_inputs"):
                await self.materialize_evaluation_models(task, run_dir, emit)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "analysis_input"
            ):
                await self.materialize_dataset_analysis_input(task, run_dir, emit)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "dataset_viewer_input"
            ):
                await self.materialize_dataset_viewer_input(task, run_dir, emit)
                materialized_task_spec = True
            if task.get("adapter_id") == "builtin.dataset-labeler":
                await self.materialize_dataset_label_inputs(task, run_dir, emit)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "data_processing_input"
            ):
                await self.materialize_data_processing_input(task, run_dir, emit)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "data_conversion_input"
            ):
                await self.materialize_data_conversion_input(task, run_dir, emit)
                materialized_task_spec = True
            if task.get("adapter_id") == "builtin.dataset-materializer":
                await self.materialize_dataset_replica_input(task, run_dir, emit)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "validation_data_input"
            ):
                await self.materialize_training_validation_data_input(task, run_dir)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get("tokenizer"):
                await self.materialize_training_tokenizer(task, run_dir)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "tokenized_data_inputs"
            ):
                await self.materialize_tokenized_data_inputs(task, run_dir)
                materialized_task_spec = True
            if (task.get("payload") or {}).get("training_selection_inputs"):
                await self.materialize_training_selections(task, run_dir, emit)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "tokenized_corpora"
            ):
                await self.materialize_tokenized_corpora(task, run_dir)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "training_data_view_inputs"
            ):
                await self.materialize_training_data_views(task, run_dir)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "prefix_training_data_view"
            ):
                await self.materialize_training_data_view_prefix(task)
                materialized_task_spec = True
            if (task.get("payload") or {}).get("supervision_sources"):
                await self.materialize_supervision_sources(task)
                materialized_task_spec = True
            if (task.get("payload") or {}).get("supervision_corpus"):
                await self.materialize_supervision_corpus(task)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "supervision_training_view_inputs"
            ):
                await self.materialize_supervision_training_views(task)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "resume_checkpoint"
            ):
                await self.materialize_training_checkpoint(task, run_dir)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "reference_checkpoint"
            ):
                await self.materialize_training_checkpoint(
                    task,
                    run_dir,
                    binding_key="reference_checkpoint",
                    environment_key="LLM_PLATFORM_REFERENCE_CHECKPOINT_DIR",
                )
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get("validation_set"):
                await self.materialize_training_validation(task, run_dir)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "model_specification"
            ):
                await self.materialize_model_specification(task, run_dir)
                materialized_task_spec = True
            if isinstance(command, dict) and (task.get("payload") or {}).get("code"):
                await self.materialize_code_version(task, run_dir)
                materialized_task_spec = True
            if materialized_task_spec:
                assert spec_path is not None
                spec_path.write_text(
                    json.dumps(
                        {"task": task, "attempt": attempt, "allocation": allocation},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                await emit(
                    "system.validation",
                    {
                        "validation_id": "task_input_materialization",
                        "status": "passed",
                        "phase": "worker",
                        "message": "Worker 已按冻结身份完成本次任务输入物化。",
                    },
                )
            if isinstance(command, dict) and (task.get("payload") or {}).get(
                "resume_checkpoint"
            ):
                active_training_validation = "resume_checkpoint_compatible"
                await self.validate_training_checkpoint_compatibility(task, emit)
            active_training_validation = "runtime_training_invariants"
            await emit(
                "log",
                {
                    "stream": "platform",
                    "level": "info",
                    "message": f"任务输入准备完成，正在启动适配器（{task['adapter_id']}）",
                    "phase": "starting_adapter",
                    "adapter_id": task["adapter_id"],
                },
            )
            if task["adapter_id"] == "builtin.plugin-manager":
                result = await self.plugin_manager.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.suite-manager":
                result = await self.suite_runtime.manage(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.suite-runner":
                result = await self.suite_runtime.execute(task, emit, lambda: requested_state)
            elif task["adapter_id"].startswith("plugin."):
                if spec_path is None:
                    raise RuntimeError("Plugin runtime directory was not initialized")
                preparation_started = time.perf_counter()
                await emit("progress", {"phase": "preparing_model", "progress": 0.01})
                await self.materialize_inference_model(task, run_dir, emit, lambda: requested_state)
                task["payload"]["model"]["preparation_ms"] = round((time.perf_counter() - preparation_started) * 1000, 2)
                spec_path.write_text(
                    json.dumps(
                        {"task": task, "attempt": attempt, "allocation": allocation},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                result = await self.installed_plugin.run(
                    task, emit, lambda: requested_state
                )
            elif task["adapter_id"] in {
                "builtin.transformers-inference",
                "builtin.microlm-inference",
            }:
                if spec_path is None:
                    raise RuntimeError("Inference runtime directory was not initialized")
                preparation_started = time.perf_counter()
                await emit("progress", {"phase": "preparing_model", "progress": 0.01})
                await self.materialize_inference_model(task, run_dir, emit, lambda: requested_state)
                task["payload"]["model"]["preparation_ms"] = round((time.perf_counter() - preparation_started) * 1000, 2)
                spec_path.write_text(
                    json.dumps(
                        {"task": task, "attempt": attempt, "allocation": allocation},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                inference_adapter = (
                    self.microlm_inference
                    if task["adapter_id"] == "builtin.microlm-inference"
                    else self.transformers_inference
                )
                result = await inference_adapter.run(task, emit, lambda: requested_state)
                await self.heartbeat([attempt_id])
            elif task["adapter_id"] == "builtin.code-import":
                result = await self.code_import.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.data-import":
                result = await self.data_import.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] in {"builtin.data-generation.transformers", "builtin.data-generation.vllm"}:
                result = await self.data_generation.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.data-conversion":
                result = await self.data_conversion.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.data-juicer":
                result = await self.data_juicer.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.storage-maintenance":
                result = await self.storage_maintenance.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.data-mixture":
                result = await self.data_mixture.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.dataset-materializer":
                result = await self.dataset_materializer.run(
                    task, emit, lambda: requested_state
                )
            elif task["adapter_id"] == "builtin.dataset-viewer":
                result = await self.dataset_viewer.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.evaluation-viewer":
                result = await self.evaluation_viewer.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.dataset-labeler":
                result = await self.dataset_labeler.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.pretokenize":
                result = await self.pretokenize.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.tokenize-corpus":
                result = await self.tokenized_corpus.run(
                    task, emit, lambda: requested_state
                )
            elif task["adapter_id"] == "builtin.build-training-index":
                result = await self.training_data_index.run(
                    task, emit, lambda: requested_state
                )
            elif task["adapter_id"] == "builtin.supervision-scan":
                result = await self.supervision_scan.run(
                    task, emit, lambda: requested_state
                )
            elif task["adapter_id"] == "builtin.supervision-corpus":
                result = await self.supervision_corpus.run(
                    task, emit, lambda: requested_state
                )
            elif task["adapter_id"] == "builtin.build-supervision-index":
                result = await self.supervision_training_index.run(
                    task, emit, lambda: requested_state
                )
            elif task["adapter_id"] == "builtin.lm-eval":
                result = await self.lm_eval.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.model-weight":
                result = await self.model_weight.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.hf-tokenizer":
                result = await self.hf_tokenizer.run(task, emit, lambda: requested_state)
            elif task["adapter_id"] == "builtin.disk-bpe-tokenizer":
                result = await self.disk_bpe_tokenizer.run(task, emit, lambda: requested_state)
            elif task["adapter_id"].startswith("builtin.command.") or (
                task["adapter_id"] in self.config.extra_capabilities
                and isinstance((task.get("payload") or {}).get("command"), dict)
                and (task.get("payload") or {}).get("command", {}).get("executable")
            ):
                result = await self.command.run(task, emit, lambda: requested_state)
            else:
                raise RuntimeError(f"Worker cannot load adapter {task['adapter_id']}")
            active_training_validation = "output_artifact_integrity"
            await emit(
                "log",
                {
                    "stream": "platform",
                    "level": "info",
                    "message": "适配器执行完成，正在登记任务输出",
                    "phase": "finalizing_outputs",
                    "adapter_id": task["adapter_id"],
                },
            )
            if task.get("kind") == "dataset_materialize" and not result.get(
                "materialized_uri"
            ):
                dataset_output = next(
                    (
                        output
                        for output in (result.get("outputs") or [])
                        if isinstance(output, dict)
                        and str(output.get("ref") or "") == "dataset"
                        and output.get("path")
                    ),
                    None,
                )
                if dataset_output is not None:
                    result["materialized_uri"] = Path(
                        str(dataset_output["path"])
                    ).resolve().as_uri()
            uploaded_outputs = await self.upload_outputs(task, attempt_id, result)
            if uploaded_outputs:
                result["output_artifacts"] = [
                    *(result.get("output_artifacts") or []),
                    *uploaded_outputs,
                ]
                result["output_artifact_ids"] = list(
                    dict.fromkeys(
                        [
                            *(result.get("output_artifact_ids") or []),
                            *(item["artifact_id"] for item in uploaded_outputs),
                        ]
                    )
                )
            controlled = result.get("controlled_state")
            status = (
                "paused"
                if controlled == "paused"
                else "cancelled"
                if controlled == "cancelled"
                else "succeeded"
            )
            resource_summary, metrics_artifact_id = await finish_observability()
            await self._post_completion(
                task_id,
                {
                    "worker_id": self.worker_id,
                    "attempt_id": attempt_id,
                    "status": status,
                    "result": result,
                    "exit_code": 0,
                    "resource_summary": resource_summary,
                    "runtime_metrics_artifact_id": metrics_artifact_id,
                    "log_summary": log_summary,
                },
            )
        except AnalysisInterrupted as exc:
            resource_summary, metrics_artifact_id = await finish_observability()
            await self._post_completion(task_id, {
                "worker_id": self.worker_id, "attempt_id": attempt_id,
                "status": exc.state, "result": {"controlled_state": exc.state}, "exit_code": 0,
                "resource_summary": resource_summary, "runtime_metrics_artifact_id": metrics_artifact_id,
                "log_summary": log_summary,
            })
        except Exception as exc:
            safe_message = str(_redact_log_value(str(exc)))
            try:
                if task.get("kind") == "train":
                    await emit(
                        "system.validation",
                        {
                            "validation_id": active_training_validation,
                            "status": "blocked",
                            "phase": (
                                "worker"
                                if active_training_validation
                                in {
                                    "task_input_materialization",
                                    "resume_checkpoint_compatible",
                                }
                                else "output"
                                if active_training_validation == "output_artifact_integrity"
                                else "runtime"
                            ),
                            "message": safe_message,
                            "error_type": type(exc).__name__,
                        },
                    )
                await emit(
                    "log",
                    {
                        "stream": "platform",
                        "level": "error",
                        "message": safe_message,
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
            except Exception:
                pass
            resource_summary, metrics_artifact_id = await finish_observability()
            await self._post_completion(
                task_id,
                {
                    "worker_id": self.worker_id,
                    "attempt_id": attempt_id,
                    "status": "failed",
                    "error": {"type": type(exc).__name__, "message": safe_message},
                    "exit_code": 1,
                    "resource_summary": resource_summary,
                    "runtime_metrics_artifact_id": metrics_artifact_id,
                    "log_summary": log_summary,
                },
            )
        finally:
            heartbeat_stop.set()
            await heartbeat_task
            await asyncio.to_thread(self._cleanup_attempt_scratch, run_dir)

    async def loop(self) -> None:
        while True:
            try:
                if self.worker_id is None:
                    await self.register()
                await self.heartbeat()
                claimed = await self.receive_dispatch()
                if claimed:
                    await self.execute(claimed)
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.TimeoutException):
                await asyncio.sleep(2)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRYABLE_HTTP_STATUS_CODES:
                    raise
                await asyncio.sleep(2)

    async def close(self) -> None:
        await self.client.aclose()


async def async_main(config: AgentConfig) -> None:
    agent = WorkerAgent(config)
    try:
        await agent.loop()
    finally:
        await agent.close()


def _environment_flag(name: str) -> bool:
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def run() -> None:
    parser = argparse.ArgumentParser(description="Local LLM Platform worker agent")
    parser.add_argument(
        "--api-url", default=os.getenv("LLM_PLATFORM_API_URL", "http://127.0.0.1:8000")
    )
    parser.add_argument(
        "--token", default=os.getenv("LLM_PLATFORM_AGENT_TOKEN", "local-development-token")
    )
    parser.add_argument(
        "--enrollment-environment-id",
        default=os.getenv("LLM_PLATFORM_ENROLLMENT_ENVIRONMENT_ID"),
        help="Remote environment identity used for one-time Worker enrollment.",
    )
    parser.add_argument(
        "--token-file",
        default=os.getenv("LLM_PLATFORM_AGENT_TOKEN_FILE"),
        help="Service-user-owned file for the enrolled Worker identity.",
    )
    parser.add_argument(
        "--name",
        default=os.getenv("LLM_PLATFORM_WORKER_NAME", f"local-{socket.gethostname()}"),
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--inference-idle-seconds", type=float,
                        default=float(os.getenv("LLM_PLATFORM_INFERENCE_IDLE_SECONDS", "600")))
    parser.add_argument(
        "--workspace-root",
        default=os.getenv("LLM_PLATFORM_WORKER_WORKSPACE_ROOT"),
        help="Worker-local root used for project run and data paths.",
    )
    parser.add_argument(
        "--plugin-root",
        default=os.getenv("LLM_PLATFORM_WORKER_PLUGIN_ROOT"),
        help="Worker-local root for isolated installable plugin environments.",
    )
    parser.add_argument(
        "--release-root",
        default=os.getenv("LLM_PLATFORM_WORKER_RELEASE_ROOT"),
        help="Administrator-controlled root for local release delivery.",
    )
    parser.add_argument(
        "--profile",
        choices=("general", "data"),
        default=os.getenv("LLM_PLATFORM_WORKER_PROFILE", "general"),
        help="Capability and resource profile advertised by this Worker.",
    )
    parser.add_argument(
        "--disable-gpu",
        action="store_true",
        default=_environment_flag("LLM_PLATFORM_WORKER_DISABLE_GPU"),
        help="Do not discover or advertise GPUs from this Worker.",
    )
    parser.add_argument(
        "--cpu-limit",
        type=int,
        default=(
            int(os.getenv("LLM_PLATFORM_WORKER_CPU_LIMIT"))
            if os.getenv("LLM_PLATFORM_WORKER_CPU_LIMIT", "").strip()
            else None
        ),
        help="Maximum CPU count advertised to the scheduler.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Worker routing label; repeat for multiple labels.",
    )
    parser.add_argument(
        "--capability",
        action="append",
        default=[],
        help="Additional capability advertised by this Worker.",
    )
    args = parser.parse_args()
    if args.cpu_limit is not None and args.cpu_limit < 1:
        parser.error("--cpu-limit must be at least 1")
    labels: dict[str, str] = {}
    raw_labels = os.getenv("LLM_PLATFORM_WORKER_LABELS", "").strip()
    if raw_labels:
        try:
            parsed_labels = json.loads(raw_labels)
        except json.JSONDecodeError:
            parser.error("LLM_PLATFORM_WORKER_LABELS must be valid JSON")
        if not isinstance(parsed_labels, dict):
            parser.error("LLM_PLATFORM_WORKER_LABELS must be a JSON object")
        labels.update({str(key): str(value) for key, value in parsed_labels.items()})
    for item in args.label:
        key, separator, value = item.partition("=")
        if not separator or not key.strip() or not value.strip():
            parser.error("--label must use KEY=VALUE")
        labels[key.strip()] = value.strip()
    environment_capabilities = [
        value.strip()
        for value in os.getenv("LLM_PLATFORM_WORKER_CAPABILITIES", "").split(",")
        if value.strip()
    ]
    asyncio.run(
        async_main(
            AgentConfig(
                args.api_url,
                args.token,
                args.name,
                args.poll_seconds,
                labels=labels,
                extra_capabilities=[*environment_capabilities, *args.capability],
                workspace_root=args.workspace_root,
                profile=args.profile,
                disable_gpu=args.disable_gpu,
                cpu_limit=args.cpu_limit,
                plugin_root=args.plugin_root,
                release_root=args.release_root,
                enrollment_environment_id=args.enrollment_environment_id,
                token_file=args.token_file,
                inference_idle_seconds=args.inference_idle_seconds,
            )
        )
    )


if __name__ == "__main__":
    run()
