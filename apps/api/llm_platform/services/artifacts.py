from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Artifact, ArtifactLocation, Task, WorkerNode
from .project_storage import artifact_root, local_workspace_path, project_artifact_root

ARTIFACT_KIND_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,80}")
CONTENT_HASH_PATTERN = re.compile(r"[a-fA-F0-9]{64}")
AVAILABILITY_STATES = {"available", "missing", "unknown", "offline"}


class LocalArtifactStorage:
    @staticmethod
    def _validate_identity(kind: str, content_hash: str) -> None:
        if not ARTIFACT_KIND_PATTERN.fullmatch(kind):
            raise ValueError("artifact kind contains unsupported characters")
        if not CONTENT_HASH_PATTERN.fullmatch(content_hash):
            raise ValueError("content_hash must be a SHA-256 hexadecimal digest")

    @staticmethod
    def _register_location(
        session: Session,
        artifact: Artifact,
        *,
        storage_type: str,
        uri: str,
        availability_state: str,
        project_id: str | None = None,
        worker_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactLocation:
        from .storage_relocation import guard_writes
        guard_writes(session)
        from ..models import StorageCleanupPlan, StorageCleanupPlanItem
        from .model_deletion import local_path, overlaps
        new_path = local_path(uri, session.get(WorkerNode, worker_id) if worker_id else None)
        for cleaning in session.scalars(select(StorageCleanupPlanItem).join(StorageCleanupPlan,
                StorageCleanupPlan.id == StorageCleanupPlanItem.plan_id).where(
                StorageCleanupPlan.status == "executing",
                StorageCleanupPlan.summary["kind"].as_string().in_({"model_deletion", "data_cleanup"}),
                StorageCleanupPlanItem.status != "protected")):
            if cleaning.metadata_json.get("remote"):
                if worker_id == cleaning.metadata_json.get("worker_id") and urlparse(uri).scheme == "file":
                    if overlaps(PurePosixPath(unquote(urlparse(uri).path)), PurePosixPath(cleaning.path)):
                        raise ValueError("制品位置与正在回收的模型文件重叠")
            elif new_path and overlaps(new_path, Path(cleaning.path)):
                raise ValueError("制品位置与正在回收的模型文件重叠")
        if urlparse(uri).scheme == "file":
            new_path = unquote(urlparse(uri).path).replace("\\", "/").strip("/")
            for maintenance in session.scalars(select(Task).where(
                Task.adapter_id == "builtin.storage-maintenance",
                Task.observed_state.in_({"assigned", "starting", "running", "stopping"}),
            )):
                selected_worker = maintenance.payload.get("resource_request", {}).get("worker_selector", {}).get("worker_id")
                if worker_id and worker_id != selected_worker:
                    continue
                for item in maintenance.payload.get("candidates", []):
                    target = str(item["path"]).replace("\\", "/").strip("/")
                    if new_path == target or new_path.startswith(target + "/") or target.startswith(new_path + "/"):
                        raise ValueError("artifact location overlaps an active storage cleanup")
        if storage_type not in {"managed", "external"}:
            raise ValueError("storage_type must be managed or external")
        if availability_state not in AVAILABILITY_STATES:
            raise ValueError("unsupported availability_state")
        if not uri.strip():
            raise ValueError("artifact location URI is required")
        if worker_id is not None and session.get(WorkerNode, worker_id) is None:
            raise ValueError("artifact location worker does not exist")
        location = session.scalar(select(ArtifactLocation).where(ArtifactLocation.uri == uri))
        if location:
            if location.artifact_id != artifact.id:
                raise ValueError("artifact location URI already belongs to different content")
            if worker_id and location.worker_id is None:
                location.worker_id = worker_id
            if project_id and location.project_id is None:
                location.project_id = project_id
            elif project_id and location.project_id != project_id:
                raise ValueError("artifact location already belongs to a different project")
            if availability_state != "unknown":
                location.availability_state = availability_state
            if metadata:
                location.metadata_json = {**(location.metadata_json or {}), **metadata}
            return location
        location = ArtifactLocation(
            artifact_id=artifact.id,
            project_id=project_id,
            storage_type=storage_type,
            uri=uri,
            worker_id=worker_id,
            availability_state=availability_state,
            metadata_json=metadata or {},
        )
        session.add(location)
        session.flush()
        return location

    @staticmethod
    def _managed_location(
        session: Session, artifact: Artifact, project_id: str
    ) -> ArtifactLocation | None:
        from llm_platform_core.storage_layout import PUBLIC_DATA_KINDS

        from ..config import get_settings
        query = select(ArtifactLocation).where(
            ArtifactLocation.artifact_id == artifact.id,
            ArtifactLocation.storage_type == "managed",
            ArtifactLocation.availability_state == "available",
        )
        if not (get_settings().storage_root and artifact.kind in PUBLIC_DATA_KINDS):
            query = query.where(ArtifactLocation.project_id == project_id)
        return session.scalar(query)

    def managed_payload_path(
        self, session: Session, artifact: Artifact, project_id: str
    ) -> Path | None:
        """Return a verified local path for a managed artifact payload."""
        from .global_data import project_can_access_artifact

        can_use_bound_location = project_can_access_artifact(
            session, project_id, artifact.id
        )
        query = select(ArtifactLocation).where(
            ArtifactLocation.artifact_id == artifact.id,
            ArtifactLocation.storage_type == "managed",
            ArtifactLocation.availability_state == "available",
        )
        if not can_use_bound_location:
            query = query.where(ArtifactLocation.project_id == project_id)
        locations = session.scalars(query).all()
        for location in locations:
            if location.project_id is None:
                continue
            root = artifact_root(session, location.project_id, artifact.kind, create=False)
            parsed = urlparse(location.uri)
            if parsed.scheme != "file":
                continue
            raw_path = unquote(parsed.path)
            if parsed.netloc:
                raw_path = f"//{parsed.netloc}{raw_path}"
            if len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
                raw_path = raw_path[1:]
            path = Path(raw_path).resolve()
            try:
                from ..models import Project
                legacy = local_workspace_path(session.get(Project, location.project_id).workspace_uri) / ".llm-platform" / "artifacts"
                if not path.is_relative_to(root) and not path.is_relative_to(legacy):
                    path.relative_to(project_artifact_root(session, location.project_id, create=False))
            except ValueError:
                continue
            if path.is_file():
                return path
        return None

    def read_managed_archive_json(
        self,
        session: Session,
        artifact: Artifact,
        project_id: str,
        relative_path: str,
        *,
        max_bytes: int = 1024 * 1024,
    ) -> dict[str, Any] | None:
        normalized = str(relative_path or "").replace("\\", "/").strip("/")
        member_path = PurePosixPath(normalized)
        if (
            not normalized
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise ValueError("archive JSON path must be a safe relative path")
        payload_path = self.managed_payload_path(session, artifact, project_id)
        if payload_path is None:
            return None
        payload: bytes | None = None
        if zipfile.is_zipfile(payload_path):
            with zipfile.ZipFile(payload_path) as archive:
                try:
                    info = archive.getinfo(normalized)
                except KeyError:
                    return None
                if info.is_dir() or info.file_size > max_bytes:
                    raise ValueError("archive JSON member is not a bounded file")
                payload = archive.read(info)
        elif tarfile.is_tarfile(payload_path):
            with tarfile.open(payload_path) as archive:
                try:
                    member = archive.getmember(normalized)
                except KeyError:
                    return None
                if not member.isfile() or member.issym() or member.islnk() or member.size > max_bytes:
                    raise ValueError("archive JSON member is not a bounded regular file")
                source = archive.extractfile(member)
                if source is None:
                    return None
                payload = source.read(max_bytes + 1)
        if payload is None:
            return None
        if len(payload) > max_bytes:
            raise ValueError("archive JSON member exceeds the size limit")
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("archive JSON member must contain an object")
        return value

    def read_managed_archive_jsonl_page(
        self,
        session: Session,
        artifact: Artifact,
        project_id: str,
        suffix_parts: tuple[str, ...],
        *,
        offset: int,
        limit: int,
        max_member_bytes: int = 1024 * 1024 * 1024,
        max_line_bytes: int = 2 * 1024 * 1024,
        fallback_parent_parts: tuple[str, ...] = (),
        fallback_filename_prefixes: tuple[str, ...] = (),
    ) -> tuple[list[dict[str, Any]], bool, str] | None:
        """Read a bounded JSONL page from one safely matched archive member."""
        if not suffix_parts or any(not part or part in {".", ".."} or "/" in part or "\\" in part for part in suffix_parts):
            raise ValueError("archive member suffix contains unsafe path components")
        if any(not part or part in {".", ".."} or "/" in part or "\\" in part for part in fallback_parent_parts):
            raise ValueError("archive fallback parent contains unsafe path components")
        if any(not prefix or "/" in prefix or "\\" in prefix for prefix in fallback_filename_prefixes):
            raise ValueError("archive fallback filename prefix is unsafe")
        payload_path = self.managed_payload_path(session, artifact, project_id)
        if payload_path is None:
            return None

        def match_priority(name: str) -> int:
            parts = PurePosixPath(name.replace("\\", "/")).parts
            if len(parts) >= len(suffix_parts) and tuple(parts[-len(suffix_parts):]) == suffix_parts:
                return 3
            if not fallback_parent_parts or not fallback_filename_prefixes:
                return 0
            parent_width = len(fallback_parent_parts)
            has_parent = any(
                tuple(parts[index:index + parent_width]) == fallback_parent_parts
                for index in range(max(0, len(parts) - parent_width))
            )
            if not has_parent or not parts:
                return 0
            filename = parts[-1]
            if filename == "samples.jsonl":
                return 2
            if filename.endswith(".jsonl") and any(filename.startswith(prefix) for prefix in fallback_filename_prefixes):
                return 1
            return 0

        def choose_candidate(candidates: list[Any], name_of: Any) -> Any | None:
            ranked = [(match_priority(name_of(candidate)), name_of(candidate), candidate) for candidate in candidates]
            ranked = [item for item in ranked if item[0] > 0]
            if not ranked:
                return None
            # Stable paths win over historical layouts. For multiple retry files,
            # choose the latest lexical attempt/timestamp deterministically.
            return max(ranked, key=lambda item: (item[0], item[1]))[2]

        def read_page(source: BinaryIO, member_name: str) -> tuple[list[dict[str, Any]], bool, str]:
            items: list[dict[str, Any]] = []
            record_index = 0
            while len(items) <= limit:
                line = source.readline(max_line_bytes + 1)
                if not line:
                    break
                if len(line) > max_line_bytes:
                    raise ValueError("archive JSONL record exceeds the size limit")
                if not line.strip():
                    continue
                if record_index < offset:
                    record_index += 1
                    continue
                value = json.loads(line.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("archive JSONL record must contain an object")
                items.append(value)
                record_index += 1
            return items[:limit], len(items) > limit, member_name

        if zipfile.is_zipfile(payload_path):
            with zipfile.ZipFile(payload_path) as archive:
                candidate = choose_candidate(
                    [info for info in archive.infolist() if not info.is_dir()],
                    lambda info: info.filename,
                )
                if candidate is None:
                    return None
                info = candidate
                if info.file_size > max_member_bytes:
                    raise ValueError("archive JSONL member exceeds the size limit")
                with archive.open(info) as source:
                    return read_page(source, info.filename)
        if tarfile.is_tarfile(payload_path):
            with tarfile.open(payload_path) as archive:
                candidate = choose_candidate(
                    [member for member in archive.getmembers() if member.isfile() and not member.issym() and not member.islnk()],
                    lambda member: member.name,
                )
                if candidate is None:
                    return None
                member = candidate
                if member.size > max_member_bytes:
                    raise ValueError("archive JSONL member exceeds the size limit")
                source = archive.extractfile(member)
                if source is None:
                    return None
                with source:
                    return read_page(source, member.name)
        return None

    def inference_access(
        self, session: Session, artifact: Artifact, project_id: str
    ) -> dict[str, Any]:
        """Describe how an inference Worker can access an artifact."""
        if self.managed_payload_path(session, artifact, project_id) is not None:
            return {
                "type": "managed",
                "artifact_id": artifact.id,
                "worker_ids": [],
                "globally_accessible": True,
            }
        from .global_data import project_can_access_artifact

        query = select(ArtifactLocation).where(
            ArtifactLocation.artifact_id == artifact.id,
            ArtifactLocation.storage_type == "external",
            ArtifactLocation.availability_state == "available",
        )
        if not project_can_access_artifact(session, project_id, artifact.id):
            query = query.where(
                (ArtifactLocation.project_id == project_id)
                | (ArtifactLocation.project_id.is_(None))
            )
        locations = session.scalars(query).all()
        worker_ids = {location.worker_id for location in locations if location.worker_id}
        for worker in session.scalars(select(WorkerNode)).all():
            if any(
                self._supports_local_worker_bridge(session, location, worker)
                for location in locations
            ):
                worker_ids.add(worker.id)
        return {
            "type": "external" if locations else "unavailable",
            "artifact_id": artifact.id,
            "worker_ids": sorted(worker_ids),
            "globally_accessible": any(location.worker_id is None for location in locations),
        }

    @staticmethod
    def _is_windows_drive_file_uri(uri: str) -> bool:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc:
            return False
        raw_path = unquote(parsed.path)
        return bool(re.fullmatch(r"/?[A-Za-z]:/.*", raw_path))

    def _supports_local_windows_wsl_bridge(
        self,
        session: Session,
        location: ArtifactLocation,
        target_worker: WorkerNode,
    ) -> bool:
        if (
            location.storage_type != "external"
            or location.availability_state != "available"
            or not location.worker_id
            or not self._is_windows_drive_file_uri(location.uri)
        ):
            return False
        source_worker = session.get(WorkerNode, location.worker_id)
        if source_worker is None or source_worker.id == target_worker.id:
            return False
        source_labels = source_worker.labels or {}
        target_labels = target_worker.labels or {}
        return (
            source_worker.hostname.casefold() == target_worker.hostname.casefold()
            and source_labels.get("location") == "local"
            and source_labels.get("environment") == "native"
            and source_labels.get("os") == "windows"
            and target_labels.get("location") == "local"
            and target_labels.get("environment") == "wsl"
            and bool(str(target_labels.get("distro") or "").strip())
        )

    @staticmethod
    def _is_absolute_posix_file_uri(uri: str) -> bool:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc:
            return False
        raw_path = unquote(parsed.path)
        return raw_path.startswith("/") and not bool(re.fullmatch(r"/?[A-Za-z]:/.*", raw_path))

    def _supports_local_wsl_peer_bridge(
        self,
        session: Session,
        location: ArtifactLocation,
        target_worker: WorkerNode,
    ) -> bool:
        if (
            location.storage_type != "external"
            or location.availability_state != "available"
            or not location.worker_id
            or not self._is_absolute_posix_file_uri(location.uri)
        ):
            return False
        source_worker = session.get(WorkerNode, location.worker_id)
        if source_worker is None or source_worker.id == target_worker.id:
            return False
        source_labels = source_worker.labels or {}
        target_labels = target_worker.labels or {}
        source_distro = str(source_labels.get("distro") or "").strip()
        target_distro = str(target_labels.get("distro") or "").strip()
        return (
            source_worker.hostname.casefold() == target_worker.hostname.casefold()
            and source_labels.get("location") == "local"
            and source_labels.get("environment") == "wsl"
            and target_labels.get("location") == "local"
            and target_labels.get("environment") == "wsl"
            and bool(source_distro)
            and source_distro == target_distro
        )

    def _supports_local_worker_bridge(
        self,
        session: Session,
        location: ArtifactLocation,
        target_worker: WorkerNode,
    ) -> bool:
        return self._supports_local_windows_wsl_bridge(
            session, location, target_worker
        ) or self._supports_local_wsl_peer_bridge(session, location, target_worker)

    def local_worker_bridge_locations(
        self,
        session: Session,
        artifact: Artifact,
        project_id: str,
        target_worker: WorkerNode,
    ) -> list[dict[str, Any]]:
        from .global_data import project_can_access_artifact

        query = select(ArtifactLocation).where(
            ArtifactLocation.artifact_id == artifact.id,
            ArtifactLocation.storage_type == "external",
            ArtifactLocation.availability_state == "available",
        )
        if not project_can_access_artifact(session, project_id, artifact.id):
            query = query.where(
                (ArtifactLocation.project_id == project_id)
                | (ArtifactLocation.project_id.is_(None))
            )
        locations = session.scalars(query).all()
        bridged: list[dict[str, Any]] = []
        for location in locations:
            windows_wsl = self._supports_local_windows_wsl_bridge(session, location, target_worker)
            wsl_peer = self._supports_local_wsl_peer_bridge(session, location, target_worker)
            if not windows_wsl and not wsl_peer:
                continue
            access_mode = "local_windows_wsl" if windows_wsl else "local_wsl_peer"
            view = location.to_dict()
            view.update(
                {
                    "id": f"{access_mode}:{location.id}:{target_worker.id}",
                    "worker_id": target_worker.id,
                    "metadata_json": {
                        **(location.metadata_json or {}),
                        "access_mode": access_mode,
                        "source_worker_id": location.worker_id,
                    },
                }
            )
            bridged.append(view)
        return bridged

    def local_wsl_bridge_locations(
        self,
        session: Session,
        artifact: Artifact,
        project_id: str,
        target_worker: WorkerNode,
    ) -> list[dict[str, Any]]:
        """Backward-compatible alias for local Worker path projections."""
        return self.local_worker_bridge_locations(session, artifact, project_id, target_worker)

    def register_external(
        self,
        session: Session,
        *,
        kind: str,
        uri: str,
        content_hash: str,
        size_bytes: int = 0,
        media_type: str = "application/octet-stream",
        availability_state: str = "unknown",
        project_id: str | None = None,
        worker_id: str | None = None,
        created_by_task_attempt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Artifact, ArtifactLocation]:
        self._validate_identity(kind, content_hash)
        if size_bytes < 0:
            raise ValueError("artifact size cannot be negative")
        artifact = session.scalar(
            select(Artifact).where(
                Artifact.kind == kind,
                Artifact.content_hash == content_hash.lower(),
            )
        )
        if artifact is None:
            artifact = Artifact(
                kind=kind,
                uri=uri,
                content_hash=content_hash.lower(),
                size_bytes=size_bytes,
                media_type=media_type,
                created_by_task_attempt_id=created_by_task_attempt_id,
                availability_state=availability_state,
                metadata_json=metadata or {},
            )
            session.add(artifact)
            session.flush()
        location = self._register_location(
            session,
            artifact,
            storage_type="external",
            uri=uri,
            project_id=project_id,
            worker_id=worker_id,
            availability_state=availability_state,
            metadata=metadata,
        )
        if availability_state == "available":
            artifact.availability_state = "available"
        return artifact, location

    def put_bytes(
        self,
        session: Session,
        *,
        project_id: str,
        kind: str,
        payload: bytes,
        filename: str,
        media_type: str = "application/octet-stream",
        created_by_task_attempt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        digest = hashlib.sha256(payload).hexdigest()
        from .storage_relocation import guard_writes
        guard_writes(session)
        self._validate_identity(kind, digest)
        existing = session.scalar(
            select(Artifact).where(Artifact.kind == kind, Artifact.content_hash == digest)
        )
        if existing and self._managed_location(session, existing, project_id):
            return existing

        root = artifact_root(session, project_id, kind)
        artifact_dir = root / kind / digest[:2] / digest
        artifact_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or "payload.bin"
        payload_path = artifact_dir / safe_name
        temp_path = artifact_dir / f".{safe_name}.tmp"
        temp_path.write_bytes(payload)
        temp_path.replace(payload_path)

        manifest = {
            "schema_version": 1,
            "kind": kind,
            "content_hash": digest,
            "size_bytes": len(payload),
            "media_type": media_type,
            "filename": safe_name,
            "metadata": metadata or {},
        }
        manifest_path = artifact_dir / "manifest.json"
        manifest_temp = artifact_dir / ".manifest.json.tmp"
        manifest_temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest_temp.replace(manifest_path)

        artifact = existing
        if artifact is None:
            artifact = Artifact(
                kind=kind,
                uri=payload_path.as_uri(),
                content_hash=digest,
                size_bytes=len(payload),
                media_type=media_type,
                created_by_task_attempt_id=created_by_task_attempt_id,
                metadata_json=metadata or {},
            )
            session.add(artifact)
            session.flush()
        elif artifact.size_bytes == 0:
            artifact.size_bytes = len(payload)
        self._register_location(
            session,
            artifact,
            storage_type="managed",
            uri=payload_path.as_uri(),
            project_id=project_id,
            availability_state="available",
        )
        return artifact

    def put_stream(
        self,
        session: Session,
        *,
        project_id: str,
        kind: str,
        source: BinaryIO,
        filename: str,
        media_type: str = "application/octet-stream",
        created_by_task_attempt_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Artifact:
        if not ARTIFACT_KIND_PATTERN.fullmatch(kind):
            raise ValueError("artifact kind contains unsupported characters")
        from .storage_relocation import guard_writes
        guard_writes(session)
        root = artifact_root(session, project_id, kind)
        staging_dir = root / ".staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or "payload.bin"
        staging_path = staging_dir / f"{uuid4()}.tmp"
        digest = hashlib.sha256()
        size_bytes = 0
        with staging_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
                target.write(chunk)

        content_hash = digest.hexdigest()
        existing = session.scalar(
            select(Artifact).where(Artifact.kind == kind, Artifact.content_hash == content_hash)
        )
        if existing and self._managed_location(session, existing, project_id):
            staging_path.unlink(missing_ok=True)
            return existing

        artifact_dir = root / kind / content_hash[:2] / content_hash
        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload_path = artifact_dir / safe_name
        shutil.move(str(staging_path), str(payload_path))
        manifest = {
            "schema_version": 1,
            "kind": kind,
            "content_hash": content_hash,
            "size_bytes": size_bytes,
            "media_type": media_type,
            "filename": safe_name,
            "metadata": metadata or {},
        }
        manifest_path = artifact_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        artifact = existing
        if artifact is None:
            artifact = Artifact(
                kind=kind,
                uri=payload_path.as_uri(),
                content_hash=content_hash,
                size_bytes=size_bytes,
                media_type=media_type,
                created_by_task_attempt_id=created_by_task_attempt_id,
                metadata_json=metadata or {},
            )
            session.add(artifact)
            session.flush()
        elif artifact.size_bytes == 0:
            artifact.size_bytes = size_bytes
        self._register_location(
            session,
            artifact,
            storage_type="managed",
            uri=payload_path.as_uri(),
            project_id=project_id,
            availability_state="available",
        )
        return artifact


def artifact_view(
    session: Session, artifact: Artifact, project_id: str | None = None
) -> dict[str, Any]:
    query = select(ArtifactLocation).where(ArtifactLocation.artifact_id == artifact.id)
    if project_id is not None:
        from .global_data import project_can_access_artifact

        if not project_can_access_artifact(session, project_id, artifact.id):
            query = query.where(
                (ArtifactLocation.project_id == project_id)
                | (
                    (ArtifactLocation.project_id.is_(None))
                    & (ArtifactLocation.storage_type == "external")
                )
            )
    locations = session.scalars(query.order_by(ArtifactLocation.created_at)).all()
    view = artifact.to_dict()
    if project_id is not None:
        if locations:
            managed = next(
                (location for location in locations if location.storage_type == "managed"),
                locations[0],
            )
            view["uri"] = managed.uri
        else:
            view["uri"] = ""
    return {
        **view,
        "locations": [location.to_dict() for location in locations],
    }


def artifact_view_for_worker(
    session: Session,
    artifact: Artifact,
    project_id: str,
    worker: WorkerNode,
) -> dict[str, Any]:
    view = artifact_view(session, artifact, project_id)
    view["locations"] = [
        *view["locations"],
        *artifact_storage.local_worker_bridge_locations(session, artifact, project_id, worker),
    ]
    return view


artifact_storage = LocalArtifactStorage()
