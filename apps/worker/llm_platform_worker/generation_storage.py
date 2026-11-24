"""Conservative, manual-only reclamation of generation copies and legacy indexes."""
import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from urllib.parse import unquote, urlparse
from uuid import UUID

from llm_platform_core.storage_cleanup import checked_path, inspect_target


def _referenced(path, payload):
    target = path.as_posix().rstrip("/")
    for value in [*payload.get("protected_uris", []), *payload.get("generation_path_references", [])]:
        ref = unquote(urlparse(value).path) if value.startswith("file:") else value
        ref = ref.replace("\\", "/").rstrip("/")
        if ref and (target in ref or target.startswith(ref + "/")):
            return True
    return False


def generation_candidates(root, payload):
    if payload.get("automatic"):
        return
    from .data_generation_runtime import generation_checkpoint_identity

    groups = defaultdict(list)
    for owner in payload.get("generation_attempts", []):
        try:
            project, attempt = str(UUID(owner["project_id"])), str(UUID(owner["attempt_id"]))
            task = owner["task"]
            version = str(UUID(task["payload"]["output_dataset_version_id"]))
            task_id = str(UUID(task["id"]))
            signature = generation_checkpoint_identity(task)
            parent = root / "projects" / project / ".llm-platform/data-assets/data-generation"
            from llm_platform_core.storage_layout import unified_storage_root
            if unified_storage_root() == root:
                parent = root / "dataset" / "objects" / "external" / "data-generation"
            source = parent / f"{version}-{attempt}"
            checked_path(source, root)
            if not source.is_dir():
                continue
            expected = {"schema": "forgelm-generation-resume-v1", "task_id": task_id,
                        "output_dataset_version_id": version, "identity": signature}
            if json.loads((source / "resume-state.json").read_text(encoding="utf-8")) != expected:
                continue
            if {p.name for p in source.iterdir()} - {"data.partial", "skipped.records", "resume-state.json"}:
                continue
            before = inspect_target(source, root)
            hashes = []
            for name in ("data.partial", "skipped.records"):
                path = source / name
                if not path.exists() and name == "skipped.records":
                    hashes.append(None)
                    continue
                checked_path(path, root)
                with path.open("rb") as stream:
                    hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
            if inspect_target(source, root) != before:
                continue
            groups[(task_id, signature, tuple(hashes))].append((owner, source))
            index = parent / f"checkpoint-{signature}-{attempt}.sqlite"
            if index.is_file() and not _referenced(index, payload):
                checked_path(index, root)
                with closing(sqlite3.connect(f"{index.as_uri()}?mode=ro", uri=True)) as db:
                    metadata = dict(db.execute("SELECT key,value FROM metadata"))
                    columns = {r[1] for r in db.execute("PRAGMA table_info(rows)")}
                files = [source / "data.partial", source / "skipped.records"]
                fingerprint = json.dumps([[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                                          if p.exists() else [p.name, None] for p in files], sort_keys=True)
                if "row" not in columns or metadata.get("fingerprint") != fingerprint or metadata.get("identity") != signature:
                    continue
                digest = hashlib.sha256()
                for skipped, path in enumerate(files):
                    if not path.exists():
                        continue
                    with path.open("rb") as stream:
                        for line in stream:
                            if not line.endswith(b"\n"):
                                break
                            digest.update(bytes([skipped]) + line)
                if digest.hexdigest() == metadata.get("sha256") and inspect_target(source, root) == before:
                    yield index, "generation_index_cache", True, "完整来源与封存摘要一致；下次接续重建紧凑索引"
        except (OSError, ValueError, KeyError, sqlite3.Error):
            continue
    for copies in groups.values():
        copies.sort(key=lambda item: (item[0]["created_at"], item[0]["attempt_id"]))
        for owner, source in copies[:-1]:
            if owner.get("pinned") or _referenced(source, payload):
                continue
            # A sealed index still depends on this exact source directory.
            if any(source.parent.glob(f"checkpoint-*-{owner['attempt_id']}.sqlite")):
                continue
            yield source, "generation_duplicate", True, "同任务较新执行保留完全相同的输出与跳过记录；SHA-256 已核验"
