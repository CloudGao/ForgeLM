import errno
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from uuid import uuid4

import pytest
from llm_platform_worker.adapters.storage_maintenance import cleanup, inventory
from llm_platform_worker.checkpoint_copy import copy_checkpoint
from llm_platform_worker.data_generation_runtime import generation_checkpoint_identity


def copies(root):
    project, task_id, version = (str(uuid4()) for _ in range(3))
    task = {"id": task_id, "payload": {"generation": {}, "generation_input": {"artifact_hash": "input"},
            "generation_model": {"identity_hash": "model"}, "output_dataset_version_id": version}}
    parent = root / "projects" / project / ".llm-platform/data-assets/data-generation"
    owners, paths = [], []
    for i in range(2):
        attempt = str(uuid4())
        path = parent / f"{version}-{attempt}"
        path.mkdir(parents=True)
        (path / "resume-state.json").write_text(json.dumps({"schema": "forgelm-generation-resume-v1",
            "task_id": task_id, "output_dataset_version_id": version,
            "identity": generation_checkpoint_identity(task)}))
        (path / "data.partial").write_bytes(b'{"text":"keep"}\n')
        (path / "skipped.records").write_bytes(b'{"reason":"skip"}\n')
        owners.append({"project_id": project, "attempt_id": attempt, "task": task,
                       "created_at": str(i), "pinned": False})
        paths.append(path)
    return {"generation_attempts": owners}, paths


@pytest.mark.parametrize("blocker", [None, "pin", "uri", "automatic", "changed", "unknown"])
def test_generation_duplicate_keeps_recovery_copy(tmp_path, blocker):
    payload, (old, new) = copies(tmp_path)
    if blocker == "pin":
        payload["generation_attempts"][0]["pinned"] = True
    if blocker == "uri":
        payload["protected_uris"] = [old.as_uri()]
    if blocker == "automatic":
        payload["automatic"] = True
    if blocker == "unknown":
        (old / "important.bin").write_bytes(b"preserve")
    candidates = [x for x in inventory(tmp_path, payload)["items"] if x["reclaimable"]]
    if blocker in {"pin", "uri", "automatic", "unknown"}:
        assert not candidates
        return
    assert len(candidates) == 1 and candidates[0]["path"] == str(old)
    if blocker == "changed":
        (new / "data.partial").write_bytes(b"different")
    result = cleanup(tmp_path, {**payload, "candidates": candidates})
    assert old.exists() == (blocker == "changed"), result
    assert new.exists()


def test_legacy_index_requires_verified_original_and_is_rebuilt_cache(tmp_path):
    payload, (source, _) = copies(tmp_path)
    owner = payload["generation_attempts"][0]
    owner["pinned"] = True  # Pin protects original; derived cache remains rebuildable.
    signature = generation_checkpoint_identity(owner["task"])
    path = source.parent / f"checkpoint-{signature}-{owner['attempt_id']}.sqlite"
    files = [source / n for n in ("data.partial", "skipped.records")]
    digest = hashlib.sha256()
    for i, file in enumerate(files):
        for line in file.read_bytes().splitlines(keepends=True):
            digest.update(bytes([i]) + line)
    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE rows (row TEXT)")
        db.execute("CREATE TABLE metadata (key PRIMARY KEY, value)")
        db.executemany("INSERT INTO metadata VALUES (?,?)", {"identity": signature,
            "fingerprint": json.dumps([[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in files],sort_keys=True),
            "sha256": digest.hexdigest()}.items())
        db.commit()
    selected = [x for x in inventory(tmp_path, payload)["items"] if x["category"] == "generation_index_cache"]
    assert len(selected) == 1
    result = cleanup(tmp_path, {**payload, "candidates": selected})
    assert result["items"][0]["status"] == "deleted" and source.exists()
    assert all(p.exists() for p in files)


def test_atomic_checkpoint_copy_independent_and_failure_keeps_target(tmp_path, monkeypatch):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"immutable")
    if os.name == "posix":
        import fcntl
        def unsupported(*args):
            raise OSError(errno.EOPNOTSUPP, "unsupported")
        monkeypatch.setattr(fcntl, "ioctl", unsupported)
    assert copy_checkpoint(source, target) == "copy"
    target.write_bytes(b"changed")
    assert source.read_bytes() == b"immutable"
    def fail(*args):
        raise OSError("copy interrupted")
    monkeypatch.setattr("llm_platform_worker.checkpoint_copy.shutil.copyfileobj", fail)
    with pytest.raises(OSError, match="interrupted"):
        copy_checkpoint(source, target)
    assert target.read_bytes() == b"changed"
    assert not list(tmp_path.glob(".checkpoint-copy-*"))
