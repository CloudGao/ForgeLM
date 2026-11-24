import os
import pytest
from llm_platform_core.storage_cleanup import inspect_target, remove_target
from llm_platform_worker.adapters.storage_maintenance import model_deletion_storage


def test_worker_scans_and_deletes_only_exact_unchanged_target(tmp_path):
    target = tmp_path / "projects" / "p" / ".llm-platform" / "artifacts" / "model" / "weights.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"weights")
    payload = {"storage_action": "scan", "deletion_items": [{"item_id": "i", "path": str(target)}]}
    item = model_deletion_storage(tmp_path, payload)["items"][0]
    assert item["status"] == "inspected"
    payload.update(storage_action="cleanup", deletion_items=[item])
    result = model_deletion_storage(tmp_path, payload)
    assert result["reclaimed_bytes"] == 7
    assert not target.exists()


def test_worker_rechecks_fingerprint_and_blockers(tmp_path):
    target = tmp_path / ".llm-platform" / "artifacts" / "model" / "model.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"before")
    item = {"item_id": "i", "path": str(target), **inspect_target(target, tmp_path)}
    target.write_bytes(b"after")
    result = model_deletion_storage(
        tmp_path, {"storage_action": "cleanup", "deletion_items": [item]}
    )
    assert result["items"][0]["status"] == "failed"
    assert target.exists()
    item.update(inspect_target(target, tmp_path), blocked=True)
    assert (
        model_deletion_storage(tmp_path, {"storage_action": "cleanup", "deletion_items": [item]})[
            "reclaimed_bytes"
        ]
        == 0
    )


@pytest.mark.parametrize("suffix", ["../outside", "../outside/.llm-platform/artifacts/model", "."])
def test_cleanup_rejects_boundary_escape(tmp_path, suffix):
    with pytest.raises(ValueError):
        inspect_target(tmp_path / suffix, tmp_path)


def test_cleanup_rejects_nested_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    original = tmp_path / "original"
    original.write_bytes(b"keep")
    try:
        (target / "link").symlink_to(original)
    except OSError:
        pytest.skip("Host does not permit creating symlinks")
    with pytest.raises(ValueError, match="链接|重解析"):
        inspect_target(target, tmp_path)
    assert original.exists()


def test_same_size_replacement_is_detected(tmp_path):
    target = tmp_path / "file"
    target.write_bytes(b"aaa")
    item = {"root": str(tmp_path), "path": str(target), **inspect_target(target, tmp_path)}
    previous = target.stat()
    target.write_bytes(b"bbb")
    # Ensure a metadata change even on filesystems with coarse write timestamps.
    os.utime(target, ns=(previous.st_atime_ns, previous.st_mtime_ns + 2_000_000_000))
    with pytest.raises(ValueError):
        remove_target(item)


def test_nested_windows_reparse_directory_is_rejected_before_descent(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    target = tmp_path / "target"
    junction = target / "junction"
    junction.mkdir(parents=True)
    from llm_platform_core.storage_cleanup import filesystem_path

    original = Path.lstat

    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        return (
            SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            if path == junction or path == filesystem_path(junction)
            else info
        )

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="重解析"):
        inspect_target(target, tmp_path)


def test_remote_checkpoint_group_retains_logs_and_rejects_new_checkpoint(tmp_path):
    trainer = tmp_path / ".llm-platform" / "training-runs" / "run" / "trainer"
    cp = trainer / "checkpoint-10"
    cp.mkdir(parents=True)
    (cp / "optimizer.pt").write_bytes(b"optimizer")
    (trainer / "trainer_state.json").write_text('{"step": 10}')
    payload = {
        "storage_action": "scan",
        "deletion_items": [{"item_id": "group", "path": str(trainer), "checkpoint_children": True}],
    }
    inspected = model_deletion_storage(tmp_path, payload)["items"][0]
    assert inspected["size_bytes"] == 9
    item = {**inspected, "checkpoint_children": True}
    (trainer / "checkpoint-20").mkdir()
    result = model_deletion_storage(
        tmp_path, {"storage_action": "cleanup", "deletion_items": [item]}
    )
    assert result["items"][0]["status"] == "failed"
    inspected = model_deletion_storage(tmp_path, payload)["items"][0]
    result = model_deletion_storage(
        tmp_path,
        {
            "storage_action": "cleanup",
            "deletion_items": [{**inspected, "checkpoint_children": True}],
        },
    )
    assert result["reclaimed_bytes"] == 9
    assert (trainer / "trainer_state.json").exists()
    assert not cp.exists()
