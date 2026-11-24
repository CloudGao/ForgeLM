from pathlib import Path

from llm_platform_worker.adapters.storage_maintenance import cleanup, inventory


def test_cache_cleanup_preserves_hub_and_registered_artifacts(tmp_path: Path):
    cache = tmp_path / ".cache/huggingface/datasets"
    cache.mkdir(parents=True)
    (cache / "cache.arrow").write_bytes(b"cache")
    hub = tmp_path / ".cache/huggingface/hub"
    hub.mkdir()
    (hub / "weights.bin").write_bytes(b"weights")
    payload = {"protected_uris": [], "locations": [], "terminal_attempts": []}
    scan = inventory(tmp_path, payload)
    candidate = next(x for x in scan["items"] if x["reclaimable"])
    blocked = cleanup(tmp_path, {**payload, "protected_uris": [cache.as_uri()],
                                 "candidates": [candidate]})
    assert blocked["items"][0]["status"] == "skipped"
    assert cache.exists()
    (cache / "changed.arrow").write_bytes(b"new")
    changed = cleanup(tmp_path, {**payload, "candidates": [candidate]})
    assert changed["items"][0]["status"] == "skipped"
    refreshed = next(x for x in inventory(tmp_path, payload)["items"] if x["reclaimable"])
    done = cleanup(tmp_path, {**payload, "candidates": [refreshed]})
    assert done["reclaimed_bytes"] == 8
    assert not cache.exists()
    assert (hub / "weights.bin").read_bytes() == b"weights"


def test_arbitrary_path_never_authorized_by_scan_payload(tmp_path: Path):
    data = tmp_path / "important"
    data.mkdir()
    (data / "checkpoint").write_bytes(b"keep")
    result = cleanup(tmp_path, {"candidates": [{"path": str(data)}]})
    assert result["items"][0]["status"] == "skipped"
    assert (data / "checkpoint").exists()
