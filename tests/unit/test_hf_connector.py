from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from llm_platform_worker import hf_connector


class FakeRepoFile:
    def __init__(self, path: str, size: int) -> None:
        self.path = path
        self.size = size


def test_scan_repository_resolves_revision_before_listing_tree(monkeypatch) -> None:
    calls: dict = {}

    class FakeApi:
        def __init__(self, **kwargs) -> None:
            calls["client"] = kwargs

        def resolve_revision(self, repo_id: str, **kwargs):
            calls["resolve"] = {"repo_id": repo_id, **kwargs}
            return SimpleNamespace(resolved="a" * 40)

        def list_repo_tree(self, repo_id: str, **kwargs):
            calls["tree"] = {"repo_id": repo_id, **kwargs}
            return [
                FakeRepoFile("data/train.parquet", 120),
                object(),
                FakeRepoFile("README.md", 20),
            ]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, RepoFile=FakeRepoFile),
    )
    result = hf_connector.scan_repository(
        "org/corpus",
        {"revision": "main", "hf_endpoint": "https://huggingface.co", "split": "train"},
    )

    assert result["commit_hash"] == "a" * 40
    assert result["total_size_bytes"] == 140
    assert result["files"] == [
        {"path": "README.md", "size_bytes": 20},
        {"path": "data/train.parquet", "size_bytes": 120},
    ]
    assert calls["tree"]["revision"] == "a" * 40
    assert calls["tree"]["recursive"] is True
    assert result["split"] == "train"


def test_model_repository_operations_preserve_repo_type(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []

    class FakeApi:
        def __init__(self, **_kwargs) -> None:
            pass

        def resolve_revision(self, repo_id: str, **kwargs):
            calls.append({"operation": "resolve", "repo_id": repo_id, **kwargs})
            return SimpleNamespace(resolved="e" * 40)

        def list_repo_tree(self, repo_id: str, **kwargs):
            calls.append({"operation": "tree", "repo_id": repo_id, **kwargs})
            return [FakeRepoFile("config.json", 10)]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, RepoFile=FakeRepoFile),
    )
    result = hf_connector.scan_repository(
        "org/model", {"revision": "main", "repo_type": "model"}
    )

    assert result["repo_type"] == "model"
    assert all(call["repo_type"] == "model" for call in calls)


def test_metadata_cli_writes_scan_result(tmp_path, monkeypatch) -> None:
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    request_path.write_text(
        json.dumps(
            {
                "operation": "scan",
                "repo_id": "org/corpus",
                "source_config": {"revision": "main"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        hf_connector,
        "scan_repository",
        lambda repo_id, source_config: {
            "repo_id": repo_id,
            "requested_revision": source_config["revision"],
            "commit_hash": "d" * 40,
            "files": [],
            "file_count": 0,
            "total_size_bytes": 0,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf_connector",
            "--metadata-request-file",
            str(request_path),
            "--result-file",
            str(result_path),
        ],
    )

    hf_connector.main()

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["repo_id"] == "org/corpus"
    assert result["commit_hash"] == "d" * 40


def test_plan_and_download_use_same_commit_paths_directory_and_endpoint(
    tmp_path, monkeypatch
) -> None:
    calls: list[dict] = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        if kwargs["dry_run"]:
            return [
                SimpleNamespace(
                    filename="data/a.parquet",
                    commit_hash="b" * 40,
                    file_size=100,
                    is_cached=True,
                    will_download=False,
                ),
                SimpleNamespace(
                    filename="data/b.parquet",
                    commit_hash="b" * 40,
                    file_size=300,
                    is_cached=False,
                    will_download=True,
                ),
            ]
        return str(kwargs["local_dir"])

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    destination = tmp_path / "asset"
    source_config = {
        "requested_revision": "main",
        "commit_hash": "b" * 40,
        "selected_paths": ["data/a.parquet", "data/b.parquet"],
        "hf_endpoint": "https://huggingface.co",
    }
    plan = hf_connector.plan_snapshot("org/corpus", source_config, destination)

    assert plan["selected_bytes"] == 400
    assert plan["cached_bytes"] == 100
    assert plan["download_bytes"] == 300
    assert plan["disk_space_sufficient"] is True
    hf_connector.download_snapshot(plan)

    assert len(calls) == 2
    for call in calls:
        assert call["repo_id"] == "org/corpus"
        assert call["revision"] == "b" * 40
        assert call["allow_patterns"] == ["data/a.parquet", "data/b.parquet"]
        assert call["local_dir"] == destination.resolve()
        assert call["endpoint"] == "https://huggingface.co"
    assert calls[0]["dry_run"] is True
    assert calls[1]["dry_run"] is False


def test_hf_mirror_metadata_failure_falls_back_to_official_endpoint(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    class FileMetadataError(Exception):
        pass

    def snapshot_download(**kwargs):
        calls.append(kwargs["endpoint"])
        if kwargs["endpoint"] == hf_connector.HF_MIRROR_ENDPOINT:
            try:
                raise FileMetadataError("missing X-Repo-Commit")
            except FileMetadataError as exc:
                raise RuntimeError("mirror metadata failed") from exc
        return [
            SimpleNamespace(
                filename="README.md",
                commit_hash="c" * 40,
                file_size=10,
                is_cached=False,
                will_download=True,
            )
        ]

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    plan = hf_connector.plan_snapshot(
        "org/corpus",
        {
            "requested_revision": "main",
            "commit_hash": "c" * 40,
            "selected_paths": ["README.md"],
            "hf_endpoint": hf_connector.HF_MIRROR_ENDPOINT,
        },
        tmp_path / "asset",
    )

    assert calls == [hf_connector.HF_MIRROR_ENDPOINT, hf_connector.OFFICIAL_ENDPOINT]
    assert plan["requested_endpoint"] == hf_connector.HF_MIRROR_ENDPOINT
    assert plan["endpoint"] == hf_connector.OFFICIAL_ENDPOINT
    assert plan["endpoint_warning"]


def test_mirror_download_disables_xet_before_loading_hub(tmp_path, monkeypatch):
    import os

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    calls = []

    def download(**kwargs):
        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    hf_connector.download_snapshot({"repo_id": "org/public", "repo_type": "dataset", "commit_hash": "a" * 40,
        "requested_endpoint": hf_connector.HF_MIRROR_ENDPOINT, "endpoint": hf_connector.OFFICIAL_ENDPOINT,
        "selected_paths": ["data.parquet"], "target_dir": str(tmp_path)})
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["allow_patterns"] == ["data.parquet"]


def test_mirror_download_fallback_keeps_commit_and_selected_files(tmp_path, monkeypatch):
    import pytest
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        if kwargs["endpoint"] == hf_connector.HF_MIRROR_ENDPOINT:
            raise TimeoutError("mirror unavailable")
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    with pytest.warns(UserWarning, match="same commit and files"):
        hf_connector.download_snapshot({"repo_id": "org/public", "repo_type": "dataset",
            "commit_hash": "a" * 40, "endpoint": hf_connector.HF_MIRROR_ENDPOINT,
            "selected_paths": ["data.parquet"], "target_dir": str(tmp_path)})
    assert [c["endpoint"] for c in calls] == [hf_connector.HF_MIRROR_ENDPOINT, hf_connector.OFFICIAL_ENDPOINT]
    assert {k: v for k, v in calls[0].items() if k != "endpoint"} == {
        k: v for k, v in calls[1].items() if k != "endpoint"}
