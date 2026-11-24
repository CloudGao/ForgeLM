import subprocess
from pathlib import Path

from llm_platform.models import RemoteWorkerEnvironment
from llm_platform.services import remote_workers


def test_remote_release_hash_needs_no_model_repository(tmp_path, monkeypatch):
    # A base Worker release must not read any private model requirements.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bootstrap = scripts / "bootstrap-remote-worker.sh"
    bootstrap.write_text("bootstrap", encoding="utf-8")
    (scripts / "remote-worker-supervisor.py").write_text("supervisor", encoding="utf-8")
    wheel = tmp_path / "platform.whl"
    wheel.write_bytes(b"platform-only")
    monkeypatch.setattr(remote_workers, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(remote_workers, "_BOOTSTRAP_SCRIPT", bootstrap)
    environment = RemoteWorkerEnvironment(desired_release="source", metadata_json={})
    first = remote_workers._release_id(environment, wheel)
    assert first.startswith("source-")
    wheel.write_bytes(b"updated-platform")
    assert remote_workers._release_id(environment, wheel) != first


def test_public_catalog_does_not_advertise_bundled_private_model(client):
    suites = client.get("/api/plugin-suites").json()
    assert all(suite["name"] != "MicroLM" for suite in suites)
    assert any(suite["name"] == "Transformers" for suite in suites)


def test_private_model_is_not_part_of_source_distribution():
    root = Path(__file__).resolve().parents[2]
    excluded = ("model_sources", "examples/plugin-suites/microlm", "configs/workflows")
    if (root / ".git").exists():
        # Local ignored model sources may coexist with a platform checkout.
        # Only tracked files enter the Git source release.
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", *excluded], cwd=root, text=True
        )
        assert not tracked.strip()
    else:
        for relative_path in excluded:
            assert not (root / relative_path).exists()
