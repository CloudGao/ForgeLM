from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest
from forgelm_plugin_sdk.cli import init, pack, write_json
from forgelm_plugin_sdk.package import file_hash, inspect_package
from llm_platform_worker.suite_runtime import SuiteRuntime, file_lock


@pytest.fixture(scope="session")
def suite_package(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite-source") / "sample"
    init(root, "example.text-tools")
    wheels = root / "wheels"
    wheels.mkdir()
    repository = Path(__file__).resolve().parents[2]
    for source in [repository / "packages/forgelm_plugin_sdk", root]:
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels), str(source)], check=True, capture_output=True)
    from email.parser import Parser
    packages = []
    for path in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            meta = Parser().parsestr(archive.read(metadata).decode())
        packages.append({"name": meta["Name"], "version": meta["Version"], "path": "wheels/" + path.name, "sha256": file_hash(path)})
    manifest = json.loads((root / "suite.json").read_text())
    target = manifest["targets"][0]
    write_json(root / target["runtimeGroups"]["default"]["lock"], {"schema": "forgelm.wheel-lock.v1", "packages": packages})
    return pack(root, root / "dist")


def test_package_rejects_modified_and_traversing_archive(suite_package, tmp_path):
    info = inspect_package(suite_package)
    assert info["manifest"]["id"] == "example.text-tools"
    with zipfile.ZipFile(suite_package) as original:
        for malicious in ["../escape", "suite.json"]:
            target = tmp_path / ("bad-" + str(uuid4()) + ".flsuite")
            with zipfile.ZipFile(target, "w") as archive:
                for entry in original.infolist():
                    if entry.filename != malicious:
                        archive.writestr(entry, original.read(entry.filename))
                archive.writestr(malicious, b"{}")
            with pytest.raises(ValueError):
                inspect_package(target)


def test_environment_allows_parallel_readers_but_blocks_uninstall(tmp_path):
    lease = tmp_path / "lease.lock"
    with file_lock(lease, shared=True), file_lock(lease, shared=True):
        with pytest.raises(ValueError, match="in use"):
            with file_lock(lease):
                pytest.fail("exclusive uninstall lease must not bypass active readers")
    with file_lock(lease):
        pass


@pytest.mark.asyncio
async def test_real_suite_install_execute_replay_and_uninstall(suite_package, tmp_path, monkeypatch):
    runtime = SuiteRuntime(tmp_path)
    package = inspect_package(suite_package)
    digest = package["digest"]
    cached = runtime.root / "blobs" / (digest + ".flsuite")
    cached.parent.mkdir()
    shutil.copyfile(suite_package, cached)
    operation_id, version_id = str(uuid4()), str(uuid4())
    spec = {"operation_id": operation_id, "suite_id": "example.text-tools", "generation": 1, "action": "install", "version_id": version_id, "target_id": package["manifest"]["targets"][0]["id"], "package_project_id": "system", "trusted_digest": digest, "package": {"artifact_id": "cached", "digest": digest, "size_bytes": package["size_bytes"]}}
    events = []
    async def emit(event, data):
        events.append((event, data))
        return "running"
    task = {"id": str(uuid4()), "payload": {"suite_operation": spec}}
    installed = await runtime.manage(task, emit)
    assert installed == await runtime.manage(task, emit)
    receipt = installed["suite_receipt"]["installations"][0]
    assert receipt["probes"]["normalize"]["ready"]
    assert "suite@" + digest in runtime.capabilities()
    attempt = tmp_path / "attempt"
    source = attempt / "suite-inputs/source"
    source.parent.mkdir(parents=True)
    source.write_text("  HELLO  \nWorld\n", encoding="utf-8")
    task = {"id": str(uuid4()), "project_id": "p", "runtime_context": {"attempt_dir": str(attempt), "allocation": {}}, "payload": {"suite_execution": {"suite_id": "example.text-tools", "version_id": version_id, "digest": digest, "node_id": "example.text-tools.normalize", "config": {"lowercase": True}, "inputs": {"source": {"artifact_id": "cached", "digest": file_hash(source), "size_bytes": source.stat().st_size}}}}}
    result = await runtime.execute(task, emit)
    assert Path(result["outputs"][0]["path"]).read_text() == "hello\nworld\n"
    from forgelm_plugin_sdk.package import extract_package
    updated_source = tmp_path / "updated-source"
    extract_package(suite_package, updated_source)
    manifest = json.loads((updated_source / "suite.json").read_text(encoding="utf-8"))
    manifest["version"] = "1.1.0"
    write_json(updated_source / "suite.json", manifest)
    updated_archive = pack(updated_source, tmp_path / "packages")
    updated_package = inspect_package(updated_archive)
    shutil.copyfile(updated_archive, runtime.root / "blobs" / (updated_package["digest"] + ".flsuite"))
    new_version_id = str(uuid4())
    update = {**spec, "operation_id": str(uuid4()), "generation": 2, "action": "update", "version_id": new_version_id, "trusted_digest": updated_package["digest"], "package": {"artifact_id": "cached", "digest": updated_package["digest"], "size_bytes": updated_package["size_bytes"]}}
    updated = await runtime.manage({"id": "update", "payload": {"suite_operation": update}}, emit)
    assert updated["suite_receipt"]["default_version_id"] == new_version_id
    assert len(updated["suite_receipt"]["installations"]) == 2
    rollback = {**spec, "operation_id": str(uuid4()), "generation": 3, "action": "rollback"}
    rolled_back = await runtime.manage({"id": "rollback", "payload": {"suite_operation": rollback}}, emit)
    assert rolled_back["suite_receipt"]["default_version_id"] == version_id
    original_invoke = runtime.invoke
    async def broken_environment(*args, **kwargs):
        raise RuntimeError("simulated environment preparation failure")
    monkeypatch.setattr(runtime, "invoke", broken_environment)
    with pytest.raises(RuntimeError, match="preparation failure"):
        await runtime.manage({"id": "repair", "payload": {"suite_operation": {**spec, "action": "repair", "generation": 4, "operation_id": str(uuid4())}}}, emit)
    monkeypatch.setattr(runtime, "invoke", original_invoke)
    current = json.loads((runtime.suite_dir(spec["suite_id"]) / "index.json").read_text(encoding="utf-8"))
    assert current["default"] == version_id and current["generation"] == 3
    uninstall = {**spec, "operation_id": str(uuid4()), "generation": 4, "action": "uninstall", "version_ids": [version_id, new_version_id]}
    lock = runtime.suite_dir(spec["suite_id"]) / "leases" / (receipt["generation"] + ".lock")
    with file_lock(lock):
        with pytest.raises(ValueError, match="in use"):
            await runtime.manage({"id": "test", "payload": {"suite_operation": uninstall}}, emit)
    removed = await runtime.manage({"id": "test", "payload": {"suite_operation": uninstall}}, emit)
    assert removed["suite_receipt"]["installations"] == []
    assert source.exists()
    assert cached.exists()
    assert "suite@" + digest not in runtime.capabilities()


def test_package_api_registration_and_lifecycle_plans(client, suite_package):
    from llm_platform.database import SessionLocal
    from llm_platform.models import Task, WorkerNode
    from llm_platform.suite_models import SuiteVersion
    package = inspect_package(suite_package)
    target = package["manifest"]["targets"][0]
    response = client.post("/api/suite-package-imports", files={"file": ("sample.flsuite", suite_package.read_bytes(), "application/zip")})
    assert response.status_code == 201, response.text
    imported = response.json()
    response = client.post(f"/api/suite-package-imports/{imported['id']}/register")
    assert response.status_code == 201, response.text
    version = response.json()
    assert not version["trusted"]
    assert len(version["nodes"]) == 1
    again = client.post(f"/api/suite-package-imports/{imported['id']}/register")
    assert again.json()["id"] == version["id"]
    with SessionLocal() as session:
        worker = WorkerNode(name="suite-api-" + uuid4().hex, hostname="test", status="online", capabilities=["builtin.suite-manager", "builtin.suite-runner", "builtin.local-process"], resources={"cpu_count": 4, "suite_runtime": {k: target[k] for k in ("os", "arch", "python")}})
        session.add(worker)
        session.commit()
        worker_id = worker.id
    plan = client.post("/api/suite-operation-plans", json={"action": "install", "version_id": version["id"], "worker_ids": [worker_id]})
    assert plan.status_code == 201, plan.text
    body = {"plan_id": plan.json()["id"], "idempotency_key": str(uuid4())}
    assert client.post("/api/suite-operations", json=body).status_code == 409
    body["trust_digest"] = version["digest"]
    response = client.post("/api/suite-operations", json=body)
    assert response.status_code == 201, response.text
    operation = response.json()
    repeated = client.post("/api/suite-operations", json=body)
    assert repeated.json()["id"] == operation["id"]
    with SessionLocal() as session:
        task = session.get(Task, operation["items"][0]["task_id"])
        assert task.project_id and task.payload["suite_operation"]["trusted_digest"] == version["digest"]
        assert session.get(SuiteVersion, version["id"]).trusted
    assert client.post("/api/suite-operation-plans", json={"action": "update", "version_id": version["id"], "worker_ids": [worker_id]}).status_code == 409
    assert any(node["suite_version_id"] == version["id"] for node in client.get("/api/suite-nodes").json())


def test_invalid_package_import_never_registers_code(client, tmp_path):
    response = client.post("/api/suite-package-imports", files={"file": ("not-a-package.flsuite", b"not zip", "application/zip")})
    assert response.status_code == 422


def test_package_receipts_admission_outputs_and_config_immutability(client, suite_package, tmp_path):
    from llm_platform.database import SessionLocal
    from llm_platform.models import Project, Task, WorkerNode
    from llm_platform.services.artifacts import artifact_storage
    from llm_platform.services.suite_packages import (
        can_allocate,
        claim_environment,
        create_revision,
        finalize_operation,
        finalize_run,
        make_plan,
        run_revision,
        start_operation,
        sync_status,
        validate_canvas_node,
    )
    from llm_platform.suite_models import SuiteVersion
    imported = client.post("/api/suite-package-imports", files={"file": ("sample.flsuite", suite_package.read_bytes())}).json()
    version_id = client.post(f"/api/suite-package-imports/{imported['id']}/register").json()["id"]
    package = inspect_package(suite_package)
    target = package["manifest"]["targets"][0]
    with SessionLocal() as session:
        project = Project(name="suite-e2e-" + uuid4().hex, target_model_name="test", workspace_uri=str(tmp_path))
        worker = WorkerNode(name="suite-e2e-" + uuid4().hex, hostname="test", status="online", capabilities=["builtin.suite-manager", "builtin.suite-runner", "suite@" + package["digest"]], resources={"cpu_count": 4, "memory_available_bytes": 8 * 1024**3, "suite_runtime": {k: target[k] for k in ("os", "arch", "python")}})
        session.add_all([project, worker])
        session.flush()
        plan = make_plan(session, "install", version_id, [worker.id])
        operation = start_operation(session, plan, uuid4().hex, package["digest"], False)
        task = session.scalar(__import__('sqlalchemy').select(Task).where(Task.payload["suite_operation"]["operation_id"].as_string() == operation.id))
        spec = task.payload["suite_operation"]
        installed = {"version_id": version_id, "digest": package["digest"], "generation": uuid4().hex, "target_id": target["id"], "groups": {"default": {"python": "/isolated/python", "lock_digest": package["locks"][target["id"] + "/default"]["digest"]}}, "probes": {"normalize": {"ready": True}}}
        receipt = {**{k: spec[k] for k in ("operation_id", "suite_id", "generation", "action")}, "default_version_id": version_id, "installations": [installed]}
        bad = {**receipt, "installations": [{**installed, "groups": {}}]}
        with pytest.raises(ValueError, match="runtime groups"):
            finalize_operation(session, task, {"suite_receipt": bad})
        finalize_operation(session, task, {"suite_receipt": receipt})
        task.observed_state = "succeeded"
        sync_status(session, task)
        session.flush()
        revision = create_revision(session, project.id, version_id, "example.text-tools.normalize", {})
        assert revision.config == {"lowercase": False}
        revised = create_revision(session, project.id, version_id, revision.node_id, {"lowercase": True}, revision.id)
        assert revision.config == {"lowercase": False} and revised.previous_revision_id == revision.id
        import io
        artifact = artifact_storage.put_stream(session, project_id=project.id, kind="suite_input", source=io.BytesIO(b"HELLO"), filename="source.txt", media_type="text/plain", metadata={"format": "text/plain"})
        execution = run_revision(session, revision, {"source": artifact.id}, worker.id)
        assert execution.payload["suite_execution"]["inputs"]["source"]["digest"] == artifact.content_hash
        assert can_allocate(session, worker, execution)
        execution.observed_state = "running"
        claim_environment(session, worker, execution)
        session.flush()
        uninstall = make_plan(session, "uninstall", version_id, [worker.id])
        assert uninstall.spec["items"][0]["blocked"]
        node = {"id": "second", "config": {"suite_version_id": version_id, "suite_node_id": revision.node_id, "suite_config": {}, "suite_worker_id": worker.id, "suite_inputs": {"source": {"source_node_id": "first", "output_ref": "result"}}}}
        assert validate_canvas_node(session, node, [])
        assert not validate_canvas_node(session, node, [{"source": "first", "target": "second"}])
        output = artifact_storage.put_stream(session, project_id=project.id, kind="suite_output", source=io.BytesIO(b"hello"), filename="output.txt", media_type="text/plain", metadata={"format": "text/plain"})
        execution.result = {"suite_execution": execution.payload["suite_execution"], "environment": {k: installed[k] for k in ("generation", "target_id", "groups", "digest")}, "output_artifacts": [{"kind": "suite_output", "ref": "result", "artifact_id": output.id}]}
        finalize_run(session, execution)
        execution.result = {**execution.result, "output_artifacts": []}
        with pytest.raises(ValueError, match="output ports"):
            finalize_run(session, execution)
        session.get(SuiteVersion, version_id).enabled = False
        assert not can_allocate(session, worker, execution)
        session.rollback()
