from uuid import uuid4

from llm_platform.database import SessionLocal
from llm_platform.models import RemoteWorkerEnvironment, RemoteWorkerOperation
from llm_platform.services.remote_workers import _environment_file


def test_download_policy_preserves_environment_and_is_frozen_during_upgrade(client):
    with SessionLocal() as session:
        env = RemoteWorkerEnvironment(name=uuid4().hex, ssh_host="host.test", ssh_port=22,
            ssh_user="worker", credential_ref="TEST_DOWNLOAD_KEY", host_key="test",
            host_key_fingerprint="test", control_plane_url="http://localhost:8000",
            install_root="/opt/worker", workspace_root="/data/worker",
            metadata_json={"deployment": {"runtime": "data"}, "keep": "unchanged"})
        session.add(env)
        session.commit()
        identity = env.id
    url = f"/api/execution-environments/remote/{identity}/download-policy"
    result = client.put(url, json={"download_policy": "domestic"})
    assert result.status_code == 200, result.text
    assert result.json()["download_policy"] == "domestic"
    with SessionLocal() as session:
        env = session.get(RemoteWorkerEnvironment, identity)
        assert env.metadata_json["keep"] == "unchanged"
        assert env.metadata_json["deployment"]["runtime"] == "data"
        config = _environment_file(env, "test-enrollment")
        assert 'HF_ENDPOINT="https://hf-mirror.com"' in config
        assert 'PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"' in config
        session.add(RemoteWorkerOperation(environment_id=identity, kind="upgrade", status="queued"))
        session.commit()
    assert client.put(url, json={"download_policy": "official"}).status_code == 409
