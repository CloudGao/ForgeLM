from llm_platform.models import RemoteWorkerEnvironment
from llm_platform.services.remote_workers import _environment_file
from llm_platform_core.storage_layout import storage_layout

MOUNTS = """10 1 0:1 / / rw - overlay overlay rw
11 10 8:2 /user-disk /root/autodl-tmp rw - ext4 /dev/sdb rw
"""


def test_autodl_defaults_use_real_data_mount():
    layout = storage_layout("/var/lib/forgelm/workspaces", MOUNTS)
    assert layout["provider"] == "autodl"
    assert layout["workspace_root"] == "/root/autodl-tmp/forgelm"
    assert layout["system_disk"]["device"] != layout["data_disk"]["device"]
    assert storage_layout("/root/autodl-tmp/custom", MOUNTS)["workspace_root"] == "/root/autodl-tmp/custom"


def test_no_data_mount_or_same_filesystem_does_not_redirect():
    for text in ("", MOUNTS.splitlines()[0], MOUNTS.replace("8:2", "0:1")):
        layout = storage_layout("/srv/work", text)
        assert layout["provider"] == "generic"
        assert layout["workspace_root"] == "/srv/work"


def test_worker_caches_and_temporary_files_follow_workspace():
    env = RemoteWorkerEnvironment(id="test", name="autodl", profile="general",
        install_root="/opt/worker", workspace_root="/root/autodl-tmp/forgelm",
        control_plane_url="http://localhost:8000", labels={})
    config = _environment_file(env, "")
    for key in ("HF_HOME", "PIP_CACHE_DIR", "UV_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR"):
        assert f'{key}="/root/autodl-tmp/forgelm/.' in config
    assert 'LLM_PLATFORM_AGENT_TOKEN_FILE="/opt/worker/state/agent-token"' in config
