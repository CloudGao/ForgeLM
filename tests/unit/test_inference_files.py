import asyncio
import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pytest

from llm_platform_worker.main import AgentConfig, WorkerAgent


def test_two_attempts_and_new_worker_reuse_files_and_detect_damage(tmp_path):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("model/config.json", "{}")
        archive.writestr("model/model.safetensors", "weights")
    content = data.getvalue()
    calls = []
    denied = False
    def handler(request):
        calls.append(request.url.path)
        if denied:
            return httpx.Response(403)
        if request.url.path.endswith("payload"):
            return httpx.Response(200, content=content)
        return httpx.Response(200, json={"id": "a", "content_hash": hashlib.sha256(content).hexdigest(), "locations": []})
    async def run():
        nonlocal denied
        worker = WorkerAgent(AgentConfig("http://test", "token", "worker"))
        await worker.client.aclose()
        worker.client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
        def task(): return {"project_id": "p", "payload": {"model": {"artifact_id": "a"}}}
        first, second = task(), task()
        await asyncio.gather(worker.materialize_inference_model(first, tmp_path / "runs/t1/a1"),
                             worker.materialize_inference_model(second, tmp_path / "runs/t2/a2"))
        assert sum(p.endswith("payload") for p in calls) == 1
        assert first["payload"]["model"]["runtime_source"] == second["payload"]["model"]["runtime_source"]
        await worker.client.aclose()
        worker = WorkerAgent(AgentConfig("http://test", "token", "new-worker"))
        await worker.client.aclose()
        worker.client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(handler))
        third = task()
        await worker.materialize_inference_model(third, tmp_path / "runs/t3/a3")
        assert third["payload"]["model"]["file_cache_hit"]
        source = Path(third["payload"]["model"]["runtime_source"])
        (source / "model.safetensors").write_text("damaged")
        await worker.materialize_inference_model(task(), tmp_path / "runs/t4/a4")
        assert sum(p.endswith("payload") for p in calls) == 2
        denied = True
        with pytest.raises(httpx.HTTPStatusError):
            await worker.materialize_inference_model(task(), tmp_path / "runs/t5/a5")
        await worker.client.aclose()
    asyncio.run(run())
