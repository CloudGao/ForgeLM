from __future__ import annotations

import gzip
import hashlib

import httpx
import pytest
from llm_platform_worker.adapters.data_import import AnalysisInterrupted
from llm_platform_worker.adapters.data_mixture import DataMixtureAdapter
from llm_platform_worker.main import WorkerAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("pause", [False, True])
async def test_selection_materialization_checks_hashes_names_and_control(tmp_path, pause):
    payload = gzip.compress(b'{"type":"header"}\n')
    digest = hashlib.sha256(payload).hexdigest()
    downloads = []

    def handler(request):
        if request.url.path.endswith("/payload"):
            downloads.append(request.url.path)
            return httpx.Response(200, content=payload)
        kind = (
            "data_mixture_manifest"
            if request.url.path.endswith("/mixture")
            else "dataset_record_manifest"
        )
        return httpx.Response(200, json={"kind": kind, "content_hash": digest})

    async def emit(*_args):
        return "paused" if pause else "running"

    binding = {
        "manifest_artifact_id": "mixture",
        "manifest_hash": digest,
        "record_manifest_artifact_id": "records",
        "record_manifest_hash": digest,
    }
    task = {
        "project_id": "project",
        "payload": {"training_selection_inputs": [binding, dict(binding)]},
    }
    async with httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(handler)
    ) as client:
        agent = object.__new__(WorkerAgent)
        agent.client, agent.worker_id = client, "worker"
        agent.data_mixture = DataMixtureAdapter(client)
        if pause:
            with pytest.raises(AnalysisInterrupted):
                await agent.materialize_training_selections(task, tmp_path, emit)
            assert not list((tmp_path / "selection-inputs").iterdir())
        else:
            await agent.materialize_training_selections(task, tmp_path, emit)
            assert len(downloads) == 2
            for item in task["payload"]["training_selection_inputs"]:
                assert item["runtime_path"].endswith(".jsonl.gz")
                assert item["runtime_record_manifest_path"].endswith(".jsonl.gz")
