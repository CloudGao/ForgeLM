import asyncio
from types import SimpleNamespace

import httpx

from llm_platform_worker.main import WorkerAgent


def test_dispatch_wait_keeps_heartbeats_and_finishes_cleanup_before_execution(monkeypatch):
    async def scenario():
        delivered = asyncio.Event()
        order = []

        async def transport(request):
            assert request.url.path == "/api/agent/workers/worker/dispatch"
            await delivered.wait()
            order.append("delivered")
            return httpx.Response(200, json={"task": {"id": "task"}})

        agent = WorkerAgent.__new__(WorkerAgent)
        agent.worker_id = "worker"
        agent.client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(transport))

        async def heartbeat():
            order.append("cleanup_started")
            delivered.set()
            await asyncio.sleep(.01)
            order.append("cleanup_finished")

        agent.heartbeat = heartbeat
        real_wait = asyncio.wait

        async def fast_wait(futures, timeout):
            assert timeout == 5
            return await real_wait(futures, timeout=.001)

        monkeypatch.setattr(asyncio, "wait", fast_wait)
        try:
            response = await agent.receive_dispatch()
            order.append("execute")
            assert response["task"]["id"] == "task"
            assert order == ["cleanup_started", "delivered", "cleanup_finished", "execute"]
        finally:
            await agent.client.aclose()

    asyncio.run(scenario())


def test_old_control_plane_falls_back_only_when_dispatch_is_unsupported():
    async def scenario():
        agent = WorkerAgent.__new__(WorkerAgent)
        agent.worker_id = "worker"
        agent.config = SimpleNamespace(poll_seconds=0)
        agent.client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(
            lambda request: httpx.Response(404)))
        calls = []

        async def claim():
            calls.append(True)
            return {"task": {"id": "legacy"}}

        agent.claim = claim
        try:
            assert (await agent.receive_dispatch())["task"]["id"] == "legacy"
            assert calls == [True]
        finally:
            await agent.client.aclose()

    asyncio.run(scenario())


def test_cancelled_dispatch_wait_closes_the_pending_request():
    async def scenario():
        connected, closed = asyncio.Event(), asyncio.Event()

        async def transport(request):
            connected.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        agent = WorkerAgent.__new__(WorkerAgent)
        agent.worker_id = "worker"
        agent.client = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(transport))
        try:
            pending = asyncio.create_task(agent.receive_dispatch())
            await connected.wait()
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            assert closed.is_set()
        finally:
            await agent.client.aclose()

    asyncio.run(scenario())


def test_control_plane_restart_502_retries_without_exiting_worker(monkeypatch):
    async def scenario():
        agent = WorkerAgent.__new__(WorkerAgent)
        agent.worker_id = "worker"
        calls = []

        async def heartbeat():
            calls.append("heartbeat")
            if len(calls) == 1:
                response = httpx.Response(502, request=httpx.Request("POST", "http://test/heartbeat"))
                response.raise_for_status()
            raise asyncio.CancelledError()

        async def sleep(delay):
            assert delay == 2
            calls.append("retry")

        agent.heartbeat = heartbeat
        monkeypatch.setattr(asyncio, "sleep", sleep)
        result = await asyncio.gather(agent.loop(), return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert calls == ["heartbeat", "retry", "heartbeat"]

    asyncio.run(scenario())
