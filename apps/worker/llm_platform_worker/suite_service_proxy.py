"""Relay local model streams through the authenticated Agent connection."""
import asyncio

import httpx


class StreamClosed(Exception):
    pass


async def relay_service(client, task, port, token, emit):
    worker_id = task["runtime_context"]["worker_id"]
    attempt_id = task["active_attempt_id"]
    scope = {"worker_id": worker_id, "attempt_id": attempt_id}
    pending, seen = {}, set()
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", headers={"Authorization": "Bearer " + token}, trust_env=False, timeout=30) as local:
        async def forward(request):
            sequence = len(request["chunks"])
            async def publish(event, data):
                nonlocal sequence
                sequence += 1
                body = {**scope, "sequence": sequence, "event": event, "data": data}
                attempt = 0
                while True:
                    try:
                        response = await client.post(f"/api/agent/suite-service-requests/{request['id']}/events", json=body)
                        if response.status_code in {404, 409}:
                            raise StreamClosed("Service request no longer accepts events")
                        response.raise_for_status()
                        if event == "chunk" and response.json().get("status") in {"failed", "cancelled"}:
                            raise StreamClosed("Service output was rejected")
                        return
                    except httpx.HTTPError:
                        attempt += 1
                        if attempt % 12 == 0:
                            await emit("log", {"level": "warning", "message": "服务输出等待控制面确认；保留当前序号重试"})
                        await asyncio.sleep(min(5, 0.5 * attempt))
            try:
                async with local.stream("POST", "/generate", json={"request_id": request["id"], **request["payload"]}, timeout=None) as response:
                    response.raise_for_status()
                    completed = False
                    async for line in response.aiter_lines():
                        if len(line) > 1024 * 1024:
                            raise ValueError("Service response exceeds frame limit")
                        import json
                        message = json.loads(line)
                        await publish(message["event"], message["data"])
                        if message["event"] in {"done", "error"}:
                            completed = True
                    if not completed:
                        await publish("error", {"message": "服务连接结束但未返回终态"})
            except asyncio.CancelledError:
                raise
            except StreamClosed:
                await local.post("/cancel", json={"request_id": request["id"]})
            except Exception as exc:
                try:
                    await publish("error", {"message": str(exc)})
                except Exception:
                    await emit("log", {"level": "error", "message": "服务流传输失败，等待控制面对账"})
        try:
            while True:
                try:
                    response = await client.get(f"/api/agent/suite-services/{task['id']}/requests", params=scope)
                    response.raise_for_status()
                    for request in response.json():
                        if request["cancel_requested"]:
                            await local.post("/cancel", json={"request_id": request["id"]})
                        if request["id"] not in seen:
                            seen.add(request["id"])
                            pending[request["id"]] = asyncio.create_task(forward(request))
                    for key in list(pending):
                        if pending[key].done():
                            await pending.pop(key)
                            seen.discard(key)
                except httpx.HTTPError as exc:
                    await emit("log", {"level": "warning", "message": f"服务请求通道暂不可用：{type(exc).__name__}"})
                await asyncio.sleep(0.25)
        finally:
            for future in pending.values():
                future.cancel()
            await asyncio.gather(*pending.values(), return_exceptions=True)
