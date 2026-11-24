from __future__ import annotations

import io
import json
import queue
import shutil
import threading
from pathlib import Path

import httpx
import pytest
from forgelm_plugin_sdk import Context, Request
from forgelm_plugin_sdk.serve import serve


def test_service_auth_stream_cancel_and_shutdown(tmp_path):
    class Plugin:
        stopped = False
        def start(self, *_):
            pass
        def health(self):
            return {"ready": True}
        def generate(self, request):
            yield request["prompt"]
            yield " done"
        def drain(self, deadline):
            pass
        def shutdown(self):
            self.stopped = True
    class Stream(io.StringIO):
        def write(self, value):
            event = json.loads(value)
            if event["event"] == "service.ready":
                ready.put(event["data"]["port"])
            return len(value)
    plugin, ready = Plugin(), queue.Queue()
    context = Context({"output_dir": str(tmp_path)}, Stream())
    thread = threading.Thread(target=serve, args=(plugin, {"service_token": "secret", "inputs": {}, "config": {}}, context))
    thread.start()
    try:
        port = ready.get(timeout=5)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
            assert client.get("/health").status_code == 403
            client.headers["Authorization"] = "Bearer secret"
            assert client.get("/health").json() == {"ready": True}
            assert client.post("/generate", content="broken").status_code == 400
            response = client.post("/generate", json={"request_id": "one", "prompt": "hello"})
            events = [json.loads(line) for line in response.text.splitlines()]
            assert events == [{"event": "chunk", "data": {"text": "hello"}}, {"event": "chunk", "data": {"text": " done"}}, {"event": "done", "data": {"finish_reason": "stop"}}]
            assert client.post("/cancel", json={"request_id": "one"}).status_code == 204
    finally:
        context.cancelled.set()
        thread.join(timeout=5)
    assert not thread.is_alive() and plugin.stopped
