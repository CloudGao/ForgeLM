"""Local-only authenticated service protocol. Worker proxies all requests."""
from __future__ import annotations

import hmac
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve(plugin, raw: dict, context):
    for name in ("start", "health", "generate", "drain", "shutdown"):
        if not callable(getattr(plugin, name, None)):
            raise ValueError(f"Service component lacks {name}()")
    token = raw["service_token"]
    active, cancelled = set(), set()
    lock = threading.Lock()
    plugin.start(context, raw.get("inputs", {}).get("model"), raw.get("config", {}))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(403)
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != "/health":
                self.send_error(404)
                return
            data = json.dumps(plugin.health(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400)
                return
            if not 0 < length <= 1024 * 1024:
                self.send_error(413)
                return
            try:
                request = json.loads(self.rfile.read(length))
                request_id = request["request_id"]
                if not isinstance(request_id, str) or not 1 <= len(request_id) <= 180:
                    raise ValueError("Invalid request ID")
            except (ValueError, TypeError, KeyError):
                self.send_error(400)
                return
            if self.path == "/cancel":
                with lock:
                    if request_id in active:
                        cancelled.add(request_id)
                self.send_response(204)
                self.end_headers()
                return
            if self.path != "/generate" or context.cancelled.is_set():
                self.send_error(503)
                return
            with lock:
                if request_id in active or len(active) >= 32:
                    self.send_error(409)
                    return
                active.add(request_id)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                output = plugin.generate(request)
                if isinstance(output, (dict, str)):
                    output = [output]
                for chunk in output:
                    if context.cancelled.is_set() or request_id in cancelled:
                        break
                    data = {"event": "chunk", "data": {"text": chunk} if isinstance(chunk, str) else chunk}
                    encoded = json.dumps(data, allow_nan=False).encode() + b"\n"
                    if len(encoded) > 1024 * 1024:
                        raise ValueError("Service chunk exceeds limit")
                    self.wfile.write(encoded)
                    self.wfile.flush()
                self.wfile.write(json.dumps({"event": "done", "data": {"finish_reason": "cancelled" if request_id in cancelled or context.cancelled.is_set() else "stop"}}).encode() + b"\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                try:
                    self.wfile.write(json.dumps({"event": "error", "data": {"message": str(exc)}}).encode() + b"\n")
                except OSError:
                    pass
            finally:
                with lock:
                    active.discard(request_id)
                    cancelled.discard(request_id)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.timeout = 0.25
    server.daemon_threads = True
    try:
        health = plugin.health()
        if not isinstance(health, dict) or health.get("ready") is not True:
            raise ValueError("Service did not become healthy")
        context.emit("service.ready", {"port": server.server_port, "health": health})
        while not context.cancelled.is_set():
            server.handle_request()
        plugin.drain(time.time() + 4)
        deadline = time.monotonic() + 4
        while active and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        server.server_close()
        plugin.shutdown()
