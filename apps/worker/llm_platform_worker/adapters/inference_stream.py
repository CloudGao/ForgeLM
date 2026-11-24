"""Bounded snapshot streaming; network latency must not stall GPU decoding."""
import asyncio
import threading
import time
from datetime import UTC, datetime


async def run_inference(adapter, task, emit, desired_state):
    if desired_state() in {"cancelled", "paused"}:
        return {"controlled_state": desired_state()}
    lock = threading.Lock()
    latest = None
    def publish(payload):
        nonlocal latest
        with lock:
            latest = payload
    adapter._publish = publish
    adapter._desired_state = desired_state
    adapter._active = True
    future = None
    try:
        await emit("progress", {"phase": "loading_model", "progress": 0.05})
        future = asyncio.create_task(asyncio.to_thread(adapter._generate_sync, task))
        while not future.done():
            await asyncio.wait({future}, timeout=0.1)
            with lock:
                snapshot, latest = latest, None
            if snapshot is not None:
                await emit("inference.snapshot", snapshot)
        result = await future
        if desired_state() in {"cancelled", "paused"}:
            adapter._prefix = None
            return {"controlled_state": desired_state()}
        await emit("inference.output", {**result, "phase": "finalizing_outputs"})
        await emit("progress", {"phase": "finalizing_outputs", "progress": 0.95,
                               "output_tokens": result["output_tokens"]})
        return result
    except BaseException:
        adapter._prefix = None
        if future is not None and not future.done():
            adapter._desired_state = lambda: "cancelled"
            await asyncio.shield(future)
        raise
    finally:
        adapter._last_used = time.monotonic()
        adapter._active = False
        adapter._publish = lambda payload: None


class TokenStream:
    def __init__(self, adapter, tokenizer, started, input_tokens=0, created_at=None):
        self.adapter, self.tokenizer, self.started = adapter, tokenizer, started
        self.tokens = []
        self.first_ms = None
        self.request_offset_ms = None
        self.request_first_ms = None
        if created_at:
            try:
                self.request_offset_ms = (datetime.now(UTC) - datetime.fromisoformat(created_at).replace(tzinfo=UTC)).total_seconds() * 1000
            except (ValueError, TypeError):
                pass
        self.last = 0.0
        self.input_tokens = input_tokens
        self.skip_prompt = True

    def put(self, value):
        # HF first sends the complete prompt, then one token per call.
        if self.skip_prompt:
            self.skip_prompt = False
            return
        self.tokens.extend(int(x) for x in value.reshape(-1).tolist())
        self.snapshot(self.tokens)

    def snapshot(self, tokens, force=False):
        now = time.perf_counter()
        if self.first_ms is None and tokens:
            self.first_ms = (now - self.started) * 1000
            if self.request_offset_ms is not None:
                self.request_first_ms = self.request_offset_ms + self.first_ms if self.request_offset_ms >= 0 else None
        if not force and now - self.last < 0.25:
            return
        self.last = now
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        # An incomplete UTF-8 token can temporarily decode to a replacement character.
        self.adapter._publish({"phase": "generating", "output": text.rstrip("\ufffd"),
            "output_tokens": len(tokens), "input_tokens": self.input_tokens,
            "performance": {"ttft_ms": self.first_ms, "request_ttft_ms": self.request_first_ms}})

    def end(self):
        self.snapshot(self.tokens, force=True)
