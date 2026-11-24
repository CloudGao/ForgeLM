"""Bounded subprocess protocol with group cleanup and Windows job ownership."""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import signal
import subprocess


def windows_job(pid: int):
    if os.name != "nt":
        return lambda: None
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    class Limits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong), ("flags", w.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("active", w.DWORD), ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [("counter" + str(i), ctypes.c_ulonglong) for i in range(6)]
    class Extended(ctypes.Structure):
        _fields_ = [("basic", Limits), ("io", IO), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    info = Extended()
    info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    handle = kernel.OpenProcess(0x0100 | 0x0001, False, pid)
    try:
        if not job or not handle or not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) or not kernel.AssignProcessToJobObject(job, handle):
            if job:
                kernel.CloseHandle(job)
            raise OSError(ctypes.get_last_error(), "Cannot own suite process with a Windows job")
    finally:
        if handle:
            kernel.CloseHandle(handle)
    return lambda: kernel.CloseHandle(job)


async def run_process(executable, args, cwd, env, emit, desired, timeout, *, protocol=False, on_event=None):
    inherited = {k: v for k, v in os.environ.items() if k.upper() in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LANG", "LC_ALL", "LD_LIBRARY_PATH"}}
    inherited.update(env)
    inherited["FORGELM_PARENT_PID"] = str(os.getpid())
    process = await asyncio.create_subprocess_exec(str(executable), *map(str, args), cwd=str(cwd), env=inherited,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        limit=1024 * 1024 + 1, start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    def close_job():
        pass
    readers = []
    queue = asyncio.Queue(maxsize=256)
    result, seq, ended = [], 0, 0
    async def pump(stream, name):
        try:
            while line := await stream.readline():
                await queue.put((name, line))
        except Exception as exc:
            await queue.put(("exception", exc))
        finally:
            await queue.put((name, None))
    try:
        close_job = windows_job(process.pid)
        readers = [asyncio.create_task(pump(process.stdout, "stdout")), asyncio.create_task(pump(process.stderr, "stderr"))]
        deadline = asyncio.get_running_loop().time() + timeout
        cancelled_at = None
        while ended < 2:
            now = asyncio.get_running_loop().time()
            if now > deadline:
                raise TimeoutError("Suite subprocess exceeded its deadline")
            state = desired() if desired else "running"
            if state in {"paused", "cancelled"}:
                if cancelled_at is None:
                    cancelled_at = now
                    if process.stdin and not process.stdin.is_closing():
                        process.stdin.write(b'{"command":"pause"}\n' if state == "paused" else b'{"command":"cancel"}\n')
                        try:
                            await process.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                if now - cancelled_at > (300 if state == "paused" else 5):
                    if state == "paused":
                        raise TimeoutError("Training did not acknowledge a safe checkpoint pause")
                    return {"controlled_state": state}
            try:
                stream, line = await asyncio.wait_for(queue.get(), timeout=0.25)
            except TimeoutError:
                continue
            if stream == "exception":
                raise line
            if line is None:
                ended += 1
                continue
            text = line.decode("utf-8", errors="strict").rstrip()
            if protocol and stream == "stdout":
                event = json.loads(text)
                if event.get("protocol") != "forgelm.plugin.v1" or event.get("seq") != seq + 1 or not isinstance(event.get("data"), dict):
                    raise ValueError("Invalid suite component event protocol/sequence")
                seq += 1
                kind, data = event["event"], event["data"]
                if kind == "result":
                    result.append(data)
                if on_event:
                    await on_event(kind, data)
                if kind in {"training.resolved", "training.checkpoint"} and on_event and process.stdin and not process.stdin.is_closing():
                    process.stdin.write((json.dumps({"command": "ack", "seq": seq}) + "\n").encode())
                    await process.stdin.drain()
                await emit(kind, data)
            else:
                await emit("log", {"stream": stream, "message": text})
        code = await process.wait()
        if cancelled_at is not None:
            if state == "paused":
                if code == 0 and len(result) == 1 and result[0].get("controlled_state") == "paused":
                    return result[0]
                raise ValueError("Training exited without acknowledging a safe checkpoint pause")
            return {"controlled_state": "cancelled"}
        if code:
            raise RuntimeError(f"Suite subprocess exited with {code}")
        if protocol and len(result) != 1:
            raise ValueError("Component must return exactly one result")
        return result[0] if protocol else {"exit_code": 0}
    finally:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        close_job()
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
