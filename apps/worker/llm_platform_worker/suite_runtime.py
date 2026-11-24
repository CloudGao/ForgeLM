"""Versioned suite environments; package code never loads into the Worker."""
from __future__ import annotations

import asyncio
import json
import os
import platform
import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from forgelm_plugin_sdk.package import (
    extract_package,
    file_hash,
    identity,
    inspect_package,
    safe_path,
)

from .adapters.command import CommandAdapter


def token(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,179}", value):
        raise ValueError("Invalid operation/generation token")
    return value


@contextmanager
def file_lock(path: Path, shared: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes
                class Overlapped(ctypes.Structure):
                    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t), ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
                kernel.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(Overlapped)]
                overlapped = Overlapped()
                native = msvcrt.get_osfhandle(handle.fileno())
                if not kernel.LockFileEx(native, 1 | (0 if shared else 2), 0, 1, 0, ctypes.byref(overlapped)):
                    raise OSError(ctypes.get_last_error(), "Suite lease is held")
            else:
                import fcntl
                fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Suite environment is in use; drain its tasks before removing") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                kernel.UnlockFileEx(native, 0, 1, 0, ctypes.byref(overlapped))
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def read_json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def environment_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def match_target(package: dict, target_id: str) -> dict:
    target = next((t for t in package["manifest"]["targets"] if t["id"] == target_id), None)
    arch = platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64")
    if not target or (target["os"], target["arch"], target["python"]) != (platform.system().lower(), arch, f"{sys.version_info.major}.{sys.version_info.minor}"):
        raise ValueError("Suite target does not match Worker OS, architecture and Python ABI")
    if target["accelerator"] == "cuda":
        # Driver probing is read-only and uses an argument array.
        import subprocess
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout=10, text=True)
        def numeric(v):
            return tuple(int(i) for i in v.strip().split("."))
        if not output.strip() or any(numeric(v) < numeric(target["minDriver"]) for v in output.splitlines()):
            raise ValueError("CUDA driver is below the suite minimum")
    return target


class SuiteRuntime:
    def __init__(self, root: Path, client=None):
        self.root = (root / "suite-packages").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.command = CommandAdapter()

    def suite_dir(self, suite_id: str) -> Path:
        return self.root / "deployments" / identity(suite_id)

    async def fetch(self, artifact: dict, project_id: str, destination: Path):
        if destination.is_file() and file_hash(destination) == artifact["digest"]:
            return
        if self.client is None:
            raise ValueError("Artifact is not cached and no control-plane client is available")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
        try:
            async with self.client.stream("GET", f"/api/agent/artifacts/{artifact['artifact_id']}/payload", params={"project_id": project_id}, timeout=None) as response:
                response.raise_for_status()
                with temp.open("wb") as stream:
                    total = 0
                    async for data in response.aiter_bytes():
                        total += len(data)
                        if total > int(artifact["size_bytes"]):
                            raise ValueError("Artifact exceeds its declared size")
                        stream.write(data)
            if file_hash(temp) != artifact["digest"]:
                raise ValueError("Artifact content hash mismatch")
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)

    async def invoke(self, task, executable, args, emit, desired, *, cwd: Path, timeout: float = 1800, plugin=False, on_event=None):
        from .suite_process import run_process
        env = {k: v for k, v in (task.get("payload", {}).get("command", {}).get("env", {}) or {}).items() if k in {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"}}
        env.update({"PYTHONPATH": "", "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        return await run_process(executable, args, cwd, env, emit, desired, timeout, protocol=plugin, on_event=on_event)

    async def manage(self, task: dict, emit, desired=None):
        spec = task["payload"]["suite_operation"]
        suite_dir = self.suite_dir(spec["suite_id"])
        operation_id = token(spec["operation_id"])
        ledger = suite_dir / "operations" / f"{operation_id}.json"
        with file_lock(suite_dir / ".operation.lock"):
            previous = read_json(ledger)
            if previous:
                return {"suite_receipt": previous}
            index_path = suite_dir / "index.json"
            index = read_json(index_path, {"generation": 0, "versions": {}, "default": None})
            if index.get("operation_id") == operation_id:
                receipt = {"operation_id": operation_id, "generation": index["generation"], "suite_id": spec["suite_id"], "default_version_id": index["default"], "installations": list(index["versions"].values()), "action": index["action"]}
                atomic_json(ledger, receipt)
                return {"suite_receipt": receipt}
            if int(spec["generation"]) <= index["generation"]:
                raise ValueError("Stale suite operation generation")
            action = spec["action"]
            if action in {"install", "update", "rollback", "repair"}:
                ref = spec["package"]
                if spec.get("trusted_digest") != ref["digest"]:
                    raise ValueError("Package digest has not been trusted for this operation")
                archive = self.root / "blobs" / (ref["digest"] + ".flsuite")
                await self.fetch(ref, spec["package_project_id"], archive)
                package = await asyncio.to_thread(inspect_package, archive)
                if package["manifest"]["id"] != spec["suite_id"] or package["digest"] != ref["digest"]:
                    raise ValueError("Package identity mismatch")
                target = match_target(package, spec["target_id"])
                version_id = spec["version_id"]
                existing = index["versions"].get(version_id)
                if existing and action != "repair":
                    installed = existing
                else:
                    required = package["expanded_bytes"] + package["installed_bytes"] + 128 * 1024**2
                    if shutil.disk_usage(suite_dir).free < required:
                        raise ValueError("Insufficient free disk space for the isolated suite environment")
                    generation = uuid4().hex
                    generation_dir = suite_dir / "generations" / generation
                    generation_dir.mkdir(parents=True)
                    published = False
                    try:
                        await emit("progress", {"phase": "extracting_suite", "progress": 0.1})
                        unpacked = generation_dir / "package"
                        await asyncio.to_thread(extract_package, archive, unpacked)
                        probes, groups = {}, {}
                        for group, group_spec in target["runtimeGroups"].items():
                            env_dir = generation_dir / "envs" / group
                            control = await self.invoke(task, sys.executable, ["-m", "venv", str(env_dir)], emit, desired, cwd=generation_dir)
                            if control.get("controlled_state"):
                                return control
                            python = environment_python(env_dir)
                            frozen = read_json(unpacked / group_spec["lock"])
                            requirements = generation_dir / f"{group}.requirements.txt"
                            requirements.write_text("\n".join(f"{(unpacked / p['path']).as_uri()} --hash=sha256:{p['sha256']}" for p in frozen["packages"]) + "\n", encoding="utf-8")
                            control = await self.invoke(task, python, ["-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes", "-r", requirements], emit, desired, cwd=generation_dir)
                            if control.get("controlled_state"):
                                return control
                            await self.invoke(task, python, ["-m", "pip", "check"], emit, desired, cwd=generation_dir)
                            groups[group] = {"python": str(python), "lock_digest": file_hash(unpacked / group_spec["lock"])}
                        for component in package["manifest"]["components"]:
                            request_path = generation_dir / (component["id"] + ".probe.json")
                            atomic_json(request_path, {"mode": "probe", "component": component, "execution_kinds": sorted({n["execution"]["kind"] for n in package["nodes"] if n["component"] == component["id"]}), "output_dir": str(generation_dir / "probe-output"), "run_id": task["id"]})
                            result = await self.invoke(task, groups[component["runtimeGroup"]]["python"], ["-m", "forgelm_plugin_sdk.runner", "--request", request_path], emit, desired, cwd=generation_dir, timeout=60, plugin=True)
                            if result.get("controlled_state"):
                                return result
                            probes[component["id"]] = result["probe"]
                        installed = {"suite_id": spec["suite_id"], "version_id": version_id, "version": package["manifest"]["version"], "digest": ref["digest"], "generation": generation, "target_id": target["id"], "groups": groups, "probes": probes, "package": package}
                        atomic_json(generation_dir / "receipt.json", installed)
                        published = True
                    finally:
                        if not published:
                            generation_dir.resolve().relative_to((suite_dir / "generations").resolve())
                            shutil.rmtree(generation_dir)
                index["versions"][version_id] = installed
                index["default"] = version_id
            elif action == "uninstall":
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for version_id in spec["version_ids"]:
                        receipt = index["versions"].get(version_id)
                        if receipt:
                            stack.enter_context(file_lock(suite_dir / "leases" / (receipt["generation"] + ".lock")))
                    for version_id in spec["version_ids"]:
                        receipt = index["versions"].pop(version_id, None)
                        if receipt:
                            target = suite_dir / "generations" / token(receipt["generation"])
                            target.resolve().relative_to((suite_dir / "generations").resolve())
                            if target.is_symlink():
                                raise ValueError("Refusing linked environment cleanup")
                            if target.exists():
                                shutil.rmtree(target)
                    if index["default"] in spec["version_ids"]:
                        index["default"] = None
            else:
                raise ValueError("Unsupported suite operation")
            index["generation"] = int(spec["generation"])
            # One persisted index is the activation point. Include operation ID
            # so a lost acknowledgement can be recovered without reinstalling.
            index["operation_id"] = operation_id
            index["action"] = action
            atomic_json(index_path, index)
            receipt = {"operation_id": operation_id, "generation": index["generation"], "suite_id": spec["suite_id"], "default_version_id": index["default"], "installations": list(index["versions"].values()), "action": action}
            atomic_json(ledger, receipt)
            return {"suite_receipt": receipt}

    async def execute(self, task, emit, desired=None):
        spec = task["payload"]["suite_execution"]
        suite_dir = self.suite_dir(spec["suite_id"])
        index = read_json(suite_dir / "index.json", {})
        receipt = index.get("versions", {}).get(spec["version_id"])
        if not receipt or receipt["digest"] != spec["digest"]:
            raise ValueError("Exact suite package is not installed")
        with file_lock(suite_dir / "leases" / (receipt["generation"] + ".lock"), shared=True):
            # Recheck after acquiring lease to close the uninstall/execute race.
            current = read_json(suite_dir / "index.json", {}).get("versions", {}).get(spec["version_id"])
            if not current or current["generation"] != receipt["generation"]:
                raise ValueError("Suite deployment changed before acquiring the environment")
            package = receipt["package"]
            node = next(n for n in package["nodes"] if n["id"] == spec["node_id"])
            if spec.get("node_digest", node["digest"]) != node["digest"]:
                raise ValueError("Frozen NodeSpec digest does not match installed package")
            component = next(c for c in package["manifest"]["components"] if c["id"] == node["component"])
            attempt = Path(task["runtime_context"]["attempt_dir"])
            output_dir = attempt / "suite-outputs"
            inputs = {}
            for name, ref in spec.get("inputs", {}).items():
                destination = attempt / "suite-inputs" / safe_path(name)
                await self.fetch(ref, task["project_id"], destination)
                inputs[name] = {**ref, "path": str(destination)}
            resume = task.get("suite_resume") or spec.get("continuation") or {}
            if resume:
                from forgelm_plugin_sdk.artifacts import unpack
                destination = attempt / "suite-resume.archive"
                await self.fetch(resume["artifact"], task["project_id"], destination)
                restored = attempt / "suite-resume"
                await asyncio.to_thread(unpack, destination, restored)
                resume = {**resume, "path": str(restored)}
            request = {"component": component, "node": node, "config": spec["config"], "inputs": inputs, "resources": task["runtime_context"].get("allocation", {}), "output_dir": str(output_dir), "run_id": task["id"], "resume": resume}
            request["previous_resolution"] = task.get("suite_resolution") or {}
            request["require_execution_ack"] = True
            environment = {k: receipt[k] for k in ("generation", "target_id", "groups", "digest")}
            resolved = None
            checkpoints = 0
            relay = None
            if node["execution"]["kind"] == "service":
                import secrets
                request.update({"mode": "service", "service_token": secrets.token_urlsafe(32)})
            async def service_event(event, data):
                nonlocal relay, resolved, checkpoints
                if event == "training.resolved":
                    if resolved is not None or data.get("resolved_config") != spec["config"]:
                        raise ValueError("Invalid or repeated execution resolution")
                    resolved = {**data, "suite_execution": spec, "component": component, "environment": environment,
                                "continuation": {k: v for k, v in resume.items() if k != "path"}}
                    atomic_json(attempt / "execution-resolved.json", resolved)
                    await emit("execution.resolved", resolved)
                elif event == "training.checkpoint":
                    if not resolved or not node.get("training"):
                        raise ValueError("Checkpoint requires resolved training execution")
                    path = output_dir / safe_path(data["path"])
                    path.resolve().relative_to(output_dir.resolve())
                    if path.is_symlink() or not path.is_dir() or any(p.is_symlink() for p in path.rglob("*")):
                        raise ValueError("Checkpoint contains invalid paths or links")
                    await emit("training.checkpoint.upload", {"path": str(path), "metadata": data["metadata"]})
                    checkpoints += 1
                if event == "service.ready":
                    if relay is not None or node["execution"]["kind"] != "service" or type(data.get("port")) is not int or not 1 <= data["port"] <= 65535:
                        raise ValueError("Invalid service endpoint announcement")
                    from .suite_service_proxy import relay_service
                    relay = asyncio.create_task(relay_service(self.client, task, data["port"], request["service_token"], emit))
            request_path = attempt / "suite-request.json"
            atomic_json(request_path, request)
            try:
                result = await self.invoke(task, receipt["groups"][component["runtimeGroup"]]["python"], ["-m", "forgelm_plugin_sdk.runner", "--request", request_path], emit, desired, cwd=attempt, timeout=float(spec.get("timeout_seconds", 86400)), plugin=True, on_event=service_event)
            finally:
                if relay:
                    relay.cancel()
                    await asyncio.gather(relay, return_exceptions=True)
            if result.get("controlled_state"):
                if result["controlled_state"] == "paused" and (not resolved or not checkpoints):
                    raise ValueError("Safe pause requires a persisted checkpoint from this attempt")
                return result
            if node.get("training") and (not resolved or not checkpoints):
                raise ValueError("Training requires an execution receipt and complete checkpoint")
            outputs = []
            for output in result.get("outputs", []):
                path = output_dir / safe_path(output["path"])
                path.resolve().relative_to(output_dir.resolve())
                if path.is_symlink() or not path.exists() or path.is_dir() and any(p.is_symlink() for p in path.rglob("*")):
                    raise ValueError("Output contains invalid paths or links")
                outputs.append({"path": str(path), "ref": output["port"], "kind": "suite_output", "metadata": {**output.get("metadata", {}), "format": output["format"], "suite_execution": spec}})
            return {"outputs": outputs, "suite_execution": spec, "environment": environment, "execution_receipt": resolved, "metadata": result.get("metadata", {})}

    def capabilities(self) -> list[str]:
        result = ["builtin.suite-manager", "builtin.suite-runner"]
        for path in (self.root / "deployments").glob("*/index.json"):
            try:
                for receipt in read_json(path, {}).get("versions", {}).values():
                    generation = path.parent / "generations" / token(receipt["generation"])
                    if (generation / "receipt.json").is_file() and all(Path(g["python"]).is_file() for g in receipt["groups"].values()):
                        result.append("suite@" + receipt["digest"])
            except (OSError, ValueError, KeyError):
                continue
        return result
