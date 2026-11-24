"""Effective CPU/memory capacity, including container quotas when exposed."""
from __future__ import annotations

import math
import os
import platform
import subprocess
from pathlib import Path

import psutil


def training_snapshot_memory_available() -> int:
    """Bound reversible optimizer snapshots by physical RAM on both WSL hosts."""
    available = effective_capacity()["memory_available_bytes"]
    if "microsoft" not in platform.release().lower():
        return available
    powershell = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not powershell.is_file():
        return 0
    try:
        result = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-Command",
             "[Console]::Write((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory)"],
            capture_output=True, text=True, timeout=30, check=True)
        return min(available, int(result.stdout.strip()) * 1024)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def effective_capacity(root: Path = Path("/sys/fs/cgroup")) -> dict[str, int]:
    host_cpus = os.cpu_count() or 1
    try:
        affinity = len(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.Error):
        affinity = host_cpus
    cpus = min(host_cpus, affinity or host_cpus)
    # Inspect ancestors too: a parent quota can be tighter than the leaf quota.
    cpu_roots = {root, root / "cpu", root / "cpu,cpuacct"}
    memory_roots = {root, root / "memory"}
    for line in _read(Path("/proc/self/cgroup")).splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3 or ".." in Path(parts[2]).parts:
            continue
        controllers, relative = parts[1].split(","), parts[2].lstrip("/")
        for base in ([root] if not parts[1] else [root / parts[1], *(root / c for c in controllers)]):
            leaf = base / relative
            for candidate in (leaf, *leaf.parents):
                if candidate == base or base in candidate.parents:
                    if not parts[1] or "cpu" in controllers:
                        cpu_roots.add(candidate)
                    if not parts[1] or "memory" in controllers:
                        memory_roots.add(candidate)
    for base in cpu_roots:
        values = _read(base / "cpu.max").split()
        if len(values) != 2:
            values = [_read(base / "cpu.cfs_quota_us"), _read(base / "cpu.cfs_period_us")]
        try:
            quota, period = map(int, values)
            if quota > 0 and period > 0:
                cpus = min(cpus, max(1, math.floor(quota / period)))
        except ValueError:
            pass
    memory = psutil.virtual_memory()
    total, available = int(memory.total), int(memory.available)
    for base in memory_roots:
        for limit_name, usage_name in (("memory.max", "memory.current"), ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            try:
                limit = int(_read(base / limit_name))
                usage = int(_read(base / usage_name))
                stats = {}
                for line in _read(base / "memory.stat").splitlines():
                    parts = line.split()
                    if len(parts) == 2:
                        stats[parts[0]] = int(parts[1])
                inactive_file = stats.get("total_inactive_file", stats.get("inactive_file", 0))
                # Reclaimable file cache is not pinned working memory.
                working_set = max(0, usage - max(0, inactive_file))
                if 0 < limit <= total:
                    total = limit
                    available = min(available, max(0, limit - working_set))
            except ValueError:
                pass
    return {"cpu_count": cpus, "host_cpu_count": host_cpus,
            "memory_total_bytes": total, "memory_available_bytes": available}


def processing_parallelism(requested, allocation: dict, capacity: dict) -> int:
    limit = min(256, int(capacity["cpu_count"]))
    if allocation.get("cpus"):
        limit = min(limit, max(1, int(allocation["cpus"])))
    # Leave memory for the parent, Arrow buffers and dataset export.
    memory_limit = max(1, int(capacity["memory_available_bytes"]) // (2 * 1024**3))
    limit = max(1, min(limit, memory_limit))
    if requested in (None, "auto"):
        return limit
    if isinstance(requested, bool) or not 1 <= int(requested) <= 256:
        raise ValueError("Data-Juicer np must be auto or an integer between 1 and 256")
    return min(int(requested), limit)


def dataset_loading_parallelism(processes: int, capacity: dict) -> int:
    # Cold Parquet decoding expands compressed row groups in each loader. Its
    # working set is much larger than the streaming map/hash operators' buffers.
    memory_limit = max(1, int(capacity["memory_available_bytes"]) // (8 * 1024**3))
    return max(1, min(processes, 8, memory_limit))
