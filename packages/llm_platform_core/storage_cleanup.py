"""Exact, non-following filesystem inspection shared by storage executors."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from pathlib import Path


def filesystem_path(path: str | Path) -> Path:
    """Use the Windows verbatim namespace after boundary checks, preserving names.

    WSL files can contain trailing dots that normal Win32 paths silently trim.
    This changes OS I/O syntax only, never stored paths or artifact URIs.
    """
    value = str(Path(path).absolute())
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return Path(value)
    return Path("\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value)


def checked_path(path: str | Path, root: str | Path) -> Path:
    target, boundary = Path(path).absolute(), Path(root).absolute()
    if target == boundary or not target.is_relative_to(boundary):
        raise ValueError("清理路径超出平台托管边界")
    # Check lexical ancestors before resolving, including Windows junctions.
    for parent in (target, *target.parents):
        if parent.exists() or parent.is_symlink():
            info = parent.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("清理路径含符号链接或重解析点")
    if not target.resolve().is_relative_to(boundary.resolve()):
        raise ValueError("清理路径解析后越界")
    return target


def inspect_target(path: str | Path, root: str | Path) -> dict:
    target = filesystem_path(checked_path(path, root))
    digest = hashlib.sha256()
    total = count = 0
    if not target.exists():
        return {"size_bytes": 0, "file_count": 0, "fingerprint": "missing"}
    entries = [target]
    if target.is_dir():
        for current, dirs, files in os.walk(
            target, followlinks=False, onerror=lambda error: (_ for _ in ()).throw(error)
        ):
            dirs.sort()
            files.sort()
            # Reject junctions before os.walk can descend into their destinations.
            for name in dirs:
                info = (Path(current) / name).lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                    raise ValueError("清理目录含符号链接或重解析点")
            entries.extend(Path(current) / name for name in dirs + files)
    for entry in entries:
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("清理目录含符号链接或重解析点")
        if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
            raise ValueError("清理目录含非常规文件")
        relative = str(entry.relative_to(target))
        digest.update(
            f"{relative}|{info.st_mode}|{info.st_size}|{info.st_mtime_ns}|{info.st_ino}\n".encode()
        )
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
            count += 1
    return {"size_bytes": total, "file_count": count, "fingerprint": digest.hexdigest()}


def remove_target(item: dict) -> dict:
    """The caller supplies a server-authorized, recently revalidated plan item."""
    path = checked_path(item["path"], item["root"])
    current = inspect_target(path, item["root"])
    if any(current[key] != item[key] for key in current):
        raise ValueError("文件在预览后发生变化，请重新预览")
    if current["fingerprint"] == "missing":
        return {"status": "missing", "actual_reclaimed_bytes": 0}
    physical = filesystem_path(path)
    def retry_readonly(function, raw_path, error):
        candidate = Path(raw_path)
        if candidate != physical and not candidate.is_relative_to(physical):
            raise error
        info = candidate.lstat()
        if (os.name != "nt" or not isinstance(error, PermissionError)
                or not stat.S_ISREG(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
                or not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_READONLY):
            raise error
        candidate.chmod(info.st_mode | stat.S_IWRITE)
        function(raw_path)

    if physical.is_dir():
        # Python 3.11 uses exc_info tuples; keep the same checked retry policy.
        shutil.rmtree(physical, onerror=lambda function, raw_path, info:
                      retry_readonly(function, raw_path, info[1]))
    else:
        try:
            physical.unlink()
        except PermissionError as exc:
            retry_readonly(os.unlink, physical, exc)
    return {"status": "deleted", "actual_reclaimed_bytes": current["size_bytes"]}


def checkpoint_group(path: str | Path, root: str | Path) -> tuple[dict, list[dict]]:
    """Inspect only checkpoint-* children, retaining Trainer logs and configuration."""
    target = checked_path(path, root)
    if target.name != "trainer":
        raise ValueError("恢复点分组必须位于 trainer 目录")
    if not target.exists():
        return {"size_bytes": 0, "file_count": 0, "fingerprint": "missing"}, []
    children = [
        {"path": str(child), "root": str(root), **inspect_target(child, root)}
        for child in sorted(target.glob("checkpoint-*"))
    ]
    digest = hashlib.sha256()
    for child in children:
        digest.update(f"{child['path']}|{child['fingerprint']}\n".encode())
    return {
        "size_bytes": sum(child["size_bytes"] for child in children),
        "file_count": sum(child["file_count"] for child in children),
        "fingerprint": digest.hexdigest(),
    }, children


def remove_checkpoint_group(item: dict) -> dict:
    stats, children = checkpoint_group(item["path"], item["root"])
    if any(stats[key] != item[key] for key in stats):
        raise ValueError("恢复点在预览后发生变化，请重新预览")
    for child in children:
        remove_target(child)
    return {
        "status": "deleted" if children else "missing",
        "actual_reclaimed_bytes": stats["size_bytes"],
    }
