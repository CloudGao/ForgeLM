"""Materialize model/dataset archive artifacts without extracting links."""
from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

from .package import safe_path


def unpack(source: Path, destination: Path, maximum_bytes: int = 64 * 1024**3):
    destination.mkdir(parents=True, exist_ok=False)
    seen, total = set(), 0
    def target(name, size):
        nonlocal total
        while name.startswith("./"):
            name = name[2:]
        if name in {"", "."}:
            return destination
        name = safe_path(name.rstrip("/"))
        if name.casefold() in seen:
            raise ValueError("Duplicate archive path")
        seen.add(name.casefold())
        total += size
        if total > maximum_bytes or len(seen) > 100000:
            raise ValueError("Artifact archive exceeds extraction limits")
        path = destination / name
        path.resolve().relative_to(destination.resolve())
        return path
    try:
        if zipfile.is_zipfile(source):
            import stat
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    if stat.S_ISLNK(member.external_attr >> 16):
                        raise ValueError("Artifact links are forbidden")
                    path = target(member.filename, member.file_size)
                    if member.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as incoming, path.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
        else:
            with tarfile.open(source) as archive:
                for member in archive:
                    if not member.isfile() and not member.isdir():
                        raise ValueError("Artifact links/devices are forbidden")
                    path = target(member.name, member.size)
                    if member.isdir():
                        path.mkdir(parents=True, exist_ok=True)
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as incoming, path.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
        return destination
    except BaseException:
        # Only the fresh directory created above is removed.
        shutil.rmtree(destination)
        raise
