from __future__ import annotations

import os
from pathlib import Path


def advise_sequential(file_descriptor: int) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_SEQUENTIAL"):
        return
    try:
        os.posix_fadvise(file_descriptor, 0, 0, os.POSIX_FADV_SEQUENTIAL)
    except OSError:
        pass


def release_file_cache_descriptor(
    file_descriptor: int, offset: int = 0, length: int = 0
) -> bool:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return False
    try:
        os.posix_fadvise(
            file_descriptor,
            max(int(offset), 0),
            max(int(length), 0),
            os.POSIX_FADV_DONTNEED,
        )
    except OSError:
        return False
    return True


def release_file_cache(path: Path) -> bool:
    try:
        file_descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        return release_file_cache_descriptor(file_descriptor)
    finally:
        os.close(file_descriptor)
