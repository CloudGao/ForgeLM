"""Resolve frozen processing inputs directly; never build a linked/copy directory."""
import hashlib
from pathlib import PurePosixPath

from llm_platform_core.storage_cleanup import checked_path


def selected_processing_files(root, selection, get_desired_state=None):
    from .adapters.data_import import AnalysisInterrupted

    if selection.get("schema") != "forgelm-processing-file-selection-v1" or not selection.get("files"):
        raise ValueError("Invalid frozen processing file selection")
    files, seen = [], set()
    for item in selection["files"]:
        if get_desired_state and get_desired_state() != "running":
            raise AnalysisInterrupted(get_desired_state())
        name = item["path"]
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name or name in seen:
            raise ValueError("Unsafe or duplicate processing selection path")
        seen.add(name)
        path = root / name
        checked_path(path, root)
        before = path.stat()
        if not path.is_file() or before.st_size != item["size_bytes"]:
            raise ValueError("Processing source size changed")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                if get_desired_state and get_desired_state() != "running":
                    raise AnalysisInterrupted(get_desired_state())
                digest.update(block)
        after = path.stat()
        if digest.hexdigest() != item["sha256"] or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Processing source content changed")
        files.append(str(path))
    return files
