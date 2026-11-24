"""Independent atomic checkpoint copies, using filesystem COW when available."""
import errno
import os
import shutil
import tempfile
from pathlib import Path


def copy_checkpoint(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError("Checkpoint source must be a regular file")
    if destination.is_symlink():
        raise ValueError("Checkpoint destination cannot be a symlink")
    mode = "copy"
    temporary = None
    try:
        with source.open("rb") as src, tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".checkpoint-copy-", delete=False
        ) as dst:
            temporary = Path(dst.name)
            if os.name == "posix":
                import fcntl

                try:
                    fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
                    mode = "reflink"
                except OSError as exc:
                    if exc.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.ENOSYS}:
                        raise
            if mode == "copy":
                shutil.copyfileobj(src, dst, 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        temporary.replace(destination)
        return mode
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
