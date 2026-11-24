"""Stage repository-pinned GPU wheels through the control plane's network."""

from __future__ import annotations

import hashlib
import re
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from .ssh_credentials import credential_directory

_DOWNLOAD_LOCK = threading.Lock()


def stage_runtime_wheels(requirements: Path, destination: Path, remote_directory: str, log) -> list[Path]:
    from llm_platform_core.storage_layout import platform_state_root

    from ..config import get_settings
    cache = (platform_state_root(get_settings().storage_root) / "cache" / "runtime-wheels"
             if get_settings().storage_root else credential_directory().parent / "runtime-wheel-cache")
    uploads = []
    lines = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([\w-]+) @ (https://\S+)", line.strip())
        if not match:
            lines.append(line)
            continue
        package, url = match.groups()
        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename):
            raise ValueError("Invalid pinned runtime wheel filename")
        wheel = cache / hashlib.sha256(url.encode()).hexdigest() / filename
        with _DOWNLOAD_LOCK:
            if not wheel.is_file():
                wheel.parent.mkdir(parents=True, exist_ok=True)
                partial = wheel.with_suffix(".partial")
                size = 0
                deadline = time.monotonic() + 900
                log(f"控制面下载固定依赖：{package}")
                try:
                    with httpx.stream("GET", url, follow_redirects=True, timeout=30) as response:
                        response.raise_for_status()
                        with partial.open("wb") as output:
                            for chunk in response.iter_bytes(1024 * 1024):
                                size += len(chunk)
                                if size > 2 * 1024**3 or time.monotonic() > deadline:
                                    raise ValueError("Runtime wheel download exceeded limits")
                                output.write(chunk)
                                if size % (32 * 1024**2) == 0:
                                    log(f"{package} 已下载 {size // 1024**2} MiB")
                    with zipfile.ZipFile(partial) as archive:
                        if archive.testzip() is not None:
                            raise ValueError("Runtime wheel archive failed integrity validation")
                    partial.replace(wheel)
                finally:
                    partial.unlink(missing_ok=True)
            log(f"固定依赖已缓存，准备 SSH 上传：{package}")
        uploads.append(wheel)
        lines.append(f"{package} @ file://{remote_directory}/{filename}")
    staged = destination / requirements.name
    staged.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [staged, *uploads]
