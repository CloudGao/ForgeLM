from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

SUPPORTED_SOURCE_TYPES = {"http", "https", "huggingface", "git", "local"}


def _result(
    *,
    source_type: str,
    source_uri: str,
    accessible: bool,
    files: list[dict[str, Any]] | None = None,
    total_size_bytes: int | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    free = shutil.disk_usage(storage_root or Path.cwd()).free
    size_known = total_size_bytes is not None
    enough_space = not size_known or total_size_bytes <= free
    messages = list(warnings or [])
    if not enough_space:
        messages.append("制品库可用空间小于来源的已知大小")
    return {
        "source_type": source_type,
        "source_uri": source_uri,
        "accessible": accessible,
        "files": files or [],
        "file_count": len(files or []),
        "total_size_bytes": total_size_bytes,
        "disk_free_bytes": free,
        "disk_space_sufficient": enough_space,
        "warnings": messages,
        "error": error,
    }


def preflight_source(
    source_type: str,
    source_uri: str,
    source_config: dict[str, Any],
    *,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    source_type = source_type.lower()
    source_uri = source_uri.strip()
    if source_type not in SUPPORTED_SOURCE_TYPES:
        return _result(
            source_type=source_type,
            source_uri=source_uri,
            accessible=False,
            error="当前内置导入器不支持该来源类型",
            storage_root=storage_root,
        )
    try:
        if source_type == "local":
            return _preflight_local(source_type, source_uri, storage_root)
        if source_type in {"http", "https"}:
            return _preflight_http(source_type, source_uri, storage_root)
        if source_type == "huggingface":
            return _preflight_huggingface(source_type, source_uri, source_config, storage_root)
        return _preflight_git(source_type, source_uri, storage_root)
    except (OSError, ValueError, httpx.HTTPError, subprocess.SubprocessError) as exc:
        return _result(
            source_type=source_type,
            source_uri=source_uri,
            accessible=False,
            error=str(exc),
            storage_root=storage_root,
        )


def _preflight_local(
    source_type: str, source_uri: str, storage_root: Path | None
) -> dict[str, Any]:
    from llm_platform_core.storage_layout import resolve_relocated_path
    path = Path(resolve_relocated_path(source_uri)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"本地路径不存在：{path}")
    paths = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    files = [
        {
            "path": item.name if path.is_file() else item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
        }
        for item in paths[:200]
    ]
    total = sum(item.stat().st_size for item in paths)
    warnings = ["文件列表只展示前 200 项"] if len(paths) > 200 else []
    return _result(
        source_type=source_type,
        source_uri=str(path),
        accessible=True,
        files=files,
        total_size_bytes=total,
        warnings=warnings,
        storage_root=storage_root,
    )


def _preflight_http(source_type: str, source_uri: str, storage_root: Path | None) -> dict[str, Any]:
    parsed = urlparse(source_uri)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("地址必须以 http:// 或 https:// 开头")
    with httpx.Client(follow_redirects=True, timeout=10) as client:
        response = client.head(source_uri)
        if response.status_code in {403, 405}:
            with client.stream("GET", source_uri, headers={"Range": "bytes=0-0"}) as streamed:
                streamed.raise_for_status()
                headers = dict(streamed.headers)
                response_url = streamed.url
        else:
            response.raise_for_status()
            headers = dict(response.headers)
            response_url = response.url
    size_header = headers.get("content-range", "").partition("/")[2] or headers.get(
        "content-length"
    )
    size = int(size_header) if size_header and size_header.isdigit() else None
    filename = Path(unquote(parsed.path)).name or "download.bin"
    warnings = [] if size is not None else ["服务器未提供文件大小，下载前无法精确检查空间"]
    return _result(
        source_type=source_type,
        source_uri=str(response_url),
        accessible=True,
        files=[{"path": filename, "size_bytes": size}],
        total_size_bytes=size,
        warnings=warnings,
        storage_root=storage_root,
    )


def _preflight_huggingface(
    source_type: str,
    repo_id: str,
    source_config: dict[str, Any],
    storage_root: Path | None,
) -> dict[str, Any]:
    del source_type, repo_id, source_config, storage_root
    raise ValueError("Hugging Face 来源必须使用 Worker 仓库解析和下载计划接口")


def _preflight_git(source_type: str, source_uri: str, storage_root: Path | None) -> dict[str, Any]:
    process = subprocess.run(
        ["git", "ls-remote", source_uri, "HEAD"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    commit = process.stdout.split()[0] if process.stdout.split() else None
    return _result(
        source_type=source_type,
        source_uri=source_uri,
        accessible=True,
        files=[{"path": "repository", "commit": commit}],
        total_size_bytes=None,
        warnings=["Git 远端不提供仓库下载大小，空间将在 Worker 执行时检查"],
        storage_root=storage_root,
    )
