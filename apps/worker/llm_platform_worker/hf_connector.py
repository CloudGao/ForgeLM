from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import warnings
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def huggingface_connector_available() -> bool:
    try:
        from huggingface_hub import HfApi, snapshot_download

        return (
            hasattr(HfApi, "resolve_revision")
            and "dry_run" in inspect.signature(snapshot_download).parameters
        )
    except (ImportError, ModuleNotFoundError, TypeError, ValueError):
        return False


def endpoint_from_config(source_config: dict[str, Any]) -> str:
    endpoint = (
        str(
            source_config.get("hf_endpoint") or os.getenv("HF_ENDPOINT") or "https://huggingface.co"
        )
        .strip()
        .rstrip("/")
    )
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Hugging Face endpoint must be a complete HTTP(S) URL")
    return endpoint


def repo_type_from_config(source_config: dict[str, Any]) -> str:
    repo_type = str(source_config.get("repo_type") or "dataset").strip().lower()
    if repo_type not in {"dataset", "model"}:
        raise ValueError("Hugging Face repo_type must be dataset or model")
    return repo_type


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None


def _selected_paths(values: Any) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("selected_paths must be a list of repository file paths")
    selected: list[str] = []
    for raw_value in values:
        value = str(raw_value or "").strip()
        path = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or value.startswith("/")
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"Invalid Hugging Face repository path: {value!r}")
        if value not in selected:
            selected.append(value)
    if not selected:
        raise ValueError("At least one Hugging Face file must be selected")
    return selected


def _is_file_metadata_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ == "FileMetadataError":
            return True
        current = current.__cause__ or current.__context__
    return False


def _mirror_can_retry(exc: Exception) -> bool:
    import httpx
    return _is_file_metadata_error(exc) or isinstance(exc, (httpx.HTTPError, TimeoutError, ConnectionError))


def scan_repository(repo_id: str, source_config: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi, RepoFile

    if not repo_id or "/" not in repo_id:
        raise ValueError("Hugging Face source must be a repository ID such as org/name")
    endpoint = endpoint_from_config(source_config)
    repo_type = repo_type_from_config(source_config)
    requested_revision = str(source_config.get("revision") or "main").strip() or "main"
    def resolve(active_endpoint):
        client = HfApi(endpoint=active_endpoint, token=_token())
        return client, client.resolve_revision(repo_id, repo_type=repo_type, revision=requested_revision)

    try:
        api, resolved = resolve(endpoint)
    except Exception as exc:
        if endpoint != HF_MIRROR_ENDPOINT or not _mirror_can_retry(exc):
            raise
        endpoint = OFFICIAL_ENDPOINT
        warnings.warn("HF-Mirror revision lookup failed; using official Hugging Face", stacklevel=2)
        api, resolved = resolve(endpoint)
    commit_hash = str(resolved.resolved)
    files = [
        {"path": item.path, "size_bytes": int(item.size)}
        for item in api.list_repo_tree(
            repo_id,
            repo_type=repo_type,
            revision=commit_hash,
            recursive=True,
        )
        if isinstance(item, RepoFile)
    ]
    files.sort(key=lambda item: item["path"])
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "requested_revision": requested_revision,
        "commit_hash": commit_hash,
        "endpoint": endpoint,
        "files": files,
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "config": str(source_config.get("config") or "").strip() or None,
        "split": str(source_config.get("split") or "").strip() or None,
    }


def plan_snapshot(
    repo_id: str,
    source_config: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    commit_hash = str(source_config.get("commit_hash") or "").strip()
    if not commit_hash:
        raise ValueError("A resolved Hugging Face commit hash is required")
    selected_paths = _selected_paths(source_config.get("selected_paths"))
    endpoint = endpoint_from_config(source_config)
    repo_type = repo_type_from_config(source_config)
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    requested_endpoint = endpoint
    endpoint_warning = None

    def dry_run(active_endpoint: str):
        return snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=commit_hash,
            allow_patterns=selected_paths,
            local_dir=destination,
            endpoint=active_endpoint,
            token=_token(),
            dry_run=True,
        )

    try:
        raw_plan = dry_run(endpoint)
    except Exception as exc:
        if endpoint != HF_MIRROR_ENDPOINT or not _mirror_can_retry(exc):
            raise
        endpoint = OFFICIAL_ENDPOINT
        endpoint_warning = (
            "HF-Mirror could not provide the requested frozen file metadata; "
            "this frozen plan uses the official Hugging Face endpoint instead."
        )
        raw_plan = dry_run(endpoint)
    if not isinstance(raw_plan, list):
        raise RuntimeError("Hugging Face dry-run returned an unexpected result")
    planned_paths = {str(item.filename) for item in raw_plan}
    expected_paths = set(selected_paths)
    if planned_paths != expected_paths:
        missing = sorted(expected_paths - planned_paths)
        unexpected = sorted(planned_paths - expected_paths)
        raise ValueError(
            "Hugging Face dry-run did not resolve the exact selected file set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    commit_hashes = {str(item.commit_hash) for item in raw_plan}
    if commit_hashes != {commit_hash}:
        raise ValueError("Hugging Face dry-run resolved a different repository commit")
    selected_bytes = sum(int(item.file_size) for item in raw_plan)
    download_bytes = sum(int(item.file_size) for item in raw_plan if item.will_download)
    cached_bytes = selected_bytes - download_bytes
    disk_free_bytes = shutil.disk_usage(destination).free
    return {
        "repo_id": repo_id,
        "repo_type": repo_type,
        "requested_revision": str(source_config.get("requested_revision") or "main"),
        "commit_hash": commit_hash,
        "requested_endpoint": requested_endpoint,
        "endpoint": endpoint,
        "endpoint_warning": endpoint_warning,
        "selected_paths": selected_paths,
        "file_count": len(raw_plan),
        "selected_bytes": selected_bytes,
        "cached_bytes": cached_bytes,
        "download_bytes": download_bytes,
        "disk_free_bytes": disk_free_bytes,
        "disk_space_sufficient": download_bytes <= disk_free_bytes,
        "target_dir": str(destination),
        "files": [
            {
                "path": str(item.filename),
                "size_bytes": int(item.file_size),
                "is_cached": bool(item.is_cached),
                "will_download": bool(item.will_download),
            }
            for item in raw_plan
        ],
        "config": str(source_config.get("config") or "").strip() or None,
        "split": str(source_config.get("split") or "").strip() or None,
    }


def download_snapshot(plan: dict[str, Any]) -> str:
    # The mirror can expose Xet metadata whose CAS service does not accept the
    # mirror-issued token. Use Hub's supported HTTP path for mirror requests,
    # including plans whose metadata fell back to the official endpoint.
    # Set this before importing huggingface_hub, which reads it at import time.
    if plan.get("requested_endpoint", plan.get("endpoint")) == HF_MIRROR_ENDPOINT:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    selected_paths = _selected_paths(plan.get("selected_paths"))
    destination = Path(str(plan.get("target_dir") or "")).resolve()
    if not str(plan.get("commit_hash") or "").strip():
        raise ValueError("A resolved Hugging Face commit hash is required")
    download_args = dict(
        repo_id=str(plan["repo_id"]),
        repo_type=repo_type_from_config(plan),
        revision=str(plan["commit_hash"]),
        allow_patterns=selected_paths,
        local_dir=destination,
        endpoint=str(plan["endpoint"]),
        token=_token(),
        dry_run=False,
    )
    try:
        result = snapshot_download(**download_args)
    except Exception as exc:
        if download_args["endpoint"] != HF_MIRROR_ENDPOINT or not _mirror_can_retry(exc):
            raise
        warnings.warn("HF-Mirror download failed; retrying official endpoint with the same commit and files", stacklevel=2)
        result = snapshot_download(**{**download_args, "endpoint": OFFICIAL_ENDPOINT})
    return str(result)


def execute_metadata_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = str(request.get("operation") or "").strip()
    repo_id = str(request.get("repo_id") or "").strip()
    source_config = request.get("source_config") or {}
    if not isinstance(source_config, dict):
        raise ValueError("Hugging Face source_config must be an object")
    if operation == "scan":
        return scan_repository(repo_id, source_config)
    if operation == "plan":
        destination = Path(str(request.get("destination") or "")).resolve()
        return plan_snapshot(repo_id, source_config, destination)
    raise ValueError("Unsupported Hugging Face metadata operation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a frozen Hugging Face operation")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--plan-file")
    operation.add_argument("--metadata-request-file")
    parser.add_argument("--result-file")
    args = parser.parse_args()
    if args.metadata_request_file:
        if not args.result_file:
            parser.error("--result-file is required with --metadata-request-file")
        request = json.loads(Path(args.metadata_request_file).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("Hugging Face metadata request file must contain an object")
        result = execute_metadata_request(request)
        result_path = Path(args.result_file).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(f"{result_path.suffix}.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(result_path)
        return
    plan = json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Hugging Face plan file must contain an object")
    result = download_snapshot(plan)
    print(json.dumps({"snapshot_path": result, "commit_hash": plan["commit_hash"]}))


if __name__ == "__main__":
    main()
