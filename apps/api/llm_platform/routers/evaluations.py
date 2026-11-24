from __future__ import annotations

import csv
import io
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from llm_platform_core.inference import (
    MICROLM_INFERENCE_ADAPTER,
    resolve_inference_profile,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_session
from ..models import (
    Artifact,
    Dataset,
    DatasetVersion,
    EvalDatasetVersion,
    EvalSuiteVersion,
    EvaluationRun,
    ModelVersion,
    ModelWeightVersion,
    Project,
    ProjectDataBinding,
    Task,
)
from ..schemas import (
    BenchmarkSuiteCreate,
    EvalDatasetCreate,
    EvalDatasetFromDataVersion,
    EvalSuiteCreate,
    EvaluationStart,
)
from ..services.artifacts import artifact_storage
from ..services.command_data_inputs import freeze_command_code
from ..services.benchmarks import (
    benchmark_by_id,
    canonical_hash,
    freeze_benchmark,
    list_benchmark_suite_presets,
    list_benchmarks,
)
from ..services.global_data import project_can_manage_asset, project_has_asset_access
from ..services.tasks import create_task
from .common import add_audit, get_or_404, require_executable

router = APIRouter(tags=["evaluations"])

MAX_EVALUATION_UPLOAD_BYTES = 256 * 1024 * 1024
EVALUATION_UPLOAD_SUFFIXES = {".jsonl", ".ndjson", ".csv", ".tsv", ".parquet"}
EVALUATION_EXECUTION_ONLY_INFERENCE_KEYS = {"batch_size", "device"}
MICROLM_EVALUATION_BACKEND = "microlm-native"
MODEL_ARTIFACT_PLACEHOLDER = "__FORGELM_MODEL_ARTIFACT__"


def _normalize_evaluation_batch_size(value: Any) -> str | int:
    if value is None or (isinstance(value, str) and not value.strip()):
        return "auto"
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    if isinstance(value, bool):
        normalized = 0
    else:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = 0
    if normalized < 1 or normalized > 65536:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_evaluation_batch_size",
                "message": "Batch size 必须为 auto 或 1 到 65536 的整数",
            },
        )
    return normalized


def _normalize_evaluation_sampling(inference_config: dict[str, Any]) -> dict[str, Any]:
    if inference_config.get("limit") not in (None, "", 0, "0"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "legacy_evaluation_limit_not_supported",
                "message": "标准评测不再接受题数 limit；请填写使用题目比例（%）和抽样 Seed",
            },
        )
    raw_percentage = inference_config.pop("sample_percentage", 100.0)
    raw_seed = inference_config.pop("sample_seed", 42)
    inference_config.pop("limit", None)
    if isinstance(raw_percentage, bool):
        percentage = math.nan
    else:
        try:
            percentage = float(raw_percentage)
        except (TypeError, ValueError):
            percentage = math.nan
    if not math.isfinite(percentage) or percentage <= 0 or percentage > 100:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_evaluation_sample_percentage",
                "message": "使用题目比例必须大于 0% 且不超过 100%",
            },
        )
    if isinstance(raw_seed, bool):
        numeric_seed = math.nan
        seed = -1
    else:
        try:
            numeric_seed = float(raw_seed)
            seed = int(numeric_seed)
        except (TypeError, ValueError):
            numeric_seed = math.nan
            seed = -1
    if (
        not math.isfinite(numeric_seed)
        or numeric_seed != seed
        or seed < 0
        or seed > 2_147_483_647
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_evaluation_sample_seed",
                "message": "抽样 Seed 必须为 0 到 2147483647 的整数",
            },
        )
    return {
        "schema_version": "forgelm.evaluation-sampling.v1",
        "method": "stable_hash_v1",
        "percentage": percentage,
        "seed": seed,
    }


def _parse_json_form(value: str, label: str, expected_type: type) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_evaluation_metadata", "message": f"{label}不是有效 JSON"},
        ) from exc
    if not isinstance(parsed, expected_type):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_evaluation_metadata", "message": f"{label}类型不正确"},
        )
    return parsed


def _validate_schema_config(schema_config: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        str(key): str(value).strip()
        for key, value in schema_config.items()
        if value is not None and str(value).strip()
    }
    missing = [key for key in ("prompt_field", "answer_field") if not normalized.get(key)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "missing_evaluation_field_mapping",
                "message": "自定义评测集必须配置 Prompt 字段和答案字段",
                "fields": missing,
            },
        )
    return normalized


def _mapping_value(item: dict[str, Any], path: str) -> Any:
    current: Any = item
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def _has_mapping_value(item: dict[str, Any], path: str) -> bool:
    value = _mapping_value(item, path)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def _preview_record(item: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for key, value in list(item.items())[:24]:
        if isinstance(value, str):
            preview[str(key)] = value[:1000]
        elif isinstance(value, (int, float, bool)) or value is None:
            preview[str(key)] = value
        else:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
            preview[str(key)] = rendered[:1000]
    return preview


def _inspect_uploaded_evaluation(
    filename: str,
    payload: bytes,
    schema_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in EVALUATION_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_evaluation_format",
                "message": "评测集仅支持 JSONL、NDJSON、CSV、TSV 或 Parquet",
            },
        )
    required_paths = [
        str((schema_config or {}).get(key) or "").strip()
        for key in ("prompt_field", "answer_field")
    ]
    required_paths = [value for value in required_paths if value]
    fields: set[str] = set()
    preview: list[dict[str, Any]] = []
    count = 0
    valid_count = 0

    def inspect_item(item: dict[str, Any]) -> None:
        nonlocal count, valid_count
        count += 1
        fields.update(str(key) for key in item)
        if len(preview) < 5:
            preview.append(_preview_record(item))
        if all(_has_mapping_value(item, path) for path in required_paths):
            valid_count += 1

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(io.BytesIO(payload))
            fields.update(parquet.schema_arrow.names)
            for batch in parquet.iter_batches(batch_size=8192):
                for item in batch.to_pylist():
                    if isinstance(item, dict):
                        inspect_item(item)
            file_format = "parquet"
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_evaluation_parquet",
                    "message": f"无法读取 Parquet 评测集：{exc}",
                },
            ) from exc
    else:
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "evaluation_not_utf8", "message": "评测集必须使用 UTF-8 编码"},
            ) from exc
        if suffix in {".jsonl", ".ndjson"}:
            for line_number, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "invalid_evaluation_jsonl",
                            "message": f"评测集第 {line_number} 行不是有效 JSON",
                        },
                    ) from exc
                if not isinstance(item, dict):
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "invalid_evaluation_sample",
                            "message": f"评测集第 {line_number} 行必须是 JSON 对象",
                        },
                    )
                inspect_item(item)
            file_format = "jsonl"
        else:
            delimiter = "\t" if suffix == ".tsv" else ","
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            if not reader.fieldnames or not any(str(name or "").strip() for name in reader.fieldnames):
                raise HTTPException(
                    status_code=422,
                    detail={"code": "missing_evaluation_header", "message": "CSV/TSV 缺少表头"},
                )
            fields.update(str(name).strip() for name in reader.fieldnames if str(name or "").strip())
            for row in reader:
                if any(str(value or "").strip() for value in row.values()):
                    inspect_item(dict(row))
            file_format = "tsv" if suffix == ".tsv" else "csv"
    if count == 0:
        raise HTTPException(
            status_code=422,
            detail={"code": "empty_evaluation_dataset", "message": "评测集没有非空样本"},
        )
    missing_paths = [path for path in required_paths if not any(path == field or path.startswith(f"{field}.") for field in fields)]
    return {
        "filename": Path(filename).name,
        "format": file_format,
        "size_bytes": len(payload),
        "sample_count": count,
        "valid_sample_count": valid_count if required_paths else count,
        "invalid_sample_count": count - valid_count if required_paths else 0,
        "fields": sorted(fields),
        "missing_mapping_fields": missing_paths,
        "preview": preview,
    }


def _latest_version(session: Session, model: type, project_id: str, name: str) -> int:
    return int(
        session.scalar(
            select(func.coalesce(func.max(model.version), 0)).where(
                model.project_id == project_id,
                model.name == name,
            )
        )
        or 0
    )


def _normalize_suite_entries(
    session: Session, project_id: str, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not entries:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_datasets", "message": "评测套件至少包含一个评测集"},
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict) or not raw_entry.get("dataset_version_id"):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_dataset_ref", "message": "套件条目缺少评测集版本 ID"},
            )
        dataset_id = str(raw_entry["dataset_version_id"])
        dataset = session.get(EvalDatasetVersion, dataset_id)
        if dataset is None or not project_has_asset_access(
            session, project_id, "eval_dataset_version", dataset_id
        ):
            raise HTTPException(
                status_code=404,
                detail={"code": "evaluation_dataset_not_found", "id": dataset_id},
            )
        if dataset_id in seen:
            raise HTTPException(
                status_code=422,
                detail={"code": "duplicate_dataset", "id": dataset_id},
            )
        seen.add(dataset_id)
        try:
            weight = float(raw_entry.get("weight", 1))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_weight", "id": dataset_id},
            ) from exc
        if weight <= 0:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_weight", "id": dataset_id},
            )
        entry = {"dataset_version_id": dataset_id, "weight": weight}
        if dataset.benchmark_config:
            entry.update(
                {
                    "benchmark_id": dataset.benchmark_config.get("id"),
                    "task_id": dataset.benchmark_config.get("task_id"),
                    "config_hash": dataset.compatibility_hash,
                }
            )
        normalized.append(entry)
    return normalized


def _suite_snapshot(
    session: Session, project_id: str, suites: list[EvalSuiteVersion]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, int]:
    suite_snapshots: list[dict[str, Any]] = []
    benchmarks_by_hash: dict[str, dict[str, Any]] = {}
    evaluation_inputs_by_id: dict[str, dict[str, Any]] = {}
    dataset_count = 0
    benchmark_count = 0
    for suite in suites:
        entries = _normalize_suite_entries(session, project_id, suite.datasets or [])
        frozen_entries: list[dict[str, Any]] = []
        for entry in entries:
            dataset_count += 1
            dataset = session.get(EvalDatasetVersion, entry["dataset_version_id"])
            assert dataset is not None
            frozen_entry = {
                **entry,
                "name": dataset.name,
                "version": dataset.version,
                "source_type": dataset.source_type,
                "task_type": dataset.task_type,
                "dimensions": dataset.dimensions,
                "license": dataset.license,
                "sample_count": dataset.sample_count,
                "contamination_tags": deepcopy(dataset.contamination_tags or []),
                "source_config": deepcopy(dataset.source_config or {}),
                "schema_config": deepcopy(dataset.schema_config or {}),
                "scoring_config": deepcopy(dataset.scoring_config or {}),
                "benchmark_config": deepcopy(dataset.benchmark_config or {}),
            }
            artifact = session.get(Artifact, dataset.artifact_id) if dataset.artifact_id else None
            if artifact is not None:
                frozen_entry["artifact"] = {
                    "id": artifact.id,
                    "kind": artifact.kind,
                    "content_hash": artifact.content_hash,
                    "size_bytes": artifact.size_bytes,
                }
            frozen_entries.append(frozen_entry)
            if dataset.benchmark_config:
                benchmark_count += 1
                key = str(dataset.compatibility_hash or dataset.id)
                current = benchmarks_by_hash.get(key)
                if current:
                    current["weight"] += entry["weight"]
                else:
                    benchmarks_by_hash[key] = {
                        **deepcopy(dataset.benchmark_config),
                        "dataset_version_id": dataset.id,
                        "dataset_version": dataset.version,
                        "weight": entry["weight"],
                    }
            else:
                current = evaluation_inputs_by_id.get(dataset.id)
                if current:
                    current["weight"] += entry["weight"]
                else:
                    evaluation_inputs_by_id[dataset.id] = {
                        "eval_dataset_version_id": dataset.id,
                        "name": dataset.name,
                        "version": dataset.version,
                        "source_type": dataset.source_type,
                        "task_type": dataset.task_type,
                        "license": dataset.license,
                        "dimensions": deepcopy(dataset.dimensions or []),
                        "sample_count": dataset.sample_count,
                        "contamination_tags": deepcopy(dataset.contamination_tags or []),
                        "source_config": deepcopy(dataset.source_config or {}),
                        "schema_config": deepcopy(dataset.schema_config or {}),
                        "scoring_config": deepcopy(dataset.scoring_config or {}),
                        "artifact_id": artifact.id if artifact is not None else None,
                        "artifact_hash": artifact.content_hash if artifact is not None else None,
                        "weight": entry["weight"],
                    }
        suite_snapshots.append(
            {
                "id": suite.id,
                "name": suite.name,
                "version": suite.version,
                "datasets": frozen_entries,
                "scoring_config": deepcopy(suite.scoring_config or {}),
                "release_gate": deepcopy(suite.release_gate or {}),
            }
        )
    return (
        suite_snapshots,
        list(benchmarks_by_hash.values()),
        list(evaluation_inputs_by_id.values()),
        dataset_count,
        benchmark_count,
    )


def _route_evaluation_resource_request(
    session: Session,
    project_id: str,
    resource_request: dict[str, Any],
    evaluation_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    request = deepcopy(resource_request)
    eligible_worker_ids: set[str] | None = None
    for item in evaluation_inputs:
        artifact_id = str(item.get("artifact_id") or "")
        if not artifact_id:
            continue
        artifact = session.get(Artifact, artifact_id)
        expected_kind = str(item.get("artifact_kind") or "dataset_snapshot")
        if artifact is None or artifact.kind != expected_kind:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "evaluation_dataset_snapshot_missing",
                    "message": f"评测输入 {item.get('name') or artifact_id} 的制品不存在",
                },
            )
        access = artifact_storage.inference_access(session, artifact, project_id)
        if access["type"] == "unavailable":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "evaluation_dataset_snapshot_unavailable",
                    "message": f"评测输入 {item.get('name') or artifact_id} 没有可用 Worker 位置",
                },
            )
        if access["type"] == "external" and not access["globally_accessible"]:
            workers = {str(value) for value in access["worker_ids"]}
            eligible_worker_ids = (
                workers if eligible_worker_ids is None else eligible_worker_ids & workers
            )
    if eligible_worker_ids is None:
        return request
    selector = dict(request.get("worker_selector") or {})
    if selector.get("worker_id"):
        eligible_worker_ids &= {str(selector["worker_id"])}
    selected_worker_ids = {str(value) for value in selector.get("worker_ids") or []}
    if selected_worker_ids:
        eligible_worker_ids &= selected_worker_ids
    if not eligible_worker_ids:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "evaluation_worker_location_mismatch",
                "message": "没有一个 Worker 能同时访问全部评测数据与模型权重",
            },
        )
    if len(eligible_worker_ids) == 1:
        selector.pop("worker_ids", None)
        selector["worker_id"] = next(iter(eligible_worker_ids))
    else:
        selector.pop("worker_id", None)
        selector["worker_ids"] = sorted(eligible_worker_ids)
    request["worker_selector"] = selector
    return request


def _normalized_model_runtimes(
    inference_config: dict[str, Any], model_ids: list[str]
) -> dict[str, dict[str, Any]]:
    runtimes = inference_config.get("models") or {}
    if not isinstance(runtimes, dict):
        runtimes = {}
    if len(model_ids) == 1 and model_ids[0] not in runtimes and inference_config.get("model_args"):
        runtimes = {
            **runtimes,
            model_ids[0]: {
                "backend": inference_config.get("backend", "hf"),
                "model_args": inference_config["model_args"],
            },
        }
    normalized: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        runtime = runtimes.get(model_id)
        if not isinstance(runtime, dict):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_model_runtime",
                    "message": f"模型 {model_id} 缺少 lm-eval 运行参数",
                },
            )
        backend = str(runtime.get("backend") or runtime.get("model") or "hf").strip()
        if backend not in {
            "hf",
            "vllm",
            "local-completions",
            "local-chat-completions",
            MICROLM_EVALUATION_BACKEND,
        }:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unsupported_model_backend",
                    "message": f"模型 {model_id} 使用了不支持的 lm-eval backend",
                },
            )
        model_args = str(runtime.get("model_args") or "").strip()
        if not model_args:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "missing_model_args",
                    "message": f"模型 {model_id} 必须提供 lm-eval model_args",
                },
            )
        if "trust_remote_code" in model_args.lower() or runtime.get("trust_remote_code"):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "remote_code_not_supported",
                    "message": "标准 Benchmark 适配器不会自动执行远程代码",
                },
            )
        normalized_runtime: dict[str, Any] = {"backend": backend, "model_args": model_args}
        auth_env_var = str(runtime.get("auth_env_var") or "").strip()
        if auth_env_var:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", auth_env_var):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_endpoint_auth_reference",
                        "message": f"模型 {model_id} 的认证环境变量名无效",
                    },
                )
            normalized_runtime["auth_env_var"] = auth_env_var
        capabilities = runtime.get("capabilities")
        if isinstance(capabilities, list):
            normalized_runtime["capabilities"] = [str(item) for item in capabilities]
        endpoint_mode = str(runtime.get("endpoint_mode") or "").strip()
        if endpoint_mode:
            normalized_runtime["endpoint_mode"] = endpoint_mode
        inference_adapter_id = str(runtime.get("inference_adapter_id") or "").strip()
        if inference_adapter_id:
            normalized_runtime["inference_adapter_id"] = inference_adapter_id
        normalized[model_id] = normalized_runtime
    return normalized


def _model_evaluation_runtime(
    session: Session, model: ModelVersion
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Resolve Benchmark execution from the same frozen profile as Playground."""

    metadata = model.metadata_json or {}
    registered_runtime = metadata.get("lm_eval")
    if isinstance(registered_runtime, dict) and (
        metadata.get("openai_endpoint") or model.model_weight_version_id
    ):
        return deepcopy(registered_runtime), None, {}

    profile = resolve_inference_profile(metadata)
    adapter_id = str(profile.get("adapter_id") or "").strip()
    if adapter_id == MICROLM_INFERENCE_ADAPTER:
        if not model.artifact_id:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "model_artifact_unavailable",
                    "message": f"模型 {model.name} 的 MicroLM 原生制品不存在",
                },
            )
        artifact = session.get(Artifact, model.artifact_id)
        if artifact is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "model_artifact_unavailable",
                    "message": f"模型 {model.name} 的 MicroLM 原生制品记录不存在",
                },
            )
        runtime = {
            "backend": MICROLM_EVALUATION_BACKEND,
            "model_args": f"pretrained={MODEL_ARTIFACT_PLACEHOLDER}",
            "capabilities": ["generate", "loglikelihood", "loglikelihood_rolling"],
            "inference_adapter_id": MICROLM_INFERENCE_ADAPTER,
        }
        model_input = {
            "name": model.name,
            "model_version_id": model.id,
            "artifact_id": artifact.id,
            "artifact_kind": artifact.kind,
            "artifact_hash": artifact.content_hash,
            "model": {
                "id": model.id,
                "project_id": model.project_id,
                "name": model.name,
                "version": model.version,
                "kind": model.kind,
                "artifact_id": model.artifact_id,
                "tokenizer_version_id": model.tokenizer_version_id,
                "metadata": deepcopy(metadata),
                "runtime_source": None,
                "revision": profile.get("revision") or metadata.get("revision"),
            },
        }
        return runtime, model_input, deepcopy(profile.get("resource_request") or {})

    repository = (
        profile.get("pretrained")
        or metadata.get("pretrained")
        or metadata.get("huggingface_id")
        or metadata.get("hf_repo_id")
        or metadata.get("repository_id")
    )
    if repository and adapter_id in {"", "builtin.transformers-inference"}:
        revision = profile.get("revision") or metadata.get("revision")
        return (
            {
                "backend": "hf",
                "model_args": f"pretrained={repository}"
                + (f",revision={revision}" if revision else ""),
            },
            None,
            deepcopy(profile.get("resource_request") or {}),
        )
    if isinstance(registered_runtime, dict):
        return deepcopy(registered_runtime), None, {}
    if adapter_id:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "evaluation_bridge_unavailable",
                "message": f"模型 {model.name} 的推理适配器 {adapter_id} 尚未提供 Benchmark bridge",
            },
        )
    raise HTTPException(
        status_code=409,
        detail={
            "code": "model_inference_profile_missing",
            "message": f"模型 {model.name} 没有可用于 Playground 和评测的推理配置",
        },
    )


def _merge_profile_resource_requests(
    requested: dict[str, Any], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    merged = deepcopy(requested)
    selector = dict(merged.get("worker_selector") or {})
    capabilities = list(selector.get("capabilities") or [])
    for profile in profiles:
        for key in ("gpus", "cpus"):
            if key in profile:
                merged[key] = max(int(merged.get(key, 0) or 0), int(profile.get(key, 0) or 0))
        profile_selector = profile.get("worker_selector") or {}
        capabilities.extend(profile_selector.get("capabilities") or [])
    if capabilities:
        selector["capabilities"] = list(dict.fromkeys(str(value) for value in capabilities))
        merged["worker_selector"] = selector
    return merged


@router.get("/benchmarks/catalog")
def benchmark_catalog() -> list[dict[str, Any]]:
    return list_benchmarks()


@router.get("/benchmarks/suite-presets")
def benchmark_suite_presets() -> list[dict[str, Any]]:
    return list_benchmark_suite_presets()


@router.get("/projects/{project_id}/evaluation-datasets")
def list_eval_datasets(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    bound_version_ids = set(
        session.scalars(
            select(ProjectDataBinding.asset_version_id).where(
                ProjectDataBinding.project_id == project_id,
                ProjectDataBinding.asset_type == "eval_dataset_version",
            )
        ).all()
    )
    return [
        {
            **row.to_dict(),
            "binding_scope": "owned" if row.project_id == project_id else "bound",
        }
        for row in session.scalars(
            select(EvalDatasetVersion)
            .where(
                (EvalDatasetVersion.project_id == project_id)
                | (EvalDatasetVersion.id.in_(bound_version_ids))
            )
            .order_by(EvalDatasetVersion.created_at.desc())
        ).all()
    ]


@router.post("/projects/{project_id}/evaluation-datasets", status_code=201)
def create_eval_dataset(
    project_id: str, payload: EvalDatasetCreate, session: Session = Depends(get_session)
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    latest = _latest_version(session, EvalDatasetVersion, project_id, payload.name)
    row = EvalDatasetVersion(project_id=project_id, version=latest + 1, **payload.model_dump())
    session.add(row)
    session.commit()
    return row.to_dict()


@router.post("/projects/{project_id}/evaluation-datasets/upload", status_code=201)
def upload_eval_dataset(
    project_id: str,
    name: Annotated[str, Form(min_length=1, max_length=180)],
    task_type: Annotated[str, Form()],
    license: Annotated[str, Form()],
    dimensions: Annotated[str, Form()],
    contamination_tags: Annotated[str, Form()],
    schema_config: Annotated[str, Form()],
    scoring_config: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    clean_name = name.strip()
    if not clean_name:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_evaluation_name", "message": "评测集名称不能为空"},
        )
    filename = Path(file.filename or "evaluation.jsonl").name
    payload = file.file.read(MAX_EVALUATION_UPLOAD_BYTES + 1)
    if len(payload) > MAX_EVALUATION_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "evaluation_file_too_large", "message": "评测集不能超过 256 MiB"},
        )
    parsed_dimensions = [
        str(value).strip()
        for value in _parse_json_form(dimensions, "能力维度", list)
        if str(value).strip()
    ]
    parsed_contamination = [
        str(value).strip()
        for value in _parse_json_form(contamination_tags, "污染标签", list)
        if str(value).strip()
    ]
    parsed_schema = _validate_schema_config(
        _parse_json_form(schema_config, "字段映射", dict)
    )
    parsed_scoring = _parse_json_form(scoring_config, "评分配置", dict)
    inspection = _inspect_uploaded_evaluation(filename, payload, parsed_schema)
    if inspection["missing_mapping_fields"] or inspection["invalid_sample_count"]:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_evaluation_field_mapping",
                "message": "Prompt 或答案字段在部分评测样本中不存在或为空",
                "missing_fields": inspection["missing_mapping_fields"],
                "invalid_sample_count": inspection["invalid_sample_count"],
                "sample_count": inspection["sample_count"],
            },
        )
    file_format = str(inspection["format"])
    sample_count = int(inspection["sample_count"])
    artifact = artifact_storage.put_bytes(
        session,
        project_id=project_id,
        kind="dataset_snapshot",
        payload=payload,
        filename=filename,
        media_type=(
            "application/vnd.apache.parquet"
            if file_format == "parquet"
            else "application/x-ndjson"
            if file_format == "jsonl"
            else "text/tab-separated-values"
            if file_format == "tsv"
            else "text/csv"
        ),
        metadata={
            "asset_type": "evaluation_dataset",
            "name": clean_name,
            "filename": filename,
            "format": file_format,
            "sample_count": sample_count,
            "fields": inspection["fields"],
        },
    )
    source_config = {
        "kind": "upload",
        "filename": filename,
        "format": file_format,
        "content_hash": artifact.content_hash,
        "size_bytes": len(payload),
        "fields": inspection["fields"],
        "preview": inspection["preview"],
    }
    config_hash = canonical_hash(
        {
            "artifact_hash": artifact.content_hash,
            "task_type": task_type,
            "schema_config": parsed_schema,
            "scoring_config": parsed_scoring,
        }
    )
    row = EvalDatasetVersion(
        project_id=project_id,
        name=clean_name,
        version=_latest_version(session, EvalDatasetVersion, project_id, clean_name) + 1,
        source_type="upload",
        task_type=task_type,
        license=license.strip() or "unknown",
        dimensions=parsed_dimensions,
        sample_count=sample_count,
        contamination_tags=parsed_contamination,
        source_config=source_config,
        schema_config=parsed_schema,
        scoring_config=parsed_scoring,
        compatibility_hash=config_hash,
        artifact_id=artifact.id,
    )
    session.add(row)
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="evaluation_dataset.upload",
        subject_type="eval_dataset_version",
        subject_id=row.id,
        payload={"artifact_id": artifact.id, "content_hash": artifact.content_hash},
    )
    session.commit()
    return row.to_dict()


@router.post("/projects/{project_id}/evaluation-datasets/from-data-version", status_code=201)
def create_eval_dataset_from_data_version(
    project_id: str,
    payload: EvalDatasetFromDataVersion,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    data_version = session.get(DatasetVersion, payload.dataset_version_id)
    dataset = session.get(Dataset, data_version.dataset_id) if data_version is not None else None
    if data_version is None or dataset is None or not project_can_manage_asset(
        session, project_id, "dataset_version", payload.dataset_version_id
    ):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "dataset_version_not_found",
                "message": "数据中心版本不存在或属于其他项目",
            },
        )
    artifact = session.get(Artifact, data_version.artifact_id) if data_version.artifact_id else None
    if artifact is None or artifact.kind != "dataset_snapshot":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_snapshot_missing",
                "message": "所选数据版本还没有可用于评测的不可变数据快照",
            },
        )
    access = artifact_storage.inference_access(session, artifact, project_id)
    if access["type"] == "unavailable":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "dataset_snapshot_unavailable",
                "message": "所选数据快照没有托管副本或可用 Worker 位置",
            },
        )
    parsed_schema = _validate_schema_config(payload.schema_config)
    source_config = {
        "kind": "dataset_version",
        "dataset_id": dataset.id,
        "dataset_version_id": data_version.id,
        "dataset_version": data_version.version,
        "source_type": dataset.source_type,
        "source_uri": dataset.source_uri,
        "content_hash": artifact.content_hash,
        "source_snapshot": deepcopy(data_version.source_snapshot or {}),
    }
    config_hash = canonical_hash(
        {
            "artifact_hash": artifact.content_hash,
            "task_type": payload.task_type,
            "schema_config": parsed_schema,
            "scoring_config": payload.scoring_config,
        }
    )
    clean_name = payload.name.strip()
    if not clean_name:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_evaluation_name", "message": "评测集名称不能为空"},
        )
    row = EvalDatasetVersion(
        project_id=project_id,
        name=clean_name,
        version=_latest_version(session, EvalDatasetVersion, project_id, clean_name) + 1,
        source_type=dataset.source_type,
        task_type=payload.task_type,
        license=(payload.license or dataset.license or "unknown").strip(),
        dimensions=payload.dimensions,
        sample_count=data_version.document_count,
        contamination_tags=payload.contamination_tags,
        source_config=source_config,
        schema_config=parsed_schema,
        scoring_config=payload.scoring_config,
        compatibility_hash=config_hash,
        artifact_id=artifact.id,
    )
    session.add(row)
    session.flush()
    add_audit(
        session,
        project_id=project_id,
        action="evaluation_dataset.create_from_data_version",
        subject_type="eval_dataset_version",
        subject_id=row.id,
        payload={"dataset_version_id": data_version.id, "artifact_id": artifact.id},
    )
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return row.to_dict()


@router.get("/projects/{project_id}/evaluation-suites")
def list_eval_suites(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    return [
        row.to_dict()
        for row in session.scalars(
            select(EvalSuiteVersion)
            .where(EvalSuiteVersion.project_id == project_id)
            .order_by(EvalSuiteVersion.created_at.desc())
        ).all()
    ]


def _resolve_catalog_datasets(
    session: Session, project_id: str, benchmark_ids: list[str], overrides: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[EvalDatasetVersion]]:
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise HTTPException(
            status_code=422,
            detail={"code": "duplicate_benchmark", "message": "Benchmark 不能重复选择"},
        )
    dataset_entries: list[dict[str, Any]] = []
    imported: list[EvalDatasetVersion] = []
    for benchmark_id in benchmark_ids:
        if benchmark_by_id(benchmark_id) is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "benchmark_not_found", "id": benchmark_id},
            )
        try:
            benchmark = freeze_benchmark(benchmark_id, overrides.get(benchmark_id))
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_benchmark_override", "message": str(exc)},
            ) from exc
        dataset = session.scalar(
            select(EvalDatasetVersion).where(
                EvalDatasetVersion.project_id == project_id,
                EvalDatasetVersion.source_type == "benchmark",
                EvalDatasetVersion.compatibility_hash == benchmark["config_hash"],
            )
        )
        if dataset is None:
            name = str(benchmark["name"])
            dataset = EvalDatasetVersion(
                project_id=project_id,
                name=name,
                version=_latest_version(session, EvalDatasetVersion, project_id, name) + 1,
                source_type="benchmark",
                task_type=str(benchmark["task_type"]),
                license=str(benchmark["license"]),
                dimensions=list(benchmark["dimensions"]),
                sample_count=0,
                contamination_tags=["public-benchmark"],
                benchmark_config=benchmark,
                compatibility_hash=str(benchmark["config_hash"]),
            )
            session.add(dataset)
            session.flush()
        imported.append(dataset)
        dataset_entries.append(
            {
                "dataset_version_id": dataset.id,
                "weight": 1.0,
                "benchmark_id": benchmark_id,
                "task_id": benchmark["task_id"],
                "config_hash": benchmark["config_hash"],
            }
        )
    return dataset_entries, imported


@router.post("/projects/{project_id}/evaluation-suites", status_code=201)
def create_eval_suite(
    project_id: str, payload: EvalSuiteCreate, session: Session = Depends(get_session)
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    # Validate exact references before resolving new directory selections. Both are
    # committed with the suite, so a failed request cannot leave partial imports.
    entries = _normalize_suite_entries(session, project_id, payload.datasets) if payload.datasets else []
    if payload.benchmark_ids and any(not entry.get("benchmark_id") for entry in entries):
        raise HTTPException(status_code=422, detail={
            "code": "incompatible_evaluation_engines",
            "message": "所选评测集使用不同执行器，请分别创建套件",
        })
    catalog_entries, _ = _resolve_catalog_datasets(session, project_id, payload.benchmark_ids, {})
    entries = _normalize_suite_entries(session, project_id, [*entries, *catalog_entries])
    scoring_config = deepcopy(payload.scoring_config)
    if all(entry.get("benchmark_id") for entry in entries):
        scoring_config["asset_type"] = "benchmark"
    else:
        scoring_config.pop("asset_type", None)
    latest = _latest_version(session, EvalSuiteVersion, project_id, payload.name)
    row = EvalSuiteVersion(
        project_id=project_id,
        version=latest + 1,
        name=payload.name,
        datasets=entries,
        scoring_config=scoring_config,
        release_gate=payload.release_gate,
    )
    session.add(row)
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return row.to_dict()


@router.post("/projects/{project_id}/benchmark-suites", status_code=201)
def create_benchmark_suite(
    project_id: str,
    payload: BenchmarkSuiteCreate,
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    dataset_entries, imported = _resolve_catalog_datasets(
        session, project_id, payload.benchmark_ids, payload.overrides
    )
    suite = EvalSuiteVersion(
        project_id=project_id,
        name=payload.name,
        version=_latest_version(session, EvalSuiteVersion, project_id, payload.name) + 1,
        datasets=dataset_entries,
        scoring_config={**payload.scoring_config, "asset_type": "benchmark"},
        release_gate=payload.release_gate,
    )
    session.add(suite)
    session.commit()
    return {
        **suite.to_dict(),
        "imported_datasets": [dataset.to_dict() for dataset in imported],
    }


@router.get("/projects/{project_id}/evaluations")
def list_evaluations(project_id: str, session: Session = Depends(get_session)) -> list[dict]:
    return [
        row.to_dict()
        for row in session.scalars(
            select(EvaluationRun)
            .where(EvaluationRun.project_id == project_id)
            .order_by(EvaluationRun.created_at.desc())
        ).all()
    ]


def _sample_answer(value: Any, choices: list[Any], *, normalize_by_length: bool = False) -> Any:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, str) and choices:
        try:
            numeric_index = int(value.strip())
        except ValueError:
            numeric_index = -1
        if 0 <= numeric_index < len(choices):
            return {"index": numeric_index, "label": chr(65 + numeric_index), "value": choices[numeric_index]}
    if isinstance(value, int) and 0 <= value < len(choices):
        return {"index": value, "label": chr(65 + value), "value": choices[value]}
    if isinstance(value, str) and len(value) == 1 and "A" <= value.upper() <= "Z":
        index = ord(value.upper()) - 65
        if 0 <= index < len(choices):
            return {"index": index, "label": chr(65 + index), "value": choices[index]}
    if isinstance(value, list) and choices and len(value) == len(choices):
        scored: list[float | None] = []
        for index, item in enumerate(value):
            raw_score = item[0] if isinstance(item, list) and item else None
            try:
                parsed = float(raw_score)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None and normalize_by_length:
                parsed /= max(1, len(str(choices[index])))
            scored.append(parsed)
        if scored and all(item is not None for item in scored):
            index = max(range(len(scored)), key=lambda candidate: float(scored[candidate]))
            return {"index": index, "label": chr(65 + index), "value": choices[index]}
    return value


def _sample_question(document: dict[str, Any]) -> Any:
    question = next(
        (document.get(key) for key in ("question", "query", "prompt", "input", "text", "ctx", "prefix") if document.get(key) not in (None, "")),
        None,
    )
    context = next(
        (document.get(key) for key in ("context", "passage", "article", "content") if document.get(key) not in (None, "")),
        None,
    )
    if context is None or context == question:
        return question
    context_text = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
    if question is None:
        return context_text
    question_text = question if isinstance(question, str) else json.dumps(question, ensure_ascii=False)
    return f"材料：{context_text}\n\n问题：{question_text}"


def _evaluation_sample_view(raw: dict[str, Any], metric: str) -> dict[str, Any]:
    document = raw.get("doc") if isinstance(raw.get("doc"), dict) else {}
    choices = document.get("choices") or document.get("choice") or document.get("endings") or []
    if not choices and all(document.get(key) is not None for key in ("A", "B", "C", "D")):
        choices = [document[key] for key in ("A", "B", "C", "D")]
    if not isinstance(choices, list):
        choices = []
    question = _sample_question(document)
    raw_score = raw.get(metric)
    if not isinstance(raw_score, (int, float)):
        raw_score = next((value for key, value in raw.items() if key.startswith(f"{metric},") and isinstance(value, (int, float))), None)
    filtered = raw.get("filtered_resps") if "filtered_resps" in raw else raw.get("resps")
    model_answer = _sample_answer(filtered, choices, normalize_by_length=metric.lower() == "acc_norm")
    reference = _sample_answer(raw.get("target", document.get("answer")), choices)
    metrics = {key: value for key, value in raw.items() if isinstance(value, (int, float, bool)) and key not in {"doc_id"}}
    return {
        "id": str(raw.get("doc_id") if raw.get("doc_id") is not None else raw.get("id") or ""),
        "question": question,
        "choices": choices,
        "reference_answer": reference,
        "model_answer": model_answer,
        "metric": metric,
        "score": raw_score,
        "status": "observed" if raw_score is None else "passed" if raw_score >= 1 else "failed",
        "metrics": metrics,
        "metadata": {key: document[key] for key in ("subject", "category", "topic") if document.get(key) is not None},
    }


@router.get("/projects/{project_id}/evaluations/{run_id}/samples")
def evaluation_samples(
    project_id: str,
    run_id: str,
    model_version_id: str,
    task_id: str,
    offset: int = Query(default=0, ge=0, le=1_000_000),
    limit: int = Query(default=50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> dict:
    get_or_404(session, Project, project_id, "项目")
    run = get_or_404(session, EvaluationRun, run_id, "评测运行")
    if run.project_id != project_id:
        raise HTTPException(status_code=404, detail={"code": "evaluation_not_found", "message": "评测运行不存在"})
    model_results = (run.summary or {}).get("models") or {}
    task_result = (model_results.get(model_version_id) or {}).get("tasks", {}).get(task_id)
    if model_version_id not in (run.model_version_ids or []) or not isinstance(task_result, dict):
        raise HTTPException(status_code=404, detail={"code": "evaluation_item_not_found", "message": "该评测运行没有此模型与项目结果"})
    execution_task = get_or_404(session, Task, run.task_id, "评测任务")
    artifact_id = str((execution_task.result or {}).get("artifact_id") or "")
    artifact = session.get(Artifact, artifact_id) if artifact_id else None
    if artifact is None or artifact.kind != "evaluation_report":
        raise HTTPException(status_code=409, detail={"code": "evaluation_samples_not_saved", "message": "该评测运行没有保存逐题报告"})
    try:
        page = artifact_storage.read_managed_archive_jsonl_page(
            session,
            artifact,
            project_id,
            (model_version_id, task_id, "samples.jsonl"),
            offset=offset,
            limit=limit,
            fallback_parent_parts=(model_version_id, task_id),
            fallback_filename_prefixes=("samples_",),
        )
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_evaluation_samples", "message": f"逐题报告无法读取：{exc}"}) from exc
    if page is None:
        raise HTTPException(status_code=409, detail={"code": "evaluation_samples_not_saved", "message": "该评测项目没有保存逐题报告"})
    rows, has_more, member_name = page
    metric = str(task_result.get("metric") or "score")
    return {"items": [_evaluation_sample_view(row, metric) for row in rows], "offset": offset, "limit": limit, "has_more": has_more, "source": member_name}


@router.post("/projects/{project_id}/evaluations", status_code=201)
def start_evaluation(
    project_id: str, payload: EvaluationStart, session: Session = Depends(get_session)
) -> dict:
    if not payload.model_version_ids or not payload.suite_version_ids:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_selection", "message": "至少选择一个模型和评测套件"},
        )
    primary_model_version_id = payload.primary_model_version_id or payload.model_version_ids[0]
    if primary_model_version_id not in payload.model_version_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_primary_model",
                "message": "主评模型必须包含在本轮所选模型中",
            },
        )
    models = [session.get(ModelVersion, model_id) for model_id in payload.model_version_ids]
    missing_models = [
        model_id
        for model_id, model in zip(payload.model_version_ids, models, strict=True)
        if model is None or model.project_id != project_id
    ]
    if missing_models:
        raise HTTPException(
            status_code=404, detail={"code": "models_not_found", "ids": missing_models}
        )
    suites = [session.get(EvalSuiteVersion, suite_id) for suite_id in payload.suite_version_ids]
    missing_suites = [
        suite_id
        for suite_id, suite in zip(payload.suite_version_ids, suites, strict=True)
        if suite is None or suite.project_id != project_id
    ]
    if missing_suites:
        raise HTTPException(
            status_code=404, detail={"code": "suites_not_found", "ids": missing_suites}
        )
    concrete_suites = [suite for suite in suites if suite is not None]
    empty_suites = [suite.id for suite in concrete_suites if not suite.datasets]
    if empty_suites:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_evaluation_suite",
                "message": "请先添加评测集或从 Benchmark 目录创建套件",
                "ids": empty_suites,
            },
        )
    suite_snapshots, benchmarks, evaluation_inputs, dataset_count, benchmark_count = _suite_snapshot(
        session, project_id, concrete_suites
    )
    all_benchmarks = bool(benchmarks) and benchmark_count == dataset_count
    adapter_id = payload.adapter_id or (
        "builtin.lm-eval" if all_benchmarks else "builtin.command.evaluation"
    )
    require_executable(adapter_id, payload.command)
    frozen_code = None
    if payload.code_version_id:
        if not adapter_id.startswith("builtin.command."):
            raise HTTPException(422, detail={"code": "command_code_required", "message": "自定义代码版本仅用于命令评测"})
        frozen_code = freeze_command_code(session, project_id, payload.code_version_id)
    inference_config = deepcopy(payload.inference_config)
    evaluation_model_inputs: list[dict[str, Any]] = []
    profile_resource_requests: list[dict[str, Any]] = []
    if adapter_id == "builtin.lm-eval" or frozen_code is not None:
        resolved_runtimes: dict[str, dict[str, Any]] = {}
        for model in models:
            if model is None:
                continue
            runtime, model_input, profile_request = _model_evaluation_runtime(session, model)
            resolved_runtimes[model.id] = runtime
            if model_input is not None:
                evaluation_model_inputs.append(model_input)
            if profile_request:
                profile_resource_requests.append(profile_request)
        inference_config["models"] = resolved_runtimes
    comparison_hash: str | None = None
    benchmark_signature: dict[str, Any] = {}
    if adapter_id == "builtin.lm-eval":
        if not benchmarks or not all_benchmarks:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "unsupported_benchmark_suite",
                    "message": "lm-eval 适配器只支持完全由标准 Benchmark 组成的套件",
                },
            )
        model_runtimes = _normalized_model_runtimes(inference_config, payload.model_version_ids)
        inference_config["models"] = model_runtimes
        inference_config["batch_size"] = _normalize_evaluation_batch_size(
            inference_config.get("batch_size")
        )
        inference_config["sampling"] = _normalize_evaluation_sampling(inference_config)
        unsupported_chat_tasks = [
            item["task_id"]
            for item in benchmarks
            if item.get("task_type") != "generation"
            and any(
                runtime["backend"] == "local-chat-completions"
                for runtime in model_runtimes.values()
            )
        ]
        if unsupported_chat_tasks:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "endpoint_capability_mismatch",
                    "message": "Chat Completions 端点不能执行需要 Log-likelihood 的评测项目",
                    "task_ids": unsupported_chat_tasks,
                },
            )
        primary_runtime = model_runtimes[primary_model_version_id]
        comparison_config = {
            "framework": "lm-eval",
            "benchmarks": [
                {
                    "task_id": item["task_id"],
                    "config_hash": item["config_hash"],
                    "weight": item["weight"],
                    "primary_metric": item["primary_metric"],
                }
                for item in benchmarks
            ],
            "primary_protocol": {
                "backend": primary_runtime["backend"],
                "endpoint_mode": primary_runtime.get("endpoint_mode"),
            },
            "inference": {
                key: value
                for key, value in inference_config.items()
                if key not in {"models", *EVALUATION_EXECUTION_ONLY_INFERENCE_KEYS}
            },
            "scoring": payload.scoring_config,
        }
        comparison_hash = canonical_hash(comparison_config)
        benchmark_signature = {
            "framework": "lm-eval",
            "catalog_config_hash": comparison_hash,
            "status": "pending",
        }
    config_snapshot = {
        "primary_model_version_id": primary_model_version_id,
        "baseline_model_version_ids": [
            model_id
            for model_id in payload.model_version_ids
            if model_id != primary_model_version_id
        ],
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "kind": model.kind,
                "artifact_id": model.artifact_id,
                "model_weight_version_id": model.model_weight_version_id,
                "runtime": inference_config.get("models", {}).get(model.id),
            }
            for model in models
            if model is not None
        ],
        "suites": suite_snapshots,
        "benchmarks": benchmarks,
        "evaluation_inputs": evaluation_inputs,
        "inference_config": inference_config,
        "scoring_config": deepcopy(payload.scoring_config),
        "comparison_config_hash": comparison_hash,
        **({"code": frozen_code} if frozen_code else {}),
    }
    model_weight_inputs: list[dict[str, Any]] = []
    for model in models:
        if model is None or not model.model_weight_version_id:
            continue
        weight = session.get(ModelWeightVersion, model.model_weight_version_id)
        artifact = session.get(Artifact, weight.artifact_id) if weight and weight.artifact_id else None
        if weight is None or weight.state != "ready" or artifact is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "model_weight_unavailable",
                    "message": f"模型 {model.name} 引用的全局权重当前不可用",
                },
            )
        if adapter_id == "builtin.lm-eval" and "lm_eval_hf" not in set(
            weight.runtime_capabilities or []
        ):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "model_weight_runtime_incompatible",
                    "message": f"模型 {model.name} 的权重不能由当前安全 Transformers/lm-eval 后端加载",
                },
            )
        model_weight_inputs.append(
            {
                "name": model.name,
                "model_version_id": model.id,
                "model_weight_version_id": weight.id,
                "artifact_id": artifact.id,
                "artifact_kind": "model_weights",
                "content_hash": artifact.content_hash,
                "identity_hash": weight.identity_hash,
            }
        )
    config_snapshot["model_weight_inputs"] = deepcopy(model_weight_inputs)
    merged_resource_request = _merge_profile_resource_requests(
        payload.resource_request, profile_resource_requests
    )
    resource_request = _route_evaluation_resource_request(
        session,
        project_id,
        merged_resource_request,
        [
            *(evaluation_inputs if adapter_id != "builtin.lm-eval" else []),
            *model_weight_inputs,
            *evaluation_model_inputs,
        ],
    )
    task = create_task(
        session,
        project_id=project_id,
        kind="evaluate",
        title="标准 Benchmark 评测" if adapter_id == "builtin.lm-eval" else "模型评测",
        adapter_id=adapter_id,
        launcher_id=payload.launcher_id,
        payload={
            "model_version_ids": payload.model_version_ids,
            "primary_model_version_id": primary_model_version_id,
            "suite_version_ids": payload.suite_version_ids,
            "inference_config": inference_config,
            "scoring_config": payload.scoring_config,
            "evaluation_inputs": evaluation_inputs,
            "model_weight_inputs": model_weight_inputs,
            "evaluation_model_inputs": evaluation_model_inputs,
            "benchmark_config": {
                "framework": "lm-eval",
                "tasks": benchmarks,
                "comparison_config_hash": comparison_hash,
            }
            if adapter_id == "builtin.lm-eval"
            else {},
            "config_snapshot": config_snapshot,
            "resource_request": resource_request,
            "command": payload.command,
            **({"code": frozen_code} if frozen_code else {}),
        },
    )
    run = EvaluationRun(
        project_id=project_id,
        task_id=task.id,
        status="queued",
        model_version_ids=payload.model_version_ids,
        suite_version_ids=payload.suite_version_ids,
        inference_config=inference_config,
        scoring_config=payload.scoring_config,
        config_snapshot=config_snapshot,
        benchmark_signature=benchmark_signature,
        compatibility_hash=comparison_hash,
    )
    session.add(run)
    session.flush()
    from ..services.transactions import commit_or_flush
    commit_or_flush(session)
    return {**run.to_dict(), "task": task.to_dict()}
