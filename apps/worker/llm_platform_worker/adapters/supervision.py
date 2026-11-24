from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import re
import shutil
import struct
import unicodedata
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from llm_platform_core.storage_layout import data_asset_root, derived_asset_root

from .command import Emit, GetDesiredState
from .data_import import AnalysisInterrupted

SUPERVISION_RECORD_FORMAT = "forgelm-supervision-record-index-v1"
SFT_CORPUS_FORMAT = "forgelm-sft-corpus-v1"
PREFERENCE_CORPUS_FORMAT = "forgelm-preference-corpus-v1"
SFT_VIEW_FORMAT = "forgelm-sft-training-index-v1"
PREFERENCE_VIEW_FORMAT = "forgelm-preference-training-index-v1"


def supervision_available() -> bool:
    try:
        import jinja2  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError:
        return False
    return True


def _restore_declared_eos_token(tokenizer: Any, tokenizer_binding: dict[str, Any]) -> int | None:
    """Restore EOS semantics for historical tokenizer artifacts without changing the vocabulary."""

    special_tokens = [str(value) for value in tokenizer_binding.get("special_tokens") or []]
    declared_eos = special_tokens[2] if len(special_tokens) >= 3 else None
    if tokenizer.eos_token_id is None and declared_eos:
        if declared_eos in tokenizer.get_vocab():
            tokenizer.eos_token = declared_eos
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None


def _require_sft_span_eos(
    ids: list[int], spans: list[tuple[int, int]], eos_token_id: int
) -> None:
    if any(length <= 0 or ids[start + length - 1] != eos_token_id for start, length in spans):
        raise ValueError("every SFT assistant generation span must end with eos_token_id")


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _file_spec(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return {"path": path.name, "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _controlled(get_desired_state: GetDesiredState | None) -> None:
    desired = get_desired_state() if get_desired_state else "running"
    if desired in {"paused", "cancelled"}:
        raise AnalysisInterrupted(desired)


def _source_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    suffixes = {".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet"}
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def _iter_records(root: Path, get_desired_state: GetDesiredState | None, *, records_layout: str = "auto") -> Iterator[tuple[str, int, Any]]:
    for path in _source_files(root):
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            with path.open("r", encoding="utf-8", errors="replace") as stream:
                for index, line in enumerate(stream):
                    if index % 4096 == 0:
                        _controlled(get_desired_state)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield relative, index, json.loads(line)
                    except json.JSONDecodeError:
                        yield relative, index, line
        elif suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            values = list(value.values()) if records_layout == "object_values" and isinstance(value, dict) else value if isinstance(value, list) else [value]
            for index, record in enumerate(values):
                yield relative, index, record
        elif suffix in {".csv", ".tsv"}:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t" if suffix == ".tsv" else ",")
                for index, record in enumerate(reader):
                    if index % 4096 == 0:
                        _controlled(get_desired_state)
                    yield relative, index, dict(record)
        elif suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("structured Parquet supervision data requires pyarrow") from exc
            parquet = pq.ParquetFile(path)
            position = 0
            for batch in parquet.iter_batches(batch_size=2048, use_threads=True):
                _controlled(get_desired_state)
                for record in batch.to_pylist():
                    yield relative, position, record
                    position += 1


def _field(record: Any, path: str | None, default: Any = None) -> Any:
    if not path:
        return default
    value = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _field_values(record: Any, path: str | None) -> list[Any]:
    """Resolve a dotted path through objects and lists without executing expressions."""
    if not path:
        return []
    values = [record]
    for part in path.split("."):
        next_values: list[Any] = []
        for value in values:
            candidates = value if isinstance(value, list) else [value]
            for candidate in candidates:
                if isinstance(candidate, dict) and part in candidate:
                    resolved = candidate[part]
                    next_values.extend(resolved if isinstance(resolved, list) else [resolved])
        values = next_values
        if not values:
            break
    return values


def _content_from_parts(record: dict[str, Any], parts: Any) -> str:
    if not isinstance(parts, list) or not parts:
        raise ValueError("content_parts_missing")
    output: list[str] = []
    for part in parts:
        if not isinstance(part, dict) or len(part) != 1:
            raise ValueError("content_part_invalid")
        if "literal" in part:
            output.append(str(part["literal"]))
            continue
        field = part.get("field")
        if not isinstance(field, dict) or not str(field.get("path") or "").strip():
            raise ValueError("content_part_invalid")
        values = [str(value).strip() for value in _field_values(record, str(field["path"]))
                  if value is not None and str(value).strip()]
        if field.get("unique", True):
            values = list(dict.fromkeys(values))
        if not values and field.get("required", True):
            raise ValueError(f"content_field_missing:{field['path']}")
        output.append(str(field.get("join") or "").join(values))
    content = "".join(output).strip()
    if not content:
        raise ValueError("empty_content")
    return content


def _message(value: Any, *, role_field: str, content_field: str, role_map: dict[str, str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError("message_not_object")
    raw_role = str(value.get(role_field) or "").strip()
    role = role_map.get(raw_role, raw_role)
    content = value.get(content_field)
    if role not in {"system", "user", "assistant"}:
        raise ValueError("unknown_role")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty_content")
    return {"role": role, "content": content}


def _validate_messages(messages: list[dict[str, str]], *, require_assistant: bool = True) -> None:
    if not messages:
        raise ValueError("empty_messages")
    if any(message["role"] == "system" for message in messages[1:]):
        raise ValueError("system_not_first")
    conversational = [message["role"] for message in messages if message["role"] != "system"]
    if require_assistant and "assistant" not in conversational:
        raise ValueError("assistant_missing")
    for previous, current in zip(conversational, conversational[1:], strict=False):
        if previous == current:
            raise ValueError("invalid_role_order")


def _conversation(record: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record_not_object")
    messages_path = str(mapping.get("messages_path") or mapping.get("messages_field") or "messages")
    values = _field(record, messages_path)
    if isinstance(values, list) and mapping.get("prompt_parts") is None and mapping.get("completion_parts") is None:
        role_field = str(mapping.get("role_field") or "role")
        content_field = str(mapping.get("content_field") or "content")
        role_map = {str(key): str(value) for key, value in (mapping.get("role_map") or {}).items()}
        messages = [
            _message(value, role_field=role_field, content_field=content_field, role_map=role_map)
            for value in values
        ]
    else:
        system = _field(record, str(mapping.get("system_field") or ""))
        instruction = (
            _content_from_parts(record, mapping["prompt_parts"])
            if mapping.get("prompt_parts") is not None
            else _field(
                record,
                str(mapping.get("prompt_field") or mapping.get("instruction_field") or "prompt"),
            )
        )
        extra_input = _field(record, str(mapping.get("input_field") or ""))
        completion = (
            _content_from_parts(record, mapping["completion_parts"])
            if mapping.get("completion_parts") is not None
            else _field(
                record,
                str(mapping.get("completion_field") or mapping.get("output_field") or "completion"),
            )
        )
        if mapping.get("choices_field"):
            choices = _field(record, str(mapping["choices_field"]))
            if isinstance(choices, str):
                choices = next(csv.reader([choices.strip().strip("[]")]))
            if not isinstance(choices, list) or not choices:
                raise ValueError("choices_missing")
            translations = mapping.get("choice_text_map") or {}
            choices = [str(translations.get(str(choice).strip(), str(choice).strip())) for choice in choices]
            answer = _field(record, str(mapping.get("answer_index_field") or ""))
            if not isinstance(answer, int) or isinstance(answer, bool):
                raise ValueError("answer_index_invalid")
            answer -= int(mapping.get("answer_index_base") or 0)
            if not 0 <= answer < len(choices):
                raise ValueError("answer_index_out_of_range")
            completion = choices[answer]
            explanation = _field(record, str(mapping.get("explanation_field") or ""))
            if isinstance(explanation, str) and explanation.strip():
                completion += "。" + explanation.strip()
        if not isinstance(instruction, str) or not isinstance(completion, str):
            raise ValueError("conversation_fields_missing")
        user_content = instruction
        if isinstance(extra_input, str) and extra_input.strip():
            user_content = f"{instruction}\n{extra_input}"
        messages = []
        if isinstance(system, str) and system.strip():
            messages.append({"role": "system", "content": system})
        messages.extend(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": completion},
            ]
        )
    _validate_messages(messages)
    return {"messages": messages}


def _message_list(value: Any, mapping: dict[str, Any], *, default_role: str | None = None) -> list[dict]:
    if isinstance(value, str):
        if not default_role or not value.strip():
            raise ValueError("empty_content")
        return [{"role": default_role, "content": value}]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("message_list_missing")
    role_field = str(mapping.get("role_field") or "role")
    content_field = str(mapping.get("content_field") or "content")
    role_map = {str(key): str(item) for key, item in (mapping.get("role_map") or {}).items()}
    return [
        _message(item, role_field=role_field, content_field=content_field, role_map=role_map)
        for item in value
    ]


def _preference(record: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record_not_object")
    prompt = _message_list(
        _field(record, str(mapping.get("prompt_path") or mapping.get("prompt_field") or "prompt")),
        mapping,
        default_role="user",
    )
    chosen = _message_list(
        _field(record, str(mapping.get("chosen_path") or mapping.get("chosen_field") or "chosen")),
        mapping,
        default_role="assistant",
    )
    rejected = _message_list(
        _field(record, str(mapping.get("rejected_path") or mapping.get("rejected_field") or "rejected")),
        mapping,
        default_role="assistant",
    )
    _validate_messages(prompt, require_assistant=False)
    if chosen[0]["role"] != "assistant" or rejected[0]["role"] != "assistant":
        raise ValueError("completion_not_assistant")
    if chosen == rejected:
        raise ValueError("chosen_rejected_identical")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def _canonical_record(raw: Any, kind: str, mapping: dict[str, Any]) -> dict[str, Any]:
    return _conversation(raw, mapping) if kind == "conversation" else _preference(raw, mapping)


def _lookup_scalar(record: Any, path: str, *, error: str) -> str:
    values = _field_values(record, path)
    if len(values) != 1 or isinstance(values[0], (dict, list)):
        raise ValueError(error)
    value = str(values[0]).strip()
    if not value:
        raise ValueError(error)
    return value


def _build_lookup_index(
    root: Path,
    config: dict[str, Any],
    get_desired_state: GetDesiredState | None,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    key_path = str(config.get("lookup_key") or "")
    fields = dict(config.get("fields") or {})
    for _logical_path, _record_index, raw in _iter_records(root, get_desired_state):
        if not isinstance(raw, dict):
            raise ValueError(f"lookup_record_not_object:{config.get('name')}")
        key = _lookup_scalar(raw, key_path, error=f"lookup_key_invalid:{config.get('name')}")
        if key in index:
            raise ValueError(f"lookup_key_duplicate:{config.get('name')}:{key}")
        index[key] = {alias: _field(raw, path) for alias, path in fields.items()}
    return index


def _apply_lookups(
    raw: Any,
    configs: list[dict[str, Any]],
    indexes: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("record_not_object")
    enriched = dict(raw)
    namespace: dict[str, Any] = {}
    for config, index in zip(configs, indexes, strict=True):
        name = str(config.get("name") or "")
        try:
            key = _lookup_scalar(
                raw,
                str(config.get("source_key") or ""),
                error=f"lookup_source_key_invalid:{name}",
            )
        except ValueError:
            if config.get("required", True):
                raise
            namespace[name] = {}
            continue
        matched = index.get(key)
        if matched is None:
            if config.get("required", True):
                raise ValueError(f"lookup_match_missing:{name}")
            matched = {}
        namespace[name] = matched
    enriched["_lookup"] = namespace
    return enriched


def _question_hash(record: dict) -> str:
    prompt = "\n".join(m["content"] for m in record.get("messages", []) if m["role"] == "user")
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", prompt)).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _filter_conversation(record: dict, filters: dict) -> None:
    messages = record.get("messages") or []
    for message in messages:
        text = message["content"]
        letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
        chinese = len(re.findall(r"[\u3400-\u9fff]", text))
        if letters and chinese / len(letters) < float(filters.get("min_chinese_ratio", 0)):
            raise ValueError("non_chinese_content")
        if len(text) < int(filters.get("min_message_chars", 0)):
            raise ValueError("message_too_short")
    prompt = "\n".join(m["content"] for m in messages if m["role"] == "user")
    if filters.get("exclude_math_tasks") and re.search(r"解方程|求导|积分|微分|几何证明|计算下列|求[xyXY]的值|数学题|[0-9]\s*[+*/=]\s*[0-9]", prompt):
        raise ValueError("math_task")
    if filters.get("exclude_language_learning") and re.search(r"翻译成英文|翻译成英语|英译汉|汉译英|英语作文|英文作文|英语语法|英文翻译", prompt):
        raise ValueError("language_learning_task")
    split = filters.get("hash_split")
    if split:
        bucket = int(_question_hash(record)[:16], 16) % 10000
        if not int(split["start"]) <= bucket < int(split["end"]):
            raise ValueError("outside_selected_split")


def _indexed_canonical_records(
    source_root: Path,
    records_path: Path,
    *,
    kind: str,
    mapping: dict[str, Any],
    get_desired_state: GetDesiredState | None,
) -> Iterator[dict[str, Any]]:
    """Resolve an ordered pointer index without copying source text into the index."""

    with records_path.open("r", encoding="utf-8") as pointers:
        pointer = json.loads(line) if (line := pointers.readline()) else None
        if pointer is None:
            return
        # Read-only compatibility with early v1 artifacts that embedded canonical rows.
        if "messages" in pointer or "prompt" in pointer:
            yield {key: value for key, value in pointer.items() if key != "source"}
            for line in pointers:
                row = json.loads(line)
                yield {key: value for key, value in row.items() if key != "source"}
            return
        for logical_path, record_index, raw in _iter_records(source_root, get_desired_state, records_layout=str(mapping.get("records_layout") or "auto")):
            if pointer is None:
                break
            if (
                logical_path != str(pointer.get("logical_path") or "")
                or record_index != int(pointer.get("record_index") or 0)
            ):
                continue
            canonical = _canonical_record(raw, kind, mapping)
            semantic_hash = hashlib.sha256(_canonical_json(canonical)).hexdigest()
            if semantic_hash != str(pointer.get("semantic_hash") or ""):
                raise ValueError("supervision source record no longer matches its frozen identity")
            yield canonical
            pointer = json.loads(line) if (line := pointers.readline()) else None
        if pointer is not None:
            raise ValueError("supervision record index points outside the materialized dataset")


class SupervisionScanAdapter:
    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        runtime = task.get("runtime_context") or {}
        identity_hash = str(payload.get("identity_hash") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", identity_hash):
            raise ValueError("supervision scan requires a frozen SHA-256 identity")
        data_inputs = list(payload.get("data_inputs") or [])
        if not data_inputs and isinstance(payload.get("data_input"), dict):
            data_inputs = [payload["data_input"]]
        data_input = data_inputs[0] if data_inputs else {}
        sources = data_input.get("runtime_sources") or []
        if len(sources) != 1:
            raise ValueError("supervision scan requires exactly one materialized dataset")
        source_root = Path(str(sources[0].get("runtime_path") or "")).resolve()
        kind = str(payload.get("kind") or "")
        mapping = dict(payload.get("mapping") or {})
        lookup_configs = list(mapping.get("_lookups") or [])
        if len(data_inputs) != 1 + len(lookup_configs):
            raise ValueError("supervision scan lookup bindings do not match frozen mapping")
        lookup_indexes: list[dict[str, dict[str, Any]]] = []
        for position, config in enumerate(lookup_configs, start=1):
            lookup_sources = data_inputs[position].get("runtime_sources") or []
            if len(lookup_sources) != 1:
                raise ValueError("supervision lookup requires exactly one materialized dataset")
            lookup_indexes.append(
                _build_lookup_index(
                    Path(str(lookup_sources[0].get("runtime_path") or "")).resolve(),
                    config,
                    get_desired_state,
                )
            )
        filters = dict(payload.get("filters") or {})
        if kind not in {"conversation", "preference"}:
            raise ValueError("unsupported supervision kind")
        project_workspace = Path(str(runtime.get("project_workspace") or "")).resolve()
        storage_root = (
            data_asset_root(project_workspace) / "supervision-records"
        ).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        destination = (storage_root / identity_hash).resolve()
        if destination.parent != storage_root:
            raise ValueError("supervision record destination escaped project storage")
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes)
            return self._result(destination, manifest, hashlib.sha256(manifest_bytes).hexdigest())
        partial = (storage_root / f".{identity_hash}.{task['id']}.partial").resolve()
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        loop = asyncio.get_running_loop()

        def build() -> tuple[dict[str, Any], str]:
            accepted = 0
            rejected = 0
            reasons: Counter[str] = Counter()
            previews: list[dict[str, Any]] = []
            seen: set[str] = set()
            records_path = partial / "records.jsonl"
            rejections_path = partial / "rejections.jsonl"
            with records_path.open("wb") as output, rejections_path.open("wb") as rejections:
                for processed, (logical_path, record_index, raw) in enumerate(
                    _iter_records(source_root, get_desired_state, records_layout=str(mapping.get("records_layout") or "auto")), start=1
                ):
                    try:
                        mapped_raw = (
                            _apply_lookups(raw, lookup_configs, lookup_indexes)
                            if lookup_configs
                            else raw
                        )
                        canonical = _canonical_record(mapped_raw, kind, mapping)
                        if kind == "conversation":
                            _filter_conversation(canonical, filters)
                        if filters.get("max_records") and accepted >= int(filters["max_records"]):
                            raise ValueError("sample_limit")
                        semantic_hash = hashlib.sha256(_canonical_json(canonical)).hexdigest()
                        dedup_key = _question_hash(canonical) if filters.get("deduplicate_questions") and kind == "conversation" else semantic_hash
                        if (filters.get("exact_deduplicate") or filters.get("deduplicate_questions")) and dedup_key in seen:
                            raise ValueError("exact_duplicate")
                        seen.add(dedup_key)
                        source = {
                            "logical_path": logical_path,
                            "record_index": record_index,
                            "semantic_hash": semantic_hash,
                        }
                        record = {**canonical, "source": source} if lookup_configs else source
                        output.write(_canonical_json(record))
                        accepted += 1
                        if len(previews) < 20:
                            previews.append({**canonical, "source": source})
                    except (TypeError, ValueError) as exc:
                        reason = str(exc) or "invalid_record"
                        reasons[reason] += 1
                        rejected += 1
                        rejections.write(
                            _canonical_json(
                                {
                                    "logical_path": logical_path,
                                    "record_index": record_index,
                                    "reason": reason,
                                }
                            )
                        )
                    if processed % 2048 == 0:
                        asyncio.run_coroutine_threadsafe(
                            emit(
                                "progress",
                                {
                                    "phase": "scanning_supervision_records",
                                    "progress": 0.5,
                                    "processed_records": processed,
                                    "accepted_records": accepted,
                                    "rejected_records": rejected,
                                },
                            ),
                            loop,
                        ).result()
            if accepted == 0:
                raise ValueError("no source record can be mapped to the selected supervision schema")
            report = {
                "total_record_count": accepted + rejected,
                "accepted_record_count": accepted,
                "rejected_record_count": rejected,
                "rejection_reasons": dict(reasons),
            }
            manifest = {
                "format": SUPERVISION_RECORD_FORMAT,
                "format_version": 1,
                "identity_hash": identity_hash,
                "kind": kind,
                "mapping": mapping,
                "materialization_mode": (
                    "canonical_rows_v1" if lookup_configs else "source_pointers_v1"
                ),
                "filters": filters,
                "validation_report": report,
                "sample_preview": previews,
                "files": {
                    "records": _file_spec(records_path),
                    "rejections": _file_spec(rejections_path),
                },
            }
            manifest_bytes = _canonical_json(manifest)
            (partial / "manifest.json").write_bytes(manifest_bytes)
            partial.replace(destination)
            return manifest, hashlib.sha256(manifest_bytes).hexdigest()

        try:
            manifest, manifest_hash = await asyncio.to_thread(build)
        except AnalysisInterrupted as exc:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            return {"controlled_state": exc.state}
        except Exception:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            raise
        await emit("progress", {"phase": "completed", "progress": 1.0, **manifest["validation_report"]})
        return self._result(destination, manifest, manifest_hash)

    @staticmethod
    def _result(destination: Path, manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
        report = manifest["validation_report"]
        size_bytes = sum(int(spec["size_bytes"]) for spec in manifest["files"].values())
        return {
            "identity_hash": manifest["identity_hash"],
            "format": manifest["format"],
            "accepted_record_count": report["accepted_record_count"],
            "rejected_record_count": report["rejected_record_count"],
            "validation_report": report,
            "sample_preview": manifest.get("sample_preview") or [],
            "warnings": [],
            "outputs": [
                {
                    "ref": "supervision_record_index",
                    "kind": "supervision_record_index",
                    "storage_type": "external",
                    "uri": destination.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": size_bytes,
                    "media_type": "application/vnd.forgelm.supervision-record-index-v1",
                    "availability_state": "available",
                    "metadata": {"format": manifest["format"], "identity_hash": manifest["identity_hash"]},
                }
            ],
        }


def _flat_token_values(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _tokenized_chat(tokenizer: Any, body: str, messages: list[dict], *, generation_prompt: bool) -> tuple[list[int], list[int]]:
    rendered = tokenizer.apply_chat_template(
        messages,
        chat_template=body,
        tokenize=True,
        add_generation_prompt=generation_prompt,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    ids = _flat_token_values(rendered.get("input_ids"))
    masks = _flat_token_values(
        rendered.get("assistant_masks", rendered.get("assistant_tokens_mask"))
    )
    return ids, masks


def _drop_oldest_conversation_turn(messages: list[dict]) -> list[dict] | None:
    prefix = 1 if messages and messages[0].get("role") == "system" else 0
    if len(messages) - prefix <= 2:
        return None
    removable = 2 if len(messages) - prefix >= 4 else 1
    return [*messages[:prefix], *messages[prefix + removable :]]


def _truncate_sft_prompt(ids: list[int], mask: list[int], max_length: int) -> tuple[list[int], list[int]]:
    overflow = len(ids) - max_length
    first_assistant = next((index for index, value in enumerate(mask) if value), len(ids))
    if overflow <= 0:
        return ids, mask
    if overflow >= first_assistant:
        return ids, mask
    return ids[:1] + ids[1 + overflow :], mask[:1] + mask[1 + overflow :]


def _spans(mask: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate([*mask, 0]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            spans.append((start, index - start))
            start = None
    return spans


def _sft_windows(ids: list[int], mask: list[int], max_length: int):
    """Keep each predictable assistant token once, with masked overlap as context."""
    if len(ids) != len(mask):
        raise ValueError("assistant mask length does not match token length")
    overlap = min(128, max_length // 4)
    start = 0
    supervised_end = 0
    while start < len(ids):
        end = min(start + max_length, len(ids))
        window_mask = list(mask[start:end])
        for index in range(max(1, supervised_end - start)):
            window_mask[index] = 0
        if any(window_mask):
            yield ids[start:end], window_mask
        supervised_end = end
        if end == len(ids):
            break
        start = end - overlap


class SupervisionCorpusAdapter:
    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        from transformers import AutoTokenizer

        payload = task.get("payload") or {}
        runtime = task.get("runtime_context") or {}
        identity_hash = str(payload.get("identity_hash") or "")
        objective = str(payload.get("objective_kind") or "")
        format_name = str(payload.get("format") or "")
        if objective not in {"sft", "dpo"} or format_name not in {
            SFT_CORPUS_FORMAT,
            PREFERENCE_CORPUS_FORMAT,
        }:
            raise ValueError("unsupported supervision corpus objective or format")
        tokenizer_binding = dict(payload.get("tokenizer") or {})
        tokenizer_root = Path(str(tokenizer_binding.get("runtime_path") or ""))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, trust_remote_code=False)
        eos_token_id = _restore_declared_eos_token(tokenizer, tokenizer_binding)
        body = str((payload.get("chat_template") or {}).get("template_body") or "")
        config = dict(payload.get("configuration") or {})
        max_length = int(config.get("max_length") or 0)
        min_completion = int(config.get("min_completion_tokens") or 1)
        overflow_policy = str(config.get("overflow_policy") or "drop_overflow_v1")
        dtype_policy = str(config.get("token_dtype") or "auto")
        if max_length <= 0 or not body:
            raise ValueError("supervision corpus requires a template and positive max_length")
        if objective == "sft" and eos_token_id is None:
            raise ValueError("SFT supervision corpus tokenizer must define eos_token_id")
        vocab_size = len(tokenizer)
        dtype = np.dtype("<u2" if dtype_policy == "uint16" or (dtype_policy == "auto" and vocab_size <= 65536) else "<u4")
        project_workspace = Path(str(runtime.get("project_workspace") or "")).resolve()
        storage_root = (
            derived_asset_root(project_workspace) / "supervision-corpora"
        ).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        destination = (storage_root / identity_hash).resolve()
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            manifest_bytes = manifest_path.read_bytes()
            return self._result(destination, json.loads(manifest_bytes), hashlib.sha256(manifest_bytes).hexdigest())
        partial = (storage_root / f".{identity_hash}.{task['id']}.partial").resolve()
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)

        def build() -> tuple[dict[str, Any], str]:
            tokens_path = partial / "tokens.bin"
            index_path = partial / ("examples.idx" if objective == "sft" else "pairs.idx")
            spans_path = partial / "loss-spans.idx"
            raw_tokens = 0
            supervised_tokens = 0
            assistant_span_count = 0
            assistant_eos_span_count = 0
            count = 0
            dropped = 0
            seen_questions: set[str] = set()
            token_offset = 0
            span_offset = 0
            sources = sorted(payload.get("supervision_sources") or [], key=lambda item: int(item["source_index"]))
            source_ranges: list[dict[str, int | str]] = []
            with tokens_path.open("wb") as token_stream, index_path.open("wb") as index_stream, spans_path.open("wb") as span_stream:
                for source in sources:
                    source_row_start = count
                    root = Path(str(source.get("runtime_record_index_path") or "")).resolve()
                    records_path = root / "records.jsonl"
                    if not records_path.is_file():
                        raise ValueError("supervision record index was not materialized")
                    data_input_index = int(source.get("data_input_index") or 0)
                    data_inputs = payload.get("data_inputs") or []
                    if data_input_index >= len(data_inputs):
                        raise ValueError("supervision source data binding is missing")
                    runtime_sources = data_inputs[data_input_index].get("runtime_sources") or []
                    if len(runtime_sources) != 1:
                        raise ValueError("supervision source requires one materialized dataset")
                    source_root = Path(
                        str(runtime_sources[0].get("runtime_path") or "")
                    ).resolve()
                    for record in _indexed_canonical_records(
                        source_root,
                        records_path,
                        kind=str(source.get("kind") or ""),
                        mapping=dict(source.get("mapping") or {}),
                        get_desired_state=get_desired_state,
                    ):
                        if record:
                            _controlled(get_desired_state)
                            if objective == "sft":
                                question_key = _question_hash(record)
                                if config.get("deduplicate_questions") and question_key in seen_questions:
                                    dropped += 1
                                    continue
                                messages = list(record["messages"])
                                ids, mask = _tokenized_chat(tokenizer, body, messages, generation_prompt=False)
                                if overflow_policy == "conversation_tail_v1":
                                    while len(ids) > max_length:
                                        messages = _drop_oldest_conversation_turn(messages)
                                        if messages is None:
                                            break
                                        ids, mask = _tokenized_chat(
                                            tokenizer, body, messages, generation_prompt=False
                                        )
                                elif overflow_policy == "prompt_completion_v1":
                                    ids, mask = _truncate_sft_prompt(ids, mask, max_length)
                                if (len(ids) > max_length and overflow_policy != "sliding_window_v1") or len(ids) != len(mask):
                                    dropped += 1
                                    continue
                                spans = _spans(mask)
                                supervised = sum(length for _start, length in spans)
                                if not spans or supervised < min_completion:
                                    dropped += 1
                                    continue
                                _require_sft_span_eos(ids, spans, int(eos_token_id))
                                if config.get("deduplicate_questions"):
                                    seen_questions.add(question_key)
                                windows = (_sft_windows(ids, mask, max_length)
                                           if overflow_policy == "sliding_window_v1" else [(ids, mask)])
                                for window_ids, window_mask in windows:
                                    window_spans = _spans(window_mask)
                                    assistant_span_count += len(window_spans)
                                    assistant_eos_span_count += sum(
                                        window_ids[start + length - 1] == int(eos_token_id)
                                        for start, length in window_spans
                                    )
                                    token_stream.write(np.asarray(window_ids, dtype=dtype).tobytes())
                                    for start, length in window_spans:
                                        span_stream.write(struct.pack("<QQ", start, length))
                                    index_stream.write(struct.pack("<QQQQ", token_offset, len(window_ids), span_offset, len(window_spans)))
                                    token_offset += len(window_ids)
                                    span_offset += len(window_spans)
                                    raw_tokens += len(window_ids)
                                    supervised_tokens += sum(length for _, length in window_spans)
                                    count += 1
                            else:
                                prompt_messages = list(record["prompt"])
                                prompt_ids, _ = _tokenized_chat(tokenizer, body, prompt_messages, generation_prompt=True)
                                chosen_full, _ = _tokenized_chat(tokenizer, body, [*prompt_messages, *record["chosen"]], generation_prompt=False)
                                rejected_full, _ = _tokenized_chat(tokenizer, body, [*prompt_messages, *record["rejected"]], generation_prompt=False)
                                if overflow_policy == "conversation_tail_v1":
                                    while max(len(chosen_full), len(rejected_full)) > max_length:
                                        prompt_messages = _drop_oldest_conversation_turn(prompt_messages)
                                        if prompt_messages is None:
                                            break
                                        prompt_ids, _ = _tokenized_chat(tokenizer, body, prompt_messages, generation_prompt=True)
                                        chosen_full, _ = _tokenized_chat(tokenizer, body, [*prompt_messages, *record["chosen"]], generation_prompt=False)
                                        rejected_full, _ = _tokenized_chat(tokenizer, body, [*prompt_messages, *record["rejected"]], generation_prompt=False)
                                if chosen_full[: len(prompt_ids)] != prompt_ids or rejected_full[: len(prompt_ids)] != prompt_ids:
                                    dropped += 1
                                    continue
                                chosen = chosen_full[len(prompt_ids) :]
                                rejected = rejected_full[len(prompt_ids) :]
                                if overflow_policy == "prompt_completion_v1":
                                    keep_prompt = max_length - max(len(chosen), len(rejected))
                                    if keep_prompt > 0 and len(prompt_ids) > keep_prompt:
                                        prompt_ids = prompt_ids[-keep_prompt:]
                                if min(len(chosen), len(rejected)) < min_completion or max(len(chosen_full), len(rejected_full)) > max_length:
                                    if overflow_policy == "prompt_completion_v1" and max(
                                        len(prompt_ids) + len(chosen), len(prompt_ids) + len(rejected)
                                    ) <= max_length:
                                        pass
                                    else:
                                        dropped += 1
                                        continue
                                prompt_offset = token_offset
                                token_stream.write(np.asarray(prompt_ids, dtype=dtype).tobytes())
                                token_offset += len(prompt_ids)
                                chosen_offset = token_offset
                                token_stream.write(np.asarray(chosen, dtype=dtype).tobytes())
                                token_offset += len(chosen)
                                rejected_offset = token_offset
                                token_stream.write(np.asarray(rejected, dtype=dtype).tobytes())
                                token_offset += len(rejected)
                                index_stream.write(
                                    struct.pack(
                                        "<QQQQQQ",
                                        prompt_offset,
                                        len(prompt_ids),
                                        chosen_offset,
                                        len(chosen),
                                        rejected_offset,
                                        len(rejected),
                                    )
                                )
                                raw_tokens += len(prompt_ids) + len(chosen) + len(rejected)
                                supervised_tokens += len(chosen) + len(rejected)
                            if objective != "sft":
                                count += 1
                    source_ranges.append(
                        {
                            "source_index": int(source["source_index"]),
                            "version_id": str(source.get("version_id") or ""),
                            "row_start": source_row_start,
                            "row_count": count - source_row_start,
                            "weight": int(source.get("weight") or 1),
                        }
                    )
            if count == 0:
                raise ValueError("no supervision record remained after template rendering and length handling")
            files = {"tokens": _file_spec(tokens_path), "index": _file_spec(index_path)}
            if objective == "sft":
                files["loss_spans"] = _file_spec(spans_path)
            else:
                spans_path.unlink(missing_ok=True)
            manifest = {
                "format": format_name,
                "format_version": 1,
                "identity_hash": identity_hash,
                "identity_snapshot": payload.get("identity_snapshot") or {},
                "objective_kind": objective,
                "token_dtype": dtype.str,
                "max_length": max_length,
                "example_count": count if objective == "sft" else 0,
                "pair_count": count if objective == "dpo" else 0,
                "raw_token_count": raw_tokens,
                "supervised_token_count": supervised_tokens,
                "eos_token_id": eos_token_id,
                "assistant_span_count": assistant_span_count if objective == "sft" else 0,
                "assistant_eos_span_count": (
                    assistant_eos_span_count if objective == "sft" else 0
                ),
                "assistant_eos_coverage": (
                    assistant_eos_span_count / assistant_span_count
                    if objective == "sft" and assistant_span_count
                    else None
                ),
                "dropped_record_count": dropped,
                "sources": source_ranges,
                "index_record": (
                    "little-endian uint64 token_offset,length,span_offset,span_count"
                    if objective == "sft"
                    else "little-endian uint64 prompt_offset,length,chosen_offset,length,rejected_offset,length"
                ),
                "files": files,
                "warnings": ([{"code": "records_dropped", "count": dropped}] if dropped else []),
            }
            manifest_bytes = _canonical_json(manifest)
            (partial / "manifest.json").write_bytes(manifest_bytes)
            partial.replace(destination)
            return manifest, hashlib.sha256(manifest_bytes).hexdigest()

        try:
            manifest, manifest_hash = await asyncio.to_thread(build)
        except AnalysisInterrupted as exc:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            return {"controlled_state": exc.state}
        except Exception:
            await asyncio.to_thread(shutil.rmtree, partial, True)
            raise
        await emit("progress", {"phase": "completed", "progress": 1.0, "raw_tokens": manifest["raw_token_count"]})
        return self._result(destination, manifest, manifest_hash)

    @staticmethod
    def _result(destination: Path, manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
        size_bytes = sum(int(spec["size_bytes"]) for spec in manifest["files"].values())
        return {
            "identity_hash": manifest["identity_hash"],
            "format": manifest["format"],
            "objective_kind": manifest["objective_kind"],
            "example_count": manifest["example_count"],
            "pair_count": manifest["pair_count"],
            "raw_token_count": manifest["raw_token_count"],
            "supervised_token_count": manifest["supervised_token_count"],
            "eos_token_id": manifest.get("eos_token_id"),
            "assistant_span_count": manifest.get("assistant_span_count", 0),
            "assistant_eos_span_count": manifest.get("assistant_eos_span_count", 0),
            "assistant_eos_coverage": manifest.get("assistant_eos_coverage"),
            "size_bytes": size_bytes,
            "warnings": manifest.get("warnings") or [],
            "outputs": [
                {
                    "ref": "supervision_corpus",
                    "kind": "supervision_corpus",
                    "storage_type": "external",
                    "uri": destination.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": size_bytes,
                    "media_type": "application/vnd.forgelm.supervision-corpus-v1",
                    "availability_state": "available",
                    "metadata": {"format": manifest["format"], "identity_hash": manifest["identity_hash"]},
                }
            ],
        }


class SupervisionTrainingIndexAdapter:
    @staticmethod
    def _order_identity(
        corpus_identity_hash: str, sample_order_policy: str, seed: int
    ) -> tuple[str, int]:
        snapshot = {
            "schema_version": "forgelm.supervision-sample-order.v1",
            "corpus_identity_hash": corpus_identity_hash,
            "sample_order_policy": sample_order_policy,
            "seed": seed,
        }
        digest = hashlib.sha256(_canonical_json(snapshot)).digest()
        return digest.hex(), int.from_bytes(digest[:16], "little")

    @staticmethod
    def _row_token_counts(
        root: Path, manifest: dict[str, Any], objective: str
    ) -> tuple[np.ndarray, np.ndarray]:
        files = manifest.get("files") or {}
        index_spec = files.get("index") or {}
        index_path = root / str(index_spec.get("path") or "")
        if objective == "sft":
            spans_spec = files.get("loss_spans") or {}
            spans_path = root / str(spans_spec.get("path") or "")
            if not index_path.is_file() or not spans_path.is_file():
                raise ValueError("SFT corpus row statistics are unavailable")
            index = np.memmap(index_path, dtype="<u8", mode="r").reshape(-1, 4)
            spans = np.memmap(spans_path, dtype="<u8", mode="r").reshape(-1, 2)
            span_prefix = np.zeros(len(spans) + 1, dtype=np.uint64)
            np.cumsum(spans[:, 1], dtype=np.uint64, out=span_prefix[1:])
            span_starts = np.asarray(index[:, 2], dtype=np.int64)
            span_ends = span_starts + np.asarray(index[:, 3], dtype=np.int64)
            raw = np.asarray(index[:, 1], dtype=np.uint64)
            supervised = span_prefix[span_ends] - span_prefix[span_starts]
            return raw, supervised
        if not index_path.is_file():
            raise ValueError("preference corpus row statistics are unavailable")
        index = np.memmap(index_path, dtype="<u8", mode="r").reshape(-1, 6)
        raw = np.asarray(index[:, 1] + index[:, 3] + index[:, 5], dtype=np.uint64)
        supervised = np.asarray(index[:, 3] + index[:, 5], dtype=np.uint64)
        return raw, supervised

    async def run(
        self,
        task: dict[str, Any],
        emit: Emit,
        get_desired_state: GetDesiredState | None = None,
    ) -> dict[str, Any]:
        payload = task.get("payload") or {}
        runtime = task.get("runtime_context") or {}
        identity_hash = str(payload.get("identity_hash") or "")
        objective = str(payload.get("objective_kind") or "")
        binding = payload.get("supervision_corpus") or {}
        root = Path(str(binding.get("runtime_path") or "")).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        config = dict(payload.get("configuration") or {})
        budget_unit = str(config.get("budget_unit") or "")
        budget_value = int(config.get("budget_value") or 0)
        consumption = str(config.get("consumption_policy") or "")
        sample_order_policy = str(config.get("sample_order_policy") or "sequential_v1")
        sample_offset = int(config.get("sample_offset") or 0)
        seed = int(config.get("seed") or 0)
        reshuffle_each_epoch = bool(config.get("reshuffle_each_epoch"))
        if reshuffle_each_epoch and (sample_order_policy != "global_shuffle_v1" or budget_unit not in {"examples", "preference_pairs"}):
            raise ValueError("per-epoch shuffle requires global order and a row budget")
        available = int(manifest.get("example_count") if objective == "sft" else manifest.get("pair_count") or 0)
        if available <= 0 or budget_value <= 0 or sample_offset < 0:
            raise ValueError("supervision training view has no usable rows or budget")
        if sample_order_policy not in {"sequential_v1", "global_shuffle_v1"}:
            raise ValueError(f"unsupported supervision sample order policy: {sample_order_policy}")
        if consumption == "unique_only" and sample_offset >= available:
            raise ValueError("supervision training view sample_offset leaves no unique rows")
        if budget_unit in {"examples", "preference_pairs"}:
            selected = budget_value
        elif budget_unit == "steps":
            selected = (
                budget_value
                * int(config.get("micro_batch_size") or 1)
                * int(config.get("gradient_accumulation_steps") or 1)
            )
        elif budget_unit == "supervised_tokens" and objective == "sft":
            average = max(int(manifest.get("supervised_token_count") or 0) / available, 1)
            selected = int(np.ceil(budget_value / average))
        else:
            selected = available
        if consumption == "unique_only":
            selected = min(selected, available - sample_offset)
        corpus_identity_hash = str(manifest.get("identity_hash") or binding.get("identity_hash") or "")
        order_identity_hash, order_seed = self._order_identity(
            corpus_identity_hash, sample_order_policy, seed
        )
        if reshuffle_each_epoch:
            order_identity_hash = hashlib.sha256(
                f"{order_identity_hash}:reshuffle_each_epoch_v1".encode()
            ).hexdigest()
        base_order: np.ndarray | None = None
        row_raw: np.ndarray | None = None
        row_supervised: np.ndarray | None = None
        if sample_order_policy == "global_shuffle_v1":
            base_order = np.random.Generator(np.random.PCG64(order_seed)).permutation(available)
            row_raw, row_supervised = self._row_token_counts(root, manifest, objective)
            if len(row_raw) != available or len(row_supervised) != available:
                raise ValueError("supervision corpus row statistics do not match its row count")
            if budget_unit == "supervised_tokens" and objective == "sft":
                if consumption == "unique_only":
                    ordered_tokens = row_supervised[base_order[sample_offset:]]
                    cumulative = np.cumsum(ordered_tokens, dtype=np.uint64)
                    selected = (
                        len(cumulative)
                        if not len(cumulative) or int(cumulative[-1]) < budget_value
                        else int(np.searchsorted(cumulative, budget_value, side="left")) + 1
                    )
                else:
                    cycle_start = sample_offset % available
                    rotated = np.concatenate((base_order[cycle_start:], base_order[:cycle_start]))
                    cycle_tokens = row_supervised[rotated]
                    cycle_total = int(np.sum(cycle_tokens, dtype=np.uint64))
                    if cycle_total <= 0:
                        raise ValueError("SFT corpus has no supervised tokens")
                    complete_cycles = (budget_value - 1) // cycle_total
                    remaining_tokens = budget_value - complete_cycles * cycle_total
                    cumulative = np.cumsum(cycle_tokens, dtype=np.uint64)
                    selected = complete_cycles * available + int(
                        np.searchsorted(cumulative, remaining_tokens, side="left")
                    ) + 1
        project_workspace = Path(str(runtime.get("project_workspace") or "")).resolve()
        storage_root = (
            derived_asset_root(project_workspace) / "supervision-training-views"
        ).resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        destination = (storage_root / identity_hash).resolve()
        manifest_path = destination / "manifest.json"
        if manifest_path.is_file():
            raw = manifest_path.read_bytes()
            return self._result(destination, json.loads(raw), hashlib.sha256(raw).hexdigest())
        partial = (storage_root / f".{identity_hash}.{task['id']}.partial").resolve()
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        order_path = partial / "sample-order.idx"
        written = 0
        selected_raw_tokens = 0
        selected_supervised_tokens = 0
        with order_path.open("wb") as stream:
            if base_order is not None:
                logical_position = sample_offset
                remaining = selected
                current_epoch = -1
                while remaining:
                    _controlled(get_desired_state)
                    chunk_size = min(1_048_576, remaining)
                    if reshuffle_each_epoch:
                        epoch, position = divmod(logical_position, available)
                        if epoch != current_epoch:
                            base_order = np.random.Generator(np.random.PCG64(
                                np.random.SeedSequence([order_seed, epoch])
                            )).permutation(available)
                            current_epoch = epoch
                        chunk_size = min(chunk_size, available - position)
                    if consumption == "unique_only":
                        rows = base_order[logical_position : logical_position + chunk_size]
                    else:
                        positions = (
                            np.arange(chunk_size, dtype=np.uint64) + logical_position
                        ) % available
                        rows = base_order[positions]
                    encoded_rows = np.asarray(rows, dtype="<u8")
                    stream.write(encoded_rows.tobytes(order="C"))
                    assert row_raw is not None and row_supervised is not None
                    selected_raw_tokens += int(np.sum(row_raw[rows], dtype=np.uint64))
                    selected_supervised_tokens += int(
                        np.sum(row_supervised[rows], dtype=np.uint64)
                    )
                    written += len(rows)
                    logical_position += len(rows)
                    remaining -= len(rows)
            else:
                sources = [
                    {
                        **source,
                        "cursor": 0,
                        "score": 0,
                        "offset": (
                            seed + int(source.get("source_index") or 0) * 1_000_003
                        )
                        % int(source.get("row_count") or 1),
                    }
                    for source in manifest.get("sources") or []
                    if int(source.get("row_count") or 0) > 0
                ]
                if not sources:
                    sources = [
                        {
                            "source_index": 0,
                            "row_start": 0,
                            "row_count": available,
                            "weight": 1,
                            "cursor": 0,
                            "score": 0,
                            "offset": seed % available,
                        }
                    ]
                for position in range(sample_offset + selected):
                    _controlled(get_desired_state)
                    active = [
                        source
                        for source in sources
                        if consumption != "unique_only"
                        or int(source["cursor"]) < int(source["row_count"])
                    ]
                    if not active:
                        break
                    total_weight = sum(int(source["weight"]) for source in active)
                    for source in active:
                        source["score"] = int(source["score"]) + int(source["weight"])
                    chosen = max(
                        active,
                        key=lambda source: (int(source["score"]), -int(source["source_index"])),
                    )
                    chosen["score"] = int(chosen["score"]) - total_weight
                    row = int(chosen["row_start"]) + (
                        int(chosen["cursor"]) + int(chosen["offset"])
                    ) % int(chosen["row_count"])
                    chosen["cursor"] = int(chosen["cursor"]) + 1
                    if position >= sample_offset:
                        stream.write(struct.pack("<Q", row))
                        written += 1
        if base_order is None:
            selected_raw_tokens = int(manifest.get("raw_token_count") or 0)
            selected_supervised_tokens = int(manifest.get("supervised_token_count") or 0)
        view_manifest = {
            "format": payload.get("format"),
            "format_version": 1,
            "identity_hash": identity_hash,
            "identity_snapshot": payload.get("identity_snapshot") or {},
            "objective_kind": objective,
            "corpus": binding,
            "configuration": config,
            "sample_order_algorithm": (
                "sha256_seeded_numpy_pcg64_per_epoch_v1"
                if reshuffle_each_epoch else
                "sha256_seeded_numpy_pcg64_permutation_v1"
                if sample_order_policy == "global_shuffle_v1"
                else "weighted_source_offset_v1"
            ),
            "order_identity_hash": order_identity_hash,
            "sample_range": {"start": sample_offset, "end": sample_offset + written},
            "row_count": written,
            "example_count": written if objective == "sft" else 0,
            "pair_count": written if objective == "dpo" else 0,
            "raw_token_count": selected_raw_tokens,
            "supervised_token_count": selected_supervised_tokens,
            "files": {"sample_order": _file_spec(order_path)},
            "warnings": [],
        }
        manifest_bytes = _canonical_json(view_manifest)
        (partial / "manifest.json").write_bytes(manifest_bytes)
        partial.replace(destination)
        await emit("progress", {"phase": "completed", "progress": 1.0, "row_count": selected})
        return self._result(destination, view_manifest, hashlib.sha256(manifest_bytes).hexdigest())

    @staticmethod
    def _result(destination: Path, manifest: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
        size_bytes = sum(int(spec["size_bytes"]) for spec in manifest["files"].values())
        return {
            "identity_hash": manifest["identity_hash"],
            "format": manifest["format"],
            "objective_kind": manifest["objective_kind"],
            "example_count": manifest["example_count"],
            "pair_count": manifest["pair_count"],
            "raw_token_count": manifest["raw_token_count"],
            "supervised_token_count": manifest["supervised_token_count"],
            "size_bytes": size_bytes,
            "warnings": manifest.get("warnings") or [],
            "outputs": [
                {
                    "ref": "supervision_training_view",
                    "kind": "supervision_training_index",
                    "storage_type": "external",
                    "uri": destination.as_uri(),
                    "content_hash": manifest_hash,
                    "size_bytes": size_bytes,
                    "media_type": "application/vnd.forgelm.supervision-training-index-v1",
                    "availability_state": "available",
                    "metadata": {"format": manifest["format"], "identity_hash": manifest["identity_hash"]},
                }
            ],
        }
