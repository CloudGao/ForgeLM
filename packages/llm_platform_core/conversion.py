from __future__ import annotations

from typing import Any

CONVERSION_ENGINE_VERSION = "document-conversion-v1"

DEFAULT_CONVERSION_OPERATORS: list[dict[str, dict[str, Any]]] = [
    {"detect_format": {}},
    {"extract_pdf": {}},
    {"extract_docx": {"include_tables": True}},
    {"extract_epub": {}},
    {"normalize_whitespace": {"max_blank_lines": 1}},
    {"filter_empty": {"min_characters": 1}},
]

CONVERSION_OPERATOR_CATALOG: list[dict[str, Any]] = [
    {
        "name": "detect_format",
        "label": "自动识别格式",
        "category": "识别",
        "description": "按文件签名和容器结构识别 PDF、DOCX、EPUB。",
        "parameters": {},
    },
    {
        "name": "extract_pdf",
        "label": "提取 PDF 文本",
        "category": "抽取",
        "description": "从文本型 PDF 逐页提取正文；不执行 OCR。",
        "parameters": {},
    },
    {
        "name": "extract_docx",
        "label": "提取 DOCX",
        "category": "抽取",
        "description": "按文档顺序提取段落，可包含表格单元格。",
        "parameters": {"include_tables": {"type": "boolean", "default": True}},
    },
    {
        "name": "extract_epub",
        "label": "提取 EPUB",
        "category": "抽取",
        "description": "按 EPUB spine 阅读顺序提取无 DRM 章节正文。",
        "parameters": {},
    },
    {
        "name": "normalize_whitespace",
        "label": "规范空白",
        "category": "加工",
        "description": "统一换行、移除行尾空白并限制连续空行。",
        "parameters": {
            "max_blank_lines": {"type": "integer", "minimum": 0, "maximum": 10, "default": 1}
        },
    },
    {
        "name": "filter_empty",
        "label": "过滤空记录",
        "category": "加工",
        "description": "移除空白或低于最小字符数的记录。",
        "parameters": {
            "min_characters": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1_000_000,
                "default": 1,
            }
        },
    },
    {
        "name": "chunk_text",
        "label": "定长文本分块",
        "category": "加工",
        "description": "对长记录做确定性字符窗口切分，并保留来源定位。",
        "parameters": {
            "max_characters": {
                "type": "integer",
                "minimum": 256,
                "maximum": 1_000_000,
                "default": 8_192,
            },
            "overlap_characters": {
                "type": "integer",
                "minimum": 0,
                "maximum": 999_999,
                "default": 0,
            },
        },
    },
]

_EXTRACTORS = {"extract_pdf", "extract_docx", "extract_epub"}
_POST_PROCESSORS = {"normalize_whitespace", "filter_empty", "chunk_text"}
_KNOWN_OPERATORS = {item["name"] for item in CONVERSION_OPERATOR_CATALOG}


def _integer_parameter(
    parameters: dict[str, Any], name: str, default: int, minimum: int, maximum: int
) -> int:
    raw = parameters.get(name, default)
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def validate_conversion_operators(
    operators: list[dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    if not isinstance(operators, list) or not operators:
        raise ValueError("conversion recipe must contain at least one operator")
    if len(operators) > 32:
        raise ValueError("conversion recipe cannot contain more than 32 operators")

    normalized: list[dict[str, dict[str, Any]]] = []
    names: list[str] = []
    for index, operator in enumerate(operators):
        if not isinstance(operator, dict) or len(operator) != 1:
            raise ValueError(f"operator {index + 1} must contain exactly one operator name")
        name, raw_parameters = next(iter(operator.items()))
        if name not in _KNOWN_OPERATORS:
            raise ValueError(f"operator {index + 1} is not supported: {name}")
        if raw_parameters is None:
            parameters: dict[str, Any] = {}
        elif isinstance(raw_parameters, dict):
            parameters = dict(raw_parameters)
        else:
            raise ValueError(f"operator {index + 1} parameters must be an object")

        allowed: set[str]
        if name == "extract_docx":
            allowed = {"include_tables"}
            include_tables = parameters.get("include_tables", True)
            if not isinstance(include_tables, bool):
                raise ValueError("include_tables must be a boolean")
            parameters = {"include_tables": include_tables}
        elif name == "normalize_whitespace":
            allowed = {"max_blank_lines"}
            parameters = {
                "max_blank_lines": _integer_parameter(parameters, "max_blank_lines", 1, 0, 10)
            }
        elif name == "filter_empty":
            allowed = {"min_characters"}
            parameters = {
                "min_characters": _integer_parameter(parameters, "min_characters", 1, 1, 1_000_000)
            }
        elif name == "chunk_text":
            allowed = {"max_characters", "overlap_characters"}
            maximum = _integer_parameter(parameters, "max_characters", 8_192, 256, 1_000_000)
            overlap = _integer_parameter(parameters, "overlap_characters", 0, 0, 999_999)
            if overlap >= maximum:
                raise ValueError("overlap_characters must be smaller than max_characters")
            parameters = {"max_characters": maximum, "overlap_characters": overlap}
        else:
            allowed = set()
            parameters = {}
        unexpected = sorted(set(raw_parameters or {}) - allowed)
        if unexpected:
            raise ValueError(f"operator {name} has unsupported parameters: {unexpected}")
        if name in names:
            raise ValueError(f"conversion operator cannot be repeated: {name}")
        names.append(name)
        normalized.append({name: parameters})

    if names[0] != "detect_format":
        raise ValueError("detect_format must be the first conversion operator")
    if not (_EXTRACTORS & set(names)):
        raise ValueError("conversion recipe must contain at least one document extractor")
    first_post = next((index for index, name in enumerate(names) if name in _POST_PROCESSORS), None)
    if first_post is not None and any(name in _EXTRACTORS for name in names[first_post + 1 :]):
        raise ValueError("document extractors must appear before processing operators")
    return normalized


def conversion_operator_names(operators: list[dict[str, Any]]) -> list[str]:
    return [str(next(iter(operator))) for operator in validate_conversion_operators(operators)]
