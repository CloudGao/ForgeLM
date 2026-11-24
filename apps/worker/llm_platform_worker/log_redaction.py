"""Shared redaction for API logs and local subprocess journals."""
import re
from typing import Any

MAX_LOG_VALUE_CHARS = 65_536
SENSITIVE_KEY = re.compile(r"(?:authorization|password|passwd|secret|token|api[_-]?key|access[_-]?key)", re.I)
SENSITIVE_TEXT = re.compile(
    r"(?i)\b(authorization|password|passwd|secret|token|api[_-]?key|access[_-]?key)"
    r"(\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"
)


def redact_log_value(value: Any, *, key: str = "") -> Any:
    if SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_log_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_log_value(v) for v in value]
    if isinstance(value, str):
        return SENSITIVE_TEXT.sub(r"\1\2[REDACTED]", value)[:MAX_LOG_VALUE_CHARS]
    return value
