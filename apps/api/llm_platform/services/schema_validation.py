from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    def __init__(self, message: str, path: tuple[str | int, ...] = ()) -> None:
        super().__init__(message)
        self.path = path


def _matches(instance: Any, schema: dict[str, Any]) -> bool:
    try:
        validate_schema_instance(instance, schema)
    except SchemaValidationError:
        return False
    return True


def validate_schema_instance(
    instance: Any,
    schema: dict[str, Any],
    path: tuple[str | int, ...] = (),
) -> None:
    """Validate the intentionally small JSON Schema subset accepted by the catalog."""
    supported = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "enum",
        "const",
        "allOf",
        "if",
        "then",
        "else",
        "description",
        "title",
        "default",
    }
    unknown = set(schema) - supported
    if unknown:
        raise SchemaValidationError(
            f"Schema uses unsupported keywords: {', '.join(sorted(unknown))}", path
        )
    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"must equal {schema['const']!r}", path)
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"must be one of {schema['enum']!r}", path)

    expected = schema.get("type")
    type_matches = {
        "object": isinstance(instance, dict),
        "array": isinstance(instance, list),
        "string": isinstance(instance, str),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "number": isinstance(instance, (int, float)) and not isinstance(instance, bool),
        "boolean": isinstance(instance, bool),
        "null": instance is None,
    }
    if expected is not None:
        if expected not in type_matches:
            raise SchemaValidationError(f"Schema type {expected!r} is unsupported", path)
        if not type_matches[expected]:
            raise SchemaValidationError(f"must be {expected}", path)

    if isinstance(instance, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        for key in required:
            if key not in instance:
                raise SchemaValidationError("is required", (*path, str(key)))
        if schema.get("additionalProperties") is False:
            extras = set(instance) - set(properties)
            if extras:
                key = sorted(extras)[0]
                raise SchemaValidationError("additional property is not allowed", (*path, key))
        for key, value in instance.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, dict):
                validate_schema_instance(value, child_schema, (*path, key))
    if isinstance(instance, list):
        if len(instance) < int(schema.get("minItems", 0)):
            raise SchemaValidationError("has too few items", path)
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            raise SchemaValidationError("has too many items", path)
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(instance):
                validate_schema_instance(item, item_schema, (*path, index))
    if isinstance(instance, str):
        if len(instance) < int(schema.get("minLength", 0)):
            raise SchemaValidationError("is too short", path)
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            raise SchemaValidationError("is too long", path)
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError(f"must be at least {schema['minimum']}", path)
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaValidationError(f"must be at most {schema['maximum']}", path)
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise SchemaValidationError(f"must be greater than {schema['exclusiveMinimum']}", path)
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            raise SchemaValidationError(f"must be less than {schema['exclusiveMaximum']}", path)

    for child in schema.get("allOf") or []:
        validate_schema_instance(instance, child, path)
    condition = schema.get("if")
    if isinstance(condition, dict):
        branch = schema.get("then") if _matches(instance, condition) else schema.get("else")
        if isinstance(branch, dict):
            validate_schema_instance(instance, branch, path)
