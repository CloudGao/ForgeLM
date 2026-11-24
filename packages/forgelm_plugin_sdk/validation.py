from __future__ import annotations

from typing import Any


def check_schema(schema: dict[str, Any]) -> None:
    """Reject unsupported keywords throughout a schema, even unused branches."""
    from copy import deepcopy
    if not isinstance(schema, dict):
        raise SchemaValidationError("Schema must be an object")
    value = deepcopy(schema)
    if "type" in value and (not isinstance(value["type"], str) or value["type"] not in {"object", "array", "string", "integer", "number", "boolean", "null"}):
        raise SchemaValidationError("Schema type is unsupported")
    if "additionalProperties" in value and type(value["additionalProperties"]) is not bool:
        raise SchemaValidationError("additionalProperties must be boolean")
    if "required" in value and (not isinstance(value["required"], list) or any(not isinstance(k, str) for k in value["required"])):
        raise SchemaValidationError("required must contain property names")
    if "allOf" in value and not isinstance(value["allOf"], list):
        raise SchemaValidationError("allOf must be an array")
    if "enum" in value and (not isinstance(value["enum"], list) or not value["enum"]):
        raise SchemaValidationError("enum must be a nonempty array")
    for key in ("minItems", "maxItems", "minLength", "maxLength"):
        if key in value and (type(value[key]) is not int or value[key] < 0):
            raise SchemaValidationError(f"{key} must be a nonnegative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in value and type(value[key]) not in {float, int}:
            raise SchemaValidationError(f"{key} must be numeric")
    if value.get("$schema") == "https://json-schema.org/draft/2020-12/schema":
        value.pop("$schema")
    for key in ("properties",):
        if key in value:
            if not isinstance(value[key], dict):
                raise SchemaValidationError("properties must be an object")
            for child in value[key].values():
                check_schema(child)
    for key in ("items", "if", "then", "else"):
        if key in value:
            check_schema(value[key])
    for child in value.get("allOf", []):
        check_schema(child)
    try:
        validate_schema_instance(None, value)
    except SchemaValidationError as exc:
        if "unsupported" in str(exc):
            raise
    if "default" in value:
        validate_schema_instance(value["default"], value)


def defaults(instance: Any, schema: dict) -> Any:
    from copy import deepcopy
    if isinstance(instance, dict):
        result = deepcopy(instance)
        for key, child in schema.get("properties", {}).items():
            if key not in result and "default" in child:
                result[key] = deepcopy(child["default"])
            if key in result:
                result[key] = defaults(result[key], child)
        return result
    if isinstance(instance, list):
        return [defaults(value, schema.get("items", {})) for value in instance]
    return instance


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
        "$schema",
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
