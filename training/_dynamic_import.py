from __future__ import annotations

import importlib


def resolve_local_symbol(class_path: str, *, field: str) -> object:
    """Resolve one explicit local ``module:qualname`` without code discovery."""

    module_name, separator, qualname = str(class_path).partition(":")
    if separator != ":" or not module_name or not qualname:
        raise ValueError(f"{field} must use the form 'python.module:qualname'")
    try:
        value: object = importlib.import_module(module_name)
    except Exception as exc:
        raise ValueError(
            f"failed to import {field} module {module_name!r}: {exc}"
        ) from exc
    try:
        for attribute in qualname.split("."):
            value = getattr(value, attribute)
    except AttributeError as exc:
        raise ValueError(f"{field} {class_path!r} does not resolve") from exc
    return value
