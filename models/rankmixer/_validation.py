from __future__ import annotations

from typing import Any

from models._validation import (
    finite_nonnegative_float as nonnegative_float,
    finite_positive_float as positive_float,
    numeric_float as _numeric_float,
)


def require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def require_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def positive_int(value: Any, name: str) -> int:
    require_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def dropout_probability(value: Any, name: str) -> float:
    converted = nonnegative_float(value, name)
    if converted >= 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return converted


def nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value
