from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


def validate_dense_stack(
    hidden_units: Sequence[int],
    *,
    field_name: str,
    activation: str,
    dropout: float,
) -> None:
    validate_positive_int_sequence(hidden_units, field_name=field_name)
    if activation not in {"relu", "gelu", "silu"}:
        raise ValueError("activation must be relu, gelu or silu")
    validate_dropout(dropout)


def validate_head_dimensions(hidden_dim: int, dropout: float) -> None:
    if int(hidden_dim) <= 0:
        raise ValueError("head_hidden_dim must be positive")
    validate_dropout(dropout, field_name="head_dropout")


def validate_dropout(dropout: float, *, field_name: str = "dropout") -> None:
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError(f"{field_name} must be in [0, 1)")


def validate_mlp_head(head_type: str, hidden_dim: int, dropout: float) -> None:
    if head_type not in {"linear", "mlp"}:
        raise ValueError("head_type must be linear or mlp")
    validate_head_dimensions(hidden_dim, dropout)


def normalize_output_label_names(names: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(name) for name in names)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("output_label_names must be non-empty and unique")
    return normalized


def validate_model_dim_heads(model_dim: int, num_heads: int) -> None:
    if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
        raise ValueError("model_dim must be positive and divisible by num_heads")


def validate_hstu_attention(
    *,
    model_dim: int,
    num_heads: int,
    value_dim: int,
    normalization: str,
) -> None:
    validate_model_dim_heads(model_dim, num_heads)
    if num_heads * value_dim != model_dim:
        raise ValueError("num_heads * value_dim must equal model_dim")
    if normalization not in {"token_count", "none"}:
        raise ValueError("unsupported attention_normalization")


def validate_history_padding_side(padding_side: str) -> None:
    if padding_side not in {"left", "right"}:
        raise ValueError("history_padding_side must be left or right")


def validate_masked_mean_reduction(reduction: str) -> None:
    if reduction != "masked_mean":
        raise ValueError("history_reduction must be masked_mean")


def validate_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be positive")


def finite_nonnegative_float(value: Any, name: str) -> float:
    converted = numeric_float(value, name)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def finite_positive_float(value: Any, name: str) -> float:
    converted = numeric_float(value, name)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def numeric_float(value: Any, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def validate_positive_int_sequence(
    values: Sequence[int],
    *,
    field_name: str,
) -> None:
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{field_name} must contain positive values")


def validate_disjoint_slot_groups(
    groups: Sequence[Sequence[int]],
    *,
    context: str,
) -> None:
    flattened = tuple(slot_id for group in groups for slot_id in group)
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"{context} must be disjoint")


def require_history_group(input_spec: Any, name: str) -> Any:
    history = input_spec.histories.get(name)
    if history is None:
        raise ValueError(f"unknown history group {name!r}")
    return history


def require_categorical_history_group(
    input_spec: Any,
    name: str,
    *,
    context: str,
) -> Any:
    history = require_history_group(input_spec, name)
    if history.direct_slot_ids:
        raise ValueError(f"{context} requires a categorical-only history group")
    return history


def validate_known_weights(
    weights: Mapping[str, float],
    names: Sequence[str],
    *,
    field_name: str = "loss_weights",
) -> None:
    unknown = sorted(set(weights) - set(names))
    if unknown:
        raise ValueError(f"{field_name} contains unknown tasks: {unknown}")


def validate_standard_multitask_contract(
    input_spec: Any,
    task_names: Sequence[str],
    *,
    candidate_isolation: bool,
    context: str,
) -> tuple[str, ...]:
    normalized = tuple(str(name) for name in task_names)
    if not candidate_isolation:
        raise ValueError(f"{context} requires candidate isolation")
    if normalized != tuple(input_spec.task_names):
        raise ValueError(f"{context} task names must match RecAgoraInputSpec.task_names")
    return normalized


def validate_history_order(
    history: Any,
    *,
    context: str,
    truncate_side: str | None = None,
) -> None:
    if history.event_order != "oldest_to_newest" or (
        truncate_side is not None and history.truncate_side != truncate_side
    ):
        suffix = (
            "oldest_to_newest"
            if truncate_side is None
            else f"oldest_to_newest with {truncate_side} truncation"
        )
        raise ValueError(f"{context} requires {suffix}")


def validate_left_padding(padding_side: str, *, context: str) -> None:
    if padding_side != "left":
        raise ValueError(f"{context} requires left-padded model history")


def validate_dcn_cross(cross: Mapping[str, object], fusion: str) -> str:
    cross_type = str(cross.get("type", "low_rank"))
    if cross_type not in {"low_rank", "full_rank"}:
        raise ValueError("cross.type must be low_rank or full_rank")
    if fusion not in {"parallel", "stacked"}:
        raise ValueError("fusion must be parallel or stacked")
    if cross_type == "full_rank" and "rank" in cross:
        raise ValueError("cross.rank is only valid for low_rank")
    return cross_type


def validate_din_attention_pairs(
    input_spec: Any,
    pairs: Sequence[tuple[int, int]],
) -> tuple[tuple[int, ...], int]:
    paired_history = validate_din_pair_membership(
        input_spec,
        pairs,
        context="DIN",
    )
    pair_dims = {
        (input_spec.slot(target).output_dim, input_spec.slot(history).output_dim)
        for target, history in pairs
    }
    if len(pair_dims) != 1 or next(iter(pair_dims))[0] != next(iter(pair_dims))[1]:
        raise ValueError("DIN attention pairs must share one embedding dimension")
    return paired_history, next(iter(pair_dims))[0]


def validate_din_pair_membership(
    input_spec: Any,
    pairs: Sequence[tuple[int, int]],
    *,
    context: str,
) -> tuple[int, ...]:
    paired_history = tuple(history for _, history in pairs)
    if len(set(paired_history)) != len(paired_history):
        raise ValueError(f"{context} history slots must not appear in multiple pairs")
    candidate_slots = set(input_spec.candidate.slot_ids)
    if any(target not in candidate_slots for target, _ in pairs):
        raise ValueError(f"{context} attention targets must be selected candidate slots")
    return paired_history


def require_din_pairs(
    pairs: Sequence[tuple[int, int]],
    *,
    context: str,
) -> None:
    if not pairs:
        raise ValueError(f"{context} requires attention_pairs")


def parse_din_pair(
    pair: Mapping[str, int] | Sequence[int],
    *,
    reject_unknown_mapping_keys: bool = False,
) -> tuple[int, int]:
    if isinstance(pair, Mapping):
        unknown = set(pair) - {"target", "history"}
        if (
            "target" not in pair
            or "history" not in pair
            or (reject_unknown_mapping_keys and bool(unknown))
        ):
            raise ValueError("DIN pair mapping requires target and history")
        return int(pair["target"]), int(pair["history"])
    if len(pair) != 2:
        raise ValueError("DIN pair requires target and history")
    return int(pair[0]), int(pair[1])
