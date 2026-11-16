from __future__ import annotations

from typing import Any

import torch

from recagora import RankingOutput


def validate_ranking_output(
    output: Any,
    *,
    task_names: tuple[str, ...],
    batch_size: int,
    list_len: int,
) -> RankingOutput:
    """Validate the model-independent output contract at the DMP boundary."""
    if type(output) is not dict:
        raise TypeError("ranking output must be a built-in dict")
    if tuple(output) != ("logits", "aux"):
        raise ValueError(
            "ranking output keys must be exactly ('logits', 'aux') in that order"
        )

    logits = output["logits"]
    if type(logits) is not dict:
        raise TypeError("ranking output['logits'] must be a built-in dict")
    if tuple(logits) != tuple(task_names):
        raise ValueError(
            "ranking logits keys must exactly match configured tasks: "
            f"expected {tuple(task_names)!r}, got {tuple(logits)!r}"
        )
    expected_shape = (int(batch_size), int(list_len))
    for name, value in logits.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"ranking logit {name!r} must be a Tensor")
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"ranking logit {name!r} must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )

    aux = output["aux"]
    if type(aux) is not dict:
        raise TypeError("ranking output['aux'] must be a built-in dict")
    _validate_aux(aux, path="aux")
    return output


def split_ranking_output(output: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Split an already validated structured ranking output."""
    if type(output) is not dict or tuple(output) != ("logits", "aux"):
        raise TypeError(
            "ranking output must be a built-in {'logits': ..., 'aux': ...} dict"
        )
    logits = output["logits"]
    aux = output["aux"]
    if type(logits) is not dict or type(aux) is not dict:
        raise TypeError("ranking output logits and aux must be built-in dicts")
    return logits, aux


def _validate_aux(value: Any, *, path: str) -> None:
    if isinstance(value, torch.Tensor):
        return
    if type(value) is dict:
        for name, item in value.items():
            if not isinstance(name, str):
                raise TypeError(f"{path} keys must be strings")
            _validate_aux(item, path=f"{path}.{name}")
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(value):
            _validate_aux(item, path=f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} must contain only Tensor leaves and built-in dict/list/tuple "
        f"containers, got {type(value).__name__}"
    )
