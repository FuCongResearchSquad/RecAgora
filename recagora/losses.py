"""Candidate-aware reductions shared by opt-in ranking models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CandidateLossContext:
    """Host-side valid counts supplied by the training runtime.

    ``world_size`` compensates for averaged dense gradients when a mean loss
    uses the global valid-candidate denominator.  This object performs no
    communication and does not belong to the model's state dictionary.
    """

    local_valid_count: int
    global_valid_count: int
    world_size: int = 1

    def __post_init__(self) -> None:
        for name in ("local_valid_count", "global_valid_count", "world_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"CandidateLossContext.{name} must be an integer")
        if self.local_valid_count < 0 or self.global_valid_count < 0:
            raise ValueError("valid candidate counts must be non-negative")
        if self.local_valid_count > self.global_valid_count:
            raise ValueError("local_valid_count cannot exceed global_valid_count")
        if self.world_size < 1:
            raise ValueError("world_size must be positive")


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    context: CandidateLossContext | None = None,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Compute BCE on retained candidates without evaluating discarded labels.

    With no mask this is the original PyTorch BCE call.  With a mask, selection
    happens *before* BCE, so invalid positions cannot inject NaN via ``0 * NaN``.
    Empty selections return a graph-connected zero.  ``sum`` is never rescaled;
    ``mean`` uses local counts unless the runtime supplies a global context.
    """

    if reduction not in {"mean", "sum"}:
        raise ValueError("candidate BCE reduction must be 'mean' or 'sum'")
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("candidate BCE logits and labels must be tensors")
    if logits.shape != labels.shape:
        raise ValueError(
            "candidate BCE labels must have the same shape as logits: "
            f"{tuple(labels.shape)} != {tuple(logits.shape)}"
        )
    if context is not None and not isinstance(context, CandidateLossContext):
        raise TypeError("context must be CandidateLossContext or None")
    if valid_mask is None:
        if context is not None:
            raise ValueError("candidate loss context requires a valid_mask")
        return F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("candidate valid_mask must be a bool tensor")
    if valid_mask.shape != logits.shape:
        raise ValueError(
            "candidate valid_mask must have the same shape as logits: "
            f"{tuple(valid_mask.shape)} != {tuple(logits.shape)}"
        )
    if valid_mask.device != logits.device or labels.device != logits.device:
        raise ValueError("candidate logits, labels, and valid_mask must be on the same device")
    selected_logits = logits[valid_mask]
    selected_labels = labels[valid_mask]
    local_count = selected_logits.numel()
    if context is not None and context.local_valid_count != local_count:
        raise ValueError("candidate loss context local_valid_count does not match valid_mask")
    summed = F.binary_cross_entropy_with_logits(
        selected_logits, selected_labels, reduction="sum"
    )
    if reduction == "sum":
        return summed
    if context is None:
        return summed / max(local_count, 1)
    return summed * (context.world_size / max(context.global_valid_count, 1))
