from __future__ import annotations

from collections.abc import Mapping

import torch


ModelOutput = torch.Tensor | Mapping[str, torch.Tensor]


def logits_by_name(
    model_output: ModelOutput,
    *,
    default_name: str,
) -> dict[str, torch.Tensor]:
    if isinstance(model_output, torch.Tensor):
        return {default_name: model_output}
    if not isinstance(model_output, Mapping):
        raise TypeError("model output must be a Tensor or a mapping of logits")

    out: dict[str, torch.Tensor] = {}
    for name, value in model_output.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model output {name!r} must be a Tensor")
        out[str(name)] = value
    if not out:
        raise ValueError("model output mapping cannot be empty")
    return out


def binary_auc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    labels = labels.detach().flatten().to(torch.float32).cpu()
    scores = scores.detach().flatten().to(torch.float32).cpu()
    if labels.numel() != scores.numel():
        raise ValueError("labels and scores must contain the same number of elements")
    if not bool(torch.isfinite(labels).all()):
        raise ValueError("labels must contain only finite values")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must contain only finite values")
    positives = labels.gt(0.5)
    positive_count = int(positives.sum().item())
    negative_count = int(labels.numel() - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None

    positive_scores = scores[positives]
    negative_scores = scores[~positives]
    # Sort the smaller class, then use vectorized binary searches to count
    # strict wins and ties for the opposite class. This is the exact pairwise
    # AUC definition, retains 0.5 tie credit, and bounds each worker's sort
    # memory better than materializing one rank/group tensor per prediction.
    if negative_count <= positive_count:
        sorted_negative_scores = torch.sort(negative_scores).values
        lower = torch.searchsorted(
            sorted_negative_scores,
            positive_scores,
            right=False,
        )
        upper = torch.searchsorted(
            sorted_negative_scores,
            positive_scores,
            right=True,
        )
        pair_wins = lower.to(torch.float64).add_(
            (upper - lower).to(torch.float64),
            alpha=0.5,
        ).sum()
    else:
        sorted_positive_scores = torch.sort(positive_scores).values
        lower = torch.searchsorted(
            sorted_positive_scores,
            negative_scores,
            right=False,
        )
        upper = torch.searchsorted(
            sorted_positive_scores,
            negative_scores,
            right=True,
        )
        pair_wins = (positive_count - upper).to(torch.float64).add_(
            (upper - lower).to(torch.float64),
            alpha=0.5,
        ).sum()
    return float(pair_wins.item() / (positive_count * negative_count))
