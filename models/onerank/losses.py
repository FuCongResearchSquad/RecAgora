"""Loss functions shared by configurable OneRank architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import torch

from recagora import CandidateLossContext

from .blocks import listwise_softmax_cross_entropy


def pointwise_binary_cross_entropy_sum(
    labels: torch.Tensor,
    logits: torch.Tensor,
    *,
    epsilon: float = 1.0e-7,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """TF source probability-clipped BCE summed over requests and items."""
    if valid_mask is not None:
        logits = logits[valid_mask]
        labels = labels[valid_mask]
    probabilities = torch.sigmoid(logits.to(torch.float32)).clamp(
        min=float(epsilon),
        max=1.0 - float(epsilon),
    )
    targets = labels.to(device=logits.device, dtype=torch.float32)
    return -(
        targets * probabilities.log()
        + (1.0 - targets) * (1.0 - probabilities).log()
    ).sum()


def ranking_task_losses(
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    *,
    point_loss_weight: float,
    list_loss_weight: float,
    task_loss_weights: Mapping[str, float],
    candidate_valid_mask: torch.Tensor | None = None,
    candidate_loss_context: CandidateLossContext | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum the source's pointwise BCE and request-wise softmax CE per task."""
    _validate_candidate_mask(
        logits,
        candidate_valid_mask,
        candidate_loss_context,
    )
    components: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for name, task_logits in logits.items():
        task_labels = labels[name]
        point = pointwise_binary_cross_entropy_sum(
            task_labels,
            task_logits,
            valid_mask=candidate_valid_mask,
        )
        listwise = listwise_softmax_cross_entropy(
            task_labels,
            task_logits,
            valid_mask=candidate_valid_mask,
        )
        task_total = float(point_loss_weight) * point + float(list_loss_weight) * listwise
        weighted = float(task_loss_weights.get(name, 1.0)) * task_total
        components[f"{name}/pointwise"] = point
        components[f"{name}/listwise"] = listwise
        components[f"{name}/total"] = weighted
        total = weighted if total is None else total + weighted
    return cast(torch.Tensor, total), components


def _validate_candidate_mask(
    logits: Mapping[str, torch.Tensor],
    valid_mask: torch.Tensor | None,
    context: CandidateLossContext | None,
) -> None:
    if valid_mask is None:
        if context is not None:
            raise ValueError("candidate loss context requires a valid mask")
        return
    if valid_mask.dtype != torch.bool:
        raise TypeError("candidate valid mask must be bool")
    if any(value.shape != valid_mask.shape for value in logits.values()):
        raise ValueError("candidate valid mask must match every task logits shape")
    if any(value.device != valid_mask.device for value in logits.values()):
        raise ValueError("candidate valid mask and logits must share a device")
    if (
        context is not None
        and context.local_valid_count != int(valid_mask.sum().item())
    ):
        raise ValueError("candidate loss context local count does not match valid mask")


@dataclass(frozen=True)
class NTPResult:
    """NTP loss and metrics produced by the TensorFlow in-batch objective."""

    loss: torch.Tensor
    positive_score_mean: torch.Tensor
    negative_score_mean: torch.Tensor
    positive_negative_difference_mean: torch.Tensor
    recalls: Mapping[int, torch.Tensor]


def next_item_targets(sequence: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Create the source's next-position target sequence and validity mask."""
    return (
        torch.cat((sequence[:, 1:], sequence[:, -1:]), dim=1),
        torch.cat((valid[:, 1:], torch.zeros_like(valid[:, :1])), dim=1),
    )


def first_future_match_targets(
    match: torch.Tensor,
    sequence: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the earliest valid future target matching each history position."""
    batch_size, sequence_length = valid.shape
    future = torch.triu(
        torch.ones((sequence_length, sequence_length), device=valid.device, dtype=torch.bool),
        diagonal=1,
    )
    candidates = match.to(torch.bool) & future[None] & valid[:, None, :]
    positions = torch.arange(sequence_length, device=valid.device)[None, None, :]
    sentinel = torch.full_like(positions, sequence_length)
    target_index = torch.where(candidates, positions, sentinel).amin(dim=-1)
    target_valid = target_index < sequence_length
    safe_index = target_index.clamp_max(sequence_length - 1)
    targets = torch.gather(
        sequence,
        dim=1,
        index=safe_index[..., None].expand(batch_size, sequence_length, sequence.shape[-1]),
    )
    return targets, target_valid


def in_batch_sampled_softmax(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    current_valid: torch.Tensor,
    target_valid: torch.Tensor,
    temperature: torch.Tensor | float,
    recall_ks: Sequence[int] = (1, 5, 10),
    dedup_threshold: float | None = None,
) -> NTPResult:
    """Port of OneRank's per-position in-batch sampled-softmax NTP loss.

    The batch axis supplies negatives, and each sequence contributes the mean
    over its valid positions before losses are summed over the batch.  This is
    intentionally not a global mean reduction.
    """
    # [S, B(query), B(key)] as in the TensorFlow implementation.
    logits = torch.einsum("lbd,lcd->lbc", hidden.transpose(0, 1), targets.transpose(0, 1)) / temperature
    valid = current_valid.transpose(0, 1).to(dtype=logits.dtype) * target_valid.transpose(0, 1).to(dtype=logits.dtype)
    batch_size = logits.shape[1]
    eye = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
    negative = (~eye)[None].expand_as(logits)
    negative = negative & current_valid.transpose(0, 1)[:, :, None]
    negative = negative & target_valid.transpose(0, 1)[:, None, :]

    dedup = torch.zeros_like(negative)
    if dedup_threshold is not None:
        key_similarity = torch.matmul(
            targets.detach().transpose(0, 1),
            targets.detach().transpose(0, 1).transpose(-1, -2),
        )
        dedup = (key_similarity > float(dedup_threshold)) & (~eye)[None]
        negative = negative & ~dedup

    positive = logits.diagonal(dim1=1, dim2=2)
    negative_logits = logits.masked_fill(~negative, -1.0e6)
    denominator = torch.logaddexp(positive, torch.logsumexp(negative_logits, dim=-1))
    loss_by_position = (denominator - positive) * valid
    valid_by_sample = valid.transpose(0, 1).sum(dim=1).clamp_min(1.0)
    loss = (loss_by_position.transpose(0, 1).sum(dim=1) / valid_by_sample).sum()

    positive_valid = positive.transpose(0, 1) * valid.transpose(0, 1)
    positive_mean = (positive_valid.sum(dim=1) / valid_by_sample).mean()
    negative_scores = logits.masked_fill(~negative, 0.0)
    negative_count = negative.to(dtype=logits.dtype).sum(dim=-1).clamp_min(1.0)
    negative_mean_by_position = negative_scores.sum(dim=-1) / negative_count
    negative_valid = negative_mean_by_position.transpose(0, 1) * valid.transpose(0, 1)
    negative_mean = (negative_valid.sum(dim=1) / valid_by_sample).mean()
    difference_mean = ((positive_valid - negative_valid).sum(dim=1) / valid_by_sample).mean()

    recalls: dict[int, torch.Tensor] = {}
    recall_logits = logits.masked_fill(dedup, -1.0e9)
    for value in recall_ks:
        if value <= 0:
            raise ValueError("recall_ks must be positive")
        top_k = min(int(value), batch_size)
        selected = recall_logits.topk(top_k, dim=-1).indices
        hit = (selected == torch.arange(batch_size, device=logits.device)[None, :, None]).any(dim=-1)
        recalls[int(value)] = (hit.to(dtype=logits.dtype) * valid).sum() / valid.sum().clamp_min(1.0)
    return NTPResult(loss, positive_mean, negative_mean, difference_mean, recalls)
