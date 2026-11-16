"""Runtime-only candidate mask capability checks and loss normalization."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from recagora import RecAgoraSparseBatch, CandidateLossContext
from utils.runtime.distributed import DistributedContext


def require_candidate_mask_support(model: torch.nn.Module, fields: Sequence[str]) -> None:
    """Refuse opt-in filtering unless the model implements its full semantics."""
    if not fields:
        return
    public_model = getattr(model, "model", model)
    if not bool(getattr(public_model, "supports_candidate_mask", False)):
        raise ValueError(
            f"{type(public_model).__name__} does not support candidate masking; "
            "leave negative_mask_fields empty until forward, loss and auxiliary "
            "objectives have been adapted"
        )
    if getattr(public_model, "candidate_mask_all_empty_policy", None) not in {
        "skip_update", "preserve_objectives"
    }:
        raise ValueError("candidate-mask models must declare an all-empty objective policy")


def prepare_candidate_mask_step(
    batch: RecAgoraSparseBatch,
    *,
    context: DistributedContext,
    device: torch.device,
    task_names: Sequence[str],
) -> tuple[RecAgoraSparseBatch, dict[str, Any] | None]:
    """Count on CPU, then share one small count vector across ranks if enabled.

    This is called after worker prepack and before H2D/model forward.  No mask
    means no new tensor allocation or collective on the legacy path.
    """
    valid = getattr(batch, "candidate_valid_mask", None)
    if valid is None:
        return batch, None
    if valid.device.type != "cpu":
        raise ValueError("candidate-mask step counts must be prepared before batch H2D")
    if valid.dtype != torch.bool or tuple(valid.shape) != (batch.batch_size, batch.list_len):
        raise ValueError("candidate_valid_mask must be bool[B,L]")
    local_valid_count = int(valid.sum().item())
    counts = [
        valid.numel(),
        local_valid_count,
        batch.batch_size,
        int((~valid.any(dim=1)).sum().item()),
        *(int(((batch.labels[name] > 0.5) & ~valid).sum().item()) for name in task_names),
    ]
    if context.enabled:
        if not dist.is_initialized():
            raise RuntimeError("candidate-mask distributed counts need an initialized process group")
        count_tensor = torch.tensor(counts, dtype=torch.int64, device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        counts = count_tensor.cpu().tolist()
    world_size = context.world_size if context.enabled else 1
    loss_context = CandidateLossContext(
        local_valid_count=local_valid_count,
        global_valid_count=int(counts[1]),
        world_size=world_size,
    )
    return replace(batch, candidate_loss_context=loss_context), {
        "total_candidates": int(counts[0]),
        "valid_candidates": int(counts[1]),
        "masked_candidates": int(counts[0] - counts[1]),
        "requests": int(counts[2]),
        "empty_requests": int(counts[3]),
        "masked_positive_counts": {
            name: int(counts[4 + index]) for index, name in enumerate(task_names)
        },
    }


def skip_empty_candidate_update(model: torch.nn.Module, batch: RecAgoraSparseBatch) -> bool:
    loss_context = batch.candidate_loss_context
    if loss_context is None or loss_context.global_valid_count:
        return False
    public_model = getattr(model, "model", model)
    return getattr(public_model, "candidate_mask_all_empty_policy", None) == "skip_update"


def accumulate_candidate_mask_counts(
    total: dict[str, Any], current: Mapping[str, Any]
) -> None:
    for name in ("total_candidates", "valid_candidates", "masked_candidates", "requests", "empty_requests"):
        total[name] = int(total.get(name, 0)) + int(current[name])
    positives = total.setdefault("masked_positive_counts", {})
    for name, value in current["masked_positive_counts"].items():
        positives[name] = int(positives.get(name, 0)) + int(value)


def negative_mask_resume_warning(
    metadata: Mapping[str, Any], current_fields: Sequence[str]
) -> str | None:
    """Diagnose changed sampling without imposing a new checkpoint load gate."""
    train_config = metadata.get("train_config")
    if not isinstance(train_config, Mapping) or "negative_mask_fields" not in train_config:
        return (
            "checkpoint has no negative_mask_fields; resumed candidate filtering is not verified"
            if current_fields else None
        )
    saved = train_config.get("negative_mask_fields") or ()
    if sorted(set(saved)) != sorted(set(current_fields)):
        return (
            "negative_mask_fields changed on resume: "
            f"saved={list(saved)!r} current={list(current_fields)!r}; "
            "the candidate training population has changed"
        )
    return None
