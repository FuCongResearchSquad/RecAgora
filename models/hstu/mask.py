from __future__ import annotations

import torch


def build_target_aware_mask(
    history_valid: torch.Tensor,
    *,
    list_len: int,
    candidate_isolation: bool,
    candidate_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build context + causal-history + isolated-target attention."""

    batch_size, history_length = history_valid.shape
    total = 1 + history_length + list_len
    candidate_start = 1 + history_length
    valid = torch.ones((batch_size, total), dtype=torch.bool, device=history_valid.device)
    valid[:, 1 : 1 + history_length] = history_valid
    if candidate_valid is not None:
        if candidate_valid.dtype != torch.bool:
            raise TypeError("candidate_valid must be bool")
        if candidate_valid.shape != (batch_size, list_len):
            raise ValueError("candidate_valid must have shape [batch_size, list_len]")
        if candidate_valid.device != history_valid.device:
            raise ValueError("candidate_valid and history_valid must share a device")
        valid[:, candidate_start:] = candidate_valid
    allowed = torch.zeros((batch_size, total, total), dtype=torch.bool, device=history_valid.device)
    # Every valid history/target token may consume the request context.  The
    # context token itself remains isolated from history and targets.
    allowed[:, :, 0] = valid
    history_start = 1
    history_end = history_start + history_length
    indices = torch.arange(history_length, device=history_valid.device)
    causal = indices[None, :] <= indices[:, None]
    allowed[:, history_start:history_end, history_start:history_end] = (
        causal[None] & history_valid[:, :, None] & history_valid[:, None, :]
    )
    allowed[:, candidate_start:, history_start:history_end] = history_valid[:, None, :]
    if candidate_isolation:
        allowed[:, candidate_start:, candidate_start:] = torch.eye(
            list_len, dtype=torch.bool, device=history_valid.device
        )[None]
    else:
        allowed[:, candidate_start:, candidate_start:] = True
    allowed &= valid[:, :, None] & valid[:, None, :]
    return allowed, valid, candidate_start
