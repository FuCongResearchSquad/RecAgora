"""A deliberately small request-ranking model implemented with plain PyTorch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class OneTransLiteBatch:
    """Natural dense-tensor inputs used by the framework-independent model."""

    query: torch.Tensor
    candidates: torch.Tensor
    history: torch.Tensor
    history_valid: torch.Tensor
    labels: torch.Tensor | None = None


class OneTransLite(nn.Module):
    """A compact history/context ranker with no RecAgora dependency."""

    def __init__(
        self,
        query_dim: int,
        candidate_dim: int,
        history_dim: int,
        model_dim: int = 32,
        num_heads: int = 4,
        ffn_dim: int = 64,
        num_tasks: int = 3,
    ) -> None:
        super().__init__()
        self.query_projection = nn.Linear(query_dim, model_dim)
        self.candidate_projection = nn.Linear(candidate_dim, model_dim)
        self.history_projection = nn.Linear(history_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.candidate_history_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
        )
        self.task_head = nn.Linear(model_dim, num_tasks)

    def forward(self, batch: OneTransLiteBatch) -> torch.Tensor:
        query = self.query_projection(batch.query).unsqueeze(1)
        candidates = self.candidate_projection(batch.candidates)
        history = self.history_projection(batch.history)

        empty_history = ~batch.history_valid.any(dim=1)
        safe_history_valid = batch.history_valid.clone()
        safe_history_valid[:, 0] = safe_history_valid[:, 0] | empty_history
        invalid_history = ~safe_history_valid
        history = history.masked_fill(empty_history[:, None, None], 0.0)
        history = self.history_encoder(
            history,
            src_key_padding_mask=invalid_history,
        )

        request_candidates = candidates + query
        history_context, _ = self.candidate_history_attention(
            request_candidates,
            history,
            history,
            key_padding_mask=invalid_history,
            need_weights=False,
        )
        history_context = history_context.masked_fill(
            empty_history[:, None, None],
            0.0,
        )
        return self.task_head(self.readout(request_candidates + history_context))

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels)
