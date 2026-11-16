from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PerHeadSwiGLU(nn.Module):
    """Independent SwiGLU parameters for every MixFormer feature head."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = model_dim * expansion_factor
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.gate_weight = nn.Parameter(
            torch.empty(num_heads, model_dim, hidden_dim)
        )
        self.gate_bias = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.value_weight = nn.Parameter(
            torch.empty(num_heads, model_dim, hidden_dim)
        )
        self.value_bias = nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.output_weight = nn.Parameter(
            torch.empty(num_heads, hidden_dim, model_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The leading axis indexes independent linear maps. Initializing the
        # whole 3D tensor would incorrectly include that axis in Xavier fan.
        for head_index in range(self.num_heads):
            nn.init.xavier_uniform_(self.gate_weight[head_index])
            nn.init.xavier_uniform_(self.value_weight[head_index])
            nn.init.xavier_uniform_(self.output_weight[head_index])
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.value_bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate = torch.einsum(
            "...hd,hde->...he", value, self.gate_weight
        ) + self.gate_bias
        projected = torch.einsum(
            "...hd,hde->...he", value, self.value_weight
        ) + self.value_bias
        hidden = self.dropout(F.silu(gate) * projected)
        return torch.einsum("...he,hed->...hd", hidden, self.output_weight)


class SwiGLU(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = model_dim * expansion_factor
        self.gate = nn.Linear(model_dim, hidden_dim)
        self.value = nn.Linear(model_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.gate, self.value, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate(value)) * self.value(value)
        return self.output(self.dropout(hidden))


class DecoupledHeadMixing(nn.Module):
    """Paper UI mask with a physically request-shared user-side path."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_user_heads: int,
        num_item_heads: int,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_user_heads = int(num_user_heads)
        self.num_item_heads = int(num_item_heads)
        self.num_heads = self.num_user_heads + self.num_item_heads
        self.chunk_dim = self.model_dim // self.num_heads

    def forward(
        self,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, candidate_count = item_tokens.shape[:2]
        user_chunks = user_tokens.reshape(
            batch_size,
            self.num_user_heads,
            self.num_heads,
            self.chunk_dim,
        )
        item_chunks = item_tokens.reshape(
            batch_size,
            candidate_count,
            self.num_item_heads,
            self.num_heads,
            self.chunk_dim,
        )

        user_mixed = user_chunks[:, :, : self.num_user_heads, :].permute(
            0, 2, 1, 3
        )
        user_mixed = user_mixed.reshape(
            batch_size,
            self.num_user_heads,
            self.num_user_heads * self.chunk_dim,
        )
        user_mixed = F.pad(
            user_mixed,
            (0, self.num_item_heads * self.chunk_dim),
        )

        user_to_item = user_chunks[:, :, self.num_user_heads :, :].permute(
            0, 2, 1, 3
        )
        user_to_item = user_to_item[:, None].expand(-1, candidate_count, -1, -1, -1)
        item_to_item = item_chunks[:, :, :, self.num_user_heads :, :].permute(
            0, 1, 3, 2, 4
        )
        item_mixed = torch.cat((user_to_item, item_to_item), dim=3).reshape(
            batch_size,
            candidate_count,
            self.num_item_heads,
            self.model_dim,
        )
        return user_mixed, item_mixed

    def forward_reference(
        self,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Naive B*C materialization used only as a parity oracle."""
        candidate_count = int(item_tokens.shape[1])
        combined = torch.cat(
            (
                user_tokens[:, None].expand(-1, candidate_count, -1, -1),
                item_tokens,
            ),
            dim=2,
        )
        mixed = combined.reshape(
            *combined.shape[:-2],
            self.num_heads,
            self.num_heads,
            self.chunk_dim,
        ).transpose(-3, -2)
        mixed = mixed.reshape(*combined.shape[:-2], self.num_heads, self.model_dim)
        mixed = mixed.clone()
        mixed[
            :,
            :,
            : self.num_user_heads,
            self.num_user_heads * self.chunk_dim :,
        ] = 0.0
        user_mixed = mixed[:, 0, : self.num_user_heads]
        item_mixed = mixed[:, :, self.num_user_heads :]
        return user_mixed, item_mixed

class QueryMixer(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_user_heads: int,
        num_item_heads: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.mix_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.head_mixing = DecoupledHeadMixing(
            model_dim=model_dim,
            num_user_heads=num_user_heads,
            num_item_heads=num_item_heads,
        )
        self.user_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.item_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.user_ffn = PerHeadSwiGLU(
            model_dim=model_dim,
            num_heads=num_user_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.item_ffn = PerHeadSwiGLU(
            model_dim=model_dim,
            num_heads=num_item_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_mixed, item_mixed = self.head_mixing(
            self.mix_norm(user_tokens),
            self.mix_norm(item_tokens),
        )
        user_tokens = user_tokens + self.dropout(user_mixed)
        item_tokens = item_tokens + self.dropout(item_mixed)
        user_tokens = user_tokens + self.dropout(
            self.user_ffn(self.user_norm(user_tokens))
        )
        item_tokens = item_tokens + self.dropout(
            self.item_ffn(self.item_norm(item_tokens))
        )
        return user_tokens, item_tokens


class PerHeadSequenceProjection(nn.Module):
    def __init__(self, *, model_dim: int, num_heads: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_heads, model_dim, model_dim))
        # K/V projections are independent per head, not one convolutional tensor.
        for head_weight in self.weight:
            nn.init.xavier_uniform_(head_weight)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,hde->bhte", sequence, self.weight)


class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_user_heads: int,
        num_item_heads: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_user_heads = int(num_user_heads)
        num_heads = int(num_user_heads + num_item_heads)
        self.sequence_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.sequence_ffn = SwiGLU(
            model_dim=model_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.key_projection = PerHeadSequenceProjection(
            model_dim=model_dim,
            num_heads=num_heads,
        )
        self.value_projection = PerHeadSequenceProjection(
            model_dim=model_dim,
            num_heads=num_heads,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = float(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
        *,
        history_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = history_valid.unsqueeze(-1)
        sequence = sequence.masked_fill(~valid, 0.0)
        sequence = sequence + self.dropout(
            self.sequence_ffn(self.sequence_norm(sequence))
        )
        sequence = sequence.masked_fill(~valid, 0.0)
        keys = self.key_projection(sequence)
        values = self.value_projection(sequence)

        safe_valid = history_valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[:, 0] |= empty
        attention_dropout = self.attention_dropout if self.training else 0.0
        user_update = F.scaled_dot_product_attention(
            user_tokens.unsqueeze(-2),
            keys[:, : self.num_user_heads],
            values[:, : self.num_user_heads],
            attn_mask=safe_valid[:, None, None, :],
            dropout_p=attention_dropout,
        ).squeeze(-2)
        item_update = F.scaled_dot_product_attention(
            item_tokens.unsqueeze(-2),
            keys[:, None, self.num_user_heads :],
            values[:, None, self.num_user_heads :],
            attn_mask=safe_valid[:, None, None, None, :],
            dropout_p=attention_dropout,
        ).squeeze(-2)
        user_tokens = user_tokens + self.dropout(user_update)
        item_tokens = item_tokens + self.dropout(item_update)
        return sequence, user_tokens, item_tokens


class OutputFusion(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_user_heads: int,
        num_item_heads: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.user_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.item_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.user_ffn = PerHeadSwiGLU(
            model_dim=model_dim,
            num_heads=num_user_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.item_ffn = PerHeadSwiGLU(
            model_dim=model_dim,
            num_heads=num_item_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_tokens = user_tokens + self.dropout(
            self.user_ffn(self.user_norm(user_tokens))
        )
        item_tokens = item_tokens + self.dropout(
            self.item_ffn(self.item_norm(item_tokens))
        )
        return user_tokens, item_tokens


class MixFormerBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_user_heads: int,
        num_item_heads: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.query_mixer = QueryMixer(
            model_dim=model_dim,
            num_user_heads=num_user_heads,
            num_item_heads=num_item_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.cross_attention = CrossAttention(
            model_dim=model_dim,
            num_user_heads=num_user_heads,
            num_item_heads=num_item_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.output_fusion = OutputFusion(
            model_dim=model_dim,
            num_user_heads=num_user_heads,
            num_item_heads=num_item_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        user_tokens: torch.Tensor,
        item_tokens: torch.Tensor,
        *,
        history_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_tokens, item_tokens = self.query_mixer(user_tokens, item_tokens)
        sequence, user_tokens, item_tokens = self.cross_attention(
            sequence,
            user_tokens,
            item_tokens,
            history_valid=history_valid,
        )
        user_tokens, item_tokens = self.output_fusion(user_tokens, item_tokens)
        return sequence, user_tokens, item_tokens
