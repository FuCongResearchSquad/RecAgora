from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from models._validation import validate_positive_int


class HSTUTokenPositionTimeEncoder(nn.Module):
    """DLRM-HSTU-style additive token encoding.

    Position and history-time embeddings are added once before the STU stack.
    Candidate targets share one position and receive no time embedding because
    the current request contract has no candidate/query exposure timestamp.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        position_bucket_count: int,
        time_bucket_count: int,
        time_bucket_transform: str,
    ) -> None:
        super().__init__()
        validate_positive_int(
            position_bucket_count,
            field_name="position_bucket_count",
        )
        validate_positive_int(time_bucket_count, field_name="time_bucket_count")
        if time_bucket_transform not in {"log1p", "identity"}:
            raise ValueError("unsupported time_bucket_transform")
        self.position_bucket_count = int(position_bucket_count)
        self.time_bucket_count = int(time_bucket_count)
        self.time_bucket_transform = str(time_bucket_transform)
        self.position = nn.Embedding(position_bucket_count, model_dim)
        self.time = nn.Embedding(time_bucket_count, model_dim)
        position_bound = math.sqrt(1.0 / float(position_bucket_count))
        time_bound = math.sqrt(1.0 / float(time_bucket_count))
        nn.init.uniform_(self.position.weight, -position_bound, position_bound)
        nn.init.uniform_(self.time.weight, -time_bound, time_bound)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        history_valid: torch.Tensor,
        timestamps: torch.Tensor,
        time_gaps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        history_length = history_valid.shape[1]
        candidate_count = token_count - history_length - 1

        target_position = history_length + 1
        positions = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=tokens.device),
                torch.arange(1, history_length + 1, device=tokens.device),
                torch.full(
                    (candidate_count,),
                    target_position,
                    dtype=torch.long,
                    device=tokens.device,
                ),
            )
        )
        encoded = tokens + self.position(positions)[None]

        if history_length == 0:
            return encoded
        # Histories are left padded and ordered oldest -> newest.  Without a
        # request/candidate exposure timestamp, the newest history event is
        # the only honest temporal anchor available to this ranker.
        latest_timestamp = timestamps[:, -1:]
        elapsed = (latest_timestamp - timestamps).clamp_min(0.0) + time_gaps.abs()
        if self.time_bucket_transform == "log1p":
            elapsed = torch.log1p(elapsed)
        time_buckets = elapsed.clamp(0.0, float(self.time_bucket_count - 1)).long()
        history_time = self.time(time_buckets)
        history_time = history_time * history_valid[..., None].to(history_time.dtype)
        time_encoding = tokens.new_zeros((batch_size, token_count, tokens.shape[-1]))
        time_encoding[:, 1 : 1 + history_length] = history_time
        return encoded + time_encoding


class HSTUBlock(nn.Module):
    """Dense PyTorch DLRM-HSTU-style STU; no ragged/fused-kernel claim."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        attention_dim: int,
        value_dim: int,
        dropout: float,
        attention_normalization: str,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.attention_dim = int(attention_dim)
        self.value_dim = int(value_dim)
        self.attention_normalization = str(attention_normalization)
        self.attention_scale = self.attention_dim**-0.5
        self.qk_width = self.num_heads * self.attention_dim
        self.value_width = self.num_heads * self.value_dim
        self.uvqk = nn.Linear(model_dim, self.value_width * 2 + self.qk_width * 2)
        self.input_norm = nn.LayerNorm(model_dim, eps=1.0e-6)
        self.attention_norm = nn.LayerNorm(self.value_width, eps=1.0e-6)
        self.output = nn.Linear(self.value_width * 3, model_dim)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.uvqk, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        value: torch.Tensor,
        *,
        allowed: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.input_norm(value)
        u, v, q, k = torch.split(
            self.uvqk(normalized),
            [self.value_width, self.value_width, self.qk_width, self.qk_width],
            dim=-1,
        )
        u = F.silu(u)
        batch_size, token_count, _ = value.shape
        q = q.reshape(batch_size, token_count, self.num_heads, self.attention_dim)
        k = k.reshape(batch_size, token_count, self.num_heads, self.attention_dim)
        v = v.reshape(batch_size, token_count, self.num_heads, self.value_dim)
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.attention_scale
        attention = F.silu(scores.masked_fill(~allowed[:, None], 0.0))
        attention = attention * allowed[:, None].to(attention.dtype)
        if self.attention_normalization == "token_count":
            attention = attention / float(token_count)
        attended = torch.einsum("bhts,bshd->bthd", attention, v).reshape(
            batch_size, token_count, self.value_width
        )
        gated = self.attention_norm(attended) * u
        projected = self.output(torch.cat((u, value[..., : self.value_width], gated), dim=-1))
        return value + self.dropout(projected) * valid[..., None].to(projected.dtype)
