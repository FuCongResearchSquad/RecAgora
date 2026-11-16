from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F


def build_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError("activation must be relu, gelu or silu")


class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        model_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.model_dim = int(model_dim)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = build_activation(activation)
        self.output_projection = nn.Linear(self.hidden_dim, self.model_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output_projection(self.activation(self.input_projection(value)))


class FieldWiseTokenizer(nn.Module):
    """One independent tokenizer per non-behavioral feature field."""

    def __init__(
        self,
        *,
        input_dims: Sequence[int],
        hidden_dim: int,
        model_dim: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.hidden_dim = int(hidden_dim)
        self.model_dim = int(model_dim)
        self.tokenizers = nn.ModuleList(
            FeatureTokenizer(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                model_dim=self.model_dim,
                activation=activation,
            )
            for input_dim in self.input_dims
        )

    def forward(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(values) != len(self.tokenizers):
            raise ValueError("EST-LCA field tokenizer received the wrong field count")
        return torch.stack(
            tuple(
                tokenizer(value)
                for tokenizer, value in zip(self.tokenizers, values, strict=True)
            ),
            dim=-2,
        )


class TokenSpecificLinear(nn.Module):
    """Independent linear weights for each non-behavioral token identity."""

    def __init__(
        self,
        *,
        num_tokens: int,
        input_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight = nn.Parameter(
            torch.empty(self.num_tokens, self.input_dim, self.output_dim)
        )
        for token_weight in self.weight:
            nn.init.xavier_uniform_(token_weight)

    def forward(self, value: torch.Tensor, *, token_offset: int) -> torch.Tensor:
        token_count = int(value.shape[-2])
        end = int(token_offset) + token_count
        if token_offset < 0 or end > self.num_tokens:
            raise ValueError("EST-LCA token-specific projection slice is out of range")
        return torch.einsum(
            "...ld,ldh->...lh",
            value,
            self.weight[int(token_offset) : end],
        )


class TokenSpecificFFN(nn.Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        model_dim: int,
        hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.model_dim = int(model_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.model_dim, self.hidden_dim)
        )
        self.input_bias = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.output_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_dim, self.model_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(self.num_tokens, self.model_dim))
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_weight in self.input_weight:
            nn.init.xavier_uniform_(token_weight)
        for token_weight in self.output_weight:
            nn.init.xavier_uniform_(token_weight)
        nn.init.zeros_(self.input_bias)
        nn.init.zeros_(self.output_bias)

    def forward(self, value: torch.Tensor, *, token_offset: int) -> torch.Tensor:
        token_count = int(value.shape[-2])
        end = int(token_offset) + token_count
        if token_offset < 0 or end > self.num_tokens:
            raise ValueError("EST-LCA token-specific FFN slice is out of range")
        hidden = torch.einsum(
            "...ld,ldh->...lh",
            value,
            self.input_weight[int(token_offset) : end],
        ) + self.input_bias[int(token_offset) : end]
        hidden = self.dropout(self.activation(hidden))
        return torch.einsum(
            "...lh,lhd->...ld",
            hidden,
            self.output_weight[int(token_offset) : end],
        ) + self.output_bias[int(token_offset) : end]


class SharedFFN(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(model_dim, hidden_dim)
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, model_dim)
        for layer in (self.input_projection, self.output_projection):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.input_projection(value)))
        return self.output_projection(hidden)


class LightweightCrossAttention(nn.Module):
    """Non-behavioral queries over request-shared behavioral K/V."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        num_non_behavior_tokens: int,
        dropout: float,
        attention_backend: str,
    ) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError("EST-LCA model_dim must be divisible by num_heads")
        if attention_backend not in {"math", "sdpa"}:
            raise ValueError("EST-LCA attention_backend must be math or sdpa")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.attention_backend = str(attention_backend)
        self.attention_dropout = float(dropout)
        self.query_projection = TokenSpecificLinear(
            num_tokens=num_non_behavior_tokens,
            input_dim=self.model_dim,
            output_dim=self.model_dim,
        )
        self.output_projection = TokenSpecificLinear(
            num_tokens=num_non_behavior_tokens,
            input_dim=self.model_dim,
            output_dim=self.model_dim,
        )
        self.key_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.value_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)

    def project_history(
        self,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._split_heads(self.key_projection(history)),
            self._split_heads(self.value_projection(history)),
        )

    def attend(
        self,
        tokens: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        history_valid: torch.Tensor,
        token_offset: int,
    ) -> torch.Tensor:
        query = self._split_heads(
            self.query_projection(tokens, token_offset=token_offset)
        )
        while keys.ndim < query.ndim:
            keys = keys.unsqueeze(1)
            values = values.unsqueeze(1)
        safe_valid = history_valid.to(dtype=torch.bool).clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[:, 0] |= empty
        mask = safe_valid.reshape(
            safe_valid.shape[0],
            *(1 for _ in range(query.ndim - 2)),
            safe_valid.shape[1],
        )
        if self.attention_backend == "math":
            attended = self._math_attention(query, keys, values, mask)
        else:
            attended = F.scaled_dot_product_attention(
                query,
                keys,
                values,
                attn_mask=mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
        attended = attended.transpose(-3, -2).contiguous().reshape(
            *tokens.shape[:-2],
            tokens.shape[-2],
            self.model_dim,
        )
        return self.output_projection(attended, token_offset=token_offset)

    def _math_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        score_dtype = (
            torch.float32
            if query.dtype in {torch.float16, torch.bfloat16}
            else query.dtype
        )
        scores = torch.matmul(
            query.to(score_dtype),
            keys.to(score_dtype).transpose(-2, -1),
        ) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(~mask, torch.finfo(score_dtype).min)
        weights = torch.softmax(scores, dim=-1).to(values.dtype)
        weights = F.dropout(
            weights,
            p=self.attention_dropout,
            training=self.training,
        )
        return torch.matmul(weights, values)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.view(
            *value.shape[:-2],
            value.shape[-2],
            self.num_heads,
            self.head_dim,
        ).transpose(-3, -2)


class ESTLCABlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        query_token_count: int,
        candidate_token_count: int,
        expansion_factor: int,
        ffn_activation: str,
        dropout: float,
        norm_eps: float,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.query_token_count = int(query_token_count)
        self.candidate_token_count = int(candidate_token_count)
        total_tokens = self.query_token_count + self.candidate_token_count
        hidden_dim = int(model_dim) * int(expansion_factor)
        self.attention_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.lca = LightweightCrossAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_non_behavior_tokens=total_tokens,
            dropout=dropout,
            attention_backend=attention_backend,
        )
        self.non_behavior_ffn = TokenSpecificFFN(
            num_tokens=total_tokens,
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            activation=ffn_activation,
            dropout=dropout,
        )
        self.history_ffn = SharedFFN(
            model_dim=model_dim,
            hidden_dim=hidden_dim,
            activation=ffn_activation,
            dropout=dropout,
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        history: torch.Tensor,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        *,
        history_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = history_valid.unsqueeze(-1)
        history = history.masked_fill(~valid, 0.0)
        normalized_history = self.attention_norm(history).masked_fill(~valid, 0.0)
        keys, values = self.lca.project_history(normalized_history)
        query_tokens = query_tokens + self.residual_dropout(
            self.lca.attend(
                self.attention_norm(query_tokens),
                keys,
                values,
                history_valid=history_valid,
                token_offset=0,
            )
        )
        candidate_tokens = candidate_tokens + self.residual_dropout(
            self.lca.attend(
                self.attention_norm(candidate_tokens),
                keys,
                values,
                history_valid=history_valid,
                token_offset=self.query_token_count,
            )
        )

        normalized_history = self.ffn_norm(history).masked_fill(~valid, 0.0)
        history = history + self.residual_dropout(
            self.history_ffn(normalized_history)
        )
        query_tokens = query_tokens + self.residual_dropout(
            self.non_behavior_ffn(
                self.ffn_norm(query_tokens),
                token_offset=0,
            )
        )
        candidate_tokens = candidate_tokens + self.residual_dropout(
            self.non_behavior_ffn(
                self.ffn_norm(candidate_tokens),
                token_offset=self.query_token_count,
            )
        )
        history = history.masked_fill(~valid, 0.0)
        return history, query_tokens, candidate_tokens


class PartitionedTaskHead(nn.Module):
    """One MLP head with an exactly partitioned first affine transform."""

    def __init__(
        self,
        *,
        request_input_dim: int,
        candidate_input_dim: int,
        hidden_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.request_input_dim = int(request_input_dim)
        self.candidate_input_dim = int(candidate_input_dim)
        self.first = nn.Linear(
            self.request_input_dim + self.candidate_input_dim,
            hidden_dim,
        )
        self.activation = build_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.first.weight)
        nn.init.zeros_(self.first.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        request_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        request_hidden = F.linear(
            request_features,
            self.first.weight[:, : self.request_input_dim],
            None,
        )
        candidate_hidden = F.linear(
            candidate_features,
            self.first.weight[:, self.request_input_dim :],
            self.first.bias,
        )
        hidden = request_hidden.unsqueeze(1) + candidate_hidden
        return self.output(self.dropout(self.activation(hidden))).squeeze(-1)

    def forward_concat_reference(
        self,
        request_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = int(candidate_features.shape[1])
        combined = torch.cat(
            (
                request_features[:, None].expand(-1, candidate_count, -1),
                candidate_features,
            ),
            dim=-1,
        )
        hidden = self.first(combined)
        return self.output(self.dropout(self.activation(hidden))).squeeze(-1)
