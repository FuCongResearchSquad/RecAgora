from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def build_cross_attention_pattern(
    *,
    history_length: int,
    merge_group_size: int,
    recent_query_size: int,
    num_ns_tokens: int,
) -> torch.Tensor:
    """Return the static LONGER cross-attention pattern with True == allowed."""

    merged_length = history_length // merge_group_size
    recent_positions = torch.arange(
        history_length - recent_query_size,
        history_length,
    )
    merged_end_positions = (
        torch.arange(merged_length) + 1
    ) * merge_group_size - 1
    recent_to_merged = (
        merged_end_positions.unsqueeze(0) <= recent_positions.unsqueeze(1)
    )
    query_length = recent_query_size + num_ns_tokens
    key_length = merged_length + num_ns_tokens
    allowed = torch.zeros((query_length, key_length), dtype=torch.bool)
    allowed[:recent_query_size, :merged_length] = recent_to_merged
    allowed[recent_query_size:, :] = True
    return allowed


class AutoSplitNSTokenizer(nn.Module):
    """Generate candidate-private non-sequence tokens from full Q+C features."""

    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int,
        num_tokens: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.num_tokens = int(num_tokens)
        self.token_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.input_dim, self.model_dim),
                nn.SiLU(),
                nn.LayerNorm(self.model_dim, eps=float(layer_norm_eps)),
            )
            for _ in range(self.num_tokens)
        )
        for token_mlp in self.token_mlps:
            linear = token_mlp[0]
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = int(candidates.shape[1])
        features = torch.cat(
            (
                query[:, None, :].expand(-1, candidate_count, -1),
                candidates,
            ),
            dim=-1,
        )
        return torch.stack(
            tuple(token_mlp(features) for token_mlp in self.token_mlps),
            dim=2,
        )


class InnerTrans(nn.Module):
    """Masked adjacent-group attention followed by concat-FFN token merge."""

    def __init__(
        self,
        *,
        model_dim: int,
        merge_group_size: int,
        expansion_factor: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.merge_group_size = int(merge_group_size)
        merged_input_dim = self.model_dim * self.merge_group_size
        self.input_norm = nn.LayerNorm(self.model_dim, eps=float(layer_norm_eps))
        self.query_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.key_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.value_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.merge_ffn = nn.Sequential(
            nn.Linear(
                merged_input_dim,
                merged_input_dim * int(expansion_factor),
            ),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(
                merged_input_dim * int(expansion_factor),
                self.model_dim,
            ),
            nn.Dropout(float(dropout)),
        )
        self.attention_dropout = float(dropout)
        _initialize_linears(self)

    def _reshape_groups(
        self,
        events: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_length, model_dim = events.shape
        merged_length = history_length // self.merge_group_size
        return (
            events.reshape(
                batch_size * merged_length,
                self.merge_group_size,
                model_dim,
            ),
            valid.reshape(
                batch_size * merged_length,
                self.merge_group_size,
            ),
        )

    def forward(
        self,
        events: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_length, model_dim = events.shape
        if history_length % self.merge_group_size:
            raise ValueError(
                "LONGER history length must be divisible by merge_group_size"
            )
        normalized = self.input_norm(events).masked_fill(
            ~valid.unsqueeze(-1),
            0.0,
        )
        grouped, group_valid = self._reshape_groups(normalized, valid)
        query = self.query_projection(grouped).unsqueeze(1)
        key = self.key_projection(grouped).unsqueeze(1)
        value = self.value_projection(grouped).unsqueeze(1)
        allowed = group_valid[:, :, None] & group_valid[:, None, :]
        attended = _scaled_dot_product_attention(
            query,
            key,
            value,
            allowed=allowed,
            dropout=self.attention_dropout,
            training=self.training,
        ).squeeze(1)
        attended = attended.masked_fill(~group_valid.unsqueeze(-1), 0.0)
        merged_length = history_length // self.merge_group_size
        concatenated = attended.reshape(
            batch_size,
            merged_length,
            self.merge_group_size * model_dim,
        )
        merged = self.merge_ffn(concatenated)
        merged_valid = group_valid.any(dim=1).reshape(batch_size, merged_length)
        return (
            merged.masked_fill(~merged_valid.unsqueeze(-1), 0.0),
            merged_valid,
        )


class HistoryCandidateCrossAttention(nn.Module):
    """Causal recent-history compression with candidate-global NS queries."""

    def __init__(
        self,
        *,
        history_length: int,
        merge_group_size: int,
        recent_query_size: int,
        num_ns_tokens: int,
        model_dim: int,
        num_heads: int,
        expansion_factor: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.input_norm = nn.LayerNorm(self.model_dim, eps=float(layer_norm_eps))
        self.query_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.key_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.value_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.ffn = _FeedForward(
            model_dim=self.model_dim,
            expansion_factor=expansion_factor,
            activation=activation,
            dropout=dropout,
        )
        self.attention_dropout = float(dropout)
        self.register_buffer(
            "structural_allowed",
            build_cross_attention_pattern(
                history_length=history_length,
                merge_group_size=merge_group_size,
                recent_query_size=recent_query_size,
                num_ns_tokens=num_ns_tokens,
            ),
            persistent=False,
        )
        _initialize_linears(self)

    def forward(
        self,
        query_tokens: torch.Tensor,
        key_value_tokens: torch.Tensor,
        *,
        query_valid: torch.Tensor,
        key_value_valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized_query = self.input_norm(query_tokens)
        normalized_key_value = self.input_norm(key_value_tokens)
        query = _split_heads(
            self.query_projection(normalized_query),
            self.num_heads,
        )
        key = _split_heads(
            self.key_projection(normalized_key_value),
            self.num_heads,
        )
        value = _split_heads(
            self.value_projection(normalized_key_value),
            self.num_heads,
        )
        allowed = (
            self.structural_allowed.to(device=query_tokens.device)[None]
            & key_value_valid[:, None, :]
        )
        attended = _merge_heads(
            _scaled_dot_product_attention(
                query,
                key,
                value,
                allowed=allowed,
                dropout=self.attention_dropout,
                training=self.training,
            )
        )
        output = query_tokens + self.ffn(attended)
        return output.masked_fill(~query_valid.unsqueeze(-1), 0.0)


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        model_dim: int,
        num_heads: int,
        expansion_factor: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.input_norm = nn.LayerNorm(self.model_dim, eps=float(layer_norm_eps))
        self.query_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.key_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.value_projection = nn.Linear(self.model_dim, self.model_dim, bias=False)
        self.ffn = _FeedForward(
            model_dim=self.model_dim,
            expansion_factor=expansion_factor,
            activation=activation,
            dropout=dropout,
        )
        self.attention_dropout = float(dropout)
        self.register_buffer(
            "causal_allowed",
            torch.ones((sequence_length, sequence_length), dtype=torch.bool).tril(),
            persistent=False,
        )
        _initialize_linears(self)

    def forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.input_norm(tokens)
        query = _split_heads(self.query_projection(normalized), self.num_heads)
        key = _split_heads(self.key_projection(normalized), self.num_heads)
        value = _split_heads(self.value_projection(normalized), self.num_heads)
        allowed = (
            self.causal_allowed.to(device=tokens.device)[None]
            & valid[:, None, :]
        )
        attended = _merge_heads(
            _scaled_dot_product_attention(
                query,
                key,
                value,
                allowed=allowed,
                dropout=self.attention_dropout,
                training=self.training,
            )
        )
        output = tokens + self.ffn(attended)
        return output.masked_fill(~valid.unsqueeze(-1), 0.0)


class LongerInteraction(nn.Module):
    def __init__(
        self,
        *,
        history_length: int,
        merge_group_size: int,
        recent_query_size: int,
        num_ns_tokens: int,
        model_dim: int,
        num_heads: int,
        num_self_layers: int,
        expansion_factor: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        sequence_length = int(recent_query_size + num_ns_tokens)
        self.cross_attention = HistoryCandidateCrossAttention(
            history_length=history_length,
            merge_group_size=merge_group_size,
            recent_query_size=recent_query_size,
            num_ns_tokens=num_ns_tokens,
            model_dim=model_dim,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            activation=activation,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.self_attention_layers = nn.ModuleList(
            CausalSelfAttentionBlock(
                sequence_length=sequence_length,
                model_dim=model_dim,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                activation=activation,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_self_layers)
        )
        self.output_norm = nn.LayerNorm(model_dim, eps=float(layer_norm_eps))

    def forward(
        self,
        recent_tokens: torch.Tensor,
        merged_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        recent_valid: torch.Tensor,
        merged_valid: torch.Tensor,
    ) -> torch.Tensor:
        ns_valid = torch.ones(
            ns_tokens.shape[:2],
            dtype=torch.bool,
            device=ns_tokens.device,
        )
        query_tokens = torch.cat((recent_tokens, ns_tokens), dim=1)
        key_value_tokens = torch.cat((merged_tokens, ns_tokens), dim=1)
        query_valid = torch.cat((recent_valid, ns_valid), dim=1)
        key_value_valid = torch.cat((merged_valid, ns_valid), dim=1)
        tokens = self.cross_attention(
            query_tokens,
            key_value_tokens,
            query_valid=query_valid,
            key_value_valid=key_value_valid,
        )
        for layer in self.self_attention_layers:
            tokens = layer(tokens, query_valid)
        return self.output_norm(tokens).masked_fill(
            ~query_valid.unsqueeze(-1),
            0.0,
        )


class _FeedForward(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        expansion_factor: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_dim * expansion_factor)
        self.network = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    allowed: torch.Tensor,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    safe_allowed = _ensure_nonempty_rows(allowed).unsqueeze(1)
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=safe_allowed,
        dropout_p=float(dropout) if training else 0.0,
    )


def _ensure_nonempty_rows(allowed: torch.Tensor) -> torch.Tensor:
    missing = ~allowed.any(dim=-1, keepdim=True)
    fallback = torch.zeros_like(allowed)
    fallback[..., 0] = True
    return allowed | (missing & fallback)


def _split_heads(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, sequence_length, model_dim = value.shape
    head_dim = model_dim // num_heads
    return value.reshape(
        batch_size,
        sequence_length,
        num_heads,
        head_dim,
    ).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, sequence_length, head_dim = value.shape
    return value.transpose(1, 2).contiguous().reshape(
        batch_size,
        sequence_length,
        num_heads * head_dim,
    )


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("LONGER activation must be relu, gelu or silu")


def _initialize_linears(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
