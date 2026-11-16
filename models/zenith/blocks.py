from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TokenwiseMultiHeadSelfAttention(nn.Module):
    """TMHSA with independent Q/K/V projections for every Prime Token."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(token_dim, num_tokens, num_heads) <= 0 or token_dim % num_heads:
            raise ValueError(
                "Zenith TMHSA dimensions must be positive and token_dim divisible "
                "by num_heads"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Zenith TMHSA dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.dropout = float(dropout)
        self.query_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.token_dim)
        )
        self.key_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.token_dim)
        )
        self.value_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.token_dim)
        )
        self.output_projection = nn.Linear(
            self.token_dim,
            self.token_dim,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initializing [T,D,D] as one tensor incorrectly treats T as a fan
        # dimension.  Each packed slice is an independent D x D projection.
        for token_index in range(self.num_tokens):
            nn.init.xavier_uniform_(self.query_weight[token_index])
            nn.init.xavier_uniform_(self.key_weight[token_index])
            nn.init.xavier_uniform_(self.value_weight[token_index])
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError("Zenith TMHSA token shape does not match config")
        leading = tokens.shape[:-2]
        rows = tokens.reshape(-1, self.num_tokens, self.token_dim)
        query = torch.einsum("btd,tdf->btf", rows, self.query_weight)
        key = torch.einsum("btd,tdf->btf", rows, self.key_weight)
        value = torch.einsum("btd,tdf->btf", rows, self.value_weight)
        query = query.reshape(
            -1,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.reshape(
            -1,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.reshape(
            -1,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            -1,
            self.num_tokens,
            self.token_dim,
        )
        return self.output_projection(attended).reshape(
            *leading,
            self.num_tokens,
            self.token_dim,
        )


class TokenwiseSwiGLU(nn.Module):
    """TSwiGLU with one independently initialized FFN per Prime Token."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(token_dim, num_tokens, expansion_factor) <= 0:
            raise ValueError("Zenith TSwiGLU dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Zenith TSwiGLU dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = self.token_dim * int(expansion_factor)
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.hidden_dim)
        )
        self.value_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.hidden_dim)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_dim, self.token_dim)
        )
        self.gate_bias = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.value_bias = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.down_bias = nn.Parameter(torch.zeros(self.num_tokens, self.token_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_index in range(self.num_tokens):
            nn.init.xavier_uniform_(self.gate_weight[token_index])
            nn.init.xavier_uniform_(self.value_weight[token_index])
            nn.init.xavier_uniform_(self.down_weight[token_index])
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.value_bias)
        nn.init.zeros_(self.down_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError("Zenith TSwiGLU token shape does not match config")
        gate = torch.einsum(
            "...td,tdh->...th",
            tokens,
            self.gate_weight,
        ) + self.gate_bias
        value = torch.einsum(
            "...td,tdh->...th",
            tokens,
            self.value_weight,
        ) + self.value_bias
        hidden = self.dropout(F.silu(gate) * value)
        return torch.einsum(
            "...th,thd->...td",
            hidden,
            self.down_weight,
        ) + self.down_bias


class ZenithBlock(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_heads: int,
        expansion_factor: int,
        attention_dropout: float,
        ffn_dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim, eps=float(norm_eps))
        self.attention = TokenwiseMultiHeadSelfAttention(
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.ffn_norm = nn.LayerNorm(token_dim, eps=float(norm_eps))
        self.feed_forward = TokenwiseSwiGLU(
            token_dim=token_dim,
            num_tokens=num_tokens,
            expansion_factor=expansion_factor,
            dropout=ffn_dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.feed_forward(self.ffn_norm(tokens))


class ZenithBackbone(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_layers: int,
        num_heads: int,
        expansion_factor: int,
        attention_dropout: float,
        ffn_dropout: float,
        norm_eps: float,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("Zenith num_layers must be positive")
        self.activation_checkpointing = bool(activation_checkpointing)
        self.blocks = nn.ModuleList(
            ZenithBlock(
                token_dim=token_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
                expansion_factor=expansion_factor,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                norm_eps=norm_eps,
            )
            for _ in range(int(num_layers))
        )
        self.final_norm = nn.LayerNorm(token_dim, eps=float(norm_eps))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        checkpointing = (
            self.activation_checkpointing and self.training and tokens.requires_grad
        )
        for block in self.blocks:
            if checkpointing:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return self.final_norm(tokens)
