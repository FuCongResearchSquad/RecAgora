from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LowRankLinear(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int, rank: int) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.output_dim = _positive_int(output_dim, "output_dim")
        self.rank = _positive_int(rank, "rank")
        self.down_projection = nn.Linear(self.input_dim, self.rank, bias=False)
        self.up_projection = nn.Linear(self.rank, self.output_dim, bias=False)
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.xavier_uniform_(self.up_projection.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"LowRankLinear expected last dimension {self.input_dim}, "
                f"got {values.shape[-1]}"
            )
        return self.up_projection(self.down_projection(values))


class CompositeLowRankAttention(nn.Module):
    """Generate every token's Q/K/V from the flattened field-token list."""

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        qkv_rank: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_tokens = _positive_int(num_tokens, "num_tokens")
        self.token_dim = _positive_int(token_dim, "token_dim")
        self.num_heads = _positive_int(num_heads, "num_heads")
        self.qkv_rank = _positive_int(qkv_rank, "qkv_rank")
        self.dropout = _dropout(dropout, "dropout")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.head_dim = self.token_dim // self.num_heads
        self.flat_dim = self.num_tokens * self.token_dim
        self.query_projection = LowRankLinear(
            input_dim=self.flat_dim,
            output_dim=self.flat_dim,
            rank=self.qkv_rank,
        )
        self.key_projection = LowRankLinear(
            input_dim=self.flat_dim,
            output_dim=self.flat_dim,
            rank=self.qkv_rank,
        )
        self.value_projection = LowRankLinear(
            input_dim=self.flat_dim,
            output_dim=self.flat_dim,
            rank=self.qkv_rank,
        )

    def project_qkv(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_tokens(tokens)
        flattened = tokens.reshape(tokens.shape[0], self.flat_dim)
        projected = tuple(
            projection(flattened)
            .reshape(
                tokens.shape[0],
                self.num_tokens,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
            for projection in (
                self.query_projection,
                self.key_projection,
                self.value_projection,
            )
        )
        return projected[0], projected[1], projected[2]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query, key, value = self.project_qkv(tokens)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return (
            output.transpose(1, 2)
            .contiguous()
            .reshape(tokens.shape[0], self.num_tokens, self.token_dim)
        )

    def _validate_tokens(self, tokens: torch.Tensor) -> None:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_tokens,
            self.token_dim,
        ):
            raise ValueError(
                "CompositeLowRankAttention expected "
                f"[rows, {self.num_tokens}, {self.token_dim}], got "
                f"{tuple(tokens.shape)}"
            )


class PerTokenFeedForward(nn.Module):
    """GELU feed-forward network with private parameters per field/task token."""

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_tokens = _positive_int(num_tokens, "num_tokens")
        self.token_dim = _positive_int(token_dim, "token_dim")
        self.expansion_factor = _positive_int(
            expansion_factor,
            "expansion_factor",
        )
        self.hidden_dim = self.token_dim * self.expansion_factor
        self.input_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.hidden_dim)
        )
        self.input_bias = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.output_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_dim, self.token_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(self.num_tokens, self.token_dim))
        self.dropout = nn.Dropout(_dropout(dropout, "dropout"))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_index in range(self.num_tokens):
            nn.init.xavier_uniform_(self.input_weight[token_index])
            nn.init.xavier_uniform_(self.output_weight[token_index])
        nn.init.zeros_(self.input_bias)
        nn.init.zeros_(self.output_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_tokens,
            self.token_dim,
        ):
            raise ValueError(
                "PerTokenFeedForward expected "
                f"[rows, {self.num_tokens}, {self.token_dim}], got "
                f"{tuple(tokens.shape)}"
            )
        hidden = torch.einsum(
            "btd,tdh->bth",
            tokens,
            self.input_weight,
        ) + self.input_bias
        hidden = self.dropout(F.gelu(hidden))
        return (
            torch.einsum("bth,thd->btd", hidden, self.output_weight)
            + self.output_bias
        )


class HiFormerCompositeBlock(nn.Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        num_heads: int,
        qkv_rank: int,
        expansion_factor: int,
        attention_dropout: float,
        ffn_dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(token_dim, eps=norm_eps)
        self.attention = CompositeLowRankAttention(
            num_tokens=num_tokens,
            token_dim=token_dim,
            num_heads=num_heads,
            qkv_rank=qkv_rank,
            dropout=attention_dropout,
        )
        self.ffn_norm = nn.LayerNorm(token_dim, eps=norm_eps)
        self.ffn = PerTokenFeedForward(
            num_tokens=num_tokens,
            token_dim=token_dim,
            expansion_factor=expansion_factor,
            dropout=ffn_dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.ffn(self.ffn_norm(tokens))


class HiFormerCompositeBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        num_layers: int,
        num_heads: int,
        qkv_rank: int,
        expansion_factor: int,
        attention_dropout: float,
        ffn_dropout: float,
        norm_eps: float,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.num_tokens = _positive_int(num_tokens, "num_tokens")
        self.token_dim = _positive_int(token_dim, "token_dim")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.activation_checkpointing = _boolean(
            activation_checkpointing,
            "activation_checkpointing",
        )
        self.blocks = nn.ModuleList(
            HiFormerCompositeBlock(
                num_tokens=self.num_tokens,
                token_dim=self.token_dim,
                num_heads=num_heads,
                qkv_rank=qkv_rank,
                expansion_factor=expansion_factor,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                norm_eps=norm_eps,
            )
            for _ in range(self.num_layers)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_tokens,
            self.token_dim,
        ):
            raise ValueError(
                "HiFormerCompositeBackbone expected "
                f"[rows, {self.num_tokens}, {self.token_dim}], got "
                f"{tuple(tokens.shape)}"
            )
        for block in self.blocks:
            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
                and tokens.requires_grad
                and tokens.device.type == "cuda"
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _dropout(value: float, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return converted


def _boolean(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value
