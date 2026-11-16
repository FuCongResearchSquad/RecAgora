from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PFFNSettings(NamedTuple):
    kind: str
    num_experts: int
    routed_top_k: int
    expert_hidden_dim: int
    gate_scale: float
    down_init_scale: float


def parse_pffn_config(
    config: Mapping[str, Any] | None,
    *,
    dense_hidden_dim: int,
) -> PFFNSettings:
    if config is None:
        return PFFNSettings("dense", 0, 0, dense_hidden_dim, 1.0, 1.0)
    if not isinstance(config, Mapping):
        raise TypeError("pffn must be a mapping")
    raw = dict(config)
    expected = {
        "type",
        "num_experts",
        "shared_experts",
        "routed_top_k",
        "expert_hidden_dim",
        "router",
        "down_init_scale",
        "sparse_training",
        "sparse_inference",
    }
    _require_exact_keys(raw, expected, "pffn")
    if raw["type"] != "sparse_pertoken_moe":
        raise ValueError("pffn.type must be 'sparse_pertoken_moe'")
    num_experts = _positive_int(raw["num_experts"], "pffn.num_experts")
    shared_experts = _positive_int(
        raw["shared_experts"],
        "pffn.shared_experts",
    )
    if shared_experts != 1 or num_experts < 2:
        raise ValueError("S-P MoE requires one shared and at least one routed expert")
    routed_top_k = _positive_int(raw["routed_top_k"], "pffn.routed_top_k")
    if routed_top_k > num_experts - shared_experts:
        raise ValueError("pffn.routed_top_k exceeds routed expert count")
    expert_hidden_dim = _positive_int(
        raw["expert_hidden_dim"],
        "pffn.expert_hidden_dim",
    )
    if not _boolean(raw["sparse_training"], "pffn.sparse_training"):
        raise ValueError("S-P MoE requires sparse_training=true")
    if not _boolean(raw["sparse_inference"], "pffn.sparse_inference"):
        raise ValueError("S-P MoE requires sparse_inference=true")

    router = raw["router"]
    if not isinstance(router, Mapping):
        raise TypeError("pffn.router must be a mapping")
    router = dict(router)
    _require_exact_keys(
        router,
        {"type", "bias", "renormalize_selected", "gate_scale"},
        "pffn.router",
    )
    if router["type"] != "full_softmax_topk":
        raise ValueError("S-P MoE router.type must be 'full_softmax_topk'")
    if _boolean(router["bias"], "pffn.router.bias"):
        raise ValueError("S-P MoE router bias must be false")
    if _boolean(
        router["renormalize_selected"],
        "pffn.router.renormalize_selected",
    ):
        raise ValueError("top-k routed gates must retain full-softmax probabilities")
    gate_scale = _positive_float(router["gate_scale"], "pffn.router.gate_scale")
    down_init_scale = _positive_float(
        raw["down_init_scale"],
        "pffn.down_init_scale",
    )
    return PFFNSettings(
        "sparse_pertoken_moe",
        num_experts,
        routed_top_k,
        expert_hidden_dim,
        gate_scale,
        down_init_scale,
    )


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, *, eps: float) -> None:
        super().__init__()
        self.dimension = _positive_int(dimension, "dimension")
        self.eps = _positive_float(eps, "eps")
        self.weight = nn.Parameter(torch.ones(self.dimension))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.dimension:
            raise ValueError(
                f"RMSNorm expected last dimension {self.dimension}, "
                f"got {values.shape[-1]}"
            )
        inverse_rms = torch.rsqrt(
            values.to(torch.float32).square().mean(dim=-1, keepdim=True)
            + self.eps
        ).to(dtype=values.dtype)
        return values * inverse_rms * self.weight.to(dtype=values.dtype)


class MultiHeadTokenMixing(nn.Module):
    """Parameter-free split/transpose/concat mixing over semantic tokens."""

    def __init__(self, *, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        self.num_tokens = _positive_int(num_tokens, "num_tokens")
        self.token_dim = _positive_int(token_dim, "token_dim")
        if self.token_dim % self.num_tokens:
            raise ValueError("token_dim must be divisible by num_tokens")
        self.head_dim = self.token_dim // self.num_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("token mixing expects [rows, tokens, dimension]")
        if tuple(tokens.shape[1:]) != (self.num_tokens, self.token_dim):
            raise ValueError(
                "token mixing shape mismatch: expected "
                f"[rows, {self.num_tokens}, {self.token_dim}], got "
                f"{tuple(tokens.shape)}"
            )
        return (
            tokens.reshape(
                tokens.shape[0],
                self.num_tokens,
                self.num_tokens,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
            .reshape(tokens.shape[0], self.num_tokens, self.token_dim)
        )


class PerTokenSwiGLU(nn.Module):
    """Token-private, bias-free SwiGLU evaluated as grouped dense matrices."""

    def __init__(
        self,
        *,
        num_tokens: int,
        input_dim: int,
        expansion_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_tokens = _positive_int(num_tokens, "num_tokens")
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.expansion_ratio = _positive_int(
            expansion_ratio,
            "expansion_ratio",
        )
        self.hidden_dim = self.input_dim * self.expansion_ratio
        self.dropout = nn.Dropout(_dropout(dropout, "dropout"))
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.input_dim, self.hidden_dim)
        )
        self.up_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.input_dim, self.hidden_dim)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_dim, self.input_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_index in range(self.num_tokens):
            nn.init.xavier_uniform_(self.gate_weight[token_index])
            nn.init.xavier_uniform_(self.up_weight[token_index])
            nn.init.xavier_uniform_(self.down_weight[token_index])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("PerTokenSwiGLU expects [rows, tokens, dimension]")
        if tuple(tokens.shape[1:]) != (self.num_tokens, self.input_dim):
            raise ValueError(
                "PerTokenSwiGLU shape mismatch: expected "
                f"[rows, {self.num_tokens}, {self.input_dim}], got "
                f"{tuple(tokens.shape)}"
            )
        gate = torch.einsum("btd,tdh->bth", tokens, self.gate_weight)
        up = torch.einsum("btd,tdh->bth", tokens, self.up_weight)
        hidden = self.dropout(F.silu(gate) * up)
        return torch.einsum("bth,thd->btd", hidden, self.down_weight)


class SparsePerTokenMoEOutput(NamedTuple):
    values: torch.Tensor
    router_probabilities: torch.Tensor
    selected_experts: torch.Tensor


class SparsePerTokenMoE(nn.Module):
    """True sparse-train/sparse-infer token-private SwiGLU experts."""

    def __init__(
        self,
        *,
        num_tokens: int,
        input_dim: int,
        settings: PFFNSettings,
    ) -> None:
        super().__init__()
        self.num_tokens = _positive_int(num_tokens, "num_tokens")
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.num_experts = settings.num_experts
        self.num_routed_experts = self.num_experts - 1
        self.shared_expert_index = self.num_experts - 1
        self.routed_top_k = settings.routed_top_k
        self.expert_hidden_dim = settings.expert_hidden_dim
        self.gate_scale = settings.gate_scale
        self.down_init_scale = settings.down_init_scale
        self.router_weight = nn.Parameter(
            torch.empty(
                self.num_tokens,
                self.input_dim,
                self.num_routed_experts,
            )
        )
        self.gate_weight = nn.Parameter(
            torch.empty(
                self.num_tokens,
                self.num_experts,
                self.input_dim,
                self.expert_hidden_dim,
            )
        )
        self.up_weight = nn.Parameter(torch.empty_like(self.gate_weight))
        self.down_weight = nn.Parameter(
            torch.empty(
                self.num_tokens,
                self.num_experts,
                self.expert_hidden_dim,
                self.input_dim,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_index in range(self.num_tokens):
            nn.init.xavier_uniform_(self.router_weight[token_index])
            for expert_index in range(self.num_experts):
                nn.init.xavier_uniform_(
                    self.gate_weight[token_index, expert_index]
                )
                nn.init.xavier_uniform_(
                    self.up_weight[token_index, expert_index]
                )
                nn.init.xavier_uniform_(
                    self.down_weight[token_index, expert_index]
                )
                with torch.no_grad():
                    self.down_weight[token_index, expert_index].mul_(
                        self.down_init_scale
                    )

    def _expert(
        self,
        values: torch.Tensor,
        *,
        token_index: int,
        expert_index: int,
    ) -> torch.Tensor:
        gate = values @ self.gate_weight[token_index, expert_index]
        up = values @ self.up_weight[token_index, expert_index]
        hidden = F.silu(gate) * up
        return hidden @ self.down_weight[token_index, expert_index]

    def forward(self, tokens: torch.Tensor) -> SparsePerTokenMoEOutput:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_tokens,
            self.input_dim,
        ):
            raise ValueError(
                "SparsePerTokenMoE expects "
                f"[rows, {self.num_tokens}, {self.input_dim}]"
            )
        logits = torch.einsum("btd,tdr->btr", tokens, self.router_weight)
        probabilities = torch.softmax(logits, dim=-1)
        selected_weights, selected_experts = torch.topk(
            probabilities,
            k=self.routed_top_k,
            dim=-1,
        )

        token_outputs: list[torch.Tensor] = []
        for token_index in range(self.num_tokens):
            token_values = tokens[:, token_index]
            token_output = self._expert(
                token_values,
                token_index=token_index,
                expert_index=self.shared_expert_index,
            )
            for expert_index in range(self.num_routed_experts):
                active = selected_experts[:, token_index] == expert_index
                rows, selections = active.nonzero(as_tuple=True)
                if rows.numel() == 0:
                    continue
                contribution = self._expert(
                    token_values.index_select(0, rows),
                    token_index=token_index,
                    expert_index=expert_index,
                )
                weights = selected_weights[rows, token_index, selections]
                contribution = contribution * (
                    weights * self.gate_scale
                ).unsqueeze(-1)
                token_output = token_output.index_add(0, rows, contribution)
            token_outputs.append(token_output)
        return SparsePerTokenMoEOutput(
            values=torch.stack(token_outputs, dim=1),
            router_probabilities=probabilities,
            selected_experts=selected_experts,
        )


def _build_channel_mixer(
    *,
    num_tokens: int,
    token_dim: int,
    expansion_ratio: int,
    dropout: float,
    settings: PFFNSettings,
) -> nn.Module:
    if settings.kind == "dense":
        return PerTokenSwiGLU(
            num_tokens=num_tokens,
            input_dim=token_dim,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
        )
    return SparsePerTokenMoE(
        num_tokens=num_tokens,
        input_dim=token_dim,
        settings=settings,
    )


def _channel_values(
    output: torch.Tensor | SparsePerTokenMoEOutput,
) -> torch.Tensor:
    return output.values if isinstance(output, SparsePerTokenMoEOutput) else output


class TokenMixerBlock(nn.Module):
    """Mixing-reverting block with two independent token-private SwiGLUs."""

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        expansion_ratio: int,
        dropout: float,
        rms_norm_eps: float,
        pffn_settings: PFFNSettings | None = None,
    ) -> None:
        super().__init__()
        settings = pffn_settings or PFFNSettings(
            "dense",
            0,
            0,
            token_dim * expansion_ratio,
            1.0,
            1.0,
        )
        self.pffn_kind = settings.kind
        self.mixing_norm = RMSNorm(token_dim, eps=rms_norm_eps)
        self.mixing = MultiHeadTokenMixing(
            num_tokens=num_tokens,
            token_dim=token_dim,
        )
        self.mixed_swiglu = _build_channel_mixer(
            num_tokens=num_tokens,
            token_dim=token_dim,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            settings=settings,
        )
        self.reverting = MultiHeadTokenMixing(
            num_tokens=num_tokens,
            token_dim=token_dim,
        )
        self.token_norm = RMSNorm(token_dim, eps=rms_norm_eps)
        self.token_swiglu = _build_channel_mixer(
            num_tokens=num_tokens,
            token_dim=token_dim,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            settings=settings,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        mixed = self.mixing(self.mixing_norm(tokens))
        mixed = mixed + _channel_values(self.mixed_swiglu(mixed))
        reverted = self.reverting(mixed)
        return residual + _channel_values(
            self.token_swiglu(self.token_norm(reverted))
        )


class TokenMixerBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_tokens: int,
        num_layers: int,
        expansion_ratio: int,
        dropout: float,
        rms_norm_eps: float,
        activation_checkpointing: bool,
        pffn_settings: PFFNSettings,
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.num_tokens = _positive_int(num_tokens, "num_tokens")
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.block_dim = (
            math.ceil(self.input_dim / self.num_tokens) * self.num_tokens
        )
        self.pad_dim = self.block_dim - self.input_dim
        self.output_dim = self.block_dim
        self.activation_checkpointing = _boolean(
            activation_checkpointing,
            "activation_checkpointing",
        )
        self.blocks = nn.ModuleList(
            TokenMixerBlock(
                num_tokens=self.num_tokens,
                token_dim=self.block_dim,
                expansion_ratio=expansion_ratio,
                dropout=dropout,
                rms_norm_eps=rms_norm_eps,
                pffn_settings=pffn_settings,
            )
            for _ in range(self.num_layers)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("TokenMixerBackbone expects [rows, tokens, dimension]")
        if tuple(tokens.shape[1:]) != (self.num_tokens, self.input_dim):
            raise ValueError(
                "TokenMixerBackbone shape mismatch: expected "
                f"[rows, {self.num_tokens}, {self.input_dim}], got "
                f"{tuple(tokens.shape)}"
            )
        if self.pad_dim:
            tokens = F.pad(tokens, (0, self.pad_dim))
        for block in self.blocks:
            if (
                self.activation_checkpointing
                and self.training
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


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
