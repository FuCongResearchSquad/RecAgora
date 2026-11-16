from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ._validation import (
    nonempty_string as _nonempty_string,
    positive_float as _positive_float,
    positive_int as _positive_int,
    require_bool as _boolean,
)


class PFFNSettings(NamedTuple):
    kind: str
    num_experts: int
    activation_checkpointing: bool
    sparse_inference: bool
    router_type: str
    training_router_initial_bias: float
    inference_router_initial_bias: float
    training_output_combination: str
    target_active_experts: float
    initial_regularization_coefficient: float
    coefficient_update_multiplier: float
    minimum_regularization_coefficient: float
    maximum_regularization_coefficient: float
    load_balance: bool


def parse_pffn_config(config: Mapping[str, Any] | None) -> PFFNSettings:
    """Validate one neutral PFFN choice without changing the dense default."""
    if config is None:
        return _dense_pffn_settings()
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raise TypeError("pffn must be a mapping")
    if not raw:
        raise ValueError("pffn must explicitly select a type")

    kind = _nonempty_string(raw.get("type"), "pffn.type")
    if kind == "dense":
        _require_exact_keys(raw, {"type"}, "pffn")
        return _dense_pffn_settings()
    if kind not in {"relu_smoe", "dtsi_moe"}:
        raise ValueError(f"unsupported pffn.type {kind!r}")

    _require_exact_keys(
        raw,
        {
            "type",
            "expert_scope",
            "expert_activation",
            "num_experts",
            "activation_checkpointing",
            "sparse_inference",
            "router",
        },
        "pffn",
    )
    if raw["expert_scope"] != "token_private":
        raise ValueError("pffn.expert_scope must be 'token_private'")
    if raw["expert_activation"] != "gelu":
        raise ValueError("pffn.expert_activation must be 'gelu'")
    num_experts = _positive_int(raw["num_experts"], "pffn.num_experts")
    if num_experts < 2:
        raise ValueError("pffn.num_experts must be at least two")
    activation_checkpointing = _boolean(
        raw["activation_checkpointing"],
        "pffn.activation_checkpointing",
    )
    sparse_inference = _boolean(
        raw["sparse_inference"],
        "pffn.sparse_inference",
    )
    if kind == "dtsi_moe" and not sparse_inference:
        raise ValueError("DTSI requires sparse_inference=true")

    router = raw["router"]
    if not isinstance(router, Mapping):
        raise TypeError("pffn.router must be a mapping")
    router = dict(router)
    common_router_keys = {
        "type",
        "bias",
        "target_active_experts",
        "initial_regularization_coefficient",
        "coefficient_update_multiplier",
        "minimum_regularization_coefficient",
        "maximum_regularization_coefficient",
        "load_balance",
    }
    shared_path_router_keys = {
        "training_activation",
        "inference_activation",
        "shared_logits",
        "normalize",
    }
    independent_router_keys = {
        "training_activation",
        "inference_activation",
        "training_router_initial_bias",
        "inference_router_initial_bias",
        "training_output_combination",
    }
    router_type = _nonempty_string(router.get("type"), "pffn.router.type")
    if kind == "relu_smoe":
        expected_router_keys = common_router_keys
    elif router_type == "shared_logits_dual_path":
        expected_router_keys = common_router_keys | shared_path_router_keys
    elif router_type == "independent_dual_router":
        expected_router_keys = common_router_keys | independent_router_keys
    else:
        raise ValueError(
            "pffn.router.type must be shared_logits_dual_path or "
            "independent_dual_router for DTSI"
        )
    _require_exact_keys(
        router,
        expected_router_keys,
        "pffn.router",
    )
    if kind == "relu_smoe" and router_type != "relu":
        raise ValueError("pffn.router.type must be 'relu'")

    training_router_initial_bias = 0.0
    inference_router_initial_bias = 0.0
    training_output_combination = "training_only"
    if kind == "dtsi_moe" and router_type == "shared_logits_dual_path":
        if router["training_activation"] != "softplus":
            raise ValueError("DTSI training_activation must be 'softplus'")
        if router["inference_activation"] != "relu":
            raise ValueError("DTSI inference_activation must be 'relu'")
        if not _boolean(router["shared_logits"], "pffn.router.shared_logits"):
            raise ValueError("DTSI shared_logits must be true")
        if not _boolean(router["normalize"], "pffn.router.normalize"):
            raise ValueError("DTSI gate normalization must be enabled")
    elif kind == "dtsi_moe":
        if router["training_activation"] != "relu":
            raise ValueError("dual-router DTSI training_activation must be 'relu'")
        if router["inference_activation"] != "relu":
            raise ValueError("dual-router DTSI inference_activation must be 'relu'")
        training_router_initial_bias = _finite_float(
            router["training_router_initial_bias"],
            "pffn.router.training_router_initial_bias",
        )
        inference_router_initial_bias = _finite_float(
            router["inference_router_initial_bias"],
            "pffn.router.inference_router_initial_bias",
        )
        training_output_combination = _nonempty_string(
            router["training_output_combination"],
            "pffn.router.training_output_combination",
        )
        if training_output_combination != "sum":
            raise ValueError(
                "dual-router DTSI training_output_combination must be 'sum'"
            )
    router_bias = _boolean(router["bias"], "pffn.router.bias")
    if router_type == "independent_dual_router" and not router_bias:
        raise ValueError("independent dual routers require bias=true")
    if router_type != "independent_dual_router" and router_bias:
        raise ValueError("pffn.router.bias must be false for the raw ReLU router")
    target_active_experts = _positive_float(
        router["target_active_experts"],
        "pffn.router.target_active_experts",
    )
    if target_active_experts > num_experts:
        raise ValueError(
            "pffn.router.target_active_experts must not exceed num_experts"
        )
    initial_coefficient = _positive_float(
        router["initial_regularization_coefficient"],
        "pffn.router.initial_regularization_coefficient",
    )
    update_multiplier = _positive_float(
        router["coefficient_update_multiplier"],
        "pffn.router.coefficient_update_multiplier",
    )
    if update_multiplier <= 1.0:
        raise ValueError(
            "pffn.router.coefficient_update_multiplier must be greater than one"
        )
    minimum_coefficient = _positive_float(
        router["minimum_regularization_coefficient"],
        "pffn.router.minimum_regularization_coefficient",
    )
    maximum_coefficient = _positive_float(
        router["maximum_regularization_coefficient"],
        "pffn.router.maximum_regularization_coefficient",
    )
    if maximum_coefficient < minimum_coefficient:
        raise ValueError(
            "pffn.router.maximum_regularization_coefficient must be no smaller "
            "than minimum_regularization_coefficient"
        )
    if not minimum_coefficient <= initial_coefficient <= maximum_coefficient:
        raise ValueError(
            "pffn.router.initial_regularization_coefficient must lie within "
            "the configured coefficient bounds"
        )
    load_balance = _boolean(router["load_balance"], "pffn.router.load_balance")
    return PFFNSettings(
        kind=kind,
        num_experts=num_experts,
        activation_checkpointing=activation_checkpointing,
        sparse_inference=sparse_inference,
        router_type=router_type,
        training_router_initial_bias=training_router_initial_bias,
        inference_router_initial_bias=inference_router_initial_bias,
        training_output_combination=training_output_combination,
        target_active_experts=target_active_experts,
        initial_regularization_coefficient=initial_coefficient,
        coefficient_update_multiplier=update_multiplier,
        minimum_regularization_coefficient=minimum_coefficient,
        maximum_regularization_coefficient=maximum_coefficient,
        load_balance=load_balance,
    )


def _dense_pffn_settings() -> PFFNSettings:
    return PFFNSettings(
        kind="dense",
        num_experts=0,
        activation_checkpointing=False,
        sparse_inference=False,
        router_type="none",
        training_router_initial_bias=0.0,
        inference_router_initial_bias=0.0,
        training_output_combination="none",
        target_active_experts=0.0,
        initial_regularization_coefficient=0.0,
        coefficient_update_multiplier=1.0,
        minimum_regularization_coefficient=0.0,
        maximum_regularization_coefficient=0.0,
        load_balance=False,
    )


class MultiHeadTokenMixer(nn.Module):
    """Parameter-free RankMixer token permutation."""

    def __init__(self, num_tokens: int, token_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.head_dim = self.token_dim // self.num_tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        return (
            tokens.reshape(
                batch_size,
                self.num_tokens,
                self.num_tokens,
                self.head_dim,
            )
            .transpose(1, 2)
            .contiguous()
            .reshape(batch_size, self.num_tokens, self.token_dim)
        )


class PerTokenFFN(nn.Module):
    """Independent dense FFN layout with one parameter set per token."""

    def __init__(self, num_tokens: int, token_dim: int, expansion_ratio: int) -> None:
        super().__init__()
        hidden_dim = token_dim * expansion_ratio
        self.ffns = nn.ModuleList(
            nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, token_dim),
            )
            for _ in range(num_tokens)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [ffn(token) for ffn, token in zip(self.ffns, tokens.unbind(dim=1))],
            dim=1,
        )


class ReLUSMoEOutput(NamedTuple):
    values: torch.Tensor
    gates: torch.Tensor


class TokenPrivateReLUSMoE(nn.Module):
    """Token-private experts with one raw ReLU router for train and inference."""

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        expansion_ratio: int,
        num_experts: int,
        activation_checkpointing: bool,
        sparse_inference: bool,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.num_experts = int(num_experts)
        hidden_dim = self.token_dim * int(expansion_ratio)
        self.activation_checkpointing = activation_checkpointing
        self.sparse_inference = sparse_inference
        # Separate Linear modules keep initialization fan-in identical to the
        # dense FFN and avoid accidental fan computation across token/expert axes.
        self.routers = nn.ModuleList(
            nn.Linear(self.token_dim, self.num_experts, bias=False)
            for _ in range(self.num_tokens)
        )
        self.experts = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    nn.Linear(self.token_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.token_dim),
                )
                for _ in range(self.num_experts)
            )
            for _ in range(self.num_tokens)
        )

    def route(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.relu(router(token_values))
                for router, token_values in zip(
                    self.routers,
                    tokens.unbind(dim=1),
                )
            ],
            dim=1,
        )

    def dense_values(
        self,
        tokens: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Reference mixture used for training and sparse-dispatch parity."""
        use_checkpoint = (
            self.activation_checkpointing
            and self.training
            and tokens.device.type == "cuda"
        )
        token_outputs: list[torch.Tensor] = []
        for token_index, token_values in enumerate(tokens.unbind(dim=1)):
            mixed = torch.zeros_like(token_values)
            for expert_index, expert in enumerate(self.experts[token_index]):
                gate_values = gates[:, token_index, expert_index]
                if use_checkpoint:
                    def _weighted(
                        values: torch.Tensor,
                        weights: torch.Tensor,
                        module: nn.Module = expert,
                    ) -> torch.Tensor:
                        return module(values) * weights.unsqueeze(-1)

                    contribution = checkpoint(
                        _weighted,
                        token_values,
                        gate_values,
                        use_reentrant=False,
                    )
                else:
                    contribution = expert(token_values) * gate_values.unsqueeze(-1)
                mixed = mixed + contribution
            token_outputs.append(mixed)
        return torch.stack(token_outputs, dim=1)

    def sparse_values(
        self,
        tokens: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate only positive routes while preserving the dense equation."""
        flat_output = tokens.new_zeros(
            (tokens.shape[0] * self.num_tokens, self.token_dim)
        )
        # Discover every active route once. Grouping the result avoids one
        # CUDA host synchronization for every static token/expert pair.
        active_routes = torch.nonzero(gates > 0, as_tuple=False)
        if active_routes.numel() == 0:
            return flat_output.reshape(
                tokens.shape[0],
                self.num_tokens,
                self.token_dim,
            )
        group_ids = active_routes[:, 1] * self.num_experts + active_routes[:, 2]
        order = torch.argsort(group_ids)
        routes = active_routes.index_select(0, order)
        group_ids = group_ids.index_select(0, order)
        counts = torch.bincount(
            group_ids,
            minlength=self.num_tokens * self.num_experts,
        ).to(device="cpu")
        offset = 0
        for group_id, count in enumerate(counts.tolist()):
            if count == 0:
                continue
            token_index = group_id // self.num_experts
            expert_index = group_id % self.num_experts
            selected_routes = routes[offset : offset + count]
            batch_indices = selected_routes[:, 0]
            selected = tokens[batch_indices, token_index]
            contribution = self.experts[token_index][expert_index](selected)
            contribution = contribution * gates[
                batch_indices,
                token_index,
                expert_index,
            ].unsqueeze(-1)
            flat_indices = batch_indices * self.num_tokens + token_index
            flat_output = flat_output.index_add(0, flat_indices, contribution)
            offset += count
        return flat_output.reshape(
            tokens.shape[0],
            self.num_tokens,
            self.token_dim,
        )

    def forward(self, tokens: torch.Tensor) -> ReLUSMoEOutput:
        gates = self.route(tokens)
        values = (
            self.sparse_values(tokens, gates)
            if self.sparse_inference and not self.training
            else self.dense_values(tokens, gates)
        )
        return ReLUSMoEOutput(values=values, gates=gates)


class TokenPrivateDTSIMoE(TokenPrivateReLUSMoE):
    """Shared-logit dense-training/sparse-inference token-private MoE."""

    def route_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                router(token_values)
                for router, token_values in zip(
                    self.routers,
                    tokens.unbind(dim=1),
                    strict=True,
                )
            ],
            dim=1,
        )

    @staticmethod
    def normalize_gates(gates: torch.Tensor) -> torch.Tensor:
        denominator = gates.sum(dim=-1, keepdim=True)
        return torch.where(
            denominator > 0,
            gates / denominator.clamp_min(torch.finfo(gates.dtype).eps),
            torch.zeros_like(gates),
        )

    def forward(self, tokens: torch.Tensor) -> ReLUSMoEOutput:
        logits = self.route_logits(tokens)
        inference_gates = torch.relu(logits)
        if self.training:
            training_gates = self.normalize_gates(F.softplus(logits))
            values = self.dense_values(tokens, training_gates)
        else:
            normalized_inference = self.normalize_gates(inference_gates)
            values = self.sparse_values(tokens, normalized_inference)
        return ReLUSMoEOutput(values=values, gates=inference_gates)


class TokenPrivateDualRouterDTSIMoE(TokenPrivateReLUSMoE):
    """Independent train/inference routers with dense train and sparse eval."""

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        expansion_ratio: int,
        num_experts: int,
        activation_checkpointing: bool,
        sparse_inference: bool,
        training_router_initial_bias: float,
        inference_router_initial_bias: float,
        training_output_combination: str,
    ) -> None:
        super().__init__(
            num_tokens=num_tokens,
            token_dim=token_dim,
            expansion_ratio=expansion_ratio,
            num_experts=num_experts,
            activation_checkpointing=activation_checkpointing,
            sparse_inference=sparse_inference,
        )
        # Reuse the parent's expert storage and sparse dispatch, but replace
        # its single router with two independently registered parameter sets.
        self.inference_routers = nn.ModuleList(
            nn.Linear(self.token_dim, self.num_experts, bias=True)
            for _ in range(self.num_tokens)
        )
        self.training_routers = nn.ModuleList(
            nn.Linear(self.token_dim, self.num_experts, bias=True)
            for _ in range(self.num_tokens)
        )
        del self.routers
        for router in self.training_routers:
            nn.init.constant_(router.bias, float(training_router_initial_bias))
        for router in self.inference_routers:
            nn.init.constant_(router.bias, float(inference_router_initial_bias))
        if training_output_combination != "sum":
            raise ValueError("dual-router DTSI currently supports sum only")
        self.training_output_combination = training_output_combination

    @staticmethod
    def _route(
        tokens: torch.Tensor,
        routers: nn.ModuleList,
    ) -> torch.Tensor:
        return torch.stack(
            [
                torch.relu(router(token_values))
                for router, token_values in zip(
                    routers,
                    tokens.unbind(dim=1),
                    strict=True,
                )
            ],
            dim=1,
        )

    def forward(self, tokens: torch.Tensor) -> ReLUSMoEOutput:
        inference_gates = self._route(tokens, self.inference_routers)
        if not self.training:
            values = self.sparse_values(tokens, inference_gates)
            return ReLUSMoEOutput(values=values, gates=inference_gates)

        training_gates = self._route(tokens, self.training_routers)
        # The paper does not publish how the two router outputs are combined.
        # Summing their gates evaluates every expert once while preserving the
        # explicit training_values + inference_values adaptation.
        values = self.dense_values(tokens, training_gates + inference_gates)
        return ReLUSMoEOutput(values=values, gates=inference_gates)


class AdaptiveSparsityCoefficient(nn.Module):
    """Bounded ReMoE-style controller stored in the model checkpoint."""

    def __init__(self, settings: PFFNSettings) -> None:
        super().__init__()
        self.target_active_experts = settings.target_active_experts
        self.update_multiplier = settings.coefficient_update_multiplier
        self.minimum_value = settings.minimum_regularization_coefficient
        self.maximum_value = settings.maximum_regularization_coefficient
        self.register_buffer(
            "value",
            torch.tensor(settings.initial_regularization_coefficient),
        )

    @torch.no_grad()
    def update(self, average_active_experts: torch.Tensor) -> None:
        target = average_active_experts.new_tensor(self.target_active_experts)
        multiplier = average_active_experts.new_tensor(self.update_multiplier)
        factor = torch.where(
            average_active_experts > target,
            multiplier,
            torch.where(
                average_active_experts < target,
                multiplier.reciprocal(),
                multiplier.new_ones(()),
            ),
        )
        updated = (self.value * factor.to(dtype=self.value.dtype)).clamp(
            min=self.minimum_value,
            max=self.maximum_value,
        )
        self.value.copy_(updated)


class RankMixerBlockOutput(NamedTuple):
    tokens: torch.Tensor
    gates: torch.Tensor | None


class RankMixerBlock(nn.Module):
    """RankMixer block whose PFFN implementation is selected by configuration."""

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        expansion_ratio: int,
        layer_norm_eps: float,
        pffn_settings: PFFNSettings | None = None,
    ) -> None:
        super().__init__()
        settings = pffn_settings or _dense_pffn_settings()
        self.pffn_kind = settings.kind
        self.token_mixer = MultiHeadTokenMixer(num_tokens, token_dim)
        self.token_norm = nn.LayerNorm(token_dim, eps=layer_norm_eps)
        if settings.kind == "dense":
            self.per_token_ffn: nn.Module = PerTokenFFN(
                num_tokens,
                token_dim,
                expansion_ratio,
            )
        elif settings.kind == "relu_smoe":
            self.per_token_ffn = TokenPrivateReLUSMoE(
                num_tokens=num_tokens,
                token_dim=token_dim,
                expansion_ratio=expansion_ratio,
                num_experts=settings.num_experts,
                activation_checkpointing=settings.activation_checkpointing,
                sparse_inference=settings.sparse_inference,
            )
        elif settings.router_type == "shared_logits_dual_path":
            self.per_token_ffn = TokenPrivateDTSIMoE(
                num_tokens=num_tokens,
                token_dim=token_dim,
                expansion_ratio=expansion_ratio,
                num_experts=settings.num_experts,
                activation_checkpointing=settings.activation_checkpointing,
                sparse_inference=settings.sparse_inference,
            )
        else:
            self.per_token_ffn = TokenPrivateDualRouterDTSIMoE(
                num_tokens=num_tokens,
                token_dim=token_dim,
                expansion_ratio=expansion_ratio,
                num_experts=settings.num_experts,
                activation_checkpointing=settings.activation_checkpointing,
                sparse_inference=settings.sparse_inference,
                training_router_initial_bias=(
                    settings.training_router_initial_bias
                ),
                inference_router_initial_bias=(
                    settings.inference_router_initial_bias
                ),
                training_output_combination=(
                    settings.training_output_combination
                ),
            )
        self.ffn_norm = nn.LayerNorm(token_dim, eps=layer_norm_eps)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_routing: bool = False,
    ) -> torch.Tensor | RankMixerBlockOutput:
        tokens = self.token_norm(tokens + self.token_mixer(tokens))
        gates: torch.Tensor | None = None
        if self.pffn_kind == "dense":
            values = self.per_token_ffn(tokens)
        else:
            mixture = self.per_token_ffn(tokens)
            values, gates = mixture
        output = self.ffn_norm(tokens + values)
        if return_routing:
            return RankMixerBlockOutput(tokens=output, gates=gates)
        return output


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


def _finite_float(value: Any, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted
