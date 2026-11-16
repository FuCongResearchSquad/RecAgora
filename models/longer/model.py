from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from recagora import (
    RecAgoraInputSpec,
    RecAgoraModelBatch,
    RecAgoraRankingModel,
    RecAgoraSparseBatch,
    RankingOutput,
    masked_binary_cross_entropy_with_logits,
    structured_ranking_output,
)
from models._validation import (
    finite_nonnegative_float,
    finite_positive_float,
    validate_dropout,
    validate_known_weights,
    validate_model_dim_heads,
    validate_positive_int,
    validate_positive_int_sequence,
    validate_standard_multitask_contract,
)

from .blocks import AutoSplitNSTokenizer, InnerTrans, LongerInteraction
from .inputs import LongerInputs


class LongerArchCoreAdaptedModel(RecAgoraRankingModel):
    """LONGER-inspired request-level ranker with isolated candidate paths."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        query_slots: Sequence[int],
        candidate_slots: Sequence[int],
        history: Mapping[str, object],
        model_dim: int,
        num_heads: int,
        merge_group_size: int,
        recent_query_size: int,
        num_ns_tokens: int,
        num_self_layers: int,
        expansion_factor: int,
        activation: str,
        dropout: float,
        layer_norm_eps: float,
        position_encoding: str,
        cross_attention_mode: str,
        candidate_isolation: bool,
        candidate_chunk_size: int,
        activation_checkpointing: bool,
        task_names: Sequence[str],
        tower_hidden_units: Sequence[int],
        tower_activation: str,
        tower_dropout: float,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        validate_model_dim_heads(model_dim, num_heads)
        for field_name, value in (
            ("merge_group_size", merge_group_size),
            ("recent_query_size", recent_query_size),
            ("num_ns_tokens", num_ns_tokens),
            ("num_self_layers", num_self_layers),
            ("expansion_factor", expansion_factor),
            ("candidate_chunk_size", candidate_chunk_size),
        ):
            validate_positive_int(value, field_name=field_name)
        validate_dropout(dropout)
        validate_dropout(tower_dropout, field_name="tower_dropout")
        layer_norm_eps = finite_positive_float(
            layer_norm_eps,
            "LONGER layer_norm_eps",
        )
        if activation not in {"relu", "gelu", "silu"}:
            raise ValueError("LONGER activation must be relu, gelu or silu")
        if tower_activation not in {"relu", "gelu", "silu"}:
            raise ValueError("LONGER tower_activation must be relu, gelu or silu")
        if cross_attention_mode != "history_causal_candidate_global":
            raise ValueError(
                "LONGER cross_attention_mode must be "
                "history_causal_candidate_global"
            )
        if type(activation_checkpointing) is not bool:
            raise TypeError("LONGER activation_checkpointing must be boolean")

        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="LONGER",
        )
        hidden_units = tuple(int(unit) for unit in tower_hidden_units)
        validate_positive_int_sequence(
            hidden_units,
            field_name="tower_hidden_units",
        )
        normalized_weights = {
            str(name): float(weight) for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(
            normalized_weights,
            task_names,
            field_name="task_weights",
        )
        effective_weights = {
            name: finite_nonnegative_float(
                normalized_weights.get(name, 1.0),
                f"task_weights.{name}",
            )
            for name in task_names
        }
        if not any(weight > 0.0 for weight in effective_weights.values()):
            raise ValueError("LONGER requires at least one positive task weight")

        inputs = LongerInputs(
            input_spec=input_spec,
            query_slots=query_slots,
            candidate_slots=candidate_slots,
            history=history,
            model_dim=model_dim,
            position_encoding=position_encoding,
        )
        if inputs.history_length % int(merge_group_size):
            raise ValueError(
                "LONGER history length must be divisible by merge_group_size"
            )
        if int(recent_query_size) > inputs.history_length:
            raise ValueError(
                "LONGER recent_query_size must not exceed the history length"
            )

        params: dict[str, Any] = {
            "query_slots": tuple(int(slot) for slot in query_slots),
            "candidate_slots": tuple(int(slot) for slot in candidate_slots),
            "history": dict(history),
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "merge_group_size": int(merge_group_size),
            "recent_query_size": int(recent_query_size),
            "num_ns_tokens": int(num_ns_tokens),
            "num_self_layers": int(num_self_layers),
            "expansion_factor": int(expansion_factor),
            "activation": activation,
            "dropout": float(dropout),
            "layer_norm_eps": float(layer_norm_eps),
            "position_encoding": position_encoding,
            "cross_attention_mode": cross_attention_mode,
            "candidate_isolation": bool(candidate_isolation),
            "candidate_chunk_size": int(candidate_chunk_size),
            "activation_checkpointing": activation_checkpointing,
            "task_names": task_names,
            "tower_hidden_units": hidden_units,
            "tower_activation": tower_activation,
            "tower_dropout": float(tower_dropout),
            "task_weights": effective_weights,
        }
        super().__init__(params, input_spec)
        self.inputs = inputs
        self.model_dim = int(model_dim)
        self.merge_group_size = int(merge_group_size)
        self.recent_query_size = int(recent_query_size)
        self.num_ns_tokens = int(num_ns_tokens)
        self.sequence_length = self.recent_query_size + self.num_ns_tokens
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.activation_checkpointing = activation_checkpointing
        self.task_names = task_names
        self.task_weights = effective_weights

        self.inner_trans = InnerTrans(
            model_dim=self.model_dim,
            merge_group_size=self.merge_group_size,
            expansion_factor=int(expansion_factor),
            activation=activation,
            dropout=float(dropout),
            layer_norm_eps=float(layer_norm_eps),
        )
        self.ns_tokenizer = AutoSplitNSTokenizer(
            input_dim=self.inputs.query_dim + self.inputs.candidate_dim,
            model_dim=self.model_dim,
            num_tokens=self.num_ns_tokens,
            layer_norm_eps=float(layer_norm_eps),
        )
        self.interaction = LongerInteraction(
            history_length=self.inputs.history_length,
            merge_group_size=self.merge_group_size,
            recent_query_size=self.recent_query_size,
            num_ns_tokens=self.num_ns_tokens,
            model_dim=self.model_dim,
            num_heads=int(num_heads),
            num_self_layers=int(num_self_layers),
            expansion_factor=int(expansion_factor),
            activation=activation,
            dropout=float(dropout),
            layer_norm_eps=float(layer_norm_eps),
        )
        tower_input_dim = self.sequence_length * self.model_dim
        self.heads = nn.ModuleDict(
            {
                name: _build_task_tower(
                    input_dim=tower_input_dim,
                    hidden_units=hidden_units,
                    activation=tower_activation,
                    dropout=float(tower_dropout),
                )
                for name in self.task_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        return structured_ranking_output(self._forward_slots(embedded))

    def _forward_slots(self, embedded: RecAgoraModelBatch) -> dict[str, torch.Tensor]:
        inputs = self.inputs(embedded)
        merged_tokens, merged_valid = self.inner_trans(
            inputs.history_events,
            inputs.history_valid,
        )
        recent_tokens, recent_valid = self._recent_history(
            inputs.history_events,
            inputs.history_valid,
        )
        chunks: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.task_names
        }
        for start in range(0, embedded.list_len, self.candidate_chunk_size):
            candidate_chunk = inputs.candidates[
                :,
                start : start + self.candidate_chunk_size,
            ]
            if self._should_checkpoint(candidate_chunk):
                chunk_logits = checkpoint(
                    self._score_candidate_chunk,
                    inputs.query,
                    candidate_chunk,
                    recent_tokens,
                    merged_tokens,
                    recent_valid,
                    merged_valid,
                    use_reentrant=False,
                )
            else:
                chunk_logits = self._score_candidate_chunk(
                    inputs.query,
                    candidate_chunk,
                    recent_tokens,
                    merged_tokens,
                    recent_valid,
                    merged_valid,
                )
            for name, logits in zip(self.task_names, chunk_logits, strict=True):
                chunks[name].append(logits)
        return {
            name: torch.cat(chunks[name], dim=1)
            for name in self.task_names
        }

    def _recent_history(
        self,
        events: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            events[:, -self.recent_query_size :],
            valid[:, -self.recent_query_size :],
        )

    def _score_candidate_chunk(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        recent_tokens: torch.Tensor,
        merged_tokens: torch.Tensor,
        recent_valid: torch.Tensor,
        merged_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size, candidate_count = candidates.shape[:2]
        ns_tokens = self.ns_tokenizer(query, candidates).reshape(
            batch_size * candidate_count,
            self.num_ns_tokens,
            self.model_dim,
        )
        recent_for_candidates = recent_tokens[:, None].expand(
            -1,
            candidate_count,
            -1,
            -1,
        ).reshape(
            batch_size * candidate_count,
            self.recent_query_size,
            self.model_dim,
        )
        merged_for_candidates = merged_tokens[:, None].expand(
            -1,
            candidate_count,
            -1,
            -1,
        ).reshape(
            batch_size * candidate_count,
            merged_tokens.shape[1],
            self.model_dim,
        )
        recent_valid_for_candidates = recent_valid[:, None].expand(
            -1,
            candidate_count,
            -1,
        ).reshape(batch_size * candidate_count, self.recent_query_size)
        merged_valid_for_candidates = merged_valid[:, None].expand(
            -1,
            candidate_count,
            -1,
        ).reshape(batch_size * candidate_count, merged_valid.shape[1])
        states = self.interaction(
            recent_for_candidates,
            merged_for_candidates,
            ns_tokens,
            recent_valid_for_candidates,
            merged_valid_for_candidates,
        ).reshape(batch_size, candidate_count, -1)
        return tuple(
            self.heads[name](states).squeeze(-1)
            for name in self.task_names
        )

    def _should_checkpoint(self, candidate_chunk: torch.Tensor) -> bool:
        return (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
            and candidate_chunk.requires_grad
            and candidate_chunk.device.type == "cuda"
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_by_task = output["logits"]
        losses: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []
        for name in self.task_names:
            logits = logits_by_task[name]
            labels = batch.labels[name].to(device=logits.device, dtype=logits.dtype)
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted.append(loss * self.task_weights[name])
        total = torch.stack(weighted).sum()
        losses["total"] = total
        return total, losses


def _build_task_tower(
    *,
    input_dim: int,
    hidden_units: Sequence[int],
    activation: str,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_units:
        linear = nn.Linear(previous, int(hidden))
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.extend((linear, _activation(activation), nn.Dropout(dropout)))
        previous = int(hidden)
    output = nn.Linear(previous, 1)
    nn.init.xavier_uniform_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError("LONGER activation must be relu, gelu or silu")
