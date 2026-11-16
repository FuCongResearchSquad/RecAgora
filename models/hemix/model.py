from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from recagora import (
    RecAgoraInputSpec,
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

from .blocks import (
    CandidateConditionedTokenizer,
    HeteroMixer,
    QueryMixedInterestExtractor,
)
from .inputs import HeMixInputs


class HeMixModel(RecAgoraRankingModel):
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
        token_dim: int,
        num_global_tokens: int,
        num_realtime_tokens: int,
        num_attention_heads: int,
        num_mix_heads: int,
        low_rank_dim: int,
        num_layers: int,
        expansion_factor: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1.0e-6,
        activation_checkpointing: bool = True,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_units: Sequence[int] = (128, 64),
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        validate_positive_int(token_dim, field_name="token_dim")
        validate_positive_int(num_global_tokens, field_name="num_global_tokens")
        validate_positive_int(num_realtime_tokens, field_name="num_realtime_tokens")
        validate_positive_int(num_attention_heads, field_name="num_attention_heads")
        validate_positive_int(num_mix_heads, field_name="num_mix_heads")
        validate_model_dim_heads(int(token_dim), int(num_attention_heads))
        validate_model_dim_heads(int(token_dim), int(num_mix_heads))
        validate_positive_int(low_rank_dim, field_name="low_rank_dim")
        validate_positive_int(num_layers, field_name="num_layers")
        validate_positive_int(expansion_factor, field_name="expansion_factor")
        validate_dropout(dropout)
        validate_dropout(attention_dropout, field_name="attention_dropout")
        validate_dropout(head_dropout, field_name="head_dropout")
        finite_positive_float(layer_norm_eps, "layer_norm_eps")
        if type(activation_checkpointing) is not bool:
            raise TypeError("activation_checkpointing must be bool")

        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="HeMix",
        )
        for index, value in enumerate(head_hidden_units):
            validate_positive_int(value, field_name=f"head_hidden_units[{index}]")
        head_hidden_units = tuple(int(value) for value in head_hidden_units)
        validate_positive_int_sequence(
            head_hidden_units,
            field_name="head_hidden_units",
        )
        task_weights = {
            str(name): float(weight) for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(task_weights, task_names, field_name="task_weights")
        for name, weight in task_weights.items():
            finite_nonnegative_float(weight, f"task_weights.{name}")

        params: dict[str, Any] = {
            "query_slots": tuple(int(slot_id) for slot_id in query_slots),
            "candidate_slots": tuple(int(slot_id) for slot_id in candidate_slots),
            "history": dict(history),
            "token_dim": int(token_dim),
            "num_global_tokens": int(num_global_tokens),
            "num_realtime_tokens": int(num_realtime_tokens),
            "num_attention_heads": int(num_attention_heads),
            "num_mix_heads": int(num_mix_heads),
            "low_rank_dim": int(low_rank_dim),
            "num_layers": int(num_layers),
            "expansion_factor": int(expansion_factor),
            "dropout": float(dropout),
            "attention_dropout": float(attention_dropout),
            "layer_norm_eps": float(layer_norm_eps),
            "activation_checkpointing": activation_checkpointing,
            "candidate_isolation": bool(candidate_isolation),
            "task_names": task_names,
            "head_hidden_units": head_hidden_units,
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
        }
        super().__init__(params, input_spec)
        self.token_dim = int(token_dim)
        self.num_global_tokens = int(num_global_tokens)
        self.num_realtime_tokens = int(num_realtime_tokens)
        self.num_ns_tokens = self.num_global_tokens + self.num_realtime_tokens
        self.num_interest_tokens = 2 * self.num_ns_tokens
        self.num_tokens = self.num_interest_tokens + self.num_ns_tokens
        self.task_names = task_names
        self.task_weights = task_weights
        self.inputs = HeMixInputs(
            input_spec=input_spec,
            query_slots=query_slots,
            candidate_slots=candidate_slots,
            history=history,
            token_dim=self.token_dim,
        )
        self.ns_tokenizer = CandidateConditionedTokenizer(
            query_dim=self.inputs.query_dim,
            candidate_dim=self.inputs.candidate_dim,
            num_tokens=self.num_ns_tokens,
            token_dim=self.token_dim,
        )
        self.interest_extractor = QueryMixedInterestExtractor(
            token_dim=self.token_dim,
            num_global_tokens=self.num_global_tokens,
            num_realtime_tokens=self.num_realtime_tokens,
            num_attention_heads=int(num_attention_heads),
            attention_dropout=float(attention_dropout),
        )
        self.mixer = HeteroMixer(
            token_dim=self.token_dim,
            num_tokens=self.num_tokens,
            num_layers=int(num_layers),
            num_mix_heads=int(num_mix_heads),
            low_rank_dim=int(low_rank_dim),
            expansion_factor=int(expansion_factor),
            dropout=float(dropout),
            layer_norm_eps=float(layer_norm_eps),
            activation_checkpointing=activation_checkpointing,
        )
        self.heads = nn.ModuleDict(
            {
                name: _TaskTower(
                    input_dim=self.token_dim,
                    hidden_units=head_hidden_units,
                    dropout=float(head_dropout),
                )
                for name in self.task_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        inputs = self.inputs(embedded)
        ns_tokens = self.ns_tokenizer(inputs.query, inputs.candidates)
        interests = self.interest_extractor(
            ns_tokens,
            global_history=inputs.global_history,
            realtime_history=inputs.realtime_history,
            global_valid=inputs.global_valid,
            realtime_valid=inputs.realtime_valid,
        )
        tokens = torch.cat(
            (
                interests.global_tokens,
                interests.realtime_tokens,
                ns_tokens,
            ),
            dim=2,
        )
        representation = self.mixer(tokens).mean(dim=2)
        return structured_ranking_output(
            {
                name: head(representation).squeeze(-1)
                for name, head in self.heads.items()
            }
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_by_task = output["logits"]
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
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
            weighted = loss * self.task_weights.get(name, 1.0)
            total = weighted if total is None else total + weighted
        losses["total"] = total  # type: ignore[assignment]
        return total, losses  # type: ignore[return-value]


class _TaskTower(nn.Sequential):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_units: Sequence[int],
        dropout: float,
    ) -> None:
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_units:
            linear = nn.Linear(previous, int(hidden))
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, nn.SiLU(), nn.Dropout(dropout)))
            previous = int(hidden)
        output = nn.Linear(previous, 1)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        super().__init__(*layers)
