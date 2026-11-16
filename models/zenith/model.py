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
    validate_positive_int,
    validate_positive_int_sequence,
    validate_standard_multitask_contract,
)

from .blocks import ZenithBackbone
from .inputs import ZenithInputs


class ZenithModel(RecAgoraRankingModel):
    """Request-level Zenith family core with TMHSA and TSwiGLU."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        item_id_slot: int,
        query_token_order: Sequence[str],
        query_groups: Mapping[str, Sequence[int]],
        candidate_token_order: Sequence[str],
        candidate_groups: Mapping[str, Sequence[int]],
        history: Mapping[str, object],
        token_dim: int,
        num_interest_tokens: int,
        num_layers: int,
        num_heads: int,
        expansion_factor: int,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        norm_eps: float = 1.0e-6,
        activation_checkpointing: bool = True,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_units: Sequence[int] = (128, 64),
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        for value, name in (
            (token_dim, "token_dim"),
            (num_interest_tokens, "num_interest_tokens"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (expansion_factor, "expansion_factor"),
        ):
            validate_positive_int(value, field_name=name)
        if token_dim % num_heads:
            raise ValueError("Zenith token_dim must be divisible by num_heads")
        validate_dropout(attention_dropout, field_name="attention_dropout")
        validate_dropout(ffn_dropout, field_name="ffn_dropout")
        validate_dropout(head_dropout, field_name="head_dropout")
        finite_positive_float(norm_eps, "norm_eps")
        for value, name in (
            (activation_checkpointing, "activation_checkpointing"),
            (candidate_isolation, "candidate_isolation"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be bool")
        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="Zenith",
        )
        validate_positive_int_sequence(
            head_hidden_units,
            field_name="head_hidden_units",
        )
        head_hidden_units = tuple(int(value) for value in head_hidden_units)
        task_weights = {
            str(name): float(weight) for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(task_weights, task_names, field_name="task_weights")
        for name, weight in task_weights.items():
            finite_nonnegative_float(weight, f"task_weights.{name}")

        params: dict[str, Any] = {
            "item_id_slot": int(item_id_slot),
            "query_token_order": tuple(str(name) for name in query_token_order),
            "query_groups": {
                str(name): tuple(int(slot_id) for slot_id in slot_ids)
                for name, slot_ids in query_groups.items()
            },
            "candidate_token_order": tuple(
                str(name) for name in candidate_token_order
            ),
            "candidate_groups": {
                str(name): tuple(int(slot_id) for slot_id in slot_ids)
                for name, slot_ids in candidate_groups.items()
            },
            "history": dict(history),
            "token_dim": int(token_dim),
            "num_interest_tokens": int(num_interest_tokens),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "expansion_factor": int(expansion_factor),
            "attention_dropout": float(attention_dropout),
            "ffn_dropout": float(ffn_dropout),
            "norm_eps": float(norm_eps),
            "activation_checkpointing": activation_checkpointing,
            "candidate_isolation": candidate_isolation,
            "task_names": task_names,
            "head_hidden_units": head_hidden_units,
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
        }
        super().__init__(params, input_spec)
        self.token_dim = int(token_dim)
        self.task_names = task_names
        self.task_weights = task_weights
        self.candidate_isolation = candidate_isolation
        self.inputs = ZenithInputs(
            input_spec=input_spec,
            item_id_slot=item_id_slot,
            query_token_order=query_token_order,
            query_groups=query_groups,
            candidate_token_order=candidate_token_order,
            candidate_groups=candidate_groups,
            history=history,
            token_dim=self.token_dim,
            num_interest_tokens=num_interest_tokens,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            norm_eps=norm_eps,
        )
        self.num_prime_tokens = self.inputs.num_prime_tokens
        self.backbone = ZenithBackbone(
            token_dim=self.token_dim,
            num_tokens=self.num_prime_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            norm_eps=norm_eps,
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
        representation = self.backbone(inputs.tokens).mean(dim=-2)
        return structured_ranking_output(
            {
                name: head(representation).squeeze(-1).reshape(
                    embedded.batch_size,
                    embedded.list_len,
                )
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
            layers.extend((linear, nn.ReLU(), nn.Dropout(dropout)))
            previous = int(hidden)
        output = nn.Linear(previous, 1)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        super().__init__(*layers)
