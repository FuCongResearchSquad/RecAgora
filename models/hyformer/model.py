from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

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
    validate_head_dimensions,
    validate_known_weights,
    validate_model_dim_heads,
    validate_positive_int,
    validate_standard_multitask_contract,
)

from .blocks import HyFormerLayer
from .inputs import HyFormerInputs


class GlobalTokenGenerator(nn.Module):
    """Generate candidate-private semantic queries from Q/C and pooled history."""

    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int,
        num_tokens: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.generators = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, model_dim),
                nn.LayerNorm(model_dim, eps=norm_eps),
                nn.SiLU(),
            )
            for _ in range(num_tokens)
        )
        for generator in self.generators:
            projection = generator[0]
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, global_info: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            tuple(generator(global_info) for generator in self.generators),
            dim=2,
        )


class HyFormerModel(RecAgoraRankingModel):
    """RecAgora HyFormer with shared history and isolated candidate queries."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        non_sequence_token_order: Sequence[str],
        non_sequence_groups: Mapping[str, Sequence[int]],
        history: Mapping[str, Any],
        model_dim: int,
        num_heads: int,
        num_layers: int,
        num_global_tokens: int,
        expansion_factor: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1.0e-6,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_dim: int | None = None,
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
    ) -> None:
        model_dim = int(model_dim)
        num_heads = int(num_heads)
        validate_model_dim_heads(model_dim, num_heads)
        validate_positive_int(num_layers, field_name="num_layers")
        validate_positive_int(
            num_global_tokens,
            field_name="num_global_tokens",
        )
        validate_positive_int(
            expansion_factor,
            field_name="expansion_factor",
        )
        validate_dropout(dropout)
        finite_positive_float(layer_norm_eps, "HyFormer LayerNorm epsilon")
        task_names = validate_standard_multitask_contract(
            input_spec,
            task_names,
            candidate_isolation=candidate_isolation,
            context="HyFormer",
        )
        head_hidden_dim = model_dim if head_hidden_dim is None else int(
            head_hidden_dim
        )
        validate_head_dimensions(head_hidden_dim, head_dropout)
        task_weights = {
            str(name): float(weight)
            for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(task_weights, task_names, field_name="task_weights")
        for name, weight in task_weights.items():
            finite_nonnegative_float(weight, f"task_weights.{name}")
        if not any(task_weights.get(name, 1.0) > 0.0 for name in task_names):
            raise ValueError("HyFormer requires at least one positive task weight")

        params: dict[str, Any] = {
            "non_sequence_token_order": tuple(
                str(name) for name in non_sequence_token_order
            ),
            "non_sequence_groups": {
                str(name): tuple(int(slot) for slot in slots)
                for name, slots in non_sequence_groups.items()
            },
            "history": dict(history),
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_layers": int(num_layers),
            "num_global_tokens": int(num_global_tokens),
            "expansion_factor": int(expansion_factor),
            "dropout": float(dropout),
            "layer_norm_eps": float(layer_norm_eps),
            "candidate_isolation": bool(candidate_isolation),
            "task_names": task_names,
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
        }
        super().__init__(params, input_spec)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = int(num_layers)
        self.num_global_tokens = int(num_global_tokens)
        self.task_names = task_names
        self.task_weights = task_weights
        self.inputs = HyFormerInputs(
            input_spec=input_spec,
            model_dim=self.model_dim,
            non_sequence_token_order=non_sequence_token_order,
            non_sequence_groups=non_sequence_groups,
            history=history,
        )
        self.sequence_projection = nn.Linear(
            self.inputs.history_event_dim,
            self.model_dim,
        )
        nn.init.xavier_uniform_(self.sequence_projection.weight)
        nn.init.zeros_(self.sequence_projection.bias)
        self.history_position_embedding = nn.Parameter(
            torch.empty(self.inputs.history_length, self.model_dim)
        )
        nn.init.normal_(self.history_position_embedding, mean=0.0, std=0.02)
        self.global_token_generator = GlobalTokenGenerator(
            input_dim=(
                self.inputs.query_dim
                + self.inputs.candidate_dim
                + self.model_dim
            ),
            model_dim=self.model_dim,
            num_tokens=self.num_global_tokens,
            norm_eps=float(layer_norm_eps),
        )
        self.layers = nn.ModuleList(
            HyFormerLayer(
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                num_global_tokens=self.num_global_tokens,
                num_non_sequence_tokens=self.inputs.num_non_sequence_tokens,
                expansion_factor=int(expansion_factor),
                dropout=float(dropout),
                norm_eps=float(layer_norm_eps),
            )
            for _ in range(self.num_layers)
        )
        self.output_norm = nn.LayerNorm(
            self.model_dim,
            eps=float(layer_norm_eps),
        )
        self.heads = nn.ModuleDict(
            {
                name: _build_task_head(
                    input_dim=self.model_dim,
                    hidden_dim=head_hidden_dim,
                    dropout=float(head_dropout),
                )
                for name in self.task_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        return structured_ranking_output(self._forward_slots(embedded))

    def _forward_slots(self, embedded: RecAgoraModelBatch) -> dict[str, torch.Tensor]:
        inputs = self.inputs(embedded)
        history_valid = inputs.history_valid
        sequence = self.sequence_projection(inputs.history_events)
        sequence = sequence + self.history_position_embedding.unsqueeze(0)
        sequence = sequence.masked_fill(
            ~history_valid.unsqueeze(-1),
            0.0,
        )
        history_weight = history_valid.unsqueeze(-1).to(dtype=sequence.dtype)
        pooled_history = (sequence * history_weight).sum(dim=1) / history_weight.sum(
            dim=1
        ).clamp_min(1.0)
        list_len = embedded.list_len
        global_info = torch.cat(
            (
                inputs.query[:, None, :].expand(-1, list_len, -1),
                inputs.candidates,
                pooled_history[:, None, :].expand(-1, list_len, -1),
            ),
            dim=-1,
        )
        global_tokens = self.global_token_generator(global_info)
        non_sequence_tokens = inputs.non_sequence_tokens
        for layer in self.layers:
            sequence, global_tokens, non_sequence_tokens = layer(
                sequence,
                global_tokens,
                non_sequence_tokens,
                history_valid=history_valid,
            )
        candidate_states = self.output_norm(global_tokens.mean(dim=2))
        return {
            name: head(candidate_states).squeeze(-1)
            for name, head in self.heads.items()
        }

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_by_task = output["logits"]
        metrics: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.task_names:
            logits = logits_by_task[name]
            labels = batch.labels[name].to(
                device=logits.device,
                dtype=logits.dtype,
            )
            point = masked_binary_cross_entropy_with_logits(
                logits,
                labels,
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            weighted = point * float(self.task_weights.get(name, 1.0))
            metrics[f"{name}_point"] = point
            metrics[f"{name}_total"] = weighted
            total = weighted if total is None else total + weighted
        metrics["total"] = total  # type: ignore[assignment]
        return total, metrics  # type: ignore[return-value]


def _build_task_head(
    *,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    for module in head:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return head
