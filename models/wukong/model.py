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
    validate_dropout,
    validate_known_weights,
    validate_positive_int,
    validate_positive_int_sequence,
    validate_standard_multitask_contract,
)

from .blocks import WukongBackbone
from .inputs import WukongInputs, WukongInputsOutput


class WukongModel(RecAgoraRankingModel):
    """Stacked optimized-FM ranking backbone over Native160 categorical fields."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history_group_name: str,
        action_slot: int,
        interaction_dim: int,
        num_layers: int,
        projection_rank: int,
        fmb_output_tokens: int,
        lcb_output_tokens: int,
        fmb_hidden_units: Sequence[int],
        lcb_dropout: float = 0.0,
        activation_checkpointing: bool = True,
        candidate_chunk_size: int = 4,
        candidate_isolation: bool = True,
        task_names: Sequence[str] = ("click", "atc", "order"),
        head_hidden_units: Sequence[int] = (256, 128),
        head_dropout: float = 0.0,
        task_weights: Mapping[str, float] | None = None,
        history_max_events: int | None = None,
    ) -> None:
        for value, name in (
            (interaction_dim, "interaction_dim"),
            (num_layers, "num_layers"),
            (projection_rank, "projection_rank"),
            (fmb_output_tokens, "fmb_output_tokens"),
            (lcb_output_tokens, "lcb_output_tokens"),
            (candidate_chunk_size, "candidate_chunk_size"),
        ):
            validate_positive_int(value, field_name=name)
        validate_positive_int_sequence(
            fmb_hidden_units,
            field_name="fmb_hidden_units",
        )
        validate_positive_int_sequence(
            head_hidden_units,
            field_name="head_hidden_units",
        )
        validate_dropout(lcb_dropout, field_name="lcb_dropout")
        validate_dropout(head_dropout, field_name="head_dropout")
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
            context="Wukong",
        )
        fmb_hidden_units = tuple(int(value) for value in fmb_hidden_units)
        head_hidden_units = tuple(int(value) for value in head_hidden_units)
        task_weights = {
            str(name): float(weight)
            for name, weight in (task_weights or {}).items()
        }
        validate_known_weights(task_weights, task_names, field_name="task_weights")
        for name, weight in task_weights.items():
            finite_nonnegative_float(weight, f"task_weights.{name}")

        params: dict[str, Any] = {
            "history_group_name": str(history_group_name),
            "action_slot": int(action_slot),
            "interaction_dim": int(interaction_dim),
            "num_layers": int(num_layers),
            "projection_rank": int(projection_rank),
            "fmb_output_tokens": int(fmb_output_tokens),
            "lcb_output_tokens": int(lcb_output_tokens),
            "fmb_hidden_units": fmb_hidden_units,
            "lcb_dropout": float(lcb_dropout),
            "activation_checkpointing": activation_checkpointing,
            "candidate_chunk_size": int(candidate_chunk_size),
            "candidate_isolation": candidate_isolation,
            "task_names": task_names,
            "head_hidden_units": head_hidden_units,
            "head_dropout": float(head_dropout),
            "task_weights": task_weights,
        }
        if history_max_events is not None:
            params["history_max_events"] = history_max_events
        super().__init__(params, input_spec)
        self.task_names = task_names
        self.task_weights = task_weights
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.inputs = WukongInputs(
            input_spec=input_spec,
            history_group_name=history_group_name,
            action_slot=int(action_slot),
            interaction_dim=int(interaction_dim),
            history_max_events=history_max_events,
        )
        self.backbone = WukongBackbone(
            num_input_tokens=self.inputs.num_fields,
            interaction_dim=int(interaction_dim),
            num_layers=int(num_layers),
            projection_rank=int(projection_rank),
            fmb_output_tokens=int(fmb_output_tokens),
            lcb_output_tokens=int(lcb_output_tokens),
            fmb_hidden_units=fmb_hidden_units,
            lcb_dropout=float(lcb_dropout),
            activation_checkpointing=activation_checkpointing,
        )
        self.heads = nn.ModuleDict(
            {
                name: _TaskTower(
                    input_dim=self.backbone.output_dim,
                    hidden_units=head_hidden_units,
                    dropout=float(head_dropout),
                )
                for name in self.task_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        inputs = self.inputs(embedded)
        outputs: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.task_names
        }
        for start in range(0, embedded.list_len, self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, embedded.list_len)
            chunk = self._score_candidate_chunk(
                inputs,
                candidate_start=start,
                candidate_end=end,
            )
            for name in self.task_names:
                outputs[name].append(chunk[name])
        return structured_ranking_output(
            {name: torch.cat(values, dim=1) for name, values in outputs.items()}
        )

    def _score_candidate_chunk(
        self,
        inputs: WukongInputsOutput,
        *,
        candidate_start: int,
        candidate_end: int,
    ) -> dict[str, torch.Tensor]:
        candidates = inputs.candidate_fields[:, candidate_start:candidate_end]
        batch_size, chunk_size = candidates.shape[:2]
        query = inputs.query_fields[:, None].expand(-1, chunk_size, -1, -1)
        history = inputs.history_fields[:, None].expand(-1, chunk_size, -1, -1)
        fields = torch.cat((candidates, query, history), dim=2)
        fields = fields.reshape(
            batch_size * chunk_size,
            self.inputs.num_fields,
            self.inputs.interaction_dim,
        )
        representation = self.backbone(fields).flatten(start_dim=1)
        return {
            name: head(representation).reshape(batch_size, chunk_size)
            for name, head in self.heads.items()
        }

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.task_names:
            logits = output["logits"][name]
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
            _init_linear(linear)
            layers.extend((linear, nn.ReLU(), nn.Dropout(dropout)))
            previous = int(hidden)
        output = nn.Linear(previous, 1)
        _init_linear(output)
        layers.append(output)
        super().__init__(*layers)


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
