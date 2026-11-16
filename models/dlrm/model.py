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
    normalize_output_label_names,
    require_categorical_history_group,
    validate_dense_stack,
    validate_known_weights,
    validate_mlp_head,
)

from .blocks import DotInteraction


class DLRMModel(RecAgoraRankingModel):
    """Categorical DLRM with dot interaction over public RecAgora slot views."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        interaction: Mapping[str, Any] | None = None,
        action: Mapping[str, Any] | None = None,
        top_hidden_units: Sequence[int] = (1024, 512, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        head_type: str = "mlp",
        head_hidden_dim: int = 128,
        head_dropout: float = 0.0,
        history_group_name: str = "history_group_1",
        output_label_names: Sequence[str] = ("click", "atc", "order"),
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        interaction_config = dict(interaction or {})
        unknown_interaction = set(interaction_config) - {"include_self"}
        if unknown_interaction:
            raise ValueError(
                "interaction has unknown fields: "
                f"{sorted(unknown_interaction)}"
            )
        include_self = bool(interaction_config.get("include_self", False))
        action_config = _parse_action_config(action)

        hidden_units = tuple(int(value) for value in top_hidden_units)
        validate_dense_stack(
            hidden_units,
            field_name="top_hidden_units",
            activation=activation,
            dropout=dropout,
        )
        validate_mlp_head(head_type, head_hidden_dim, head_dropout)

        output_label_names = normalize_output_label_names(output_label_names)
        if set(output_label_names) != set(input_spec.task_names):
            raise ValueError("output_label_names must match RecAgoraInputSpec tasks")
        loss_weights = {
            str(name): float(value) for name, value in (loss_weights or {}).items()
        }
        validate_known_weights(loss_weights, output_label_names)

        history = require_categorical_history_group(
            input_spec,
            history_group_name,
            context="DLRM",
        )

        candidate_slot_ids = tuple(input_spec.candidate.slot_ids)
        query_slot_ids = tuple(input_spec.query.slot_ids)
        history_slot_ids = tuple(history.categorical_slot_ids)
        field_slot_ids = candidate_slot_ids + query_slot_ids + history_slot_ids
        if len(field_slot_ids) != len(set(field_slot_ids)):
            raise ValueError("DLRM field slots must be unique")
        action_slot_id: int | None = None
        action_input_dim: int | None = None
        action_output_dim: int | None = None
        if action_config is not None:
            action_slot_id = int(action_config["slot_id"])
            action_input_dim = int(action_config["input_dim"])
            action_output_dim = int(action_config["output_dim"])
            if action_slot_id not in history_slot_ids:
                raise ValueError(
                    "DLRM action.slot_id must belong to the configured history group"
                )
            action_slot = input_spec.slot(action_slot_id)
            if action_slot.output_dim != action_input_dim:
                raise ValueError(
                    f"DLRM action slot {action_slot_id} output_dim="
                    f"{action_slot.output_dim} does not match action.input_dim="
                    f"{action_input_dim}"
                )
        regular_field_slot_ids = tuple(
            slot_id for slot_id in field_slot_ids if slot_id != action_slot_id
        )
        field_dims = {
            input_spec.slot(slot_id).output_dim
            for slot_id in regular_field_slot_ids
        }
        if len(field_dims) != 1:
            raise ValueError(
                f"DLRM dot interaction requires one embedding dimension, got {field_dims}"
            )
        embedding_dim = next(iter(field_dims))
        if action_output_dim is not None and action_output_dim != embedding_dim:
            raise ValueError(
                f"DLRM action.output_dim={action_output_dim} must match the "
                f"dot-interaction dimension {embedding_dim}"
            )
        non_embedding = [
            slot_id
            for slot_id in field_slot_ids
            if input_spec.slot(slot_id).encoding != "embedding"
        ]
        if non_embedding:
            raise ValueError(
                f"DLRM categorical fields must use embedding encoding: {non_embedding}"
            )

        params: dict[str, Any] = {
            "interaction": {"include_self": include_self},
            "action": action_config,
            "top_hidden_units": hidden_units,
            "activation": activation,
            "dropout": float(dropout),
            "head_type": head_type,
            "head_hidden_dim": int(head_hidden_dim),
            "head_dropout": float(head_dropout),
            "history_group_name": history_group_name,
            "output_label_names": output_label_names,
            "loss_weights": loss_weights,
        }
        super().__init__(params, input_spec)
        self.candidate_slot_ids = candidate_slot_ids
        self.query_slot_ids = query_slot_ids
        self.history_group_name = str(history_group_name)
        self.history_slot_ids = history_slot_ids
        self.field_slot_ids = field_slot_ids
        self.embedding_dim = int(embedding_dim)
        self.action_slot_id = action_slot_id
        self.action_input_dim = action_input_dim
        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.loss_weights = loss_weights

        self.interaction = DotInteraction(
            len(field_slot_ids),
            include_self=include_self,
        )
        if action_config is None:
            self.action_adapter: nn.Module | None = None
        else:
            self.action_adapter = nn.Linear(
                int(action_config["input_dim"]),
                int(action_config["output_dim"]),
                bias=bool(action_config["bias"]),
            )
            nn.init.xavier_uniform_(self.action_adapter.weight)
            if self.action_adapter.bias is not None:
                nn.init.zeros_(self.action_adapter.bias)
        layers: list[nn.Module] = []
        previous = self.interaction.output_dim
        for hidden in hidden_units:
            layers.extend((nn.Linear(previous, hidden), _activation(activation)))
            if dropout:
                layers.append(nn.Dropout(float(dropout)))
            previous = hidden
        self.top = nn.Sequential(*layers)
        self.heads = _build_task_heads(
            previous,
            head_type,
            int(head_hidden_dim),
            float(head_dropout),
            output_label_names,
        )

    @property
    def num_fields(self) -> int:
        return len(self.field_slot_ids)

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        fields = self._assemble_fields(embedded)
        hidden = self.top(self.interaction(fields))
        return structured_ranking_output(
            {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}
        )

    def _assemble_fields(self, batch: RecAgoraModelBatch) -> torch.Tensor:
        candidate = torch.stack(
            [batch.candidate[slot_id] for slot_id in self.candidate_slot_ids],
            dim=-2,
        )
        query = torch.stack(
            [batch.query[slot_id] for slot_id in self.query_slot_ids],
            dim=-2,
        )
        query = query[:, None, :, :].expand(-1, batch.list_len, -1, -1)
        history = batch.histories[self.history_group_name]
        pooled_fields: list[torch.Tensor] = []
        for slot_id in self.history_slot_ids:
            field = _segment_mean(history.categorical[slot_id])
            if slot_id == self.action_slot_id:
                if self.action_adapter is None:
                    raise RuntimeError("DLRM action adapter is not initialized")
                field = self.action_adapter(field)
            pooled_fields.append(field)
        pooled = torch.stack(pooled_fields, dim=-2)
        pooled = pooled[:, None, :, :].expand(-1, batch.list_len, -1, -1)
        return torch.cat((candidate, query, pooled), dim=-2)

    def compute_loss(
        self,
        model_output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model_output = model_output["logits"]
        losses: dict[str, torch.Tensor] = {}
        total: torch.Tensor | None = None
        for name in self.output_label_names:
            loss = masked_binary_cross_entropy_with_logits(
                model_output[name],
                batch.labels[name].to(model_output[name].dtype),
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted = loss * self.loss_weights.get(name, 1.0)
            total = weighted if total is None else total + weighted
        losses["total"] = total  # type: ignore[assignment]
        return total, losses  # type: ignore[return-value]


def _segment_mean(feature: Any) -> torch.Tensor:
    values = feature.values
    lengths = feature.lengths
    batch_size = int(lengths.numel())
    output = values.new_zeros((batch_size, int(values.shape[-1])))
    if values.shape[0]:
        owners = torch.repeat_interleave(
            torch.arange(batch_size, device=lengths.device),
            lengths,
            output_size=int(values.shape[0]),
        )
        output.index_add_(0, owners, values)
    return output / lengths.clamp_min(1).to(values.dtype).unsqueeze(-1)


def _parse_action_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    expected = {
        "slot_id",
        "adapter",
        "input_dim",
        "output_dim",
        "reduction",
        "bias",
    }
    actual = {str(name) for name in config}
    if actual != expected:
        raise ValueError(
            "DLRM action keys mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    for name in ("slot_id", "input_dim", "output_dim"):
        if type(config[name]) is not int:
            raise TypeError(f"DLRM action.{name} must be an integer")
    slot_id = int(config["slot_id"])
    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    if slot_id <= 0:
        raise ValueError("DLRM action.slot_id must be positive")
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("DLRM action dimensions must be positive")
    if config["adapter"] != "linear":
        raise ValueError("DLRM action.adapter must be linear")
    if config["reduction"] != "segment_mean":
        raise ValueError("DLRM action.reduction must be segment_mean")
    if type(config["bias"]) is not bool:
        raise TypeError("DLRM action.bias must be boolean")
    return {
        "slot_id": slot_id,
        "adapter": "linear",
        "input_dim": input_dim,
        "output_dim": output_dim,
        "reduction": "segment_mean",
        "bias": config["bias"],
    }


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    return nn.SiLU()


def _build_task_heads(
    input_dim: int,
    head_type: str,
    hidden_dim: int,
    dropout: float,
    task_names: Sequence[str],
) -> nn.ModuleDict:
    if head_type == "linear":
        return nn.ModuleDict(
            {name: nn.Linear(input_dim, 1) for name in task_names}
        )
    towers = nn.ModuleDict()
    for name in task_names:
        tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for module in tower:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        towers[name] = tower
    return towers
