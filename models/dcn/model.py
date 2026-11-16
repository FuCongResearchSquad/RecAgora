from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

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
    require_categorical_history_group,
    validate_dcn_cross,
    validate_dense_stack,
    validate_masked_mean_reduction,
    validate_mlp_head,
)


class LowRankCrossNet(nn.Module):
    """Low-rank cross layers: x' = x + x0 * (U(Vx) + b)."""

    def __init__(self, input_dim: int, *, layers: int, rank: int) -> None:
        super().__init__()
        self.v = nn.ParameterList(
            [nn.Parameter(torch.empty(rank, input_dim)) for _ in range(layers)]
        )
        self.u = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, rank)) for _ in range(layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(layers)]
        )
        for v, u in zip(self.v, self.u):
            nn.init.xavier_uniform_(v)
            nn.init.xavier_uniform_(u)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        value = x0
        for v, u, bias in zip(self.v, self.u, self.bias):
            projected = torch.matmul(value, v.transpose(0, 1))
            projected = torch.matmul(projected, u.transpose(0, 1)) + bias
            value = value + x0 * projected
        return value


class FullRankCrossNet(nn.Module):
    """Full-rank DCNv2 cross layers: x' = x + x0 * (Wx + b)."""

    def __init__(self, input_dim: int, *, layers: int) -> None:
        super().__init__()
        self.weight = nn.ParameterList(
            [nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(layers)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(layers)]
        )
        for weight in self.weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        value = x0
        for weight, bias in zip(self.weight, self.bias):
            projected = F.linear(value, weight, bias)
            value = value + x0 * projected
        return value


class DCNModel(RecAgoraRankingModel):
    """Configurable DCNv2 with low/full-rank and parallel/stacked variants."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        cross: Mapping[str, Any],
        fusion: str = "parallel",
        deep_hidden_units: Sequence[int] = (512, 256, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        head_type: str = "mlp",
        head_hidden_dim: int = 128,
        head_dropout: float = 0.0,
        history_group_name: str = "history_group_1",
        history_reduction: str = "masked_mean",
        history_max_events: int | None = None,
        output_label_names: Sequence[str] = ("click", "atc", "order"),
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        cross = dict(cross)
        cross_type = validate_dcn_cross(cross, fusion)
        validate_masked_mean_reduction(history_reduction)
        hidden_units = tuple(int(value) for value in deep_hidden_units)
        validate_dense_stack(
            hidden_units,
            field_name="deep_hidden_units",
            activation=activation,
            dropout=dropout,
        )
        validate_mlp_head(head_type, head_hidden_dim, head_dropout)
        output_label_names = tuple(str(name) for name in output_label_names)
        loss_weights = {
            str(name): float(value) for name, value in (loss_weights or {}).items()
        }
        history = require_categorical_history_group(
            input_spec,
            history_group_name,
            context="DCN",
        )
        if history_max_events is not None:
            if type(history_max_events) is not int or not 1 <= history_max_events <= history.max_length:
                raise ValueError(
                    "history_max_events must be an integer between 1 and "
                    f"the public history length ({history.max_length})"
                )
            if history_max_events == history.max_length:
                history_max_events = None

        params = {
            "cross": cross,
            "fusion": fusion,
            "deep_hidden_units": hidden_units,
            "activation": activation,
            "dropout": float(dropout),
            "head_type": head_type,
            "head_hidden_dim": int(head_hidden_dim),
            "head_dropout": float(head_dropout),
            "history_group_name": history_group_name,
            "history_reduction": history_reduction,
            "output_label_names": output_label_names,
            "loss_weights": loss_weights,
        }
        if history_max_events is not None:
            params["history_max_events"] = history_max_events
        super().__init__(params, input_spec)
        self.history_group_name = str(history_group_name)
        self.history_slot_ids = tuple(history.categorical_slot_ids)
        self.history_max_events = history_max_events
        self.fusion = str(fusion)
        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.loss_weights = loss_weights

        input_dim = (
            input_spec.query.total_dim
            + input_spec.candidate.total_dim
            + input_spec.total_dim(self.history_slot_ids)
        )
        cross_layers = int(cross.get("layers", 3))
        cross_rank = int(cross.get("rank", 64))
        if cross_layers <= 0 or (cross_type == "low_rank" and cross_rank <= 0):
            raise ValueError("cross layers and low-rank width must be positive")
        if cross_type == "low_rank":
            self.cross: nn.Module = LowRankCrossNet(
                input_dim,
                layers=cross_layers,
                rank=cross_rank,
            )
        else:
            self.cross = FullRankCrossNet(input_dim, layers=cross_layers)
        deep: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_units:
            deep.extend((nn.Linear(previous, hidden), _activation(activation)))
            if dropout:
                deep.append(nn.Dropout(float(dropout)))
            previous = hidden
        self.deep = nn.Sequential(*deep)
        head_input_dim = input_dim + previous if fusion == "parallel" else previous
        self.heads = _build_task_heads(
            head_input_dim,
            head_type,
            int(head_hidden_dim),
            float(head_dropout),
            output_label_names,
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        model_input = self._assemble_inputs(embedded)
        crossed = self.cross(model_input)
        if self.fusion == "parallel":
            fused = torch.cat((crossed, self.deep(model_input)), dim=-1)
        else:
            fused = self.deep(crossed)
        return structured_ranking_output(
            {name: head(fused).squeeze(-1) for name, head in self.heads.items()}
        )

    def _assemble_inputs(self, batch: RecAgoraModelBatch) -> torch.Tensor:
        query = batch.query.concat()[:, None, :].expand(-1, batch.list_len, -1)
        candidate = batch.candidate.concat()
        history = batch.histories[self.history_group_name]
        if self.history_max_events is None:
            pooled = torch.cat(
                [_segment_mean(history.categorical[slot_id]) for slot_id in self.history_slot_ids],
                dim=-1,
            )
        else:
            pooled = torch.cat(
                [
                    _segment_mean_recent(history.categorical[slot_id], self.history_max_events)
                    for slot_id in self.history_slot_ids
                ],
                dim=-1,
            )
        pooled = pooled[:, None, :].expand(-1, batch.list_len, -1)
        return torch.cat((candidate, query, pooled), dim=-1)

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
    output = values.new_zeros((int(lengths.numel()), int(values.shape[-1])))
    if values.shape[0]:
        owners = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=lengths.device),
            lengths,
            output_size=int(values.shape[0]),
        )
        output.index_add_(0, owners, values)
    return output / lengths.clamp_min(1).to(values.dtype).unsqueeze(-1)


def _segment_mean_recent(feature: Any, max_events: int) -> torch.Tensor:
    values = feature.values
    lengths = feature.lengths
    output = values.new_zeros((int(lengths.numel()), int(values.shape[-1])))
    if values.shape[0]:
        owners = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=lengths.device),
            lengths,
            output_size=int(values.shape[0]),
        )
        starts = torch.cumsum(lengths, dim=0) - lengths
        positions = torch.arange(values.shape[0], device=values.device) - starts[owners]
        keep = positions >= (lengths[owners] - max_events).clamp_min(0)
        output.index_add_(0, owners[keep], values[keep])
    retained = lengths.clamp_max(max_events).clamp_min(1)
    return output / retained.to(values.dtype).unsqueeze(-1)


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
