from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
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
    parse_din_pair as _parse_pair,
    require_categorical_history_group,
    require_din_pairs,
    validate_din_pair_membership,
    validate_dcn_cross,
    validate_dropout,
    validate_history_padding_side,
    validate_mlp_head,
    validate_positive_int_sequence,
    validate_positive_int,
)
from models.dcn.model import FullRankCrossNet, LowRankCrossNet


class EventwiseLocalActivationUnit(nn.Module):
    """Exact DIN local activation without materializing the 4D interaction."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        *,
        activation: str,
        weight_normalization: str,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.weight_normalization = str(weight_normalization)
        self.activation_checkpointing = bool(activation_checkpointing)
        hidden_units = tuple(int(value) for value in hidden_units)
        if self.input_dim <= 0 or not hidden_units or any(
            value <= 0 for value in hidden_units
        ):
            raise ValueError("DIN attention dimensions must be positive")
        if self.weight_normalization not in {"none", "softmax"}:
            raise ValueError("weight_normalization must be none or softmax")

        layers: list[nn.Module] = []
        previous = self.input_dim * 4
        for hidden in hidden_units:
            layers.extend((nn.Linear(previous, hidden), _activation(activation)))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, int(query.shape[1]), int(chunk_size)):
            query_chunk = query[:, start : start + int(chunk_size)]
            if (
                self.activation_checkpointing
                and self.training
                and query_chunk.device.type == "cuda"
            ):
                output = checkpoint(
                    self._aggregate_chunk,
                    query_chunk,
                    keys,
                    mask,
                    use_reentrant=False,
                )
            else:
                output = self._aggregate_chunk(query_chunk, keys, mask)
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    def _aggregate_chunk(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self._scores(query, keys)
        valid = mask[:, None, :].expand(-1, int(query.shape[1]), -1)
        if self.weight_normalization == "softmax":
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = F.softmax(scores, dim=-1) * valid.to(scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).eps
            )
        else:
            weights = scores * valid.to(scores.dtype)
        return torch.matmul(weights, keys)

    def _scores(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        first = self.network[0]
        query_weight, key_weight, difference_weight, product_weight = (
            first.weight.split(self.input_dim, dim=1)
        )
        hidden = F.linear(
            query,
            query_weight + difference_weight,
            first.bias,
        )[:, :, None, :]
        hidden = hidden + F.linear(
            keys,
            key_weight - difference_weight,
            None,
        )[:, None, :, :]
        hidden = hidden + torch.einsum(
            "bld,btd,hd->blth",
            query,
            keys,
            product_weight,
        )
        for layer in self.network[1:]:
            hidden = layer(hidden)
        return hidden.squeeze(-1)


class DINDCNv2Model(RecAgoraRankingModel):
    """Event-wise DIN history encoder followed by configurable DCNv2 blocks."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        attention_pairs: Sequence[Mapping[str, int] | Sequence[int]],
        unpaired_history: Sequence[int],
        attention_granularity: str = "eventwise",
        attention_hidden_units: Sequence[int] = (80, 40),
        attention_activation: str = "relu",
        weight_normalization: str = "none",
        candidate_chunk_size: int = 1,
        activation_checkpointing: bool = True,
        cross: Mapping[str, Any],
        fusion: str = "parallel",
        deep_hidden_units: Sequence[int] = (512, 256, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        head_type: str = "mlp",
        head_hidden_dim: int = 128,
        head_dropout: float = 0.0,
        history_group_name: str = "history_group_1",
        history_padding_side: str = "right",
        output_label_names: Sequence[str] = ("click", "atc", "order"),
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        pairs = tuple(_parse_pair(pair) for pair in attention_pairs)
        unpaired = tuple(int(slot_id) for slot_id in unpaired_history)
        attention_hidden_units = tuple(int(value) for value in attention_hidden_units)
        hidden_units = tuple(int(value) for value in deep_hidden_units)
        output_label_names = tuple(str(name) for name in output_label_names)
        cross = dict(cross)
        loss_weights = {
            str(name): float(value) for name, value in (loss_weights or {}).items()
        }
        if attention_granularity != "eventwise":
            raise ValueError("DIN-DCNv2 requires attention_granularity=eventwise")
        require_din_pairs(pairs, context="DIN-DCNv2")
        validate_positive_int(candidate_chunk_size, field_name="candidate_chunk_size")
        validate_history_padding_side(history_padding_side)
        validate_positive_int_sequence(
            hidden_units,
            field_name="deep_hidden_units",
        )
        validate_dropout(dropout)
        validate_mlp_head(head_type, head_hidden_dim, head_dropout)

        history = require_categorical_history_group(
            input_spec,
            history_group_name,
            context="DIN-DCNv2",
        )
        history_slots = tuple(history.categorical_slot_ids)
        paired_history = validate_din_pair_membership(
            input_spec,
            pairs,
            context="DIN-DCNv2",
        )
        if any(history_slot not in set(history_slots) for history_slot in paired_history):
            raise ValueError("DIN-DCNv2 pairs must use selected history slots")
        expected_unpaired = tuple(
            slot_id for slot_id in history_slots if slot_id not in set(paired_history)
        )
        if unpaired != expected_unpaired:
            raise ValueError(
                "unpaired_history must follow the remaining canonical history order"
            )
        expected_paired = tuple(
            slot_id for slot_id in history_slots if slot_id not in set(unpaired)
        )
        if paired_history != expected_paired:
            raise ValueError("attention_pairs must follow canonical history order")

        pair_dims = tuple(
            input_spec.slot(history_slot).output_dim for _, history_slot in pairs
        )
        for (target, history_slot), history_dim in zip(pairs, pair_dims):
            if input_spec.slot(target).output_dim != history_dim:
                raise ValueError(
                    "DIN-DCNv2 target/history pair dimensions must match"
                )
        event_dim = sum(pair_dims)

        cross_type = validate_dcn_cross(cross, fusion)

        params: dict[str, Any] = {
            "attention_pairs": pairs,
            "unpaired_history": unpaired,
            "attention_granularity": attention_granularity,
            "attention_hidden_units": attention_hidden_units,
            "attention_activation": attention_activation,
            "weight_normalization": weight_normalization,
            "candidate_chunk_size": int(candidate_chunk_size),
            "activation_checkpointing": bool(activation_checkpointing),
            "cross": cross,
            "fusion": fusion,
            "deep_hidden_units": hidden_units,
            "activation": activation,
            "dropout": float(dropout),
            "head_type": head_type,
            "head_hidden_dim": int(head_hidden_dim),
            "head_dropout": float(head_dropout),
            "history_group_name": history_group_name,
            "history_padding_side": history_padding_side,
            "output_label_names": output_label_names,
            "loss_weights": loss_weights,
        }
        super().__init__(params, input_spec)
        self.attention_pairs = pairs
        self.unpaired_history = unpaired
        self.history_slot_ids = history_slots
        self.pair_dims = pair_dims
        self.history_group_name = str(history_group_name)
        self.history_padding_side = str(history_padding_side)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.fusion = str(fusion)
        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.loss_weights = loss_weights

        self.attention = EventwiseLocalActivationUnit(
            event_dim,
            attention_hidden_units,
            activation=attention_activation,
            weight_normalization=weight_normalization,
            activation_checkpointing=activation_checkpointing,
        )
        input_dim = (
            input_spec.candidate.total_dim
            + input_spec.query.total_dim
            + input_spec.total_dim(history_slots)
        )
        cross_layers = int(cross.get("layers", 3))
        if cross_type == "low_rank":
            self.cross: nn.Module = LowRankCrossNet(
                input_dim,
                layers=cross_layers,
                rank=int(cross.get("rank", 64)),
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
        history = batch.histories[self.history_group_name]
        positions = torch.arange(
            history.max_length,
            device=history.lengths.device,
        )[None, :]
        if self.history_padding_side == "right":
            mask = positions < history.lengths[:, None]
        else:
            mask = positions >= history.max_length - history.lengths[:, None]

        target_event = torch.cat(
            [batch.candidate[target] for target, _ in self.attention_pairs],
            dim=-1,
        )
        history_event = torch.cat(
            [
                history.categorical[history_slot].to_padded(
                    max_length=history.max_length,
                    padding_side=self.history_padding_side,
                )
                for _, history_slot in self.attention_pairs
            ],
            dim=-1,
        )
        interest_event = self.attention(
            target_event,
            history_event,
            mask,
            chunk_size=self.candidate_chunk_size,
        )
        paired_values = {
            history_slot: value
            for (_, history_slot), value in zip(
                self.attention_pairs,
                torch.split(interest_event, self.pair_dims, dim=-1),
            )
        }
        for slot_id in self.unpaired_history:
            paired_values[slot_id] = _segment_mean(
                history.categorical[slot_id]
            )[:, None, :].expand(-1, batch.list_len, -1)
        history_input = torch.cat(
            [paired_values[slot_id] for slot_id in self.history_slot_ids],
            dim=-1,
        )
        query = batch.query.concat()[:, None, :].expand(-1, batch.list_len, -1)
        return torch.cat((batch.candidate.concat(), query, history_input), dim=-1)

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


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation {name!r}")


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
