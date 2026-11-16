from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from recagora import (
    RecAgoraInputSpec,
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
    validate_din_attention_pairs,
    validate_history_padding_side,
    validate_mlp_head,
    validate_positive_int,
)


class LocalActivationUnit(nn.Module):
    def __init__(self, input_dim: int, hidden_units: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim * 4
        for hidden in hidden_units:
            layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: torch.Tensor,
        *,
        chunk_size: int,
        weight_normalization: str,
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        list_len = int(query.shape[1])
        for start in range(0, list_len, chunk_size):
            query_chunk = query[:, start : start + chunk_size]
            chunk = int(query_chunk.shape[1])
            q = query_chunk[:, :, None, :].expand(-1, -1, keys.shape[1], -1)
            k = keys[:, None, :, :].expand(-1, chunk, -1, -1)
            interaction = torch.cat((q, k, q - k, q * k), dim=-1)
            scores = self.network(interaction).squeeze(-1)
            valid = mask[:, None, :].expand(-1, chunk, -1)
            if weight_normalization == "softmax":
                scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
                weights = F.softmax(scores, dim=-1) * valid.to(scores.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(weights.dtype).eps
                )
            else:
                weights = scores * valid.to(scores.dtype)
            outputs.append(torch.matmul(weights, keys))
        return torch.cat(outputs, dim=1)


class DINModel(RecAgoraRankingModel):
    """Candidate-aware DIN whose pair count and feature choices live in YAML."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        attention_pairs: Sequence[Mapping[str, int] | Sequence[int]],
        unpaired_history: Sequence[int],
        attention_hidden_units: Sequence[int] = (80, 40),
        deep_hidden_units: Sequence[int] = (256, 128, 64),
        attention_granularity: str = "fieldwise",
        weight_normalization: str = "softmax",
        candidate_chunk_size: int = 1,
        head_type: str = "mlp",
        head_hidden_dim: int = 64,
        head_dropout: float = 0.0,
        history_group_name: str = "history_group_1",
        history_padding_side: str = "right",
        output_label_names: Sequence[str] = ("click", "atc", "order"),
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        pairs = tuple(_parse_pair(pair) for pair in attention_pairs)
        unpaired = tuple(int(slot_id) for slot_id in unpaired_history)
        attention_hidden_units = tuple(int(value) for value in attention_hidden_units)
        deep_hidden_units = tuple(int(value) for value in deep_hidden_units)
        output_label_names = tuple(str(name) for name in output_label_names)
        attention_granularity = str(attention_granularity)
        weight_normalization = str(weight_normalization)
        loss_weights = {
            str(name): float(value) for name, value in (loss_weights or {}).items()
        }
        require_din_pairs(pairs, context="DIN")
        if any(value <= 0 for value in attention_hidden_units + deep_hidden_units):
            raise ValueError("DIN hidden units must be positive")
        validate_mlp_head(head_type, head_hidden_dim, head_dropout)
        validate_positive_int(candidate_chunk_size, field_name="candidate_chunk_size")
        if attention_granularity not in {"fieldwise", "eventwise"}:
            raise ValueError(
                "attention_granularity must be fieldwise or eventwise"
            )
        if weight_normalization not in {"none", "softmax"}:
            raise ValueError("weight_normalization must be none or softmax")
        validate_history_padding_side(history_padding_side)
        history = require_categorical_history_group(
            input_spec,
            history_group_name,
            context="DIN",
        )
        history_slots = set(history.categorical_slot_ids)
        paired_history_order, embedding_dim = validate_din_attention_pairs(
            input_spec,
            pairs,
        )
        paired_history = set(paired_history_order)
        if any(slot not in history_slots for _, slot in pairs):
            raise ValueError("DIN attention histories must be selected history slots")
        if set(unpaired) != history_slots - paired_history:
            raise ValueError("unpaired_history must exactly cover non-paired history slots")

        params: dict[str, Any] = {
            "attention_pairs": pairs,
            "unpaired_history": unpaired,
            "attention_hidden_units": attention_hidden_units,
            "deep_hidden_units": deep_hidden_units,
            "attention_granularity": attention_granularity,
            "weight_normalization": weight_normalization,
            "candidate_chunk_size": int(candidate_chunk_size),
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
        self.attention_granularity = attention_granularity
        self.weight_normalization = weight_normalization
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.history_group_name = str(history_group_name)
        self.history_padding_side = str(history_padding_side)
        self.output_label_names = output_label_names
        self.task_names = output_label_names
        self.loss_weights = loss_weights
        attention_input_dim = (
            embedding_dim
            if attention_granularity == "fieldwise"
            else embedding_dim * len(pairs)
        )
        self.attention = LocalActivationUnit(
            attention_input_dim,
            attention_hidden_units,
        )

        static_dim = (
            input_spec.candidate.total_dim
            + input_spec.query.total_dim
            + input_spec.total_dim(unpaired)
        )
        input_dim = static_dim + embedding_dim * len(pairs)
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in deep_hidden_units:
            layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
            previous = hidden
        self.deep = nn.Sequential(*layers)
        self.heads = _build_task_heads(
            previous,
            head_type,
            int(head_hidden_dim),
            float(head_dropout),
            output_label_names,
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        history = embedded.histories[self.history_group_name]
        positions = torch.arange(history.max_length, device=history.lengths.device)[None, :]
        if self.history_padding_side == "right":
            mask = positions < history.lengths[:, None]
        else:
            mask = positions >= history.max_length - history.lengths[:, None]
        padded = {
            slot_id: history.categorical[slot_id].to_padded(
                max_length=history.max_length,
                padding_side=self.history_padding_side,
            )
            for slot_id in history.categorical
        }
        if self.attention_granularity == "fieldwise":
            interests = tuple(
                self.attention(
                    embedded.candidate[target],
                    padded[history_slot],
                    mask,
                    chunk_size=self.candidate_chunk_size,
                    weight_normalization=self.weight_normalization,
                )
                for target, history_slot in self.attention_pairs
            )
        else:
            target_event = torch.cat(
                [embedded.candidate[target] for target, _ in self.attention_pairs],
                dim=-1,
            )
            history_events = torch.cat(
                [padded[history_slot] for _, history_slot in self.attention_pairs],
                dim=-1,
            )
            interests = (
                self.attention(
                    target_event,
                    history_events,
                    mask,
                    chunk_size=self.candidate_chunk_size,
                    weight_normalization=self.weight_normalization,
                ),
            )
        unpaired = torch.cat(
            [_masked_mean(padded[slot_id], mask) for slot_id in self.unpaired_history],
            dim=-1,
        )[:, None, :].expand(-1, embedded.list_len, -1)
        query = embedded.query.concat()[:, None, :].expand(-1, embedded.list_len, -1)
        model_input = torch.cat(
            (embedded.candidate.concat(), query, unpaired, *interests), dim=-1
        )
        hidden = self.deep(model_input)
        return structured_ranking_output(
            {name: head(hidden).squeeze(-1) for name, head in self.heads.items()}
        )

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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask.to(values.dtype).unsqueeze(-1)
    return masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(values.dtype).unsqueeze(-1)


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
