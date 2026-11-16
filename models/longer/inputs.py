from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch
from models._validation import (
    require_categorical_history_group,
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
)


@dataclass(frozen=True)
class LongerInputsOutput:
    query: torch.Tensor
    candidates: torch.Tensor
    history_events: torch.Tensor
    history_valid: torch.Tensor


class LongerInputs(nn.Module):
    """Project public Q/C/H views into LONGER-private dense tensors."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        query_slots: Sequence[int],
        candidate_slots: Sequence[int],
        history: Mapping[str, object],
        model_dim: int,
        position_encoding: str,
    ) -> None:
        super().__init__()
        history = dict(history)
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="LONGER")

        history_spec = require_categorical_history_group(
            input_spec,
            self.history_group_name,
            context="LONGER",
        )
        validate_history_order(
            history_spec,
            context="LONGER",
            truncate_side="tail",
        )

        self.query_slots = tuple(int(slot) for slot in query_slots)
        self.candidate_slots = tuple(int(slot) for slot in candidate_slots)
        self.sequence_slots = tuple(
            int(slot) for slot in history.get("sequence_slots", ())
        )
        _require_unique_nonempty(self.query_slots, label="query_slots")
        _require_unique_nonempty(self.candidate_slots, label="candidate_slots")
        _require_unique_nonempty(
            self.sequence_slots,
            label="history.sequence_slots",
        )
        if self.query_slots != tuple(input_spec.query.slot_ids):
            raise ValueError(
                "LONGER query_slots must preserve and exactly cover the public query view"
            )
        if self.candidate_slots != tuple(input_spec.candidate.slot_ids):
            raise ValueError(
                "LONGER candidate_slots must preserve and exactly cover the public "
                "candidate view"
            )
        if self.sequence_slots != tuple(history_spec.categorical_slot_ids):
            raise ValueError(
                "LONGER history.sequence_slots must preserve and exactly cover the "
                "public categorical history view"
            )
        validate_disjoint_slot_groups(
            (self.query_slots, self.candidate_slots, self.sequence_slots),
            context="LONGER query, candidate, and history slots",
        )
        _validate_slot_contracts(
            input_spec,
            query_slots=self.query_slots,
            candidate_slots=self.candidate_slots,
            sequence_slots=self.sequence_slots,
        )
        if position_encoding != "learned_absolute":
            raise ValueError("LONGER position_encoding must be learned_absolute")

        self.history_length = int(history_spec.max_length)
        self.model_dim = int(model_dim)
        self.query_dim = input_spec.total_dim(self.query_slots)
        self.candidate_dim = input_spec.total_dim(self.candidate_slots)
        self.history_feature_dims = tuple(
            int(input_spec.slot(slot).output_dim) for slot in self.sequence_slots
        )
        self.history_projections = nn.ModuleList(
            nn.Linear(feature_dim, self.model_dim, bias=False)
            for feature_dim in self.history_feature_dims
        )
        self.history_bias = nn.Parameter(torch.zeros(self.model_dim))
        self.position_embedding = nn.Embedding(self.history_length, self.model_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.history_projections:
            nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(self.history_bias)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, batch: RecAgoraModelBatch) -> LongerInputsOutput:
        history = batch.histories[self.history_group_name]
        valid = _suffix_valid_mask(history.lengths, self.history_length)
        events = self.history_bias.reshape(1, 1, -1).expand(
            batch.batch_size,
            self.history_length,
            -1,
        )
        for slot, projection in zip(
            self.sequence_slots,
            self.history_projections,
            strict=True,
        ):
            padded = history.categorical[slot].to_padded(
                max_length=self.history_length,
                padding_side=self.padding_side,
            )
            events = events + projection(padded)
        positions = torch.arange(self.history_length, device=events.device)
        events = events + self.position_embedding(positions).unsqueeze(0)
        events = events.masked_fill(~valid.unsqueeze(-1), 0.0)
        return LongerInputsOutput(
            query=batch.query.concat(self.query_slots),
            candidates=batch.candidate.concat(self.candidate_slots),
            history_events=events,
            history_valid=valid,
        )


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    lengths = lengths.clamp(min=0, max=width)
    positions = torch.arange(width, device=lengths.device)[None, :]
    return positions >= (width - lengths).unsqueeze(1)


def _require_unique_nonempty(slots: tuple[int, ...], *, label: str) -> None:
    if not slots:
        raise ValueError(f"LONGER {label} must not be empty")
    if len(slots) != len(set(slots)):
        raise ValueError(f"LONGER {label} must contain unique slots")


def _validate_slot_contracts(
    input_spec: RecAgoraInputSpec,
    *,
    query_slots: Sequence[int],
    candidate_slots: Sequence[int],
    sequence_slots: Sequence[int],
) -> None:
    for slots, view, role, reduction in (
        (query_slots, "query", "request", "mean"),
        (candidate_slots, "candidate", "item", "mean"),
        (sequence_slots, "history", "sequence", "none"),
    ):
        for slot in slots:
            spec = input_spec.slot(slot)
            if spec.view != view or spec.role != role:
                raise ValueError(
                    f"LONGER slot {slot} has view/role={spec.view!r}/{spec.role!r}, "
                    f"expected {view!r}/{role!r}"
                )
            if spec.encoding != "embedding" or spec.reduction != reduction:
                raise ValueError(
                    f"LONGER slot {slot} must use embedding/{reduction}"
                )
