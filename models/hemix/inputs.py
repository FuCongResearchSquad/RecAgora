from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch, HistoryView, JaggedFeatureView
from models._validation import (
    require_categorical_history_group,
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
    validate_positive_int,
)


@dataclass(frozen=True)
class HeMixInputsOutput:
    query: torch.Tensor
    candidates: torch.Tensor
    global_history: torch.Tensor
    realtime_history: torch.Tensor
    global_valid: torch.Tensor
    realtime_valid: torch.Tensor


class HistoryEventProjection(nn.Module):
    """Project aligned jagged fields without materializing their wide concat."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        slot_ids: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.slot_ids = tuple(int(slot_id) for slot_id in slot_ids)
        self.input_dims = tuple(
            int(input_spec.slot(slot_id).output_dim) for slot_id in self.slot_ids
        )
        self.input_dim = int(sum(self.input_dims))
        self.output_dim = int(output_dim)
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, self.output_dim, bias=False)
            for input_dim in self.input_dims
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6.0 / float(self.input_dim + self.output_dim))
        for projection in self.projections:
            nn.init.uniform_(projection.weight, -bound, bound)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        history: HistoryView,
        *,
        max_length: int,
        padding_side: str,
    ) -> torch.Tensor:
        first = history.categorical[self.slot_ids[0]]
        event_count = int(first.values.shape[0])
        projected = self.bias.reshape(1, -1).expand(event_count, -1)
        for slot_id, projection in zip(
            self.slot_ids,
            self.projections,
            strict=True,
        ):
            projected = projected + projection(history.categorical[slot_id].values)
        return JaggedFeatureView(
            values=projected,
            lengths=history.lengths,
        ).to_padded(
            max_length=max_length,
            padding_side=padding_side,
        )


class HeMixInputs(nn.Module):
    """Compile the public Common157 views into HeMix-private dense tensors."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        query_slots: Sequence[int],
        candidate_slots: Sequence[int],
        history: Mapping[str, object],
        token_dim: int,
    ) -> None:
        super().__init__()
        history = dict(history)
        unknown = sorted(
            set(history) - {"group", "sequence_slots", "padding_side", "recent_length"}
        )
        if unknown:
            raise ValueError(f"HeMix history contains unknown fields: {unknown}")

        self.history_group_name = str(history.get("group", "history_group_1"))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="HeMix")
        history_spec = require_categorical_history_group(
            input_spec,
            self.history_group_name,
            context="HeMix",
        )
        validate_history_order(
            history_spec,
            context="HeMix history",
            truncate_side="tail",
        )

        self.query_slots = tuple(int(slot_id) for slot_id in query_slots)
        self.candidate_slots = tuple(int(slot_id) for slot_id in candidate_slots)
        self.sequence_slots = tuple(
            int(slot_id) for slot_id in history.get("sequence_slots", ())
        )
        _require_unique_nonempty(self.query_slots, label="query_slots")
        _require_unique_nonempty(self.candidate_slots, label="candidate_slots")
        _require_unique_nonempty(self.sequence_slots, label="history.sequence_slots")
        if self.query_slots != tuple(input_spec.query.slot_ids):
            raise ValueError(
                "HeMix query_slots must preserve and exactly cover the public query view"
            )
        if self.candidate_slots != tuple(input_spec.candidate.slot_ids):
            raise ValueError(
                "HeMix candidate_slots must preserve and exactly cover the public "
                "candidate view"
            )
        if self.sequence_slots != tuple(history_spec.categorical_slot_ids):
            raise ValueError(
                "HeMix history.sequence_slots must preserve and exactly cover the "
                "public categorical history view"
            )
        validate_disjoint_slot_groups(
            (self.query_slots, self.candidate_slots, self.sequence_slots),
            context="HeMix query, candidate, and history slots",
        )

        self.history_length = int(history_spec.max_length)
        recent_length = history.get("recent_length", 20)
        validate_positive_int(recent_length, field_name="history.recent_length")
        self.recent_length = int(recent_length)
        if self.recent_length >= self.history_length:
            raise ValueError(
                "HeMix history.recent_length must be smaller than history max_length"
            )
        self.global_length = self.history_length - self.recent_length
        self.query_dim = int(input_spec.total_dim(self.query_slots))
        self.candidate_dim = int(input_spec.total_dim(self.candidate_slots))
        self.history_event_dim = int(input_spec.total_dim(self.sequence_slots))
        if self.history_event_dim != int(history_spec.event_total_dim):
            raise ValueError("HeMix history dimensions changed from the input spec")
        self.token_dim = int(token_dim)
        self.history_projection = HistoryEventProjection(
            input_spec=input_spec,
            slot_ids=self.sequence_slots,
            output_dim=self.token_dim,
        )

    def forward(self, batch: RecAgoraModelBatch) -> HeMixInputsOutput:
        history = batch.histories[self.history_group_name]
        valid = _suffix_valid_mask(history.lengths, self.history_length)
        events = self.history_projection(
            history,
            max_length=self.history_length,
            padding_side=self.padding_side,
        )
        events = events.masked_fill(~valid.unsqueeze(-1), 0.0)
        return HeMixInputsOutput(
            query=batch.query.concat(self.query_slots),
            candidates=batch.candidate.concat(self.candidate_slots),
            global_history=events[:, : self.global_length],
            realtime_history=events[:, self.global_length :],
            global_valid=valid[:, : self.global_length],
            realtime_valid=valid[:, self.global_length :],
        )


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    bounded = lengths.clamp(min=0, max=width)
    positions = torch.arange(width, device=lengths.device)[None, :]
    return positions >= (width - bounded).unsqueeze(1)


def _require_unique_nonempty(slots: tuple[int, ...], *, label: str) -> None:
    if not slots:
        raise ValueError(f"HeMix {label} must not be empty")
    if len(slots) != len(set(slots)):
        raise ValueError(f"HeMix {label} must contain unique slots")
