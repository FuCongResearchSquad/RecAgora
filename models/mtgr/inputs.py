from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch
from models._validation import (
    require_history_group,
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
)


@dataclass(frozen=True)
class MTGRInputsOutput:
    user_features: tuple[torch.Tensor, ...]
    history_events: torch.Tensor
    candidates: torch.Tensor
    history_valid: torch.Tensor
    timestamps: torch.Tensor


class MTGRInputs(nn.Module):
    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
        candidate_groups: Mapping[str, Sequence[int]],
    ) -> None:
        super().__init__()
        history = dict(history)
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.action_slot = int(history.get("action_slot", 45040))
        self.timestamp_slot = int(history.get("timestamp_slot", 45042))
        self.time_gap_slot = int(history.get("time_gap_slot", 45073))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="MTGR")
        history_spec = require_history_group(input_spec, self.history_group_name)
        validate_history_order(history_spec, context="MTGR offline history")
        self.content_slots = tuple(
            slot for slot in history_spec.categorical_slot_ids if slot != self.action_slot
        )
        if set(history_spec.direct_slot_ids) != {self.timestamp_slot, self.time_gap_slot}:
            raise ValueError("MTGR direct slots do not match selected history")
        self.intrinsic_slots = tuple(int(slot) for slot in candidate_groups["intrinsic"])
        self.cross_slots = tuple(int(slot) for slot in candidate_groups["cross"])
        flattened = self.intrinsic_slots + self.cross_slots
        validate_disjoint_slot_groups(
            (self.intrinsic_slots, self.cross_slots),
            context="MTGR candidate groups",
        )
        if set(flattened) != set(input_spec.candidate.slot_ids):
            raise ValueError("MTGR candidate groups must exactly cover candidate slots")
        self.user_slots = tuple(int(slot) for slot in input_spec.query.slot_ids)
        if not self.user_slots:
            raise ValueError("MTGR requires at least one user-side feature")
        self.history_length = int(history_spec.max_length)
        self.history_content_dim = input_spec.total_dim(self.content_slots)
        self.action_dim = input_spec.slot(self.action_slot).output_dim
        self.user_feature_dims = tuple(
            input_spec.slot(slot).output_dim for slot in self.user_slots
        )
        self.intrinsic_dim = input_spec.total_dim(self.intrinsic_slots)
        self.cross_dim = input_spec.total_dim(self.cross_slots)
        self.history_event_dim = self.history_content_dim + self.action_dim + 1
        self.candidate_dim = self.intrinsic_dim + self.cross_dim

    def forward(self, batch: RecAgoraModelBatch) -> MTGRInputsOutput:
        history = batch.histories[self.history_group_name]
        lengths = history.lengths
        content = torch.cat(
            [
                history.categorical[slot].to_padded(
                    max_length=self.history_length,
                    padding_side="left",
                )
                for slot in self.content_slots
            ],
            dim=-1,
        )
        action = history.categorical[self.action_slot].to_padded(
            max_length=self.history_length,
            padding_side="left",
        )
        timestamps = _right_align_dense(
            history.direct[self.timestamp_slot], lengths, self.history_length
        )
        time_gaps = _right_align_dense(
            history.direct[self.time_gap_slot], lengths, self.history_length
        )
        valid = _suffix_valid_mask(lengths, self.history_length)
        transformed_gaps = _nonnegative_log1p(time_gaps).unsqueeze(-1)
        events = torch.cat((content, action, transformed_gaps), dim=-1)
        return MTGRInputsOutput(
            user_features=tuple(batch.query[slot] for slot in self.user_slots),
            history_events=events,
            candidates=torch.cat(
                (
                    batch.candidate.concat(self.intrinsic_slots),
                    batch.candidate.concat(self.cross_slots),
                ),
                dim=-1,
            ),
            history_valid=valid,
            timestamps=timestamps,
        )


def _right_align_dense(values: torch.Tensor, lengths: torch.Tensor, width: int) -> torch.Tensor:
    values = values.to(device=lengths.device, dtype=torch.float32)
    positions = torch.arange(width, device=values.device)[None, :]
    source = (positions - (width - lengths).unsqueeze(1)).clamp_min(0)
    aligned = values.gather(1, source)
    return aligned.masked_fill(~_suffix_valid_mask(lengths, width), 0.0)


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device)[None, :]
    return positions >= (width - lengths).unsqueeze(1)


def _nonnegative_log1p(values: torch.Tensor) -> torch.Tensor:
    return torch.log1p(values.clamp_min(0.0))
