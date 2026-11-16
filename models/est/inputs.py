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
class ESTLCAInputsOutput:
    query_fields: tuple[torch.Tensor, ...]
    candidate_fields: tuple[torch.Tensor, ...]
    history_events: torch.Tensor
    history_valid: torch.Tensor


class ESTLCAInputs(nn.Module):
    """Map public embedded slot views into EST-LCA field and event inputs."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        query_slots: Sequence[int],
        candidate_slots: Sequence[int],
        history: Mapping[str, object],
    ) -> None:
        super().__init__()
        history = dict(history)
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="EST-LCA")

        history_spec = require_categorical_history_group(
            input_spec,
            self.history_group_name,
            context="EST-LCA",
        )
        validate_history_order(history_spec, context="EST-LCA")

        self.query_slots = tuple(int(slot) for slot in query_slots)
        self.candidate_slots = tuple(int(slot) for slot in candidate_slots)
        self.history_slots = tuple(
            int(slot) for slot in history.get("sequence_slots", ())
        )
        _require_unique_nonempty(self.query_slots, label="query_slots")
        _require_unique_nonempty(self.candidate_slots, label="candidate_slots")
        _require_unique_nonempty(
            self.history_slots,
            label="history.sequence_slots",
        )
        if self.query_slots != tuple(input_spec.query.slot_ids):
            raise ValueError(
                "EST-LCA query_slots must preserve and exactly cover the public "
                "query view"
            )
        if self.candidate_slots != tuple(input_spec.candidate.slot_ids):
            raise ValueError(
                "EST-LCA candidate_slots must preserve and exactly cover the public "
                "candidate view"
            )
        if self.history_slots != tuple(history_spec.categorical_slot_ids):
            raise ValueError(
                "EST-LCA history.sequence_slots must preserve and exactly cover the "
                "public categorical history view"
            )
        validate_disjoint_slot_groups(
            (self.query_slots, self.candidate_slots, self.history_slots),
            context="EST-LCA query, candidate, and history slots",
        )
        for slot_id in self.query_slots + self.candidate_slots + self.history_slots:
            slot = input_spec.slot(slot_id)
            if slot.encoding != "embedding":
                raise ValueError("EST-LCA only accepts embedded categorical slots")

        self.history_length = int(history_spec.max_length)
        self.query_field_dims = tuple(
            input_spec.slot(slot).output_dim for slot in self.query_slots
        )
        self.candidate_field_dims = tuple(
            input_spec.slot(slot).output_dim for slot in self.candidate_slots
        )
        self.history_event_dim = input_spec.total_dim(self.history_slots)

    def forward(self, batch: RecAgoraModelBatch) -> ESTLCAInputsOutput:
        history = batch.histories[self.history_group_name]
        history_events = torch.cat(
            [
                history.categorical[slot].to_padded(
                    max_length=self.history_length,
                    padding_side=self.padding_side,
                )
                for slot in self.history_slots
            ],
            dim=-1,
        )
        history_valid = _suffix_valid_mask(history.lengths, self.history_length)
        history_events = history_events.masked_fill(
            ~history_valid.unsqueeze(-1),
            0.0,
        )
        return ESTLCAInputsOutput(
            query_fields=tuple(batch.query[slot] for slot in self.query_slots),
            candidate_fields=tuple(
                batch.candidate[slot] for slot in self.candidate_slots
            ),
            history_events=history_events,
            history_valid=history_valid,
        )


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device)[None, :]
    clipped = lengths.clamp(min=0, max=width)
    return positions >= (width - clipped).unsqueeze(1)


def _require_unique_nonempty(slots: tuple[int, ...], *, label: str) -> None:
    if not slots:
        raise ValueError(f"EST-LCA {label} must not be empty")
    if len(slots) != len(set(slots)):
        raise ValueError(f"EST-LCA {label} must contain unique slots")
