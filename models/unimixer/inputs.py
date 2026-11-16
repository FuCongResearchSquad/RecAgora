from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn

from recagora import (
    RecAgoraInputSpec,
    RecAgoraModelBatch,
    HistoryGroupSpec,
    HistoryView,
    JaggedFeatureView,
)
from models._validation import (
    require_categorical_history_group,
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
    validate_positive_int,
)


@dataclass(frozen=True)
class UniMixerInputsOutput:
    """Candidate-local fixed-width vectors consumed by the chunk tokenizer."""

    values: torch.Tensor
    history_events: torch.Tensor
    history_valid: torch.Tensor


class CompactHistoryEventProjection(nn.Module):
    """Compress aligned history fields before the bounded H x D flatten.

    A private projection per field preserves heterogeneous embedding widths (the
    Native160 action field is 256D while the remaining categorical fields are
    32D) without ever materializing the original wide event concatenation.
    """

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        slot_ids: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.slot_ids = tuple(int(slot_id) for slot_id in slot_ids)
        if not self.slot_ids:
            raise ValueError("UniMixer history projection requires history slots")
        validate_positive_int(output_dim, field_name="history.event_dim")
        self.input_dims = tuple(
            int(input_spec.slot(slot_id).output_dim) for slot_id in self.slot_ids
        )
        self.input_dim = int(sum(self.input_dims))
        self.output_dim = int(output_dim)
        self.slot_projections = nn.ModuleDict(
            {
                str(slot_id): nn.Linear(input_dim, self.output_dim, bias=False)
                for slot_id, input_dim in zip(
                    self.slot_ids,
                    self.input_dims,
                    strict=True,
                )
            }
        )
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use one shared fan-in bound so a field split does not inflate the
        # aggregate event variance relative to a single concatenated Linear.
        bound = math.sqrt(6.0 / float(self.input_dim + self.output_dim))
        for projection in self.slot_projections.values():
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
        for slot_id in self.slot_ids:
            projected = projected + self.slot_projections[str(slot_id)](
                history.categorical[slot_id].values
            )
        return JaggedFeatureView(
            values=projected,
            lengths=history.lengths,
        ).to_padded(
            max_length=max_length,
            padding_side=padding_side,
        )


class UniMixerInputs(nn.Module):
    """Compile Native160 views into one fixed-width vector per candidate."""

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
        unknown = sorted(
            set(history)
            - {"group", "sequence_slots", "action_slot", "padding_side", "event_dim"}
        )
        if unknown:
            raise ValueError(f"UniMixer history contains unknown fields: {unknown}")

        self.history_group_name = str(history.get("group", "history_group_1"))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="UniMixer")
        history_spec = require_categorical_history_group(
            input_spec,
            self.history_group_name,
            context="UniMixer",
        )
        validate_history_order(
            history_spec,
            context="UniMixer history",
            truncate_side="tail",
        )

        self.query_slots = tuple(int(slot_id) for slot_id in query_slots)
        self.candidate_slots = tuple(int(slot_id) for slot_id in candidate_slots)
        self.sequence_slots = tuple(
            int(slot_id) for slot_id in history.get("sequence_slots", ())
        )
        self.action_slot = int(history.get("action_slot", 45040))
        self.event_dim = int(history.get("event_dim", 16))
        validate_positive_int(self.event_dim, field_name="history.event_dim")
        self._validate_contract(input_spec, history_spec)

        self.max_length = int(history_spec.max_length)
        self.query_dim = int(input_spec.total_dim(self.query_slots))
        self.candidate_dim = int(input_spec.total_dim(self.candidate_slots))
        self.history_flat_dim = self.max_length * self.event_dim
        self.output_dim = self.query_dim + self.candidate_dim + self.history_flat_dim
        self.history_projection = CompactHistoryEventProjection(
            input_spec=input_spec,
            slot_ids=self.sequence_slots,
            output_dim=self.event_dim,
        )

    def _validate_contract(
        self,
        input_spec: RecAgoraInputSpec,
        history_spec: HistoryGroupSpec,
    ) -> None:
        for slots, label in (
            (self.query_slots, "query_slots"),
            (self.candidate_slots, "candidate_slots"),
            (self.sequence_slots, "history.sequence_slots"),
        ):
            if not slots or len(slots) != len(set(slots)):
                raise ValueError(f"UniMixer {label} must be non-empty and unique")
        if self.query_slots != tuple(input_spec.query.slot_ids):
            raise ValueError(
                "UniMixer query_slots must preserve and exactly cover the public query view"
            )
        if self.candidate_slots != tuple(input_spec.candidate.slot_ids):
            raise ValueError(
                "UniMixer candidate_slots must preserve and exactly cover the public candidate view"
            )
        categorical = tuple(history_spec.categorical_slot_ids)
        if self.sequence_slots != categorical:
            raise ValueError(
                "UniMixer history.sequence_slots must preserve and exactly cover the "
                "public categorical history view"
            )
        if self.action_slot not in self.sequence_slots:
            raise ValueError(
                "UniMixer history.action_slot must belong to history.sequence_slots"
            )
        action = input_spec.slot(self.action_slot)
        if (
            action.view != "history"
            or action.role != "sequence"
            or action.encoding != "embedding"
            or action.reduction != "none"
        ):
            raise ValueError("UniMixer action slot has incompatible public semantics")
        validate_disjoint_slot_groups(
            (self.query_slots, self.candidate_slots, self.sequence_slots),
            context="UniMixer query, candidate, and history slots",
        )

    def forward(self, batch: RecAgoraModelBatch) -> UniMixerInputsOutput:
        history = batch.histories[self.history_group_name]
        history_valid = _suffix_valid_mask(history.lengths, self.max_length)
        history_events = self.history_projection(
            history,
            max_length=self.max_length,
            padding_side=self.padding_side,
        )
        history_events = history_events.masked_fill(
            ~history_valid.unsqueeze(-1),
            0.0,
        )
        history_flat = history_events.flatten(start_dim=1)
        query = batch.query.concat(self.query_slots)
        candidates = batch.candidate.concat(self.candidate_slots)
        list_len = int(batch.list_len)
        query = query[:, None, :].expand(-1, list_len, -1)
        history_flat = history_flat[:, None, :].expand(-1, list_len, -1)
        values = torch.cat(
            (
                query,
                candidates,
                history_flat,
            ),
            dim=-1,
        )
        if values.shape[-1] != self.output_dim:
            raise RuntimeError("UniMixer fixed-width input dimension changed at runtime")
        return UniMixerInputsOutput(
            values=values,
            history_events=history_events,
            history_valid=history_valid,
        )


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    bounded = lengths.clamp(min=0, max=width)
    positions = torch.arange(width, device=lengths.device)[None, :]
    return positions >= (width - bounded).unsqueeze(1)
