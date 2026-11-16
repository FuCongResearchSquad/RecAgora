from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch
from models._validation import require_history_group


@dataclass(frozen=True)
class HSTUInputsOutput:
    history_content: torch.Tensor
    history_action: torch.Tensor
    history_valid: torch.Tensor
    timestamps: torch.Tensor
    time_gaps: torch.Tensor
    candidates: torch.Tensor
    context: torch.Tensor


class HSTUInputs(nn.Module):
    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history: Mapping[str, object],
    ) -> None:
        super().__init__()
        history = dict(history)
        self.group_name = str(history.get("group", "history_group_1"))
        self.action_slot = int(history.get("action_slot", 45040))
        self.timestamp_slot = int(history.get("timestamp_slot", 45042))
        self.time_gap_slot = int(history.get("time_gap_slot", 45073))
        self.padding_side = str(history.get("padding_side", "left"))
        self.direct_validation = str(history.get("direct_validation", "strict"))
        spec = require_history_group(input_spec, self.group_name)
        configured_content = history.get("content_slots")
        self.content_slots = (
            tuple(int(slot) for slot in configured_content)
            if isinstance(configured_content, Sequence) and not isinstance(configured_content, str)
            else tuple(slot for slot in spec.categorical_slot_ids if slot != self.action_slot)
        )
        if set(self.content_slots) | {self.action_slot} != set(spec.categorical_slot_ids):
            raise ValueError("HSTU content and action slots must cover categorical history")
        if set(spec.direct_slot_ids) != {self.timestamp_slot, self.time_gap_slot}:
            raise ValueError("HSTU timestamp/time-gap slots must cover direct history")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("history.padding_side must be left or right")
        if self.direct_validation not in {"strict", "shapes_only", "off"}:
            raise ValueError("unsupported history.direct_validation")
        self.history_length = int(spec.max_length)
        max_events = history.get("max_events")
        if max_events is not None:
            if type(max_events) is not int or not 1 <= max_events <= self.history_length:
                raise ValueError(
                    "history.max_events must be an integer between 1 and "
                    f"{self.history_length}"
                )
        self.max_events = (
            max_events if max_events != self.history_length else None
        )
        self.history_input_dim = input_spec.total_dim(self.content_slots)
        self.action_input_dim = input_spec.slot(self.action_slot).output_dim
        self.candidate_input_dim = input_spec.candidate.total_dim
        self.context_input_dim = input_spec.query.total_dim

    def forward(self, batch: RecAgoraModelBatch) -> HSTUInputsOutput:
        history = batch.histories[self.group_name]
        lengths = history.lengths
        content = torch.cat(
            [
                history.categorical[slot].to_padded(
                    max_length=self.history_length,
                    padding_side=self.padding_side,
                )
                for slot in self.content_slots
            ],
            dim=-1,
        )
        action = history.categorical[self.action_slot].to_padded(
            max_length=self.history_length,
            padding_side=self.padding_side,
        )
        valid = _valid_mask(lengths, self.history_length, self.padding_side)
        if self.max_events is not None:
            positions = torch.arange(self.history_length, device=lengths.device)[None, :]
            if self.padding_side == "left":
                recent = positions >= self.history_length - lengths.clamp_max(
                    self.max_events
                )[:, None]
            else:
                recent = positions >= (lengths - self.max_events).clamp_min(0)[:, None]
            valid = valid & recent
        timestamps = self._direct(history.direct[self.timestamp_slot], lengths)
        time_gaps = self._direct(history.direct[self.time_gap_slot], lengths)
        return HSTUInputsOutput(
            history_content=content,
            history_action=action,
            history_valid=valid,
            timestamps=timestamps,
            time_gaps=time_gaps,
            candidates=batch.candidate.concat(),
            context=batch.query.concat(),
        )

    def _direct(self, value: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        value = value.to(device=lengths.device, dtype=torch.float32)
        # The formal reader validates direct time values before the batch is
        # transferred to an accelerator.  Retain the strict checks for
        # hand-built CPU batches, but never materialize CUDA scalar predicates
        # on the host during model forward.
        if self.direct_validation == "strict" and value.device.type == "cpu":
            if not bool(torch.isfinite(value).all()):
                raise ValueError("HSTU direct history must be finite")
        if self.padding_side == "left":
            return _align_dense_prefix(value, lengths, self.history_length)
        return value


def _align_dense_prefix(value: torch.Tensor, lengths: torch.Tensor, width: int) -> torch.Tensor:
    lengths = lengths.to(device=value.device, dtype=torch.long).clamp(0, width)
    positions = torch.arange(width, device=value.device)[None, :]
    offsets = width - lengths[:, None]
    source = (positions - offsets).clamp_min(0)
    valid = positions >= offsets
    return value.gather(1, source).masked_fill(~valid, 0.0)


def _valid_mask(lengths: torch.Tensor, width: int, padding_side: str) -> torch.Tensor:
    positions = torch.arange(width, device=lengths.device)[None, :]
    if padding_side == "left":
        return positions >= width - lengths[:, None]
    return positions < lengths[:, None]
