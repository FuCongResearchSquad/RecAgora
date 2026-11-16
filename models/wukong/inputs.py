from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch, JaggedFeatureView
from models._validation import require_categorical_history_group


@dataclass(frozen=True)
class WukongInputsOutput:
    query_fields: torch.Tensor
    candidate_fields: torch.Tensor
    history_fields: torch.Tensor


class FieldProjection(nn.Module):
    """Map one public field embedding to Wukong's common field width."""

    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("Wukong field dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(self.input_dim, self.output_dim, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if int(values.shape[-1]) != self.input_dim:
            raise ValueError(
                f"Wukong field projection expected {self.input_dim} dimensions, "
                f"got {values.shape[-1]}"
            )
        return self.projection(values)


class WukongInputs(nn.Module):
    """Preserve every selected categorical field as one interaction token."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history_group_name: str,
        action_slot: int,
        interaction_dim: int,
        history_max_events: int | None = None,
    ) -> None:
        super().__init__()
        if interaction_dim <= 0:
            raise ValueError("Wukong interaction_dim must be positive")
        history = require_categorical_history_group(
            input_spec,
            history_group_name,
            context="Wukong",
        )
        if history.direct_slot_ids:
            raise ValueError("Wukong architecture-core accepts categorical history only")
        self.history_group_name = str(history_group_name)
        self.action_slot = int(action_slot)
        self.interaction_dim = int(interaction_dim)
        if history_max_events is not None:
            if (
                type(history_max_events) is not int
                or not 1 <= history_max_events <= history.max_length
            ):
                raise ValueError(
                    "history_max_events must be an integer between 1 and "
                    f"{history.max_length}"
                )
            if history_max_events == history.max_length:
                history_max_events = None
        self.history_max_events = history_max_events
        self.query_slot_ids = tuple(input_spec.query.slot_ids)
        self.candidate_slot_ids = tuple(input_spec.candidate.slot_ids)
        self.history_slot_ids = tuple(history.categorical_slot_ids)
        if self.action_slot not in self.history_slot_ids:
            raise ValueError("Wukong action_slot must belong to categorical history")
        action = input_spec.slot(self.action_slot)
        if (
            action.view != "history"
            or action.role != "sequence"
            or action.encoding != "embedding"
            or action.reduction != "none"
        ):
            raise ValueError("Wukong action slot has incompatible public semantics")
        all_slots = (
            self.query_slot_ids
            + self.candidate_slot_ids
            + self.history_slot_ids
        )
        if len(all_slots) != len(set(all_slots)):
            raise ValueError("Wukong selected field slots must be unique")
        for slot_id in self.query_slot_ids:
            self._validate_slot(input_spec, slot_id, view="query", reduction="mean")
        for slot_id in self.candidate_slot_ids:
            self._validate_slot(
                input_spec,
                slot_id,
                view="candidate",
                reduction="mean",
            )
        for slot_id in self.history_slot_ids:
            self._validate_slot(
                input_spec,
                slot_id,
                view="history",
                reduction="none",
            )

        self.query_projections = self._build_projections(
            input_spec,
            self.query_slot_ids,
        )
        self.candidate_projections = self._build_projections(
            input_spec,
            self.candidate_slot_ids,
        )
        self.history_projections = self._build_projections(
            input_spec,
            self.history_slot_ids,
        )

    @property
    def num_fields(self) -> int:
        return (
            len(self.query_slot_ids)
            + len(self.candidate_slot_ids)
            + len(self.history_slot_ids)
        )

    def _build_projections(
        self,
        input_spec: RecAgoraInputSpec,
        slot_ids: tuple[int, ...],
    ) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                str(slot_id): FieldProjection(
                    input_dim=int(input_spec.slot(slot_id).output_dim),
                    output_dim=self.interaction_dim,
                )
                for slot_id in slot_ids
            }
        )

    @staticmethod
    def _validate_slot(
        input_spec: RecAgoraInputSpec,
        slot_id: int,
        *,
        view: str,
        reduction: str,
    ) -> None:
        slot = input_spec.slot(slot_id)
        if (
            slot.view != view
            or slot.encoding != "embedding"
            or slot.reduction != reduction
        ):
            raise ValueError(
                f"Wukong slot {slot_id} has incompatible public semantics"
            )

    def forward(self, batch: RecAgoraModelBatch) -> WukongInputsOutput:
        query_fields = torch.stack(
            tuple(
                self.query_projections[str(slot_id)](batch.query[slot_id])
                for slot_id in self.query_slot_ids
            ),
            dim=1,
        )
        candidate_fields = torch.stack(
            tuple(
                self.candidate_projections[str(slot_id)](
                    batch.candidate[slot_id]
                )
                for slot_id in self.candidate_slot_ids
            ),
            dim=2,
        )
        history = batch.histories[self.history_group_name]
        history_fields = torch.stack(
            tuple(
                self.history_projections[str(slot_id)](
                    segment_mean(history.categorical[slot_id])
                    if self.history_max_events is None
                    else segment_mean_recent(
                        history.categorical[slot_id], self.history_max_events
                    )
                )
                for slot_id in self.history_slot_ids
            ),
            dim=1,
        )
        return WukongInputsOutput(
            query_fields=query_fields,
            candidate_fields=candidate_fields,
            history_fields=history_fields,
        )


def segment_mean(feature: JaggedFeatureView) -> torch.Tensor:
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


def segment_mean_recent(feature: JaggedFeatureView, max_events: int) -> torch.Tensor:
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
        starts = torch.cumsum(lengths, dim=0) - lengths
        positions = torch.arange(values.shape[0], device=values.device) - starts[owners]
        keep = positions >= (lengths[owners] - max_events).clamp_min(0)
        output.index_add_(0, owners[keep], values[keep])
    retained = lengths.clamp_max(max_events).clamp_min(1)
    return output / retained.to(values.dtype).unsqueeze(-1)
