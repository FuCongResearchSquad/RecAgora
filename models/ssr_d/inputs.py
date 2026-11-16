from __future__ import annotations

import math

import torch
from torch import nn

from recagora import RecAgoraInputSpec, RecAgoraModelBatch, RecAgoraPointwiseBatch, HistoryView
from models._validation import (
    require_categorical_history_group,
    validate_history_padding_side,
)


class CandidateConditionedHistoryAdapter(nn.Module):
    """Pool one request-shared categorical history for each candidate.

    The adapter keeps the public history at request granularity. Candidate queries
    are placed back into ``[B, L, A]`` through the public pointwise indices before
    attending to one shared ``[B, T, A]`` key tensor, so candidates cannot observe
    one another.
    """

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history_group_name: str,
        attention_dim: int,
        padding_side: str,
    ) -> None:
        super().__init__()
        history = require_categorical_history_group(
            input_spec,
            history_group_name,
            context="SSR-D faithful-core",
        )
        if not history.categorical_slot_ids:
            raise ValueError("SSR-D faithful-core requires categorical history slots")
        validate_history_padding_side(padding_side)

        self.group_name = str(history_group_name)
        self.slot_ids = tuple(history.categorical_slot_ids)
        self.max_length = int(history.max_length)
        self.padding_side = str(padding_side)
        self.attention_dim = int(attention_dim)
        self.history_dim = input_spec.total_dim(self.slot_ids)
        self.candidate_dim = int(input_spec.candidate.total_dim)
        if self.history_dim != int(history.event_total_dim):
            raise ValueError("categorical history dimensions changed from the input spec")

        self.candidate_projection = nn.Linear(
            self.candidate_dim,
            self.attention_dim,
        )
        self.history_projection = nn.Linear(
            self.history_dim,
            self.attention_dim,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in (self.candidate_projection, self.history_projection):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        batch: RecAgoraModelBatch,
        pointwise: RecAgoraPointwiseBatch,
    ) -> torch.Tensor:
        history = batch.histories[self.group_name]
        events = self._padded_events(history)
        keys = self.history_projection(events)
        flat_queries = self.candidate_projection(pointwise.candidate.concat())

        queries = flat_queries.new_zeros(
            (batch.batch_size, batch.list_len, self.attention_dim)
        )
        queries = queries.index_put(
            (pointwise.request_index, pointwise.candidate_index),
            flat_queries,
            accumulate=False,
        )
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(
            self.attention_dim
        )
        valid = _sequence_valid_mask(
            history.lengths,
            max_length=self.max_length,
            padding_side=self.padding_side,
        )
        scores = scores.masked_fill(
            ~valid[:, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1) * valid[:, None, :].to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        pooled = torch.matmul(weights, events)
        return pooled[pointwise.request_index, pointwise.candidate_index]

    def _padded_events(self, history: HistoryView) -> torch.Tensor:
        fields = [
            history.categorical[slot_id].to_padded(
                    max_length=self.max_length,
                    padding_side=self.padding_side,
                )
            for slot_id in self.slot_ids
        ]
        events = torch.cat(fields, dim=-1)
        return events


class SSRDInputAdapter(nn.Module):
    """Assemble the frozen Common157 order: query, candidate, history."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        history_group_name: str,
        history_attention_dim: int,
        history_padding_side: str,
    ) -> None:
        super().__init__()
        self.query_dim = int(input_spec.query.total_dim)
        self.candidate_dim = int(input_spec.candidate.total_dim)
        self.history = CandidateConditionedHistoryAdapter(
            input_spec=input_spec,
            history_group_name=history_group_name,
            attention_dim=history_attention_dim,
            padding_side=history_padding_side,
        )
        self.output_dim = self.query_dim + self.candidate_dim + self.history.history_dim

    def forward(
        self,
        batch: RecAgoraModelBatch,
        pointwise: RecAgoraPointwiseBatch,
    ) -> torch.Tensor:
        pooled_history = self.history(batch, pointwise)
        return torch.cat(
            (
                pointwise.query.concat(),
                pointwise.candidate.concat(),
                pooled_history,
            ),
            dim=-1,
        )


def _sequence_valid_mask(
    lengths: torch.Tensor,
    *,
    max_length: int,
    padding_side: str,
) -> torch.Tensor:
    bounded = lengths.clamp(min=0, max=max_length)
    positions = torch.arange(max_length, device=lengths.device)[None, :]
    if padding_side == "left":
        return positions >= max_length - bounded[:, None]
    return positions < bounded[:, None]
