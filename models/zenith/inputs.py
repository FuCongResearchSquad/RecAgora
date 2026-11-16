from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from recagora import RecAgoraInputSpec, RecAgoraModelBatch, HistoryView, JaggedFeatureView
from models._validation import (
    require_categorical_history_group,
    validate_disjoint_slot_groups,
    validate_history_order,
    validate_left_padding,
    validate_positive_int,
)


@dataclass(frozen=True)
class ZenithInputsOutput:
    tokens: torch.Tensor
    history_events: torch.Tensor
    history_valid: torch.Tensor
    interest_tokens: torch.Tensor


class PrimeTokenProjection(nn.Module):
    """Project one complete semantic feature group into one Prime Token."""

    def __init__(self, *, input_dim: int, token_dim: int, norm_eps: float) -> None:
        super().__init__()
        validate_positive_int(input_dim, field_name="prime_token.input_dim")
        validate_positive_int(token_dim, field_name="prime_token.token_dim")
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.projection = nn.Linear(self.input_dim, self.token_dim)
        self.norm = nn.LayerNorm(self.token_dim, eps=float(norm_eps))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.norm.reset_parameters()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"PrimeTokenProjection expected input_dim={self.input_dim}, "
                f"got {values.shape[-1]}"
            )
        return self.norm(F.silu(self.projection(values)))


class HistoryEventProjection(nn.Module):
    """Project aligned Native160 categorical fields without a wide concat."""

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
            raise ValueError("Zenith history projection requires sequence slots")
        self.input_dims = tuple(
            int(input_spec.slot(slot_id).output_dim) for slot_id in self.slot_ids
        )
        self.input_dim = int(sum(self.input_dims))
        self.output_dim = int(output_dim)
        self.projections = nn.ModuleDict(
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
        # One aggregate fan-in keeps the summed field projections on the same
        # scale as a single Linear over a materialized event concatenation.
        bound = math.sqrt(6.0 / float(self.input_dim + self.output_dim))
        for projection in self.projections.values():
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
            projected = projected + self.projections[str(slot_id)](
                history.categorical[slot_id].values
            )
        return JaggedFeatureView(
            values=projected,
            lengths=history.lengths,
        ).to_padded(
            max_length=max_length,
            padding_side=padding_side,
        )


class CandidateHistoryInterest(nn.Module):
    """Three candidate-private queries over request-shared history K/V."""

    def __init__(
        self,
        *,
        token_dim: int,
        candidate_token_count: int,
        num_interest_tokens: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        for value, name in (
            (token_dim, "history_attention.token_dim"),
            (candidate_token_count, "history_attention.candidate_token_count"),
            (num_interest_tokens, "history_attention.num_interest_tokens"),
            (num_heads, "history_attention.num_heads"),
        ):
            validate_positive_int(value, field_name=name)
        if token_dim % num_heads:
            raise ValueError(
                "Zenith history attention token_dim must be divisible by num_heads"
            )
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Zenith history attention dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.candidate_token_count = int(candidate_token_count)
        self.num_interest_tokens = int(num_interest_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.dropout = float(dropout)

        self.query_mix_logits = nn.Parameter(
            torch.zeros(self.num_interest_tokens, self.candidate_token_count)
        )
        self.query_weight = nn.Parameter(
            torch.empty(self.num_interest_tokens, self.token_dim, self.token_dim)
        )
        self.key_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.output_weight = nn.Parameter(
            torch.empty(self.num_interest_tokens, self.token_dim, self.token_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.query_mix_logits)
        for interest_index in range(self.num_interest_tokens):
            nn.init.xavier_uniform_(self.query_weight[interest_index])
            nn.init.xavier_uniform_(self.output_weight[interest_index])
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        history: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_tokens.shape[-2:] != (
            self.candidate_token_count,
            self.token_dim,
        ):
            raise ValueError("Zenith candidate token shape does not match config")
        if history.shape[:2] != history_valid.shape:
            raise ValueError("Zenith history and valid mask shapes do not match")
        if (
            history.shape[0] != candidate_tokens.shape[0]
            or history.shape[-1] != self.token_dim
        ):
            raise ValueError("Zenith history shape does not match candidate tokens")

        batch_size, list_len = candidate_tokens.shape[:2]
        history_length = int(history.shape[1])
        mix = F.softmax(self.query_mix_logits, dim=-1)
        query_seed = torch.einsum("ic,blcd->blid", mix, candidate_tokens)
        query = torch.einsum("blid,idf->blif", query_seed, self.query_weight)
        key = self.key_projection(history)
        value = self.value_projection(history)

        query = query.reshape(
            batch_size,
            list_len,
            self.num_interest_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 1, 3, 2, 4)
        key = key.reshape(
            batch_size,
            history_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        value = value.reshape(
            batch_size,
            history_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        scores = torch.einsum("blhid,bhtd->blhit", query, key)
        scores = scores / math.sqrt(self.head_dim)
        valid = history_valid[:, None, None, None, :]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1)
        probabilities = probabilities * valid.to(dtype=probabilities.dtype)
        probabilities = probabilities / probabilities.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(probabilities.dtype).eps)
        probabilities = F.dropout(
            probabilities,
            p=self.dropout,
            training=self.training,
        )
        interests = torch.einsum("blhit,bhtd->blhid", probabilities, value)
        interests = interests.permute(0, 1, 3, 2, 4).reshape(
            batch_size,
            list_len,
            self.num_interest_tokens,
            self.token_dim,
        )
        return torch.einsum("blid,idf->blif", interests, self.output_weight)


class ZenithInputs(nn.Module):
    """Build 12 candidate-local Prime Tokens from one public lookup result."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        item_id_slot: int,
        query_token_order: Sequence[str],
        query_groups: Mapping[str, Sequence[int]],
        candidate_token_order: Sequence[str],
        candidate_groups: Mapping[str, Sequence[int]],
        history: Mapping[str, object],
        token_dim: int,
        num_interest_tokens: int,
        num_heads: int,
        attention_dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.item_id_slot = int(item_id_slot)
        self.query_token_order = tuple(str(name) for name in query_token_order)
        self.candidate_token_order = tuple(
            str(name) for name in candidate_token_order
        )
        self.query_groups = {
            str(name): tuple(int(slot_id) for slot_id in slot_ids)
            for name, slot_ids in query_groups.items()
        }
        self.candidate_groups = {
            str(name): tuple(int(slot_id) for slot_id in slot_ids)
            for name, slot_ids in candidate_groups.items()
        }
        self.token_dim = int(token_dim)
        self.num_interest_tokens = int(num_interest_tokens)

        history = dict(history)
        unknown = sorted(
            set(history)
            - {"group", "sequence_slots", "action_slot", "padding_side"}
        )
        if unknown:
            raise ValueError(f"Zenith history contains unknown fields: {unknown}")
        self.history_group_name = str(history.get("group", "history_group_1"))
        self.padding_side = str(history.get("padding_side", "left"))
        validate_left_padding(self.padding_side, context="Zenith")
        history_spec = require_categorical_history_group(
            input_spec,
            self.history_group_name,
            context="Zenith",
        )
        validate_history_order(
            history_spec,
            context="Zenith history",
            truncate_side="tail",
        )
        self.sequence_slots = tuple(
            int(slot_id) for slot_id in history.get("sequence_slots", ())
        )
        self.action_slot = int(history.get("action_slot", 45040))
        self.history_length = int(history_spec.max_length)
        self._validate_contract(input_spec, history_spec)

        self.query_tokenizers = nn.ModuleDict(
            {
                name: PrimeTokenProjection(
                    input_dim=input_spec.total_dim(self.query_groups[name]),
                    token_dim=self.token_dim,
                    norm_eps=norm_eps,
                )
                for name in self.query_token_order
            }
        )
        self.candidate_tokenizers = nn.ModuleDict(
            {
                name: PrimeTokenProjection(
                    input_dim=input_spec.total_dim(self.candidate_groups[name]),
                    token_dim=self.token_dim,
                    norm_eps=norm_eps,
                )
                for name in self.candidate_token_order
            }
        )
        self.item_tokenizer = PrimeTokenProjection(
            input_dim=input_spec.slot(self.item_id_slot).output_dim,
            token_dim=self.token_dim,
            norm_eps=norm_eps,
        )
        self.history_projection = HistoryEventProjection(
            input_spec=input_spec,
            slot_ids=self.sequence_slots,
            output_dim=self.token_dim,
        )
        self.history_attention = CandidateHistoryInterest(
            token_dim=self.token_dim,
            candidate_token_count=1 + len(self.candidate_token_order),
            num_interest_tokens=self.num_interest_tokens,
            num_heads=num_heads,
            dropout=attention_dropout,
        )

    @property
    def num_prime_tokens(self) -> int:
        return (
            1
            + len(self.query_token_order)
            + len(self.candidate_token_order)
            + self.num_interest_tokens
        )

    def _validate_contract(self, input_spec, history_spec) -> None:
        for order, groups, label in (
            (self.query_token_order, self.query_groups, "query"),
            (self.candidate_token_order, self.candidate_groups, "candidate"),
        ):
            if not order or len(order) != len(set(order)):
                raise ValueError(f"Zenith {label}_token_order must be non-empty and unique")
            if set(order) != set(groups):
                raise ValueError(
                    f"Zenith {label}_groups must exactly match {label}_token_order"
                )
            if any(not groups[name] for name in order):
                raise ValueError(f"Zenith {label}_groups must not contain empty groups")

        query_group_slots = tuple(
            slot_id
            for name in self.query_token_order
            for slot_id in self.query_groups[name]
        )
        candidate_group_slots = tuple(
            slot_id
            for name in self.candidate_token_order
            for slot_id in self.candidate_groups[name]
        )
        if len(query_group_slots) != len(set(query_group_slots)):
            raise ValueError("Zenith query groups must be disjoint")
        if len(candidate_group_slots) != len(set(candidate_group_slots)):
            raise ValueError("Zenith candidate groups must be disjoint")
        if set(query_group_slots) != set(input_spec.query.slot_ids):
            raise ValueError(
                "Zenith query groups must exactly cover the public query view"
            )
        if self.item_id_slot not in input_spec.candidate.slot_ids:
            raise ValueError("Zenith item_id_slot must belong to the candidate view")
        if self.item_id_slot in candidate_group_slots:
            raise ValueError("Zenith item_id_slot must remain an independent Prime Token")
        if set(candidate_group_slots) | {self.item_id_slot} != set(
            input_spec.candidate.slot_ids
        ):
            raise ValueError(
                "Zenith item ID and candidate groups must exactly cover the public "
                "candidate view"
            )
        if self.sequence_slots != tuple(history_spec.categorical_slot_ids):
            raise ValueError(
                "Zenith history.sequence_slots must preserve and exactly cover the "
                "public categorical history view"
            )
        if self.action_slot not in self.sequence_slots:
            raise ValueError(
                "Zenith history.action_slot must belong to history.sequence_slots"
            )
        action = input_spec.slot(self.action_slot)
        if (
            action.view != "history"
            or action.role != "sequence"
            or action.encoding != "embedding"
            or action.reduction != "none"
        ):
            raise ValueError("Zenith action slot has incompatible public semantics")

        validate_disjoint_slot_groups(
            (query_group_slots, candidate_group_slots, self.sequence_slots),
            context="Zenith query, candidate, and history groups",
        )
        self._validate_slot_semantics(
            input_spec,
            query_group_slots,
            view="query",
            role="request",
            reduction="mean",
        )
        self._validate_slot_semantics(
            input_spec,
            (self.item_id_slot,) + candidate_group_slots,
            view="candidate",
            role="item",
            reduction="mean",
        )
        self._validate_slot_semantics(
            input_spec,
            self.sequence_slots,
            view="history",
            role="sequence",
            reduction="none",
        )

    @staticmethod
    def _validate_slot_semantics(
        input_spec: RecAgoraInputSpec,
        slot_ids: Sequence[int],
        *,
        view: str,
        role: str,
        reduction: str,
    ) -> None:
        for slot_id in slot_ids:
            slot = input_spec.slot(slot_id)
            if (
                slot.view != view
                or slot.role != role
                or slot.encoding != "embedding"
                or slot.reduction != reduction
            ):
                raise ValueError(
                    f"Zenith slot {slot_id} has incompatible public semantics"
                )

    def forward(self, batch: RecAgoraModelBatch) -> ZenithInputsOutput:
        batch_size = batch.batch_size
        list_len = batch.list_len
        item_token = self.item_tokenizer(batch.candidate[self.item_id_slot])
        query_tokens = torch.stack(
            tuple(
                self.query_tokenizers[name](
                    batch.query.concat(self.query_groups[name])
                )[:, None, :].expand(-1, list_len, -1)
                for name in self.query_token_order
            ),
            dim=2,
        )
        candidate_tokens = torch.stack(
            tuple(
                self.candidate_tokenizers[name](
                    batch.candidate.concat(self.candidate_groups[name])
                )
                for name in self.candidate_token_order
            ),
            dim=2,
        )

        history = batch.histories[self.history_group_name]
        history_valid = _suffix_valid_mask(history.lengths, self.history_length)
        history_events = self.history_projection(
            history,
            max_length=self.history_length,
            padding_side=self.padding_side,
        ).masked_fill(~history_valid.unsqueeze(-1), 0.0)
        interest_inputs = torch.cat(
            (item_token.unsqueeze(2), candidate_tokens),
            dim=2,
        )
        interest_tokens = self.history_attention(
            interest_inputs,
            history_events,
            history_valid,
        )
        tokens = torch.cat(
            (
                item_token.unsqueeze(2),
                query_tokens,
                candidate_tokens,
                interest_tokens,
            ),
            dim=2,
        )
        expected = (
            batch_size,
            list_len,
            self.num_prime_tokens,
            self.token_dim,
        )
        if tuple(tokens.shape) != expected:
            raise RuntimeError(
                f"Zenith tokenizer produced shape {tuple(tokens.shape)}, expected {expected}"
            )
        return ZenithInputsOutput(
            tokens=tokens.reshape(
                batch_size * list_len,
                self.num_prime_tokens,
                self.token_dim,
            ),
            history_events=history_events,
            history_valid=history_valid,
            interest_tokens=interest_tokens,
        )


def _suffix_valid_mask(lengths: torch.Tensor, width: int) -> torch.Tensor:
    bounded = lengths.clamp(min=0, max=width)
    positions = torch.arange(width, device=lengths.device)[None, :]
    return positions >= (width - bounded).unsqueeze(1)
