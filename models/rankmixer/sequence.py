from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from recagora import RecAgoraInputSpec, FixedFeatureView, HistoryView
from models._validation import (
    parse_din_pair,
    require_categorical_history_group,
    require_din_pairs,
    validate_din_attention_pairs,
    validate_history_padding_side,
    validate_history_order,
    validate_model_dim_heads,
    validate_positive_int_sequence,
)
from ._validation import (
    dropout_probability as _dropout_probability,
    positive_int as _require_positive_int,
    positive_float as _require_positive_float,
    require_bool as _require_bool,
    require_int as _require_int,
)


class MaskedMeanSequenceEncoder(nn.Module):
    """Request-level mean pooling that preserves the published AVG path."""

    candidate_conditioned = False

    def __init__(
        self,
        *,
        slot_ids: Sequence[int],
        input_spec: RecAgoraInputSpec,
        max_events: int | None = None,
    ) -> None:
        super().__init__()
        self.slot_ids = tuple(int(slot_id) for slot_id in slot_ids)
        self.output_dim = input_spec.total_dim(self.slot_ids)
        self.max_events = max_events

    def forward(
        self,
        history: HistoryView,
        candidate: FixedFeatureView,
    ) -> torch.Tensor:
        del candidate
        if self.max_events is None:
            return torch.cat(
                [_jagged_masked_mean(history, slot_id) for slot_id in self.slot_ids],
                dim=-1,
            )
        return torch.cat(
            [
                _jagged_masked_mean_recent(history, slot_id, self.max_events)
                for slot_id in self.slot_ids
            ],
            dim=-1,
        )


class _LocalActivationUnit(nn.Module):
    def __init__(self, embedding_dim: int, hidden_units: Sequence[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = embedding_dim * 4
        for hidden in hidden_units:
            layers.extend((nn.Linear(previous, int(hidden)), nn.ReLU()))
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        valid_mask: torch.Tensor,
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
            valid = valid_mask[:, None, :].expand(-1, chunk, -1)
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


class DINSequenceEncoder(nn.Module):
    """Candidate-aware DIN pooling, restored to the 27-slot semantic order."""

    candidate_conditioned = True

    def __init__(
        self,
        *,
        slot_ids: Sequence[int],
        input_spec: RecAgoraInputSpec,
        attention_pairs: Sequence[Mapping[str, int] | Sequence[int]],
        attention_hidden_units: Sequence[int],
        weight_normalization: str,
        candidate_chunk_size: int,
        activation_checkpointing: bool,
        max_length: int,
        padding_side: str,
    ) -> None:
        super().__init__()
        _require_bool(activation_checkpointing, "DIN activation_checkpointing")
        self.slot_ids = tuple(int(slot_id) for slot_id in slot_ids)
        self.attention_pairs = tuple(
            parse_din_pair(pair, reject_unknown_mapping_keys=True)
            for pair in attention_pairs
        )
        self.candidate_chunk_size = _require_positive_int(
            candidate_chunk_size,
            "DIN candidate_chunk_size",
        )
        self.activation_checkpointing = activation_checkpointing
        # The factory has already matched this value to the positive public
        # history length.
        self.max_length = int(max_length)
        self.padding_side = str(padding_side)
        self.weight_normalization = str(weight_normalization)
        self.output_dim = input_spec.total_dim(self.slot_ids)

        require_din_pairs(self.attention_pairs, context="DIN sequence encoder")
        if self.weight_normalization not in {"none", "softmax"}:
            raise ValueError("DIN weight_normalization must be none or softmax")
        validate_history_padding_side(self.padding_side)
        hidden_units = tuple(
            _require_int(value, f"DIN attention_hidden_units[{index}]")
            for index, value in enumerate(attention_hidden_units)
        )
        validate_positive_int_sequence(
            hidden_units,
            field_name="DIN attention_hidden_units",
        )

        paired_history, target_dim = validate_din_attention_pairs(
            input_spec,
            self.attention_pairs,
        )
        self.unpaired_history = tuple(
            slot_id for slot_id in self.slot_ids if slot_id not in set(paired_history)
        )
        self.attention = _LocalActivationUnit(target_dim, hidden_units)
        self.target_by_history = {
            history_slot: target for target, history_slot in self.attention_pairs
        }

    def forward(
        self,
        history: HistoryView,
        candidate: FixedFeatureView,
    ) -> torch.Tensor:
        valid_mask = _sequence_valid_mask(
            history.lengths,
            max_length=self.max_length,
            padding_side=self.padding_side,
        )
        list_len = int(next(iter(candidate.values())).shape[1])
        values_by_slot: dict[int, torch.Tensor] = {}
        for slot_id in self.slot_ids:
            target_slot = self.target_by_history.get(slot_id)
            if target_slot is None:
                values_by_slot[slot_id] = _jagged_masked_mean(
                    history,
                    slot_id,
                )[:, None, :].expand(-1, list_len, -1)
                continue
            jagged = history.categorical[slot_id]
            keys = jagged.to_padded(
                max_length=self.max_length,
                padding_side=self.padding_side,
            )
            if (
                self.activation_checkpointing
                and self.training
                and keys.device.type == "cuda"
            ):
                values_by_slot[slot_id] = checkpoint(
                    self.attention,
                    candidate[target_slot],
                    keys,
                    valid_mask,
                    chunk_size=self.candidate_chunk_size,
                    weight_normalization=self.weight_normalization,
                    use_reentrant=False,
                )
            else:
                values_by_slot[slot_id] = self.attention(
                    candidate[target_slot],
                    keys,
                    valid_mask,
                    chunk_size=self.candidate_chunk_size,
                    weight_normalization=self.weight_normalization,
                )
        return torch.cat(
            [values_by_slot[slot_id] for slot_id in self.slot_ids],
            dim=-1,
        )


class _EventProjector(nn.Module):
    """Fuse aligned jagged fields without materializing their wide concat."""

    def __init__(
        self,
        *,
        slot_ids: Sequence[int],
        input_spec: RecAgoraInputSpec,
        model_dim: int,
        max_length: int,
        padding_side: str,
    ) -> None:
        super().__init__()
        self.slot_ids = tuple(int(slot_id) for slot_id in slot_ids)
        self.feature_dims = tuple(
            int(input_spec.slot(slot_id).output_dim) for slot_id in self.slot_ids
        )
        self.model_dim = int(model_dim)
        self.max_length = int(max_length)
        self.padding_side = str(padding_side)
        self.input_projections = nn.ModuleList(
            nn.Linear(feature_dim, self.model_dim, bias=False)
            for feature_dim in self.feature_dims
        )
        self.input_bias = nn.Parameter(torch.empty(self.model_dim))
        self.position_embedding = nn.Embedding(self.max_length, self.model_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(sum(self.feature_dims))
        for projection in self.input_projections:
            nn.init.uniform_(projection.weight, -bound, bound)
        nn.init.uniform_(self.input_bias, -bound, bound)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, history: HistoryView) -> tuple[torch.Tensor, torch.Tensor]:
        feature_values = [
            history.categorical[slot_id].values for slot_id in self.slot_ids
        ]
        rows = int(feature_values[0].shape[0])
        fused = self.input_bias.reshape(1, -1).expand(rows, self.model_dim)
        for values, projection in zip(
            feature_values,
            self.input_projections,
            strict=True,
        ):
            fused = fused + projection(values)
        padded = _jagged_to_padded(
            fused,
            history.lengths,
            max_length=self.max_length,
            padding_side=self.padding_side,
        )
        valid = _sequence_valid_mask(
            history.lengths,
            max_length=self.max_length,
            padding_side=self.padding_side,
        )
        positions = torch.arange(self.max_length, device=padded.device)
        padded = padded + self.position_embedding(positions).unsqueeze(0)
        return padded.masked_fill(~valid.unsqueeze(-1), 0.0), valid


class TransformerSequenceEncoder(nn.Module):
    """Request-level temporal encoder over aligned history events."""

    candidate_conditioned = False

    def __init__(
        self,
        *,
        slot_ids: Sequence[int],
        input_spec: RecAgoraInputSpec,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        max_length: int,
        padding_side: str,
        activation_checkpointing: bool,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        _require_bool(norm_first, "Transformer norm_first")
        _require_bool(
            activation_checkpointing,
            "Transformer activation_checkpointing",
        )
        _require_positive_int(num_layers, "Transformer num_layers")
        layer_norm_eps = _require_positive_float(
            layer_norm_eps,
            "Transformer layer_norm_eps",
        )
        self.output_dim = model_dim
        self.activation_checkpointing = activation_checkpointing
        _validate_attention_dimensions(self.output_dim, num_heads, ffn_dim, dropout)
        _validate_temporal_encoder_activation("Transformer", activation)
        self.projector = _EventProjector(
            slot_ids=slot_ids,
            input_spec=input_spec,
            model_dim=self.output_dim,
            max_length=int(max_length),
            padding_side=str(padding_side),
        )
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=self.output_dim,
                nhead=int(num_heads),
                dim_feedforward=int(ffn_dim),
                dropout=float(dropout),
                activation=activation,
                batch_first=True,
                norm_first=bool(norm_first),
                layer_norm_eps=float(layer_norm_eps),
            )
            for _ in range(int(num_layers))
        )
        self.output_norm = nn.LayerNorm(self.output_dim, eps=float(layer_norm_eps))

    def forward(
        self,
        history: HistoryView,
        candidate: FixedFeatureView,
    ) -> torch.Tensor:
        del candidate
        encoded, valid = self.projector(history)
        safe_valid = _safe_valid_mask(valid, padding_side=self.projector.padding_side)
        padding_mask = ~safe_valid
        for layer in self.layers:
            if (
                self.activation_checkpointing
                and self.training
                and encoded.requires_grad
                and encoded.device.type == "cuda"
            ):
                encoded = checkpoint(
                    layer,
                    encoded,
                    src_key_padding_mask=padding_mask,
                    use_reentrant=False,
                )
            else:
                encoded = layer(encoded, src_key_padding_mask=padding_mask)
        return _masked_pool(self.output_norm(encoded), valid)


class _LongerSelfBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        activation: str,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.num_heads = int(num_heads)

    def forward(
        self,
        values: torch.Tensor,
        valid: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        attention_mask: torch.Tensor | None = None
        if causal:
            base_mask = torch.triu(
                torch.ones(
                    values.shape[1],
                    values.shape[1],
                    dtype=torch.bool,
                    device=values.device,
                ),
                diagonal=1,
            )
            attention_mask = base_mask.unsqueeze(0).expand(
                values.shape[0],
                -1,
                -1,
            ).clone()
        else:
            attention_mask = torch.zeros(
                values.shape[0],
                values.shape[1],
                values.shape[1],
                dtype=torch.bool,
                device=values.device,
            )
        # Fold key validity into a per-request mask, then expose only the
        # diagonal for invalid query rows. This prevents fully masked softmax
        # rows (and their backward NaNs) while leaving every valid query's
        # causal/key contract unchanged.
        attention_mask |= ~valid[:, None, :]
        invalid_rows = ~valid
        attention_mask[invalid_rows] = True
        request_index, query_index = invalid_rows.nonzero(as_tuple=True)
        attention_mask[request_index, query_index, query_index] = False
        attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)
        attended = self.attention(
            values,
            values,
            values,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        values = self.norm1(values + attended)
        values = self.norm2(values + self.ffn(values))
        return values * valid.unsqueeze(-1).to(values.dtype)


class LongerStyleSequenceEncoder(nn.Module):
    """Sequence-only grouped-attention adapter inspired by LONGER."""

    candidate_conditioned = False

    def __init__(
        self,
        *,
        slot_ids: Sequence[int],
        input_spec: RecAgoraInputSpec,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        group_size: int,
        recent_query_size: int,
        ffn_dim: int,
        dropout: float,
        activation: str,
        max_length: int,
        causal: bool,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        _require_bool(causal, "LONGER causal")
        _require_positive_int(num_layers, "LONGER num_layers")
        _require_positive_int(group_size, "LONGER group_size")
        _require_positive_int(recent_query_size, "LONGER recent_query_size")
        layer_norm_eps = _require_positive_float(
            layer_norm_eps,
            "LONGER layer_norm_eps",
        )
        self.output_dim = model_dim
        self.group_size = group_size
        self.recent_query_size = recent_query_size
        self.causal = causal
        _validate_attention_dimensions(self.output_dim, num_heads, ffn_dim, dropout)
        if int(max_length) % self.group_size:
            raise ValueError("LONGER-style group_size must divide max_length")
        if self.recent_query_size > int(max_length):
            raise ValueError("LONGER-style recent_query_size must be within max_length")
        padding_side = "left"
        _validate_temporal_encoder_activation("LONGER-style", activation)
        self.projector = _EventProjector(
            slot_ids=slot_ids,
            input_spec=input_spec,
            model_dim=self.output_dim,
            max_length=int(max_length),
            padding_side=padding_side,
        )
        self.inner_attention = nn.MultiheadAttention(
            self.output_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.inner_norm1 = nn.LayerNorm(self.output_dim, eps=float(layer_norm_eps))
        self.inner_ffn = nn.Sequential(
            nn.Linear(self.output_dim, int(ffn_dim)),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(ffn_dim), self.output_dim),
            nn.Dropout(float(dropout)),
        )
        self.inner_norm2 = nn.LayerNorm(self.output_dim, eps=float(layer_norm_eps))
        self.cross_attention = nn.MultiheadAttention(
            self.output_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(self.output_dim, eps=float(layer_norm_eps))
        self.layers = nn.ModuleList(
            _LongerSelfBlock(
                model_dim=self.output_dim,
                num_heads=int(num_heads),
                ffn_dim=int(ffn_dim),
                dropout=float(dropout),
                activation=activation,
                layer_norm_eps=float(layer_norm_eps),
            )
            for _ in range(int(num_layers))
        )

    def forward(
        self,
        history: HistoryView,
        candidate: FixedFeatureView,
    ) -> torch.Tensor:
        del candidate
        events, valid = self.projector(history)
        batch_size, max_length, model_dim = events.shape
        group_count = max_length // self.group_size
        grouped = events.reshape(
            batch_size * group_count,
            self.group_size,
            model_dim,
        )
        group_valid = valid.reshape(batch_size * group_count, self.group_size)
        safe_group_valid = _safe_valid_mask(group_valid, padding_side="left")
        attended = self.inner_attention(
            grouped,
            grouped,
            grouped,
            key_padding_mask=~safe_group_valid,
            need_weights=False,
        )[0]
        grouped = self.inner_norm1(grouped + attended)
        grouped = self.inner_norm2(grouped + self.inner_ffn(grouped))
        grouped = grouped * group_valid.unsqueeze(-1).to(grouped.dtype)
        merged = _masked_pool(grouped, group_valid).reshape(
            batch_size,
            group_count,
            model_dim,
        )
        merged_valid = group_valid.any(dim=1).reshape(batch_size, group_count)

        recent = events[:, -self.recent_query_size :]
        recent_valid = valid[:, -self.recent_query_size :]
        safe_merged_valid = _safe_valid_mask(merged_valid, padding_side="left")
        crossed = self.cross_attention(
            recent,
            merged,
            merged,
            key_padding_mask=~safe_merged_valid,
            need_weights=False,
        )[0]
        recent = self.cross_norm(recent + crossed)
        recent = recent * recent_valid.unsqueeze(-1).to(recent.dtype)
        for layer in self.layers:
            recent = layer(recent, recent_valid, causal=self.causal)
        return _masked_pool(recent, recent_valid)


def build_sequence_encoder(
    *,
    sequence_type: str,
    params: Mapping[str, Any],
    slot_ids: Sequence[int],
    input_spec: RecAgoraInputSpec,
    history_group_name: str,
) -> nn.Module:
    history = require_categorical_history_group(
        input_spec,
        history_group_name,
        context="RankMixer",
    )
    selected_slots = tuple(int(slot_id) for slot_id in slot_ids)
    if set(selected_slots) != set(history.categorical_slot_ids):
        raise ValueError(
            "sequence input group must exactly cover categorical history slots"
        )
    raw = dict(params)
    if sequence_type == "masked_mean":
        unknown = sorted(set(raw) - {"max_events"})
        if unknown:
            raise ValueError(f"masked_mean sequence params have unknown fields: {unknown}")
        max_events = raw.get("max_events")
        if max_events is not None:
            max_events = _require_positive_int(max_events, "masked_mean max_events")
            if max_events > history.max_length:
                raise ValueError(
                    "masked_mean max_events exceeds public history length "
                    f"({history.max_length})"
                )
            if max_events == history.max_length:
                max_events = None
        return MaskedMeanSequenceEncoder(
            slot_ids=selected_slots,
            input_spec=input_spec,
            max_events=max_events,
        )
    if sequence_type == "din":
        required = {
            "attention_pairs",
            "attention_hidden_units",
            "weight_normalization",
            "candidate_chunk_size",
            "activation_checkpointing",
            "padding_side",
        }
        _require_exact_keys(raw, required=required, context="din")
        raw["max_length"] = history.max_length
        return DINSequenceEncoder(
            slot_ids=selected_slots,
            input_spec=input_spec,
            **raw,
        )
    if sequence_type == "transformer":
        required = {
            "model_dim",
            "num_layers",
            "num_heads",
            "ffn_dim",
            "dropout",
            "activation",
            "norm_first",
            "padding_side",
            "activation_checkpointing",
            "layer_norm_eps",
        }
        _require_exact_keys(raw, required=required, context="transformer")
        validate_history_order(
            history,
            context="temporal RankMixer encoders",
            truncate_side="tail",
        )
        raw["max_length"] = history.max_length
        return TransformerSequenceEncoder(
            slot_ids=selected_slots,
            input_spec=input_spec,
            **raw,
        )
    if sequence_type == "longer_style":
        required = {
            "model_dim",
            "num_heads",
            "num_layers",
            "group_size",
            "recent_query_size",
            "ffn_dim",
            "dropout",
            "activation",
            "causal",
            "layer_norm_eps",
        }
        _require_exact_keys(raw, required=required, context="longer_style")
        validate_history_order(
            history,
            context="temporal RankMixer encoders",
            truncate_side="tail",
        )
        raw["max_length"] = history.max_length
        return LongerStyleSequenceEncoder(
            slot_ids=selected_slots,
            input_spec=input_spec,
            **raw,
        )
    raise ValueError(f"unsupported RankMixer sequence_encoder.type={sequence_type!r}")


def _require_exact_keys(
    values: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(values))
    unknown = sorted(set(values) - required)
    if missing or unknown:
        raise ValueError(
            f"{context} sequence params mismatch; missing={missing}, unknown={unknown}"
        )


def _validate_attention_dimensions(
    model_dim: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
) -> None:
    _require_positive_int(model_dim, "attention model_dim")
    _require_positive_int(num_heads, "attention num_heads")
    _require_positive_int(ffn_dim, "attention ffn_dim")
    validate_model_dim_heads(model_dim, num_heads)
    _dropout_probability(dropout, "attention dropout")


def _validate_temporal_encoder_activation(context: str, activation: str) -> None:
    if activation not in {"relu", "gelu"}:
        raise ValueError(f"{context} activation must be relu or gelu")


def _jagged_masked_mean(history: HistoryView, slot_id: int) -> torch.Tensor:
    jagged = history.categorical[slot_id]
    lengths = history.lengths.to(device=jagged.values.device, dtype=torch.long)
    owners = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=jagged.values.device),
        lengths,
        output_size=jagged.values.shape[0],
    )
    pooled = jagged.values.new_zeros((lengths.numel(), jagged.values.shape[-1]))
    if jagged.values.shape[0]:
        pooled.index_add_(0, owners, jagged.values)
    return pooled / lengths.clamp_min(1).to(jagged.values.dtype).unsqueeze(-1)


def _jagged_masked_mean_recent(
    history: HistoryView,
    slot_id: int,
    max_events: int,
) -> torch.Tensor:
    jagged = history.categorical[slot_id]
    lengths = history.lengths.to(device=jagged.values.device, dtype=torch.long)
    total = jagged.values.shape[0]
    owners = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=jagged.values.device),
        lengths,
        output_size=total,
    )
    pooled = jagged.values.new_zeros((lengths.numel(), jagged.values.shape[-1]))
    if total:
        starts = torch.cumsum(lengths, dim=0) - lengths
        positions = torch.arange(total, device=jagged.values.device) - starts[owners]
        keep = positions >= (lengths[owners] - max_events).clamp_min(0)
        pooled.index_add_(0, owners[keep], jagged.values[keep])
    retained = lengths.clamp_max(max_events).clamp_min(1)
    return pooled / retained.to(jagged.values.dtype).unsqueeze(-1)


def _jagged_to_padded(
    values: torch.Tensor,
    lengths: torch.Tensor,
    *,
    max_length: int,
    padding_side: str,
) -> torch.Tensor:
    lengths = lengths.to(device=values.device, dtype=torch.long)
    rows = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=values.device),
        lengths,
        output_size=values.shape[0],
    )
    starts = torch.cumsum(lengths, dim=0) - lengths
    columns = torch.arange(values.shape[0], device=values.device) - torch.repeat_interleave(
        starts,
        lengths,
        output_size=values.shape[0],
    )
    if padding_side == "left":
        columns = columns + torch.repeat_interleave(
            max_length - lengths,
            lengths,
            output_size=values.shape[0],
        )
    padded = values.new_zeros((lengths.numel(), max_length, values.shape[-1]))
    padded[rows, columns] = values
    return padded


def _sequence_valid_mask(
    lengths: torch.Tensor,
    *,
    max_length: int,
    padding_side: str,
) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long)
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    if padding_side == "left":
        return positions >= max_length - lengths.unsqueeze(1)
    return positions < lengths.unsqueeze(1)


def _safe_valid_mask(valid: torch.Tensor, *, padding_side: str) -> torch.Tensor:
    safe = valid.clone()
    empty = ~valid.any(dim=1)
    safe_position = valid.shape[1] - 1 if padding_side == "left" else 0
    safe[empty, safe_position] = True
    return safe


def _masked_pool(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(
        values.dtype
    ).unsqueeze(-1)
