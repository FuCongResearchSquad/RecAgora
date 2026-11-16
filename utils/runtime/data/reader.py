from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from utils.runtime.data.request_batch import (
    FeatureValue,
    HistoryGroupLengths,
    RaggedFeature,
    RequestBatch,
    RequestBatchMetadata,
)
from utils.compiler.config import DTypeName, normalize_slot_id
from utils.runtime.data.errors import DataContractError
from utils.runtime.data.tensor import torch_dtype
from utils.compiler.feature_contract import ParsePlan, ReadPlan, ReadPlanFeature


@dataclass(frozen=True)
class _ListLayout:
    """Normalized view of one Arrow ListArray without materializing Python rows."""

    offsets: np.ndarray
    valid: np.ndarray
    values: pa.Array

    @property
    def lengths(self) -> np.ndarray:
        lengths = np.diff(self.offsets)
        if self.valid.all():
            return lengths
        return np.where(self.valid, lengths, 0)


class ContractFeatureReader:
    """Arrow-native parser from a FeatureContract ReadPlan to RequestBatch.

    Nested parquet columns are consumed through Arrow offsets, validity bitmaps,
    and flattened child arrays.  No feature column is converted through
    ``to_pylist``; tensors own their storage before the Arrow batch is released.
    """

    def __init__(
        self,
        read_plan: ReadPlan,
        parse_plan: ParsePlan,
        *,
        list_len: int,
        padding_value: int | float = 0,
    ) -> None:
        self.read_plan = read_plan
        self.parse_plan = parse_plan
        self.list_len = int(list_len)
        self.padding_value = padding_value

    def required_columns(self) -> tuple[str, ...]:
        return self.read_plan.columns

    def read_parquet(
        self,
        path: str | Path,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> RequestBatch:
        table = pq.read_table(path, columns=list(self.required_columns()))
        if offset or limit is not None:
            table = table.slice(offset, limit)
        return self.parse_table(table)

    def parse_table(self, table: pa.Table) -> RequestBatch:
        rows = table.num_rows
        request_id = self._parse_optional_id(
            table,
            self.parse_plan.id_fields.get("request_id"),
            "request_id",
            default=torch.arange(rows, dtype=torch.int64),
        )
        user_source = self.parse_plan.id_fields.get("user_id")
        user_id = (
            None
            if user_source is None
            else self._parse_optional_id(table, user_source, "user_id")
        )
        item_source = self.parse_plan.id_fields.get("item_id")
        # The public-data contract has no top-level item ID. These values are
        # candidate positions used only to preserve the legacy batch shape.
        item_id = (
            torch.arange(self.list_len, dtype=torch.int64).repeat(rows, 1)
            if item_source is None
            else self._parse_item_tensor(
                table,
                source_path=item_source,
                dtype="int64",
                field_name="item_id",
                missing_outer_fill=True,
            )
        )
        labels = {
            label: self._parse_item_tensor(
                table,
                source_path=label,
                dtype="float32",
                field_name=label,
            )
            for label in self.parse_plan.labels
        }
        candidate_metadata = {
            name: self._parse_item_tensor(
                table,
                source_path=name,
                dtype=_dtype_name(dtype),
                field_name=name,
            )
            for name, dtype in self.parse_plan.candidate_metadata.items()
        }
        feature_arrays = {
            feature.source_path: self._column_array(
                table, feature.source_path, feature.feature_name
            )
            for feature in (
                *self.parse_plan.features,
                *self.parse_plan.history_length_features,
            )
        }
        features = {
            feature.source_path: self._parse_feature(
                table,
                feature,
                array=feature_arrays[feature.source_path],
            )
            for feature in self.parse_plan.features
        }
        history_group_lengths = _validated_history_group_lengths(
            rows=rows,
            features=self.parse_plan.history_length_features,
            arrays=feature_arrays,
            history_groups=self.parse_plan.history_groups,
        )
        return RequestBatch(
            request_id=request_id,
            user_id=user_id,
            item_id=item_id,
            labels=labels,
            features=features,
            target=self.parse_plan.target,
            metadata=RequestBatchMetadata(),
            candidate_metadata=candidate_metadata,
            history_group_lengths=history_group_lengths,
        )

    def _parse_optional_id(
        self,
        table: pa.Table,
        source_path: str | None,
        field_name: str,
        *,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_path is None:
            if default is None:
                raise DataContractError(f"Field {field_name!r} has no source")
            return default
        array = self._column_array(table, source_path, field_name)
        if array is None or pa.types.is_null(array.type):
            return torch.zeros(table.num_rows, dtype=torch.int64)
        if _is_list_type(array.type):
            raise DataContractError(f"Field {field_name!r} must be a scalar column")
        return _primitive_tensor(array, "int64", field_name, fill_value=0)

    def _parse_item_tensor(
        self,
        table: pa.Table,
        *,
        source_path: str,
        dtype: DTypeName,
        field_name: str,
        missing_outer_fill: bool = False,
    ) -> torch.Tensor:
        array = self._column_array(table, source_path, field_name)
        if array is None or pa.types.is_null(array.type):
            if missing_outer_fill and self.parse_plan.missing_slot_policy == "fill_zero":
                return torch.zeros(
                    (table.num_rows, self.list_len), dtype=torch_dtype(dtype)
                )
            if self.parse_plan.bad_length_policy == "pad_or_truncate":
                return torch.zeros(
                    (table.num_rows, self.list_len), dtype=torch_dtype(dtype)
                )
            _raise_bad_outer_length(
                field_name,
                0,
                self.list_len,
                self.parse_plan.bad_length_policy,
            )
        assert array is not None
        outer = _list_layout(array, field_name)
        selected = _select_item_candidates(
            outer,
            list_len=self.list_len,
            field_name=field_name,
            policy=self.parse_plan.bad_length_policy,
        )
        fill_value: int | float = 0.0 if dtype.startswith("float") else 0
        if _is_list_type(outer.values.type):
            inner = _list_layout(outer.values, field_name)
            lengths = inner.lengths
            selected_lengths = _gather_or_zero(lengths, selected)
            if np.any(selected_lengths > 1):
                bad = int(selected_lengths[selected_lengths > 1][0])
                raise DataContractError(
                    f"Field {field_name!r} expected singleton item id list, got {bad}"
                )
            result = np.full(selected.size, fill_value, dtype=_numpy_dtype(dtype))
            present = (selected >= 0) & (selected_lengths == 1)
            if present.any():
                starts = inner.offsets[selected[present]]
                result[present] = _primitive_numpy(
                    pc.take(inner.values, pa.array(starts, type=pa.int64())),
                    dtype,
                    field_name,
                )
        else:
            result = np.full(selected.size, fill_value, dtype=_numpy_dtype(dtype))
            present = selected >= 0
            if present.any():
                child_valid = _validity_numpy(outer.values)
                present_indices = selected[present]
                valid_positions = child_valid[present_indices]
                target_positions = np.flatnonzero(present)[valid_positions]
                if target_positions.size:
                    result[target_positions] = _primitive_numpy(
                        pc.take(
                            outer.values,
                            pa.array(present_indices[valid_positions], type=pa.int64()),
                        ),
                        dtype,
                        field_name,
                    )
        return torch.from_numpy(result.copy()).reshape(table.num_rows, self.list_len)

    def _parse_feature(
        self,
        table: pa.Table,
        feature: ReadPlanFeature,
        *,
        array: pa.Array | None,
    ) -> FeatureValue:
        parser_kind = _canonical_parser_kind(feature)
        role = _resolved_feature_role(feature, parser_kind)
        dtype = _dtype_name(feature.dtype)
        if feature.source_path == "item_id":
            return self._parse_item_tensor(
                table,
                source_path=feature.source_path,
                dtype=dtype,
                field_name=feature.feature_name,
                missing_outer_fill=True,
            )
        if role == "sequence":
            return _parse_request_sequence_array(array, table.num_rows, feature, dtype)
        if feature.encoding == "embedding" and role == "request":
            return _parse_request_ragged_array(
                array,
                table.num_rows,
                feature,
                dtype,
                scalar_layout=parser_kind == "request_scalar",
            )
        if feature.encoding == "embedding" and role == "item":
            return _parse_item_ragged_array(
                array,
                table.num_rows,
                feature,
                dtype,
                self.list_len,
                scalar_layout=parser_kind == "item_scalar",
                bad_length_policy=self.parse_plan.bad_length_policy,
            )
        if parser_kind == "request_scalar":
            return _parse_request_dense_array(array, table.num_rows, feature, dtype)
        if parser_kind == "item_scalar":
            return _parse_item_dense_array(
                array,
                table.num_rows,
                feature,
                dtype,
                self.list_len,
                scalar_layout=True,
                bad_length_policy=self.parse_plan.bad_length_policy,
            )
        if parser_kind == "item_sequence":
            return _parse_item_ragged_array(
                array,
                table.num_rows,
                feature,
                dtype,
                self.list_len,
                scalar_layout=False,
                bad_length_policy=self.parse_plan.bad_length_policy,
                require_inner_list=True,
            )
        raise DataContractError(
            f"Feature {feature.feature_name!r} has unsupported parser_kind {parser_kind!r}"
        )

    def _column_array(
        self,
        table: pa.Table,
        source_path: str,
        field_name: str,
    ) -> pa.Array | None:
        column = _physical_column(source_path)
        if column in table.column_names:
            return table[column].combine_chunks()
        if self.parse_plan.missing_slot_policy == "fill_zero":
            return None
        raise DataContractError(
            f"Field {field_name!r} requires missing parquet column {column!r}"
        )


def _validated_history_group_lengths(
    *,
    rows: int,
    features: Sequence[ReadPlanFeature],
    arrays: Mapping[str, pa.Array | None],
    history_groups: Mapping[str, Any],
) -> Mapping[str, HistoryGroupLengths]:
    observed: dict[
        str,
        dict[str, tuple[torch.Tensor, torch.Tensor]],
    ] = {}
    direct_features: dict[str, dict[str, tuple[ReadPlanFeature, pa.Array | None]]] = {}
    for feature in features:
        group_name = feature.history_group
        if group_name is None:
            continue
        rule = history_groups.get(group_name)
        if rule is None:
            raise DataContractError(
                f"history feature {feature.feature_name!r} references unknown "
                f"history group {group_name!r}"
            )
        if feature.history_rule != rule:
            raise DataContractError(
                f"history feature {feature.feature_name!r} has a rule that differs "
                f"from group {group_name!r}"
            )
        if feature.encoding == "direct":
            group = direct_features.setdefault(group_name, {})
            if feature.source_path in group:
                raise DataContractError(
                    f"history group {group_name!r} duplicates source "
                    f"{feature.source_path!r}"
                )
            group[feature.source_path] = (
                feature,
                arrays[feature.source_path],
            )
            continue
        raw, truncated = _history_sequence_lengths(
            arrays[feature.source_path],
            rows=rows,
            feature=feature,
        )
        if bool(truncated.gt(int(rule.max_len)).any().item()):
            raise DataContractError(
                f"history feature {feature.feature_name!r} exceeds max_length "
                f"{rule.max_len} after truncation"
            )
        group = observed.setdefault(group_name, {})
        if feature.source_path in group:
            raise DataContractError(
                f"history group {group_name!r} duplicates source "
                f"{feature.source_path!r}"
            )
        group[feature.source_path] = (raw, truncated)

    canonical: dict[str, HistoryGroupLengths] = {}
    for group_name, values in observed.items():
        rule = history_groups[group_name]
        anchor_path = rule.anchor
        if anchor_path is None:
            anchor_path = next(iter(values))
        anchor = values.get(anchor_path)
        if anchor is None:
            raise DataContractError(
                f"history group {group_name!r} is missing length anchor "
                f"{anchor_path!r}; the anchor must be a ragged categorical feature"
            )
        anchor_raw, anchor_truncated = anchor
        if bool(rule.validate_aligned_lengths):
            for source_path, (raw, truncated) in values.items():
                raw_bad = torch.nonzero(raw.ne(anchor_raw), as_tuple=False).flatten()
                if int(raw_bad.numel()):
                    row = int(raw_bad[0].item())
                    raise DataContractError(
                        f"history group {group_name!r} raw lengths mismatch at row "
                        f"{row}: anchor {anchor_path!r}={int(anchor_raw[row])}, "
                        f"feature {source_path!r}={int(raw[row])}"
                    )
                truncated_bad = torch.nonzero(
                    truncated.ne(anchor_truncated), as_tuple=False
                ).flatten()
                if int(truncated_bad.numel()):
                    row = int(truncated_bad[0].item())
                    raise DataContractError(
                        f"history group {group_name!r} truncated lengths mismatch at "
                        f"row {row}: anchor {anchor_path!r}="
                        f"{int(anchor_truncated[row])}, feature {source_path!r}="
                        f"{int(truncated[row])}"
                    )
            for feature, array in direct_features.get(group_name, {}).values():
                _validate_direct_history_padding(
                    array,
                    rows=rows,
                    feature=feature,
                    canonical_lengths=anchor_truncated,
                )
        canonical[group_name] = HistoryGroupLengths(
            raw_lengths=anchor_raw,
            truncated_lengths=anchor_truncated,
        )
    return canonical


def _validate_direct_history_padding(
    array: pa.Array | None,
    *,
    rows: int,
    feature: ReadPlanFeature,
    canonical_lengths: torch.Tensor,
) -> None:
    rule = feature.history_rule
    if rule is None:
        raise DataContractError(
            f"direct history feature {feature.feature_name!r} has no history rule"
        )
    padding_side = rule.direct_padding_side
    if padding_side not in {"left", "right"}:
        raise DataContractError(
            f"direct history feature {feature.feature_name!r} requires an explicit "
            "dataset-side direct_padding_side"
        )
    max_len = int(rule.max_len)
    if array is None or pa.types.is_null(array.type):
        physical_lengths = np.zeros(rows, dtype=np.int64)
        layout = None
    else:
        layout = _list_layout(array, feature.feature_name)
        if _is_list_type(layout.values.type):
            raise DataContractError(
                f"Direct history feature {feature.feature_name!r} must have "
                "physical shape [max_length]"
            )
        physical_lengths = layout.lengths.astype(np.int64, copy=False)
    bad_width = np.flatnonzero(physical_lengths != max_len)
    if bad_width.size:
        row = int(bad_width[0])
        raise DataContractError(
            f"direct history feature {feature.feature_name!r} physical width "
            f"must be {max_len} at row {row}, got {int(physical_lengths[row])}"
        )
    if layout is None or rows == 0:
        return
    values = _primitive_numpy(
        layout.values,
        _dtype_name(feature.dtype),
        feature.feature_name,
    ).reshape(rows, max_len)
    logical_lengths = canonical_lengths.detach().cpu().numpy().astype(
        np.int64, copy=False
    )
    positions = np.arange(max_len, dtype=np.int64)[None, :]
    if padding_side == "right":
        padding_mask = positions >= logical_lengths[:, None]
    else:
        padding_mask = positions < (max_len - logical_lengths)[:, None]
    padding_value = np.asarray(rule.padding_value, dtype=values.dtype)
    bad_padding = np.argwhere(padding_mask & (values != padding_value))
    if bad_padding.size:
        row, position = (int(value) for value in bad_padding[0])
        raise DataContractError(
            f"direct history feature {feature.feature_name!r} has non-padding "
            f"value outside canonical logical length at row {row}, "
            f"position {position}"
        )
    if feature.representation == "time":
        logical_mask = ~padding_mask
        bad_finite = np.argwhere(logical_mask & ~np.isfinite(values))
        if bad_finite.size:
            row, position = (int(value) for value in bad_finite[0])
            raise DataContractError(
                f"direct time feature {feature.feature_name!r} must be finite "
                f"inside canonical logical length at row {row}, position {position}"
            )


def _history_sequence_lengths(
    array: pa.Array | None,
    *,
    rows: int,
    feature: ReadPlanFeature,
) -> tuple[torch.Tensor, torch.Tensor]:
    if array is None or pa.types.is_null(array.type):
        raw = np.zeros(rows, dtype=np.int64)
    else:
        layout = _list_layout(array, feature.feature_name)
        if _is_list_type(layout.values.type):
            raise DataContractError(
                f"Sequence feature {feature.feature_name!r} must have physical "
                "shape [S] for history length validation"
            )
        raw = layout.lengths.astype(np.int64, copy=False)
    rule = feature.history_rule
    if rule is None:
        truncated = raw
    else:
        truncated = np.minimum(raw, int(rule.max_len)).astype(
            np.int64, copy=False
        )
    return (
        torch.from_numpy(raw.copy()).to(torch.long),
        torch.from_numpy(truncated.copy()).to(torch.long),
    )


def _parse_request_sequence_array(
    array: pa.Array | None,
    rows: int,
    feature: ReadPlanFeature,
    dtype: DTypeName,
) -> FeatureValue:
    history_rule = feature.history_rule
    fill_value = feature.parse.get(
        "fill_value", 0 if history_rule is None else history_rule.padding_value
    )
    if array is None or pa.types.is_null(array.type):
        layout = None
        lengths = np.zeros(rows, dtype=np.int64)
        starts = np.zeros(rows, dtype=np.int64)
        values = pa.array([], type=_arrow_type(dtype))
    else:
        layout = _list_layout(array, feature.feature_name)
        if _is_list_type(layout.values.type):
            raise DataContractError(
                f"Sequence feature {feature.feature_name!r} must have physical "
                "shape [S]; nested [S, K] input is not supported"
            )
        lengths = layout.lengths.astype(np.int64, copy=False)
        starts = layout.offsets[:-1].astype(np.int64, copy=False)
        values = layout.values
    if history_rule is not None:
        starts, lengths = _truncate_segments(
            starts,
            lengths,
            int(history_rule.max_len),
            history_rule.truncate_side,
        )
    sequence_layout = (
        "padded"
        if feature.encoding == "direct"
        else (history_rule.layout if history_rule is not None else "ragged")
    )
    if sequence_layout == "padded":
        max_len = _outer_len(feature)
        starts, lengths = _truncate_segments(starts, lengths, max_len, "head")
        output = np.full((rows, max_len), fill_value, dtype=_numpy_dtype(dtype))
        value_indices, owners, positions = _segment_value_coordinates(starts, lengths)
        if value_indices.size:
            output[owners, positions] = _primitive_numpy(
                pc.take(values, pa.array(value_indices, type=pa.int64())),
                dtype,
                feature.feature_name,
            )
        return torch.from_numpy(output.copy())
    if sequence_layout != "ragged":
        raise DataContractError(
            f"Feature {feature.feature_name!r} has unsupported sequence layout "
            f"{sequence_layout!r}"
        )
    return _ragged_from_segments(
        values, starts, lengths, feature, dtype, level="request"
    )


def _parse_request_ragged_array(
    array: pa.Array | None,
    rows: int,
    feature: ReadPlanFeature,
    dtype: DTypeName,
    *,
    scalar_layout: bool,
) -> RaggedFeature:
    fill_value = feature.parse.get("fill_value", 0)
    if array is None or pa.types.is_null(array.type):
        if scalar_layout:
            values = torch.full((rows,), fill_value, dtype=torch_dtype(dtype))
            lengths = torch.ones(rows, dtype=torch.long)
        else:
            values = torch.empty(0, dtype=torch_dtype(dtype))
            lengths = torch.zeros(rows, dtype=torch.long)
        return RaggedFeature(values, lengths, feature.source_path, "request")
    if not _is_list_type(array.type):
        if scalar_layout:
            values = _primitive_tensor(
                array,
                dtype,
                feature.feature_name,
                fill_value=fill_value,
            )
            lengths = torch.ones(rows, dtype=torch.long)
        else:
            valid = _validity_numpy(array)
            selected = pc.filter(array, pa.array(valid))
            values = _primitive_tensor(selected, dtype, feature.feature_name)
            lengths = torch.from_numpy(valid.astype(np.int64, copy=True))
        return RaggedFeature(values, lengths, feature.source_path, "request")
    layout = _list_layout(array, feature.feature_name)
    if _is_list_type(layout.values.type):
        raise DataContractError(
            f"Feature {feature.feature_name!r} expected physical shape [K], got nested list"
        )
    lengths = layout.lengths.astype(np.int64, copy=False)
    if scalar_layout:
        if np.any(lengths > 1):
            bad = int(lengths[lengths > 1][0])
            raise DataContractError(
                f"Feature {feature.feature_name!r} expected at most one value, got {bad}"
            )
        output = np.full(rows, fill_value, dtype=_numpy_dtype(dtype))
        present = lengths == 1
        if present.any():
            output[present] = _primitive_numpy(
                pc.take(
                    layout.values,
                    pa.array(layout.offsets[:-1][present], type=pa.int64()),
                ),
                dtype,
                feature.feature_name,
            )
        return RaggedFeature(
            torch.from_numpy(output.copy()),
            torch.ones(rows, dtype=torch.long),
            feature.source_path,
            "request",
        )
    return _ragged_from_segments(
        layout.values,
        layout.offsets[:-1],
        lengths,
        feature,
        dtype,
        level="request",
    )


def _parse_item_ragged_array(
    array: pa.Array | None,
    rows: int,
    feature: ReadPlanFeature,
    dtype: DTypeName,
    list_len: int,
    *,
    scalar_layout: bool,
    bad_length_policy: str,
    require_inner_list: bool = False,
) -> RaggedFeature:
    if array is None or pa.types.is_null(array.type):
        if bad_length_policy != "pad_or_truncate":
            _raise_bad_outer_length(
                feature.feature_name, 0, list_len, bad_length_policy
            )
        segment_count = rows * list_len
        if scalar_layout:
            fill_value = feature.parse.get("fill_value", 0)
            return RaggedFeature(
                torch.full((segment_count,), fill_value, dtype=torch_dtype(dtype)),
                torch.ones(segment_count, dtype=torch.long),
                feature.source_path,
                "item",
            )
        return RaggedFeature(
            torch.empty(0, dtype=torch_dtype(dtype)),
            torch.zeros(segment_count, dtype=torch.long),
            feature.source_path,
            "item",
        )
    assert array is not None
    outer = _list_layout(array, feature.feature_name)
    selected = _select_item_candidates(
        outer,
        list_len=list_len,
        field_name=feature.feature_name,
        policy=bad_length_policy,
    )
    if not _is_list_type(outer.values.type):
        if require_inner_list:
            raise DataContractError(
                f"Feature {feature.feature_name!r} item_sequence requires a list "
                "per candidate"
            )
        starts = np.maximum(selected, 0)
        lengths = (selected >= 0).astype(np.int64)
        present = selected >= 0
        if present.any():
            child_valid = _validity_numpy(outer.values)
            lengths[present] = child_valid[selected[present]].astype(np.int64)
        values = outer.values
    else:
        inner = _list_layout(outer.values, feature.feature_name)
        starts = _gather_or_zero(inner.offsets[:-1], selected)
        lengths = _gather_or_zero(inner.lengths, selected)
        values = inner.values
    if scalar_layout:
        if np.any(lengths > 1):
            bad = int(lengths[lengths > 1][0])
            raise DataContractError(
                f"Feature {feature.feature_name!r} expected at most one value per "
                f"candidate, got {bad}"
            )
        fill_value = feature.parse.get("fill_value", 0)
        output = np.full(selected.size, fill_value, dtype=_numpy_dtype(dtype))
        present = lengths == 1
        if present.any():
            output[present] = _primitive_numpy(
                pc.take(values, pa.array(starts[present], type=pa.int64())),
                dtype,
                feature.feature_name,
            )
        return RaggedFeature(
            torch.from_numpy(output.copy()),
            torch.ones(selected.size, dtype=torch.long),
            feature.source_path,
            "item",
        )
    return _ragged_from_segments(
        values, starts, lengths, feature, dtype, level="item"
    )


def _parse_request_dense_array(
    array: pa.Array | None,
    rows: int,
    feature: ReadPlanFeature,
    dtype: DTypeName,
) -> torch.Tensor:
    fill_value = feature.parse.get("fill_value", 0)
    if array is None or pa.types.is_null(array.type):
        return torch.full((rows, 1), fill_value, dtype=torch_dtype(dtype))
    if not _is_list_type(array.type):
        return _primitive_tensor(
            array, dtype, feature.feature_name, fill_value=fill_value
        ).reshape(rows, 1)
    layout = _list_layout(array, feature.feature_name)
    lengths = layout.lengths
    if np.any(lengths > 1):
        bad = int(lengths[lengths > 1][0])
        raise DataContractError(
            f"Feature {feature.feature_name!r} expected singleton list, got {bad}"
        )
    output = np.full(rows, fill_value, dtype=_numpy_dtype(dtype))
    present = lengths == 1
    if present.any():
        output[present] = _primitive_numpy(
            pc.take(
                layout.values,
                pa.array(layout.offsets[:-1][present], type=pa.int64()),
            ),
            dtype,
            feature.feature_name,
        )
    return torch.from_numpy(output.copy()).reshape(rows, 1)


def _parse_item_dense_array(
    array: pa.Array | None,
    rows: int,
    feature: ReadPlanFeature,
    dtype: DTypeName,
    list_len: int,
    *,
    scalar_layout: bool,
    bad_length_policy: str,
) -> torch.Tensor:
    ragged = _parse_item_ragged_array(
        array,
        rows,
        feature,
        dtype,
        list_len,
        scalar_layout=scalar_layout,
        bad_length_policy=bad_length_policy,
    )
    return ragged.values.reshape(rows, list_len)


def _ragged_from_segments(
    values: pa.Array,
    starts: np.ndarray,
    lengths: np.ndarray,
    feature: ReadPlanFeature,
    dtype: DTypeName,
    *,
    level: str,
) -> RaggedFeature:
    indices, _, _ = _segment_value_coordinates(starts, lengths)
    selected = pc.take(values, pa.array(indices, type=pa.int64()))
    return RaggedFeature(
        values=_primitive_tensor(selected, dtype, feature.feature_name),
        lengths=torch.from_numpy(lengths.astype(np.int64, copy=True)),
        source_path=feature.source_path,
        level=level,
    )


def _segment_value_coordinates(
    starts: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = lengths.astype(np.int64, copy=False)
    total = int(lengths.sum())
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    prefix = np.cumsum(lengths, dtype=np.int64) - lengths
    indices = (
        np.repeat(starts.astype(np.int64, copy=False) - prefix, lengths)
        + np.arange(total, dtype=np.int64)
    )
    owners = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
    positions = np.arange(total, dtype=np.int64) - np.repeat(prefix, lengths)
    return indices, owners, positions


def _gather_or_zero(source: np.ndarray, indices: np.ndarray) -> np.ndarray:
    output = np.zeros(indices.shape, dtype=np.int64)
    present = indices >= 0
    if present.any():
        output[present] = source[indices[present]]
    return output


def _truncate_segments(
    starts: np.ndarray,
    lengths: np.ndarray,
    max_len: int,
    truncate_side: str,
) -> tuple[np.ndarray, np.ndarray]:
    if truncate_side not in {"head", "tail"}:
        raise DataContractError(
            f"truncate_side must be one of: head, tail; got {truncate_side!r}"
        )
    clipped = np.minimum(lengths, max_len).astype(np.int64, copy=False)
    if truncate_side == "tail":
        starts = starts + lengths - clipped
    return starts.astype(np.int64, copy=False), clipped


def _select_item_candidates(
    outer: _ListLayout,
    *,
    list_len: int,
    field_name: str,
    policy: str,
) -> np.ndarray:
    lengths = outer.lengths.astype(np.int64, copy=False)
    bad = np.flatnonzero(lengths != list_len)
    if bad.size and policy != "pad_or_truncate":
        _raise_bad_outer_length(field_name, int(lengths[bad[0]]), list_len, policy)
    if policy not in {"fail_fast", "pad_or_truncate", "drop_sample"}:
        raise DataContractError(f"Unsupported bad_length policy {policy!r}")
    positions = np.broadcast_to(
        np.arange(list_len, dtype=np.int64), (lengths.size, list_len)
    )
    present = positions < lengths[:, None]
    selected = outer.offsets[:-1, None] + positions
    return np.where(present, selected, -1).reshape(-1)


def _raise_bad_outer_length(
    field_name: str,
    actual: int,
    expected: int,
    policy: str,
) -> None:
    if policy == "drop_sample":
        raise DataContractError(
            f"{field_name!r} requires distributed-safe drop planning; "
            "drop_sample is not enabled in the streaming parser yet"
        )
    if policy == "fail_fast":
        raise DataContractError(
            f"Field {field_name!r} length must be {expected}, got {actual}"
        )
    if policy != "pad_or_truncate":
        raise DataContractError(f"Unsupported bad_length policy {policy!r}")


def _list_layout(array: pa.Array, field_name: str) -> _ListLayout:
    if not _is_list_type(array.type):
        raise DataContractError(
            f"Field {field_name!r} must be an Arrow ListArray, got {array.type}"
        )
    if pa.types.is_fixed_size_list(array.type):
        list_size = int(array.type.list_size)
        offsets = np.arange(len(array) + 1, dtype=np.int64) * list_size
        start = int(array.offset) * list_size
        values = array.values.slice(start, len(array) * list_size)
    else:
        raw_offsets = np.array(array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        base = int(raw_offsets[0])
        offsets = raw_offsets - base
        values = array.values.slice(base, int(raw_offsets[-1] - base))
    return _ListLayout(
        offsets=offsets,
        valid=_validity_numpy(array),
        values=values,
    )


def _validity_numpy(array: pa.Array) -> np.ndarray:
    if array.null_count == 0:
        return np.ones(len(array), dtype=np.bool_)
    return np.array(array.is_valid().to_numpy(zero_copy_only=False), dtype=np.bool_)


def _primitive_tensor(
    array: pa.Array,
    dtype: DTypeName,
    field_name: str,
    *,
    fill_value: int | float | bool | None = None,
) -> torch.Tensor:
    return torch.from_numpy(
        _primitive_numpy(array, dtype, field_name, fill_value=fill_value).copy()
    )


def _primitive_numpy(
    array: pa.Array,
    dtype: DTypeName,
    field_name: str,
    *,
    fill_value: int | float | bool | None = None,
) -> np.ndarray:
    if _is_list_type(array.type):
        raise DataContractError(
            f"Field {field_name!r} has unexpected nested value type {array.type}"
        )
    if array.null_count:
        if fill_value is None:
            raise DataContractError(
                f"Field {field_name!r} contains null scalar values"
            )
        array = pc.fill_null(array, pa.scalar(fill_value, type=array.type))
    try:
        return np.array(
            array.to_numpy(zero_copy_only=False),
            dtype=_numpy_dtype(dtype),
            copy=True,
        )
    except (pa.ArrowInvalid, TypeError, ValueError) as exc:
        raise DataContractError(
            f"Field {field_name!r} cannot be converted to {dtype}: {exc}"
        ) from exc


def _is_list_type(dtype: pa.DataType) -> bool:
    return (
        pa.types.is_list(dtype)
        or pa.types.is_large_list(dtype)
        or pa.types.is_fixed_size_list(dtype)
    )


def _arrow_type(dtype: DTypeName) -> pa.DataType:
    return {
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int64": pa.int64(),
        "int32": pa.int32(),
        "bool": pa.bool_(),
    }[dtype]


def _numpy_dtype(dtype: DTypeName) -> np.dtype[Any]:
    return np.dtype(
        {
            "float32": np.float32,
            "float64": np.float64,
            "int64": np.int64,
            "int32": np.int32,
            "bool": np.bool_,
        }[dtype]
    )


def _resolved_feature_role(feature: ReadPlanFeature, parser_kind: str) -> str:
    if feature.role is not None:
        return feature.role
    if feature.history_group is not None or feature.history_rule is not None:
        return "sequence"
    if parser_kind.startswith("item_") or feature.reader_group == "item_slots":
        return "item"
    return "request"


def _canonical_parser_kind(feature: ReadPlanFeature) -> str:
    raw = str(feature.parse.get("kind") or feature.parser_kind or "")
    legacy = {
        "request_sequence_pad_truncate": "request_sequence",
        "request_scalar_squeeze_singleton": "request_scalar",
        "item_scalar_squeeze_inner_singleton": "item_scalar",
        "item_sequence_pad_inner": "item_sequence",
    }
    return legacy.get(raw, raw)


def _physical_column(source_path: str) -> str:
    parts = str(source_path).split(".")
    if len(parts) == 2 and parts[0] in {
        "user_slots",
        "context_user_slots",
        "item_slots",
    }:
        return f"slot_{normalize_slot_id(parts[1])}"
    return str(source_path)


def _outer_len(feature: ReadPlanFeature) -> int:
    if feature.history_rule is not None:
        return int(feature.history_rule.max_len)
    value = feature.parse.get(
        "max_len",
        feature.parse.get("pad_or_truncate_outer_to", feature.max_outer_len),
    )
    if value is None:
        raise DataContractError(
            f"Feature {feature.feature_name!r} is missing outer length"
        )
    return int(value)


def _dtype_name(dtype: str) -> DTypeName:
    if dtype not in {"float32", "float64", "int64", "int32", "bool"}:
        raise DataContractError(f"Unsupported dtype: {dtype!r}")
    return dtype  # type: ignore[return-value]
