from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from recagora.model import (
    RecAgoraInputSpec,
    RecAgoraModelBatch,
    RecAgoraSlotSpec,
    BatchIDs,
    FixedFeatureSpec,
    FixedFeatureView,
    HistoryGroupSpec,
    HistoryView,
    JaggedFeatureView,
    JaggedIDView,
    NamedTensorView,
)
from .embedding import NativeEmbeddedBatch
from utils.compiler.feature_contract import EmbeddingInputFeature, FeatureContractPlans


@dataclass(frozen=True)
class _RuntimeSlotBinding:
    slot_id: int
    internal_name: str
    encoding: str
    history_group: str | None
    mapped_ids_exposed: bool


@dataclass(frozen=True)
class RecAgoraModelAdapter:
    """Private bridge from compiled plans/lookup results to the public model API."""

    input_spec: RecAgoraInputSpec
    _bindings: Mapping[int, _RuntimeSlotBinding]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_bindings", MappingProxyType(dict(self._bindings)))

    def build_batch(self, native: NativeEmbeddedBatch) -> RecAgoraModelBatch:
        prepared = native.prepared
        batch_size = prepared.batch_size
        list_len = prepared.list_len
        if list_len != self.input_spec.list_len:
            raise ValueError(
                f"runtime list_len={list_len} does not match InputSpec "
                f"list_len={self.input_spec.list_len}"
            )

        query_values: dict[int, torch.Tensor] = {}
        candidate_values: dict[int, torch.Tensor] = {}
        candidate_mapped_ids: dict[int, torch.Tensor] = {}
        categorical_by_group: dict[str, dict[int, JaggedFeatureView]] = {
            name: {} for name in self.input_spec.histories
        }
        direct_by_group: dict[str, dict[int, torch.Tensor]] = {
            name: {} for name in self.input_spec.histories
        }
        mapped_ids_by_group: dict[str, dict[int, JaggedIDView]] = {
            name: {} for name in self.input_spec.histories
        }
        history_lengths = _history_group_lengths(native)
        canonical_lengths = {
            name: _require_group_lengths(history_lengths, name)
            for name in self.input_spec.histories
        }
        for slot_id, binding in self._bindings.items():
            slot = self.input_spec.slot(slot_id)
            if slot.view == "query":
                query_values[slot_id] = _fixed_value(native, binding)
            elif slot.view == "candidate":
                value = _fixed_value(native, binding)
                candidate_values[slot_id] = value.reshape(
                    batch_size,
                    list_len,
                    value.shape[-1],
                )
                if binding.mapped_ids_exposed:
                    mapped = native.mapped_ids[slot_id]
                    if mapped.values.numel() != batch_size * list_len or not torch.all(mapped.lengths == 1):
                        raise ValueError(f"mapped candidate slot {slot_id} must have one ID per candidate")
                    candidate_mapped_ids[slot_id] = mapped.values.reshape(batch_size, list_len)
            elif slot.view == "history":
                group = binding.history_group
                if group is None:
                    raise ValueError(
                        f"history slot {slot_id} has no compiled history group"
                    )
                if binding.encoding == "embedding":
                    value = native.sequence_features[binding.internal_name]
                    categorical_by_group[group][slot_id] = JaggedFeatureView(
                        values=value.values,
                        lengths=canonical_lengths[group],
                    )
                    if binding.mapped_ids_exposed:
                        try:
                            mapped = native.mapped_ids[slot_id]
                        except KeyError as exc:
                            raise KeyError(
                                f"lookup result is missing mapped IDs for slot {slot_id}"
                            ) from exc
                        if mapped.values.ndim != 1 or mapped.values.dtype != torch.long:
                            raise ValueError(
                                f"mapped IDs for slot {slot_id} must be 1-D torch.long"
                            )
                        mapped_ids_by_group[group][slot_id] = JaggedIDView(
                            values=mapped.values,
                            lengths=canonical_lengths[group],
                        )
                else:
                    direct_by_group[group][slot_id] = native.prepared.dense[
                        binding.internal_name
                    ]
            else:
                raise ValueError(f"unsupported public feature view {slot.view!r}")

        histories = {
            name: HistoryView(
                lengths=canonical_lengths[name],
                categorical=categorical_by_group[name],
                direct=direct_by_group[name],
                max_length=spec.max_length,
                mapped_ids=mapped_ids_by_group[name],
            )
            for name, spec in self.input_spec.histories.items()
        }
        metadata = native.prepared.metadata
        request_id = _required_tensor(metadata, "request_id")
        item_id = _required_tensor(metadata, "item_id")
        user_id = metadata.get("user_id")
        if user_id is not None and not isinstance(user_id, torch.Tensor):
            raise TypeError("prepared user_id must be a Tensor")
        candidate_metadata = prepared.candidate_metadata
        if not isinstance(candidate_metadata, Mapping):
            raise TypeError("prepared candidate_metadata must be a mapping")
        labels = {
            name: _required_tensor(native.prepared.labels, name)
            for name in self.input_spec.task_names
        }
        return RecAgoraModelBatch(
            ids=BatchIDs(
                request_id=request_id,
                user_id=user_id,
                item_id=item_id,
            ),
            query=FixedFeatureView(
                query_values,
                slot_ids=self.input_spec.query.slot_ids,
                batch_size=batch_size,
                list_len=None,
            ),
            candidate=FixedFeatureView(
                candidate_values,
                slot_ids=self.input_spec.candidate.slot_ids,
                batch_size=batch_size,
                list_len=list_len,
            ),
            histories=histories,
            labels=NamedTensorView(labels),
            candidate_metadata=NamedTensorView(
                {
                    str(name): _required_tensor(candidate_metadata, str(name))
                    for name in candidate_metadata
                }
            ),
            candidate_valid_mask=prepared.candidate_valid_mask,
            candidate_mapped_ids=candidate_mapped_ids,
        )


def compile_recagora_model_adapter(
    plans: FeatureContractPlans,
) -> RecAgoraModelAdapter:
    manifest = plans.resolved_manifest
    input_by_source: dict[str, EmbeddingInputFeature] = {}
    for feature in plans.embedding_input_plan.features:
        if feature.source_path in input_by_source:
            raise ValueError(
                f"duplicate internal source binding {feature.source_path!r}"
            )
        input_by_source[feature.source_path] = feature

    slots: dict[int, RecAgoraSlotSpec] = {}
    bindings: dict[int, _RuntimeSlotBinding] = {}
    for feature in manifest.features:
        private = manifest.private_bindings[feature.slot_id]
        runtime = input_by_source.get(private.source_path)
        if runtime is None:
            raise ValueError(
                f"compiled input plan has no selected slot {feature.slot_id}"
            )
        output_dim = int(feature.output_dim or 1)
        slots[feature.slot_id] = RecAgoraSlotSpec(
            slot_id=feature.slot_id,
            view=feature.view,
            role=feature.role,
            representation=feature.representation,
            encoding=feature.encoding,
            reduction=feature.reduction,
            output_dim=output_dim,
            mapped_ids_exposed=feature.mapped_ids_exposed,
        )
        bindings[feature.slot_id] = _RuntimeSlotBinding(
            slot_id=feature.slot_id,
            internal_name=runtime.name,
            encoding=feature.encoding,
            history_group=feature.history_group,
            mapped_ids_exposed=feature.mapped_ids_exposed,
        )

    query_ids = tuple(slot_id for slot_id, slot in slots.items() if slot.view == "query")
    candidate_ids = tuple(
        slot_id for slot_id, slot in slots.items() if slot.view == "candidate"
    )
    history_specs = _compile_history_specs(plans, slots, manifest.private_bindings)
    input_spec = RecAgoraInputSpec(
        universe_version=manifest.universe_version,
        list_len=plans.adapter_plan.list_len,
        task_names=tuple(plans.parse_plan.labels),
        query=FixedFeatureSpec(
            slot_ids=query_ids,
            total_dim=sum(slots[slot_id].output_dim for slot_id in query_ids),
        ),
        candidate=FixedFeatureSpec(
            slot_ids=candidate_ids,
            total_dim=sum(slots[slot_id].output_dim for slot_id in candidate_ids),
        ),
        histories=history_specs,
        _slots=slots,
    )
    return RecAgoraModelAdapter(input_spec=input_spec, _bindings=bindings)


def compile_recagora_input_spec(plans: FeatureContractPlans) -> RecAgoraInputSpec:
    return compile_recagora_model_adapter(plans).input_spec


def _compile_history_specs(
    plans: FeatureContractPlans,
    slots: Mapping[int, RecAgoraSlotSpec],
    private_bindings: Mapping[int, object],
) -> Mapping[str, HistoryGroupSpec]:
    feature_by_source = {
        feature.source_path: feature for feature in plans.parse_plan.features
    }
    grouped: dict[str, list[int]] = {}
    for slot_id, slot in slots.items():
        if slot.view != "history":
            continue
        private = private_bindings[slot_id]
        source_path = getattr(private, "source_path")
        feature = feature_by_source[source_path]
        if feature.history_group is None:
            raise ValueError(f"history slot {slot_id} has no history group")
        grouped.setdefault(feature.history_group, []).append(slot_id)

    result: dict[str, HistoryGroupSpec] = {}
    for name, slot_ids in grouped.items():
        features = [
            feature_by_source[getattr(private_bindings[slot_id], "source_path")]
            for slot_id in slot_ids
        ]
        rules = [feature.history_rule for feature in features]
        if any(rule is None for rule in rules):
            raise ValueError(f"history group {name!r} has no data contract")
        rule = rules[0]
        if any(other != rule for other in rules[1:]):
            raise ValueError(f"history group {name!r} has inconsistent contracts")
        if rule is None:
            raise ValueError(f"history group {name!r} has no data contract")
        anchor_source = rule.anchor
        anchor_slot_ids = [
            slot_id
            for slot_id in slot_ids
            if getattr(private_bindings[slot_id], "source_path") == anchor_source
        ]
        if len(anchor_slot_ids) > 1:
            raise ValueError(
                f"history group {name!r} resolved multiple length anchors"
            )
        length_anchor_slot_id = (
            anchor_slot_ids[0]
            if anchor_slot_ids
            else _slot_id_from_source(str(anchor_source))
        )
        categorical = tuple(
            slot_id for slot_id in slot_ids if slots[slot_id].encoding == "embedding"
        )
        direct = tuple(
            slot_id for slot_id in slot_ids if slots[slot_id].encoding == "direct"
        )
        result[name] = HistoryGroupSpec(
            name=name,
            categorical_slot_ids=categorical,
            direct_slot_ids=direct,
            event_total_dim=sum(slots[slot_id].output_dim for slot_id in slot_ids),
            max_length=int(rule.max_len),
            event_order=str(getattr(rule, "event_order", "oldest_to_newest")),
            truncate_side=str(rule.truncate_side),
            padding_side=str(getattr(rule, "padding_side", "model_configured")),
            direct_padding_side=str(rule.direct_padding_side),
            padding_value=float(rule.padding_value),
            length_anchor_slot_id=length_anchor_slot_id,
        )
    return MappingProxyType(result)


def _slot_id_from_source(source_path: str) -> int:
    """Resolve a reader-only physical history anchor to its public slot ID."""

    raw = str(source_path).rsplit(".", 1)[-1]
    if raw.startswith("slot_"):
        raw = raw[len("slot_") :]
    if not raw.isdigit():
        raise ValueError(
            f"history length anchor source {source_path!r} has no numeric slot ID"
        )
    return int(raw)


def _fixed_value(
    native: NativeEmbeddedBatch,
    binding: _RuntimeSlotBinding,
) -> torch.Tensor:
    if binding.encoding == "embedding":
        try:
            return native.fixed_features[binding.internal_name]
        except KeyError as exc:
            raise KeyError(
                f"lookup result is missing selected slot {binding.slot_id}"
            ) from exc
    try:
        value = native.prepared.dense[binding.internal_name]
    except KeyError as exc:
        raise KeyError(
            f"direct input is missing selected slot {binding.slot_id}"
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"direct slot {binding.slot_id} must be a Tensor")
    return value if value.ndim >= 2 else value.unsqueeze(-1)


def _history_group_lengths(native: NativeEmbeddedBatch) -> Mapping[str, torch.Tensor]:
    preserved = native.prepared.history_group_lengths
    if not preserved:
        raise ValueError(
            "RecAgora model interface requires reader-validated canonical history lengths"
        )
    return {
        name: lengths.truncated_lengths
        for name, lengths in preserved.items()
    }


def _require_group_lengths(
    lengths: Mapping[str, torch.Tensor],
    name: str,
) -> torch.Tensor:
    value = lengths.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"runtime batch has no validated lengths for {name!r}")
    if value.ndim != 1 or value.dtype != torch.long:
        raise ValueError(
            f"runtime batch canonical lengths for {name!r} must be "
            "one-dimensional torch.long"
        )
    return value


def _required_tensor(values: Mapping[str, object], name: str) -> torch.Tensor:
    value = values.get(name)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"runtime batch has no tensor {name!r}")
    return value
