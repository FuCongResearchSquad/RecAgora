from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from recagora import RecAgoraSparseBatch
from utils.compiler.feature_contract import EmbeddingInputFeature, EmbeddingInputPlan


@dataclass(frozen=True)
class JaggedEmbedding:
    """TorchRec-neutral embedding values with model-semantic segment lengths."""

    values: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True)
class JaggedMappedIDs:
    """Private device-side view into mapped KJT values for one public slot."""

    values: torch.Tensor
    lengths: torch.Tensor


@dataclass(frozen=True)
class NativeEmbeddedBatch:
    """Lookup result before the requested pointwise/listwise fixed-feature view."""

    fixed_features: Mapping[str, torch.Tensor]
    sequence_features: Mapping[str, JaggedEmbedding]
    mapped_ids: Mapping[int, JaggedMappedIDs]
    embeddings: Any
    prepared: RecAgoraSparseBatch
    input_plan: EmbeddingInputPlan

    @property
    def batch_size(self) -> int:
        return self.prepared.batch_size

    @property
    def list_len(self) -> int:
        return self.prepared.list_len

def _embedding_by_name(embedded: Any, name: str) -> Any:
    if isinstance(embedded, Mapping):
        if name in embedded:
            return embedded[name]
        raise KeyError(f"embedding output has no feature {name!r}")
    to_dict = getattr(embedded, "to_dict", None)
    if callable(to_dict):
        return _embedding_by_name(to_dict(), name)
    raise TypeError("embedding output does not support feature-name lookup")


def build_native_embedded_batch(
    embeddings: Any,
    *,
    prepared: RecAgoraSparseBatch,
    input_plan: EmbeddingInputPlan,
) -> NativeEmbeddedBatch:
    """Normalize TorchRec output before any pointwise/listwise model-view transform."""
    fixed_features: dict[str, torch.Tensor] = {}
    sequence_features: dict[str, JaggedEmbedding] = {}
    mapped_ids: dict[int, JaggedMappedIDs] = {}
    lookup_payload = prepared._runtime_lookup_payload()
    semantic_lengths = lookup_payload.feature_segment_lengths
    for feature in input_plan.categorical_features:
        values = _embedding_values_by_name(embeddings, feature.name)
        lengths = semantic_lengths[feature.name]
        if feature.role == "sequence" and feature.history_group is not None:
            lengths = lookup_payload.sequence_group_lengths[feature.history_group]
        jagged = JaggedEmbedding(
            values=values,
            lengths=lengths.to(device=values.device, dtype=torch.long),
        )
        if feature.role == "sequence":
            sequence_features[feature.name] = jagged
            continue
        fixed_features[feature.name] = jagged_segment_reduce(
            jagged,
            reduction=feature.reduction,
        )
    sparse_values = lookup_payload.sparse_kjt.values()
    if not isinstance(sparse_values, torch.Tensor):
        raise TypeError("prepared sparse KJT values must be a Tensor")
    for slot_id, span in lookup_payload.mapped_id_spans.items():
        if span.history_group is None:
            feature = next(
                feature for feature in input_plan.categorical_features
                if feature.slot_id == slot_id
            )
            lengths = semantic_lengths[feature.name]
        else:
            lengths = lookup_payload.sequence_group_lengths[span.history_group]
        mapped_ids[int(slot_id)] = JaggedMappedIDs(
            values=sparse_values.narrow(0, int(span.start), int(span.length)),
            lengths=lengths.to(device=sparse_values.device, dtype=torch.long),
        )
    return NativeEmbeddedBatch(
        fixed_features=fixed_features,
        sequence_features=sequence_features,
        mapped_ids=mapped_ids,
        embeddings=embeddings,
        prepared=prepared,
        input_plan=input_plan,
    )


def jagged_segment_reduce(
    feature: JaggedEmbedding,
    *,
    reduction: str,
) -> torch.Tensor:
    """Reduce non-sequence multi-value embeddings without leaving autograd."""
    values = feature.values
    lengths = feature.lengths.to(device=values.device, dtype=torch.long)
    segment_count = int(lengths.numel())
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_count, device=values.device),
        lengths,
        output_size=values.shape[0],
    )
    if reduction in {"sum", "mean"}:
        pooled = values.new_zeros((segment_count, values.shape[1]))
        pooled.index_add_(0, segment_ids, values)
        if reduction == "sum":
            return pooled
        return pooled / lengths.clamp_min(1).to(values.dtype).unsqueeze(-1)
    pooled = values.new_full(
        (segment_count, values.shape[1]),
        torch.finfo(values.dtype).min,
    )
    indices = segment_ids.unsqueeze(-1).expand(-1, values.shape[1])
    pooled.scatter_reduce_(0, indices, values, reduce="amax", include_self=True)
    return torch.where(lengths.unsqueeze(-1).eq(0), torch.zeros_like(pooled), pooled)


def _embedding_values_by_name(embeddings: Any, name: str) -> torch.Tensor:
    feature = _embedding_by_name(embeddings, name)
    if isinstance(feature, torch.Tensor):
        return feature
    values = getattr(feature, "values", None)
    if callable(values):
        result = values()
        if isinstance(result, torch.Tensor):
            return result
    if isinstance(feature, Mapping):
        for key in ("values", "value", "tensor"):
            result = feature.get(key)
            if isinstance(result, torch.Tensor):
                return result
    raise TypeError(f"feature {name!r} has no tensor embedding values")


class FeatureEmbedding(nn.Module):
    """Config-driven TorchRec embedding lookup shared by ranking models."""

    def __init__(
        self,
        input_plan: EmbeddingInputPlan,
        *,
        device: torch.device | None = None,
    ):
        super().__init__()
        specs = input_plan.id_features
        embedding_config_cls, embedding_collection_cls, keyed_jagged_tensor_cls = (
            _load_torchrec_embedding_classes()
        )
        self.input_plan = input_plan
        specs_by_dim: dict[int, list[EmbeddingInputFeature]] = {}
        for spec in specs:
            specs_by_dim.setdefault(int(spec.embedding_dim), []).append(spec)
        # TorchRec EmbeddingCollection requires every table in one collection to
        # have the same embedding dimension. Keep the grouping inside the common
        # lookup layer so models still consume one feature-name keyed result.
        if device is None:
            # Preserve the established ranking construction path exactly. In
            # particular, do not pass a newer TorchRec ``device`` keyword when
            # the caller did not request deferred materialization.
            self.embedding_collection = nn.ModuleDict(
                {
                    f"dim_{embedding_dim}": embedding_collection_cls(
                        tables=_torchrec_embedding_configs(
                            embedding_config_cls,
                            tuple(dimension_specs),
                        )
                    )
                    for embedding_dim, dimension_specs in specs_by_dim.items()
                }
            )
        else:
            self.embedding_collection = nn.ModuleDict(
                {
                    f"dim_{embedding_dim}": embedding_collection_cls(
                        tables=_torchrec_embedding_configs(
                            embedding_config_cls,
                            tuple(dimension_specs),
                        ),
                        device=device,
                    )
                    for embedding_dim, dimension_specs in specs_by_dim.items()
                }
            )
        self.keyed_jagged_tensor_cls = keyed_jagged_tensor_cls
        self.table_names = input_plan.tables

    def forward(self, sparse_kjt: Any) -> Any:
        """Look up every configured feature while preserving jagged outputs."""
        result: dict[str, Any] = {}
        for collection in self.embedding_collection.values():
            current = resolve_torchrec_embedding_output(collection(sparse_kjt))
            result.update(current)
        return result


def _torchrec_embedding_configs(
    embedding_config_cls: type[Any],
    specs: tuple[Any, ...],
) -> list[Any]:
    configs: list[Any] = []
    for spec in _dedupe_embedding_specs(specs):
        configs.append(
            embedding_config_cls(
                name=spec.resolved_embedding_name,
                embedding_dim=spec.embedding_dim,
                num_embeddings=spec.vocab_size,
                feature_names=[
                    feature.name
                    for feature in specs
                    if feature.resolved_embedding_name == spec.resolved_embedding_name
                ],
            )
        )
    return configs


def _dedupe_embedding_specs(
    specs: tuple[Any, ...],
) -> tuple[Any, ...]:
    by_name: dict[str, Any] = {}
    for spec in specs:
        by_name.setdefault(spec.resolved_embedding_name, spec)
    return tuple(by_name.values())


def _load_torchrec_embedding_classes() -> tuple[type[Any], type[nn.Module], type[Any]]:
    try:
        from torchrec.modules.embedding_configs import EmbeddingConfig
        from torchrec.modules.embedding_modules import EmbeddingCollection
        from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
    except ImportError as exc:
        raise ImportError(
            "FeatureEmbedding requires torchrec/fbgemm_gpu. Run smoke tests in "
            "the AIS container or remote Linux environment."
        ) from exc
    return EmbeddingConfig, EmbeddingCollection, KeyedJaggedTensor


def resolve_torchrec_embedding_output(embedded: Any) -> Any:
    """Normalize local and distributed TorchRec lookup return wrappers."""
    if hasattr(embedded, "wait"):
        embedded = embedded.wait()
    if isinstance(embedded, tuple):
        if not embedded:
            raise TypeError("TorchRec embedding output tuple cannot be empty")
        embedded = embedded[0]
    if hasattr(embedded, "to_dict"):
        return embedded.to_dict()
    return embedded
