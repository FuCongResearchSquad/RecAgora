"""CPU-only candidate masking before the unique worker categorical prepack."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from utils.compiler.config import normalize_negative_mask_fields
from utils.compiler.feature_contract import EmbeddingInputPlan
from .request_batch import RaggedFeature, RequestBatch


def mask_request_batch_candidates(
    batch: RequestBatch,
    *,
    fields: Sequence[str],
    input_plan: EmbeddingInputPlan,
) -> RequestBatch:
    """Keep candidate positions, removing only the selected candidates' inputs.

    Labels and IDs remain unchanged for diagnostics. Consumers must respect
    ``candidate_valid_mask``. No positive-label override is applied.
    """
    fields = normalize_negative_mask_fields(fields)
    if not fields:
        return batch
    if batch.packed_categorical is not None:
        raise ValueError("candidate masking must run before categorical prepack")
    if batch.item_id.device.type != "cpu":
        raise ValueError("candidate masking requires a CPU RequestBatch")
    keep = torch.ones_like(batch.item_id, dtype=torch.bool)
    if batch.candidate_valid_mask is not None:
        keep &= batch.candidate_valid_mask
    for name in fields:
        value = batch.candidate_metadata.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"candidate masking requires metadata field {name!r}")
        if value.device.type != "cpu" or value.shape != batch.item_id.shape:
            raise ValueError(f"candidate metadata {name!r} must be CPU with shape [B, L]")
        if not bool((value.eq(0) | value.eq(1)).all().item()):
            raise ValueError(f"candidate metadata {name!r} must contain only 0 or 1")
        keep &= value.ne(1)

    features = dict(batch.features)
    processed: set[str] = set()
    for spec in input_plan.features:
        if spec.source_level != "item" or spec.source_path in processed:
            continue
        processed.add(spec.source_path)
        value = batch.feature(spec.source_path)
        if spec.encoding == "embedding":
            features[spec.source_path] = _mask_categorical(
                value, keep, source_path=spec.source_path
            )
        elif spec.encoding == "direct":
            if not isinstance(value, torch.Tensor):
                raise ValueError("candidate direct features must be dense tensors")
            _validate_dense_candidate(value, keep, spec.source_path)
            expanded_keep = keep.reshape(*keep.shape, *((1,) * (value.ndim - 2)))
            features[spec.source_path] = value.masked_fill(~expanded_keep, 0)
        else:
            raise ValueError(f"unsupported candidate feature encoding {spec.encoding!r}")
    return replace(batch, features=features, candidate_valid_mask=keep)


def _validate_dense_candidate(
    value: torch.Tensor, keep: torch.Tensor, source_path: str
) -> None:
    if (
        value.device.type != "cpu"
        or value.ndim < 2
        or value.shape[:2] != keep.shape
    ):
        raise ValueError(
            f"candidate feature {source_path!r} must be CPU with shape [B, L, ...]"
        )


def _mask_categorical(
    value: torch.Tensor | RaggedFeature,
    keep: torch.Tensor,
    *,
    source_path: str,
) -> RaggedFeature:
    keep_flat = keep.reshape(-1)
    if isinstance(value, RaggedFeature):
        if value.level != "item" or value.lengths.numel() != keep_flat.numel():
            raise ValueError(f"candidate ragged feature {source_path!r} must have B*L segments")
        if value.values.device.type != "cpu" or value.lengths.device.type != "cpu":
            raise ValueError("candidate masking requires CPU ragged features")
        value_keep = torch.repeat_interleave(keep_flat, value.lengths)
        kept_values = value.values[value_keep]
        lengths = value.lengths.masked_fill(~keep_flat, 0)
    else:
        _validate_dense_candidate(value, keep, source_path)
        # Dense categorical candidates may carry one or several IDs each.
        # Empty segments, unlike ID 0, do not issue a lookup for masked slots.
        width = 1
        for size in value.shape[2:]:
            width *= int(size)
        kept_values = value[keep].reshape(-1)
        lengths = keep_flat.to(torch.long) * width
    return RaggedFeature(
        values=kept_values,
        lengths=lengths,
        source_path=source_path,
        level="item",
        # Do not reuse the old metadata: lengths and values have changed.
    )
