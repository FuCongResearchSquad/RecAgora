"""Public RecAgora model integration API."""

from .losses import CandidateLossContext, masked_binary_cross_entropy_with_logits
from .model import (
    RecAgoraInputSpec,
    RecAgoraModelBatch,
    RecAgoraPointwiseBatch,
    RecAgoraRankingModel,
    RecAgoraSlotLookup,
    RecAgoraSlotSpec,
    RecAgoraSparseBatch,
    BatchIDs,
    FixedFeatureSpec,
    FixedFeatureView,
    HistoryGroupSpec,
    HistoryView,
    JaggedFeatureView,
    JaggedIDView,
    NamedTensorView,
)
from .output import RankingOutput, structured_ranking_output

__all__ = [
    "RecAgoraInputSpec",
    "RecAgoraModelBatch",
    "RecAgoraPointwiseBatch",
    "RecAgoraRankingModel",
    "RecAgoraSlotLookup",
    "RecAgoraSlotSpec",
    "RecAgoraSparseBatch",
    "BatchIDs",
    "CandidateLossContext",
    "FixedFeatureSpec",
    "FixedFeatureView",
    "HistoryGroupSpec",
    "HistoryView",
    "JaggedFeatureView",
    "JaggedIDView",
    "NamedTensorView",
    "RankingOutput",
    "masked_binary_cross_entropy_with_logits",
    "structured_ranking_output",
]
