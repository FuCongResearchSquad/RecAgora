from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from recagora import RecAgoraRankingModel, RecAgoraSparseBatch
from utils.runtime.data.request_batch import RequestBatch
from utils.runtime.data.prepack import prepare_embedding_model_batch
from utils.compiler.feature_contract import FeatureContractPlans
from utils.runtime.model.output_contract import (
    RankingOutput,
    split_ranking_output,
    validate_ranking_output,
)


class RecAgoraRuntimeShell(nn.Module):
    """Internal training lifecycle around a public RecAgora ranking model."""

    def __init__(
        self,
        *,
        model: RecAgoraRankingModel,
        plans: FeatureContractPlans,
    ) -> None:
        super().__init__()
        if model.slot_lookup is None:
            raise ValueError("RecAgora model must bind slot lookup before runtime wrapping")
        self.model = model
        self._plans = plans
        self._batch_prepare_timing_enabled = False

    @property
    def input_spec(self):
        return self.model.input_spec

    def prepare_model_batch(self, batch: RequestBatch) -> RecAgoraSparseBatch:
        return prepare_embedding_model_batch(
            batch,
            adapter_plan=self._plans.adapter_plan,
            input_plan=self._plans.embedding_input_plan,
            keyed_jagged_tensor_cls=self.model.slot_lookup.keyed_jagged_tensor_cls,
            collect_timing=self._batch_prepare_timing_enabled,
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        return validate_ranking_output(
            self.model(batch),
            task_names=tuple(self.model.task_names),
            batch_size=batch.batch_size,
            list_len=batch.list_len,
        )

    def compute_loss(
        self,
        output: Any,
        prepared_batch: RecAgoraSparseBatch,
    ) -> Any:
        compute_loss = getattr(self.model, "compute_loss", None)
        if not callable(compute_loss):
            raise AttributeError(
                f"Model {type(self.model).__name__} must define compute_loss"
            )
        return compute_loss(output, prepared_batch)

    def set_batch_prepare_timing_enabled(self, enabled: bool) -> None:
        self._batch_prepare_timing_enabled = bool(enabled)

    def set_embedding_timing_enabled(self, enabled: bool) -> None:
        self._slot_lookup().set_timing_enabled(enabled)

    def set_forward_timing_enabled(self, enabled: bool) -> None:
        self.set_embedding_timing_enabled(enabled)

    def consume_embedding_timing_span(self):
        return self._slot_lookup().consume_timing_span()

    def set_communication_trace_enabled(self, enabled: bool) -> None:
        self._slot_lookup().set_communication_trace_enabled(enabled)

    @property
    def last_embedding_lookup_sec(self) -> float:
        return self._slot_lookup().last_lookup_sec

    def _slot_lookup(self):
        slot_lookup = self.model.slot_lookup
        if slot_lookup is None:
            raise RuntimeError("RecAgora runtime model has no bound slot lookup")
        return slot_lookup


def unwrap_recagora_runtime_output(
    output: Any,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    return split_ranking_output(output)
