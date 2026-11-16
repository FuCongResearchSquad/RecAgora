from __future__ import annotations

import torch

from recagora import RecAgoraModelBatch, RecAgoraSlotLookup, RecAgoraSparseBatch
from utils.compiler.feature_contract import EmbeddingInputPlan
from utils.runtime.timing import AsyncDeviceTimer

from .adapter import RecAgoraModelAdapter
from .embedding import FeatureEmbedding, build_native_embedded_batch


class CompiledSlotLookup(RecAgoraSlotLookup):
    """Framework implementation behind ``RecAgoraRankingModel.get_slots``."""

    def __init__(
        self,
        *,
        input_plan: EmbeddingInputPlan,
        adapter: RecAgoraModelAdapter,
    ) -> None:
        super().__init__()
        self._input_plan = input_plan
        self._adapter = adapter
        self.feature_embedding = FeatureEmbedding(input_plan)
        self.keyed_jagged_tensor_cls = self.feature_embedding.keyed_jagged_tensor_cls
        self._timing_enabled = False
        self._last_lookup_sec = 0.0
        self._timing_span: AsyncDeviceTimer | None = None
        self._communication_trace_enabled = False

    def forward(self, batch: RecAgoraSparseBatch) -> RecAgoraModelBatch:
        if not isinstance(batch, RecAgoraSparseBatch):
            raise TypeError("CompiledSlotLookup expects an RecAgoraSparseBatch")
        payload = batch._runtime_lookup_payload()
        timer = None
        if self._timing_enabled:
            values = payload.sparse_kjt.values()
            device = (
                values.device
                if isinstance(values, torch.Tensor)
                else torch.device("cpu")
            )
            timer = AsyncDeviceTimer(device, enabled=True)
        if self._communication_trace_enabled:
            with torch.profiler.record_function("recagora.embedding_lookup"):
                embeddings = self.feature_embedding(payload.sparse_kjt)
        else:
            embeddings = self.feature_embedding(payload.sparse_kjt)
        if timer is not None:
            timer.stop()
        self._timing_span = timer
        if self._timing_span is not None and self._timing_span.ready():
            self._last_lookup_sec = self._timing_span.elapsed_ms() / 1000.0
        native = build_native_embedded_batch(
            embeddings,
            prepared=batch,
            input_plan=self._input_plan,
        )
        return self._adapter.build_batch(native)

    def set_timing_enabled(self, enabled: bool) -> None:
        self._timing_enabled = bool(enabled)
        self._last_lookup_sec = 0.0
        self._timing_span = None

    def consume_timing_span(self) -> AsyncDeviceTimer | None:
        span = self._timing_span
        self._timing_span = None
        return span

    def set_communication_trace_enabled(self, enabled: bool) -> None:
        self._communication_trace_enabled = bool(enabled)

    @property
    def last_lookup_sec(self) -> float:
        return float(self._last_lookup_sec)
