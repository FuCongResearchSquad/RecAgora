from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from recagora import (
    RecAgoraInputSpec,
    RecAgoraModelBatch,
    RecAgoraRankingModel,
    RankingOutput,
    RecAgoraSlotLookup,
    RecAgoraSparseBatch,
    structured_ranking_output,
)
from models._validation import (
    finite_positive_float,
    validate_history_order,
    validate_model_dim_heads,
)

from .v1 import JointCausalInputs, JointCausalRanker
from .v2 import TemporalPerceiverInputs, TemporalPerceiverRanker, TemporalPerceiverState


class OneRankModel(RecAgoraRankingModel):
    """One public model assembled from neutral module choices in YAML."""

    # Supported OneRank modes have only candidate-ranking objectives, so a
    # globally empty candidate set can skip the update. V2 pretrain remains
    # unsupported because its independent history NTP objective must survive.
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "model_defined"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        input_groups: Mapping[str, Any],
        slot_encoder: Mapping[str, Any],
        history_encoder: Mapping[str, Any],
        candidate_context: Mapping[str, Any],
        view_encoder: Mapping[str, Any],
        decoder: Mapping[str, Any],
        task_interaction: Mapping[str, Any],
        objectives: Mapping[str, Any],
        initialization: Mapping[str, Any] | None = None,
        acr: Mapping[str, Any] | None = None,
        model_dim: int = 256,
        num_heads: int = 4,
        ffn_kind: str = "swiglu",
        ffn_expansion: float = 4.0,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1.0e-5,
        score_temperature_init: float = 0.1,
    ) -> None:
        inputs = _plain_mapping(input_groups)
        slot_encoder = _plain_mapping(slot_encoder)
        history_encoder = _plain_mapping(history_encoder)
        candidate_context = _plain_mapping(candidate_context)
        view_encoder = _plain_mapping(view_encoder)
        decoder = _plain_mapping(decoder)
        task_interaction = _plain_mapping(task_interaction)
        objectives = _plain_mapping(objectives)
        initialization = _plain_mapping(initialization or {})
        acr = _plain_mapping(acr or {})
        validate_model_dim_heads(model_dim, num_heads)
        if ffn_kind not in {"standard", "swiglu", "convswiglu"}:
            raise ValueError("ffn_kind must be standard, swiglu, or convswiglu")
        finite_positive_float(ffn_expansion, "ffn_expansion")
        finite_positive_float(rms_norm_eps, "rms_norm_eps")
        finite_positive_float(score_temperature_init, "score_temperature_init")
        params = {
            "input_groups": inputs,
            "slot_encoder": slot_encoder,
            "history_encoder": history_encoder,
            "candidate_context": candidate_context,
            "view_encoder": view_encoder,
            "decoder": decoder,
            "task_interaction": task_interaction,
            "objectives": objectives,
            "initialization": initialization,
            "acr": acr,
            "model_dim": int(model_dim),
            "num_heads": int(num_heads),
            "ffn_kind": str(ffn_kind),
            "ffn_expansion": float(ffn_expansion),
            "use_qk_norm": bool(use_qk_norm),
            "rms_norm_eps": float(rms_norm_eps),
            "score_temperature_init": float(score_temperature_init),
        }
        super().__init__(params, input_spec)
        self.requires_init_from = bool(
            initialization.get("require_init_from", False)
        )
        self.history_group_name = str(inputs.get("history_group", "history_group_1"))
        embedding_uniform_range = initialization.get("embedding_uniform_range")
        if embedding_uniform_range is None:
            self.embedding_uniform_range: float | None = None
        else:
            self.embedding_uniform_range = finite_positive_float(
                embedding_uniform_range,
                "embedding_uniform_range",
            )
        self._validate_public_inputs()
        public_history_length = self.input_spec.histories[self.history_group_name].max_length
        self.history_max_events = int(
            history_encoder.get("max_events", public_history_length)
        )
        if not 1 <= self.history_max_events <= public_history_length:
            raise ValueError(
                "OneRank history_encoder.max_events must be between 1 and "
                f"the public history length ({public_history_length})"
            )

        self.input_groups = {
            name: tuple(int(slot) for slot in inputs[name])
            for name in ("history_slots", "candidate_slots", "query_slots")
        }
        self.action_slot = int(inputs.get("history_action_slot", 45040))
        self.timestamp_slot = int(inputs.get("history_timestamp_slot", 45073))
        self.objective_mode = str(objectives.get("mode", "ranking"))
        self.acr_enabled = bool(acr.get("enabled", False))
        if self.acr_enabled and self.objective_mode != "ranking":
            raise ValueError("ACR is only supported for OneRank P1 ranking")
        if self.acr_enabled and int(acr.get("item_slot", 5000)) not in self.input_spec.mapped_id_slot_ids:
            raise ValueError("ACR requires features.mapped_id_slots to include candidate item slot")
        if self.acr_enabled and int(acr.get("history_item_slot", 45041)) not in self.input_spec.mapped_id_slot_ids:
            raise ValueError("ACR requires features.mapped_id_slots to include history item slot")
        self.acr_item_slot = int(acr.get("item_slot", 5000))
        self.acr_history_item_slot = int(acr.get("history_item_slot", 45041))
        if self.objective_mode not in {"ranking", "next_event"}:
            raise ValueError("objectives.mode must be ranking or next_event")

        encoder_type = str(history_encoder["type"])
        if self.acr_enabled and encoder_type != "time_rope_causal":
            raise ValueError("ACR requires the temporal OneRank-v2 backbone")
        if encoder_type == "joint_causal":
            config = self._joint_config(
                slot_encoder,
                history_encoder,
                decoder,
                task_interaction,
                objectives,
                model_dim,
                num_heads,
                ffn_kind,
                ffn_expansion,
                use_qk_norm,
                rms_norm_eps,
                score_temperature_init,
            )
            self.backbone_kind = "joint_causal"
            self.backbone: nn.Module = JointCausalRanker(config)
        elif encoder_type == "time_rope_causal":
            config = self._temporal_config(
                slot_encoder,
                history_encoder,
                candidate_context,
                view_encoder,
                decoder,
                objectives,
                model_dim,
                num_heads,
                ffn_kind,
                ffn_expansion,
                use_qk_norm,
                rms_norm_eps,
                score_temperature_init,
                acr,
            )
            self.backbone_kind = "time_rope_causal"
            self.backbone = TemporalPerceiverRanker(config)
        else:
            raise ValueError(
                "history_encoder.type must be joint_causal or time_rope_causal"
            )
        self.supports_candidate_mask = self.backbone_kind == "joint_causal" or (
            self.backbone_kind == "time_rope_causal"
            and self.objective_mode == "ranking"
        )
        self.backbone_config = config
        self.task_names = tuple(config.task_names)
        self._pending_temporal_state: TemporalPerceiverState | None = None

    def train(self, mode: bool = True) -> OneRankModel:
        if not mode:
            self._pending_temporal_state = None
        return super().train(mode)

    def bind_slot_lookup(self, slot_lookup: RecAgoraSlotLookup) -> None:
        super().bind_slot_lookup(slot_lookup)
        if self.embedding_uniform_range is None:
            return
        with torch.no_grad():
            for parameter in slot_lookup.parameters():
                parameter.uniform_(
                    -self.embedding_uniform_range,
                    self.embedding_uniform_range,
                )

    def _validate_public_inputs(self) -> None:
        history = self.input_spec.histories[self.history_group_name]
        validate_history_order(history, context="OneRank")
        if history.padding_side not in {"left", "model_configured"}:
            raise ValueError("OneRank requires left/model-configured history padding")

    def _validate_groups(self) -> None:
        history = self.input_spec.histories[self.history_group_name]
        history_slots = set(self.input_groups["history_slots"])
        if history_slots | {self.action_slot} != set(history.categorical_slot_ids):
            raise ValueError(
                "OneRank history slots plus action must cover categorical history"
            )
        if history_slots & {self.action_slot}:
            raise ValueError("OneRank action slot must be separate from history slots")
        if set(self.input_groups["candidate_slots"]) != set(
            self.input_spec.candidate.slot_ids
        ):
            raise ValueError("OneRank candidate slots must cover the candidate view")
        if set(self.input_groups["query_slots"]) != set(
            self.input_spec.query.slot_ids
        ):
            raise ValueError("OneRank query slots must cover the query view")

    def _joint_config(
        self,
        slots: Mapping[str, Any],
        history: Mapping[str, Any],
        decoder: Mapping[str, Any],
        interaction: Mapping[str, Any],
        objectives: Mapping[str, Any],
        model_dim: int,
        num_heads: int,
        ffn_kind: str,
        ffn_expansion: float,
        use_qk_norm: bool,
        rms_norm_eps: float,
        score_temperature_init: float,
    ) -> SimpleNamespace:
        self._validate_groups()
        if int(slots.get("residual_dim", 0)) != 0:
            raise ValueError("joint_causal slot encoder does not use residual dimensions")
        if str(candidate_context_type := self.config["candidate_context"]["type"]) != "joint_causal":
            raise ValueError(
                f"joint_causal history requires joint_causal candidate context, got {candidate_context_type!r}"
            )
        ranking = _plain_mapping(objectives["ranking"])
        tasks = tuple(str(name) for name in ranking["tasks"])
        if self.objective_mode != "ranking":
            raise ValueError("joint_causal architecture supports ranking objective")
        return SimpleNamespace(
            history_length=self.input_spec.histories[
                self.history_group_name
            ].max_length,
            history_slot_count=len(self.input_groups["history_slots"]),
            candidate_slot_count=len(self.input_groups["candidate_slots"]),
            query_slot_count=len(self.input_groups["query_slots"]),
            slot_dim=int(slots["base_dim"]),
            action_dim=self.input_spec.slot(self.action_slot).output_dim,
            action_adapter=str(slots.get("action_adapter", "identity")),
            model_dim=int(model_dim),
            num_heads=int(num_heads),
            num_encoder_layers=int(history["layers"]),
            num_decoder_layers=int(decoder["layers"]),
            task_names=tasks,
            ffn_kind=str(ffn_kind),
            ffn_expansion=float(ffn_expansion),
            use_qk_norm=bool(use_qk_norm),
            use_position_embedding=str(history.get("position", "learned")) == "learned",
            reverse_history_after_lookup=False,
            task_attention_causal=str(interaction.get("type", "causal_attention")) == "causal_attention",
            rms_norm_eps=float(rms_norm_eps),
            score_temperature_init=float(score_temperature_init),
            point_loss_weight=float(ranking.get("point_weight", 1.0)),
            list_loss_weight=float(ranking.get("list_weight", 1.0)),
            task_loss_weights=_plain_mapping(ranking.get("task_weights", {})),
        )

    def _temporal_config(
        self,
        slots: Mapping[str, Any],
        history: Mapping[str, Any],
        candidate_context: Mapping[str, Any],
        view_encoder: Mapping[str, Any],
        decoder: Mapping[str, Any],
        objectives: Mapping[str, Any],
        model_dim: int,
        num_heads: int,
        ffn_kind: str,
        ffn_expansion: float,
        use_qk_norm: bool,
        rms_norm_eps: float,
        score_temperature_init: float,
        acr: Mapping[str, Any],
    ) -> SimpleNamespace:
        self._validate_groups()
        if str(candidate_context["type"]) != "perceiver":
            raise ValueError("time_rope_causal history requires perceiver context")
        if str(decoder["type"]) != "shared_lora_cross_attention":
            raise ValueError("temporal OneRank requires shared_lora_cross_attention")
        ranking = _plain_mapping(objectives["ranking"])
        next_event = _plain_mapping(objectives["next_event"])
        next_weights = _plain_mapping(next_event["weights"])
        base_dim = int(slots["base_dim"])
        residual_dim = int(slots["residual_dim"])
        if base_dim <= 0 or residual_dim <= 0:
            raise ValueError("temporal slot encoder requires base and residual dimensions")
        return SimpleNamespace(
            history_length=self.input_spec.histories[
                self.history_group_name
            ].max_length,
            history_slot_count=len(self.input_groups["history_slots"]),
            candidate_slot_count=len(self.input_groups["candidate_slots"]),
            query_slot_count=len(self.input_groups["query_slots"]),
            base_slot_dim=base_dim,
            lora_slot_dim=residual_dim,
            raw_slot_dim=base_dim + residual_dim,
            action_dim=self.input_spec.slot(self.action_slot).output_dim,
            model_dim=int(model_dim),
            num_heads=int(num_heads),
            num_encoder_layers=int(history["layers"]),
            num_perceiver_layers=int(candidate_context["layers"]),
            num_view_self_layers=int(view_encoder["layers"]),
            num_decoder_layers=int(decoder["layers"]),
            lora_rank=int(decoder["lora_rank"]),
            task_names=tuple(str(name) for name in ranking["tasks"]),
            ffn_kind=str(ffn_kind),
            ffn_expansion=float(ffn_expansion),
            use_qk_norm=bool(use_qk_norm),
            reverse_history_after_lookup=False,
            detach_encoder_for_perceiver=bool(candidate_context.get("detach_history", True)),
            detach_candidate_base=bool(slots.get("detach_candidate_base", True)),
            detach_query_base=bool(slots.get("detach_query_base", True)),
            rms_norm_eps=float(rms_norm_eps),
            slot_layer_norm_eps=float(slots.get("layer_norm_eps", 1e-3)),
            score_temperature_init=float(score_temperature_init),
            ntp_temperature_init=float(next_event.get("temperature_init", 0.2)),
            ntp_item_slot_index=int(next_event.get("item_slot_index", 0)),
            ntp_category_slot_index=int(next_event.get("category_slot_index", 6)),
            timestamp_invalid_value=float(history.get("timestamp_invalid_value", 4319.0)),
            time_rope_min_wavelength=float(history.get("time_rope_min_wavelength", 1.0)),
            time_rope_max_wavelength=float(history.get("time_rope_max_wavelength", 1008.0)),
            time_head_scales=tuple(float(value) for value in history.get("time_head_scales", (1.0, 1.0))),
            lora_activation_weight=float(ranking.get("lora_activation_weight", 0.01)),
            posttrain_point_loss_weight=float(ranking.get("point_weight", 1.0)),
            posttrain_list_loss_weight=float(ranking.get("list_weight", 1.0)),
            pretrain_loss_weights={
                "item": float(next_weights.get("item", 0.0)),
                "category": float(next_weights.get("category", 0.0)),
            },
            posttrain_task_loss_weights=_plain_mapping(ranking.get("task_weights", {})),
            acr_enabled=bool(acr.get("enabled", False)),
            acr_catalog_size=int(acr.get("catalog_size", 0)),
            acr_rank=int(acr.get("rank", residual_dim)),
            acr_orthogonality_weight=float(acr.get("orthogonality_weight", 0.0)),
        )

    def _build_inputs(
        self,
        embedded: RecAgoraModelBatch,
    ) -> JointCausalInputs | TemporalPerceiverInputs:
        history = embedded.histories[self.history_group_name]
        max_length = self.history_max_events
        history_slots = torch.stack(
            [
                history.categorical[slot].to_padded(
                    max_length=max_length,
                    padding_side="left",
                )
                for slot in self.input_groups["history_slots"]
            ],
            dim=2,
        )
        history_action = history.categorical[self.action_slot].to_padded(
            max_length=max_length,
            padding_side="left",
        )
        history_valid = _suffix_valid_mask(
            history.lengths,
            max_length,
            history_slots.device,
        )
        candidate_slots = torch.stack(
            [embedded.candidate[slot] for slot in self.input_groups["candidate_slots"]],
            dim=2,
        )
        query_slots = torch.stack(
            [embedded.query[slot] for slot in self.input_groups["query_slots"]],
            dim=1,
        )
        if self.backbone_kind == "joint_causal":
            return JointCausalInputs(
                history_slots=history_slots,
                history_action=history_action,
                history_valid=history_valid,
                candidate_slots=candidate_slots,
                query_slots=query_slots,
                candidate_valid_mask=embedded.candidate_valid_mask,
            )
        if max_length == history.max_length:
            timestamps = _right_to_left(
                history.direct[self.timestamp_slot],
                history.lengths,
                max_length,
            )
        else:
            timestamps = _right_to_left_recent(
                history.direct[self.timestamp_slot],
                history.lengths,
                max_length,
            )
        return TemporalPerceiverInputs(
            history_slots=history_slots,
            history_action=history_action,
            history_valid=history_valid,
            timestamps=timestamps,
            candidate_slots=candidate_slots,
            query_slots=query_slots,
            candidate_valid_mask=embedded.candidate_valid_mask,
            acr_candidate_ids=(
                embedded.candidate_mapped_ids[self.acr_item_slot]
                if self.acr_enabled else None
            ),
            acr_history_ids=(
                history.mapped_ids[self.acr_history_item_slot].to_padded(
                    max_length=max_length, padding_side="left"
                ) if self.acr_enabled else None
            ),
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        if (
            self.backbone_kind == "time_rope_causal"
            and self.training
            and self._pending_temporal_state is not None
        ):
            raise RuntimeError(
                "OneRank temporal forward cannot run before compute_loss "
                "consumes the previous forward state"
            )
        inputs = self._build_inputs(self.get_slots(batch))
        if self.backbone_kind == "joint_causal":
            return structured_ranking_output(self.backbone(inputs))
        state = self.backbone.forward_with_state(inputs)
        if self.training:
            # Keep objective-specific tensors off the public DMP output tree.
            # compute_loss consumes this exact state before the next forward.
            self._pending_temporal_state = state
        return structured_ranking_output(dict(state.logits))

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.backbone_kind == "joint_causal":
            logits = output["logits"]
            labels = _labels_for(logits, batch)
            return self.backbone.compute_loss(
                logits,
                labels,
                candidate_valid_mask=batch.candidate_valid_mask,
                candidate_loss_context=batch.candidate_loss_context,
            )
        pending_state = self._consume_pending_temporal_state()
        logits = output["logits"]
        state = TemporalPerceiverState(
            logits=logits,
            history_states=pending_state.history_states,
            history_slot_tokens=pending_state.history_slot_tokens,
            history_valid=pending_state.history_valid,
            candidate_lora_activation_loss=(
                pending_state.candidate_lora_activation_loss
            ),
            candidate_lora_activation_by_candidate=(
                pending_state.candidate_lora_activation_by_candidate
            ),
            candidate_view_tokens=pending_state.candidate_view_tokens,
            task_anchors=pending_state.task_anchors,
            acr_orthogonality_loss=pending_state.acr_orthogonality_loss,
        )
        if self.objective_mode == "next_event":
            loss, metrics = self.backbone.compute_pretrain_loss(state)
            ranking_zero = sum(value.sum() for value in logits.values()) * 0.0
            return loss + ranking_zero, metrics
        labels = _labels_for(logits, batch)
        return self.backbone.compute_posttrain_loss(
            state,
            labels,
            candidate_valid_mask=batch.candidate_valid_mask,
            candidate_loss_context=batch.candidate_loss_context,
        )

    def _consume_pending_temporal_state(self) -> TemporalPerceiverState:
        state = self._pending_temporal_state
        if state is None:
            raise RuntimeError(
                "OneRank temporal compute_loss requires one unconsumed "
                "training forward state"
            )
        self._pending_temporal_state = None
        return state


def _plain_mapping(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("OneRank module configuration must be a mapping")
    return dict(value)


def _labels_for(
    output: Mapping[str, torch.Tensor],
    batch: RecAgoraSparseBatch,
) -> dict[str, torch.Tensor]:
    return {name: batch.labels[name] for name in output}


def _suffix_valid_mask(
    lengths: torch.Tensor,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(max_length, device=device)[None, :]
    return positions >= (
        max_length - lengths.to(device=device, dtype=torch.long).clamp_max(max_length)
    ).unsqueeze(1)


def _right_to_left(
    values: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    lengths = lengths.to(device=values.device, dtype=torch.long).clamp_max(max_length)
    positions = torch.arange(max_length, device=values.device)[None, :]
    valid = positions >= (max_length - lengths).unsqueeze(1)
    source = (positions - (max_length - lengths).unsqueeze(1)).clamp_min(0)
    return values[:, :max_length].gather(1, source).masked_fill(~valid, 0)


def _right_to_left_recent(
    values: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Align the most recent direct values with left-padded jagged history."""
    lengths = lengths.to(device=values.device, dtype=torch.long)
    retained_lengths = lengths.clamp_max(max_length)
    positions = torch.arange(max_length, device=values.device)[None, :]
    valid = positions >= (max_length - retained_lengths).unsqueeze(1)
    source = (
        positions
        - (max_length - retained_lengths).unsqueeze(1)
        + (lengths - retained_lengths).unsqueeze(1)
    ).clamp_min(0)
    return values.gather(1, source).masked_fill(~valid, 0)
