from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from recagora import CandidateLossContext

from .blocks import (
    LoRATransformerCrossAttentionBlock,
    PerceiverBlock,
    TimeOrderRoPE,
    TransformerEncoderBlock,
    causal_attention_mask,
)
from .losses import (
    NTPResult,
    first_future_match_targets,
    in_batch_sampled_softmax,
    next_item_targets,
    ranking_task_losses,
)


def _init_linear(linear: nn.Linear, *, std: float = 0.02) -> None:
    nn.init.trunc_normal_(
        linear.weight,
        mean=0.0,
        std=std,
        a=-2.0 * std,
        b=2.0 * std,
    )
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


@dataclass(frozen=True)
class TemporalPerceiverInputs:
    """Embedding tensors returned by the public ``get_slots`` call.

    Each sparse-id slot is a concatenation of ``base_slot_dim`` and
    ``lora_slot_dim``.  History uses only the base part; candidate/query use
    ``stop_gradient(base) + lora_emb_proj(tail)``.
    """

    history_slots: torch.Tensor  # [B, S, 27, 40]
    history_action: torch.Tensor  # [B, S, 256]
    history_valid: torch.Tensor  # [B, S], bool
    timestamps: torch.Tensor  # [B, S]
    candidate_slots: torch.Tensor  # [B, L, 110, 40]
    query_slots: torch.Tensor  # [B, 24, 40]
    candidate_valid_mask: torch.Tensor | None = None  # [B, L], bool
    acr_candidate_ids: torch.Tensor | None = None  # [B, L], mapped item IDs
    acr_history_ids: torch.Tensor | None = None  # [B, S], mapped item IDs


@dataclass(frozen=True)
class TemporalPerceiverState:
    """Intermediate state retained for explicit pretraining/ranking loss computation."""

    logits: Mapping[str, torch.Tensor]
    history_states: torch.Tensor  # [B, S, H], excludes CLS
    history_slot_tokens: torch.Tensor  # [B, S, 27, 32], after per-slot LayerNorm
    history_valid: torch.Tensor  # [B, S]
    candidate_lora_activation_loss: torch.Tensor
    candidate_lora_activation_by_candidate: torch.Tensor  # [B, L]
    candidate_view_tokens: torch.Tensor  # [B, task, L, H]
    task_anchors: torch.Tensor  # [B, task, H]
    acr_orthogonality_loss: torch.Tensor | None = None


class TaskACR(nn.Module):
    """Task-owned catalog codes, low-rank projections, and anchored scaling."""

    def __init__(self, *, task_names: tuple[str, ...], catalog_size: int, rank: int, embedding_dim: int) -> None:
        super().__init__()
        if catalog_size <= 2 or not 0 < rank < embedding_dim:
            raise ValueError("ACR needs catalog_size > 2 and 0 < rank < embedding_dim")
        try:
            from torchrec.modules.embedding_configs import EmbeddingConfig
            from torchrec.modules.embedding_modules import EmbeddingCollection
            from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
        except ImportError as exc:
            raise ImportError("ACR requires TorchRec for task-owned catalog tables") from exc
        self.task_names = task_names
        self.rank = rank
        self.embedding_dim = embedding_dim
        self.feature_names = tuple(f"acr_{name}" for name in task_names)
        self.table_names = tuple(f"acr_catalog_{name}" for name in task_names)
        self.acr_table_specs = {
            name: (catalog_size, rank) for name in self.table_names
        }
        self.keyed_jagged_tensor_cls = KeyedJaggedTensor
        self.embedding_collection = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name=self.table_names[index],
                    embedding_dim=rank,
                    num_embeddings=catalog_size,
                    feature_names=[f"acr_{name}"],
                )
                for index, name in enumerate(task_names)
            ]
        )
        self.projection = nn.Parameter(torch.empty(len(task_names), rank, embedding_dim))
        nn.init.normal_(self.projection, std=0.02)
        self.scale_unconstrained = nn.Parameter(torch.full((len(task_names),), 0.01))

    def lookup(self, history_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        batch_size, history_length = history_ids.shape
        list_length = candidate_ids.shape[1]
        values_per_task = torch.cat((history_ids, candidate_ids), dim=1).reshape(-1)
        if values_per_task.dtype != torch.long:
            raise TypeError("ACR mapped IDs must be torch.long")
        segment_count = batch_size * (history_length + list_length)
        kjt = self.keyed_jagged_tensor_cls.from_lengths_sync(
            keys=list(self.feature_names),
            values=values_per_task.repeat(len(self.task_names)),
            lengths=torch.ones(
                segment_count * len(self.task_names),
                dtype=torch.long,
                device=values_per_task.device,
            ),
        )
        result = self.embedding_collection(kjt)
        if hasattr(result, "wait"):
            result = result.wait()
        if isinstance(result, tuple):
            result = result[0]
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        codes = [result[name].values().reshape(batch_size, history_length + list_length, self.rank)
                 for name in self.feature_names]
        return torch.stack(codes, dim=1)  # [B, task, S+L, rank]

    def adjust(self, base: torch.Tensor, code: torch.Tensor, task_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        delta = code @ self.projection[task_index]
        scale = 1.0 + torch.relu(self.scale_unconstrained[task_index])
        return scale * base.detach() + delta, delta


class TemporalPerceiverRanker(nn.Module):
    """Temporal/perceiver ranker after public sparse feature lookup.

    The object exposes two explicit loss methods instead of hiding a phase
    branch in ``forward``: ``compute_pretrain_loss`` for pretraining NTP and
    ``compute_posttrain_loss`` for ranking ranking plus activation regularization.
    This leaves the training-round choice to configuration in the framework.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        task_count = len(config.task_names)
        self.embedding_lora_projection = nn.Linear(
            config.lora_slot_dim,
            config.base_slot_dim,
            bias=False,
        )
        _init_linear(self.embedding_lora_projection)
        self.history_slot_norms = nn.ModuleList(
            nn.LayerNorm(config.base_slot_dim, eps=config.slot_layer_norm_eps)
            for _ in range(config.history_slot_count)
        )
        self.candidate_slot_norms = nn.ModuleList(
            nn.LayerNorm(config.base_slot_dim, eps=config.slot_layer_norm_eps)
            for _ in range(config.candidate_slot_count)
        )
        self.query_slot_norms = nn.ModuleList(
            nn.LayerNorm(config.base_slot_dim, eps=config.slot_layer_norm_eps)
            for _ in range(config.query_slot_count)
        )
        self.history_projection = nn.Linear(
            config.history_slot_count * config.base_slot_dim,
            config.model_dim,
            bias=False,
        )
        _init_linear(self.history_projection, std=0.08)
        self.candidate_projection = nn.Linear(
            config.candidate_slot_count * config.base_slot_dim,
            config.model_dim,
            bias=False,
        )
        _init_linear(self.candidate_projection)
        self.query_projection = nn.Linear(
            config.query_slot_count * config.base_slot_dim,
            task_count * config.model_dim,
            bias=True,
        )
        _init_linear(self.query_projection)
        self.cls_token = nn.Parameter(torch.empty(1, config.model_dim))
        self.view_tokens = nn.Parameter(torch.empty(task_count, config.model_dim))
        self.anchor_token = nn.Parameter(torch.empty(1, config.model_dim))
        for parameter in (self.cls_token, self.view_tokens, self.anchor_token):
            nn.init.trunc_normal_(parameter, mean=0.0, std=0.02, a=-0.04, b=0.04)

        time_head_count = len(config.time_head_scales)
        self.rope = TimeOrderRoPE(
            head_dim=config.model_dim // config.num_heads,
            num_heads=config.num_heads,
            num_time_heads=time_head_count,
            time_method="wavelength",
            time_min_wavelength=config.time_rope_min_wavelength,
            time_max_wavelength=config.time_rope_max_wavelength,
            order_method="base",
            time_head_scales=config.time_head_scales,
        )
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderBlock(
                config.model_dim,
                config.num_heads,
                qkv_layout="separate",
                use_qk_norm=config.use_qk_norm,
                ffn_kind=config.ffn_kind,
                ffn_expansion=config.ffn_expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_encoder_layers)
        )
        self.perceiver_blocks = nn.ModuleList(
            PerceiverBlock(
                config.model_dim,
                config.num_heads,
                use_qk_norm=config.use_qk_norm,
                ffn_kind=config.ffn_kind,
                ffn_expansion=config.ffn_expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_perceiver_layers)
        )
        self.view_self_blocks = nn.ModuleList(
            TransformerEncoderBlock(
                config.model_dim,
                config.num_heads,
                qkv_layout="separate",
                use_qk_norm=config.use_qk_norm,
                ffn_kind=config.ffn_kind,
                ffn_expansion=config.ffn_expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_view_self_layers)
        )
        self.decoder_blocks = nn.ModuleList(
            LoRATransformerCrossAttentionBlock(
                config.model_dim,
                config.num_heads,
                lora_rank=config.lora_rank,
                use_qk_norm=config.use_qk_norm,
                ffn_kind=config.ffn_kind,
                ffn_expansion=config.ffn_expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_decoder_layers)
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.score_temperature_init), dtype=torch.float32)
        )
        self.ntp_item_projection = nn.Linear(
            config.model_dim,
            config.base_slot_dim,
            bias=False,
        )
        self.ntp_category_projection = nn.Linear(
            config.model_dim,
            config.base_slot_dim,
            bias=False,
        )
        _init_linear(self.ntp_item_projection)
        _init_linear(self.ntp_category_projection)
        self.log_ntp_temperature = nn.Parameter(
            torch.tensor(math.log(config.ntp_temperature_init), dtype=torch.float32)
        )
        self.acr: TaskACR | None = None
        if config.acr_enabled:
            self.acr = TaskACR(
                task_names=config.task_names,
                catalog_size=config.acr_catalog_size,
                rank=config.acr_rank,
                embedding_dim=config.base_slot_dim,
            )
            for module in (self.history_slot_norms, self.history_projection, self.encoder_blocks):
                module.requires_grad_(False)
            self.cls_token.requires_grad_(False)

    def forward(self, inputs: TemporalPerceiverInputs) -> dict[str, torch.Tensor]:
        return dict(self.forward_with_state(inputs).logits)

    def forward_with_state(self, inputs: TemporalPerceiverInputs) -> TemporalPerceiverState:
        if self.acr is not None:
            return self._forward_with_acr(inputs)
        return self._forward_without_acr(inputs)

    def _forward_without_acr(self, inputs: TemporalPerceiverInputs, *, preserve_item_gradient: bool = False) -> TemporalPerceiverState:
        batch_size = inputs.history_slots.shape[0]
        list_length = inputs.candidate_slots.shape[1]
        history_slots = inputs.history_slots
        history_action = inputs.history_action
        history_valid = inputs.history_valid
        timestamps = inputs.timestamps
        candidate_valid_mask = inputs.candidate_valid_mask
        _validate_candidate_mask(candidate_valid_mask, inputs.candidate_slots)
        if self.config.reverse_history_after_lookup:
            history_slots = history_slots.flip(dims=(1,))
            history_action = history_action.flip(dims=(1,))
            history_valid = history_valid.flip(dims=(1,))
            timestamps = timestamps.flip(dims=(1,))

        history_slot_tokens = self._history_slot_tokens(history_slots)
        (
            candidate_slot_tokens,
            candidate_lora_activation_loss,
            candidate_lora_activation_by_candidate,
        ) = self._candidate_slot_tokens(inputs.candidate_slots, preserve_item_gradient=preserve_item_gradient)
        query_slot_tokens = self._query_slot_tokens(inputs.query_slots)
        history_tokens = self.history_projection(history_slot_tokens.flatten(start_dim=2))
        candidate_tokens = self.candidate_projection(candidate_slot_tokens.flatten(start_dim=2))
        query_tokens = self.query_projection(query_slot_tokens.flatten(start_dim=1)).reshape(
            batch_size,
            len(self.config.task_names),
            self.config.model_dim,
        )

        cls = self.cls_token[None].expand(batch_size, -1, -1)
        encoder_tokens = torch.cat((cls, history_tokens), dim=1)
        encoder_tokens = encoder_tokens + torch.cat(
            (
                torch.zeros_like(cls),
                history_action,
            ),
            dim=1,
        )
        encoder_valid = torch.cat(
            (
                torch.ones((batch_size, 1), device=history_valid.device, dtype=torch.bool),
                history_valid,
            ),
            dim=1,
        )
        rope_timestamps = self._prepare_timestamps(timestamps, history_valid)
        encoder_causal_mask = causal_attention_mask(
            encoder_tokens.shape[1],
            device=encoder_tokens.device,
            dtype=encoder_tokens.dtype,
        )
        for block in self.encoder_blocks:
            encoder_tokens = block(
                encoder_tokens,
                key_padding_mask=encoder_valid,
                attn_mask=encoder_causal_mask,
                rope=self.rope,
                timestamps=rope_timestamps,
            )
        history_states = encoder_tokens[:, 1:]

        perceiver_key_value = (
            history_states.detach()
            if self.config.detach_encoder_for_perceiver and self.acr is None
            else history_states
        )
        perceiver_tokens = candidate_tokens
        for block in self.perceiver_blocks:
            perceiver_tokens = block(
                perceiver_tokens,
                perceiver_key_value,
                key_padding_mask=history_valid,
                self_key_padding_mask=candidate_valid_mask,
            )

        task_count = len(self.config.task_names)
        view_input = torch.cat(
            (
                perceiver_tokens.reshape(batch_size * list_length, 1, self.config.model_dim),
                self.view_tokens[None].expand(batch_size * list_length, -1, -1),
            ),
            dim=1,
        )
        for block in self.view_self_blocks:
            view_input = block(view_input)
        views = view_input[:, 1:].reshape(
            batch_size,
            list_length,
            task_count,
            self.config.model_dim,
        ).permute(0, 2, 1, 3).contiguous()

        decoder_query = (query_tokens + self.anchor_token).reshape(
            batch_size * task_count,
            1,
            self.config.model_dim,
        )
        decoder_key_value = views.reshape(
            batch_size * task_count,
            list_length,
            self.config.model_dim,
        )
        decoder_key_padding_mask = (
            None
            if candidate_valid_mask is None
            else candidate_valid_mask[:, None, :]
            .expand(-1, task_count, -1)
            .reshape(batch_size * task_count, list_length)
        )
        for layer_index, block in enumerate(self.decoder_blocks):
            source_index = min(layer_index, len(self.encoder_blocks) - 1)
            decoder_query = block(
                decoder_query,
                decoder_key_value,
                decoder_key_value,
                base_projections=self.encoder_blocks[source_index].attention.projections(),
                key_padding_mask=decoder_key_padding_mask,
            )
        task_anchors = decoder_query.reshape(batch_size, task_count, self.config.model_dim)
        logits_tensor = torch.einsum(
            "btd,btld->blt",
            F.normalize(task_anchors, dim=-1),
            F.normalize(views, dim=-1),
        ) / self.log_temperature.exp()
        logits = {
            name: logits_tensor[:, :, index]
            for index, name in enumerate(self.config.task_names)
        }
        return TemporalPerceiverState(
            logits=logits,
            history_states=history_states,
            history_slot_tokens=history_slot_tokens,
            history_valid=history_valid,
            candidate_lora_activation_loss=candidate_lora_activation_loss,
            candidate_lora_activation_by_candidate=(
                candidate_lora_activation_by_candidate
            ),
            candidate_view_tokens=views,
            task_anchors=task_anchors,
        )

    def _forward_with_acr(self, inputs: TemporalPerceiverInputs) -> TemporalPerceiverState:
        acr = self.acr
        if acr is None or inputs.acr_history_ids is None or inputs.acr_candidate_ids is None:
            raise RuntimeError("ACR needs mapped history and candidate item IDs")
        batch_size, history_length = inputs.acr_history_ids.shape
        codes = acr.lookup(inputs.acr_history_ids, inputs.acr_candidate_ids)
        base_dim = self.config.base_slot_dim
        history_raw = inputs.history_slots.detach()
        candidate_raw = inputs.candidate_slots
        history_base = history_raw[:, :, 0, :base_dim]
        candidate_base = candidate_raw[:, :, 0, :base_dim]
        valid_candidates = (
            inputs.candidate_valid_mask
            if inputs.candidate_valid_mask is not None
            else torch.ones_like(inputs.acr_candidate_ids, dtype=torch.bool)
        )
        states: list[TemporalPerceiverState] = []
        orth_terms: list[torch.Tensor] = []
        logits: dict[str, torch.Tensor] = {}
        for task_index, name in enumerate(self.config.task_names):
            history_code = codes[:, task_index, :history_length]
            candidate_code = codes[:, task_index, history_length:]
            adjusted_history, history_delta = acr.adjust(history_base, history_code, task_index)
            adjusted_candidate, candidate_delta = acr.adjust(candidate_base, candidate_code, task_index)
            history_count = inputs.history_valid.sum()
            candidate_count = valid_candidates.sum()
            orth_terms.append(
                (
                    _acr_cosine_square(history_delta, history_base, inputs.history_valid) * history_count
                    + _acr_cosine_square(candidate_delta, candidate_base, valid_candidates) * candidate_count
                ) / (history_count + candidate_count).clamp_min(1)
            )

            history_item = torch.cat((adjusted_history, history_raw[:, :, 0, base_dim:]), dim=-1)
            history_slots = torch.cat((history_item.unsqueeze(2), history_raw[:, :, 1:]), dim=2)
            candidate_item = torch.cat(
                (adjusted_candidate, torch.zeros_like(candidate_raw[:, :, 0, base_dim:])),
                dim=-1,
            )
            candidate_slots = torch.cat((candidate_item.unsqueeze(2), candidate_raw[:, :, 1:]), dim=2)
            task_inputs = TemporalPerceiverInputs(
                history_slots=history_slots,
                history_action=inputs.history_action.detach(),
                history_valid=inputs.history_valid,
                timestamps=inputs.timestamps,
                candidate_slots=candidate_slots,
                query_slots=inputs.query_slots,
                candidate_valid_mask=inputs.candidate_valid_mask,
            )
            state = self._forward_without_acr(task_inputs, preserve_item_gradient=True)
            states.append(state)
            logits[name] = state.logits[name]
        first = states[0]
        return TemporalPerceiverState(
            logits=logits,
            history_states=first.history_states,
            history_slot_tokens=first.history_slot_tokens,
            history_valid=first.history_valid,
            candidate_lora_activation_loss=first.candidate_lora_activation_loss,
            candidate_lora_activation_by_candidate=first.candidate_lora_activation_by_candidate,
            candidate_view_tokens=torch.cat(
                [state.candidate_view_tokens[:, index:index + 1] for index, state in enumerate(states)],
                dim=1,
            ),
            task_anchors=torch.cat(
                [state.task_anchors[:, index:index + 1] for index, state in enumerate(states)],
                dim=1,
            ),
            acr_orthogonality_loss=torch.stack(orth_terms).mean(),
        )

    def compute_posttrain_loss(
        self,
        state: TemporalPerceiverState,
        labels: Mapping[str, torch.Tensor],
        *,
        candidate_valid_mask: torch.Tensor | None = None,
        candidate_loss_context: CandidateLossContext | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Source ranking loss: ranking point/list losses plus LoRA activation MSE."""
        ranking_loss, components = ranking_task_losses(
            state.logits,
            labels,
            point_loss_weight=self.config.posttrain_point_loss_weight,
            list_loss_weight=self.config.posttrain_list_loss_weight,
            task_loss_weights=self.config.posttrain_task_loss_weights,
            candidate_valid_mask=candidate_valid_mask,
            candidate_loss_context=candidate_loss_context,
        )
        lora_activation_loss = _candidate_activation_loss(
            state,
            candidate_valid_mask,
            candidate_loss_context,
        )
        lora_weighted = (
            float(self.config.lora_activation_weight)
            * lora_activation_loss
        )
        components = dict(components)
        components["lora_activation"] = lora_activation_loss
        components["lora_activation_weighted"] = lora_weighted
        if self.acr is None:
            return ranking_loss + lora_weighted, components
        if state.acr_orthogonality_loss is None:
            raise RuntimeError("ACR ranking state is missing orthogonality loss")
        if candidate_loss_context is not None:
            normalization = candidate_loss_context.world_size / max(candidate_loss_context.global_valid_count, 1)
        else:
            valid_count = (
                int(candidate_valid_mask.sum().item())
                if candidate_valid_mask is not None
                else next(iter(state.logits.values())).numel()
            )
            normalization = 1.0 / max(valid_count, 1)
        ranking_normalized = ranking_loss * normalization
        orth_weighted = (
            float(self.config.acr_orthogonality_weight)
            * state.acr_orthogonality_loss
        )
        components["acr/ranking_normalized"] = ranking_normalized
        components["acr/orthogonality"] = state.acr_orthogonality_loss
        components["acr/orthogonality_weighted"] = orth_weighted
        return ranking_normalized + lora_weighted * normalization + orth_weighted, components

    def compute_pretrain_loss(
        self,
        state: TemporalPerceiverState,
        *,
        recall_ks: tuple[int, ...] = (1, 5, 10),
        dedup_threshold: float = 0.995,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Item/category next-event losses over in-batch negatives."""
        temperature = self.log_ntp_temperature.exp()
        item_hidden = F.normalize(self.ntp_item_projection(state.history_states), dim=-1)
        category_hidden = F.normalize(self.ntp_category_projection(state.history_states), dim=-1)
        item_tokens = F.normalize(
            state.history_slot_tokens[:, :, self.config.ntp_item_slot_index],
            dim=-1,
        )
        category_tokens = F.normalize(
            state.history_slot_tokens[:, :, self.config.ntp_category_slot_index],
            dim=-1,
        )
        next_targets, next_valid = next_item_targets(item_tokens, state.history_valid)
        item_result = in_batch_sampled_softmax(
            item_hidden,
            next_targets,
            current_valid=state.history_valid,
            target_valid=next_valid,
            temperature=temperature,
            recall_ks=recall_ks,
            dedup_threshold=dedup_threshold,
        )
        category_match = (
            category_tokens[:, :, None, :] == category_tokens[:, None, :, :]
        ).all(dim=-1)
        category_targets, category_valid = first_future_match_targets(
            category_match,
            item_tokens,
            state.history_valid,
        )
        category_result = in_batch_sampled_softmax(
            category_hidden,
            category_targets,
            current_valid=state.history_valid,
            target_valid=category_valid,
            temperature=temperature,
            recall_ks=recall_ks,
            dedup_threshold=dedup_threshold,
        )
        total = (
            float(self.config.pretrain_loss_weights["item"]) * item_result.loss
            + float(self.config.pretrain_loss_weights["category"]) * category_result.loss
        )
        components = {
            "ntp/item": item_result.loss,
            "ntp/category": category_result.loss,
            "ntp/temperature": temperature,
        }
        components.update(_ntp_metric_components("ntp/item", item_result))
        components.update(_ntp_metric_components("ntp/category", category_result))
        return total, components

    def _history_slot_tokens(self, history_slots: torch.Tensor) -> torch.Tensor:
        base, _ = history_slots.split(
            (self.config.base_slot_dim, self.config.lora_slot_dim),
            dim=-1,
        )
        return torch.stack(
            [normalizer(base[:, :, index]) for index, normalizer in enumerate(self.history_slot_norms)],
            dim=2,
        )

    def _candidate_slot_tokens(
        self,
        candidate_slots: torch.Tensor,
        *,
        preserve_item_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, lora = candidate_slots.split(
            (self.config.base_slot_dim, self.config.lora_slot_dim),
            dim=-1,
        )
        lora_mapped = self.embedding_lora_projection(lora)
        base_for_task = base.detach() if self.config.detach_candidate_base else base
        if preserve_item_gradient:
            base_for_task = torch.cat((base[:, :, :1], base_for_task[:, :, 1:]), dim=2)
        adjusted = base_for_task + lora_mapped
        tokens = torch.stack(
            [normalizer(adjusted[:, :, index]) for index, normalizer in enumerate(self.candidate_slot_norms)],
            dim=2,
        )
        activation_by_candidate = lora_mapped.square().mean(dim=(2, 3))
        return (
            tokens,
            lora_mapped.square().mean(dim=(0, 1, 3)).mean(),
            activation_by_candidate,
        )

    def _query_slot_tokens(self, query_slots: torch.Tensor) -> torch.Tensor:
        base, lora = query_slots.split(
            (self.config.base_slot_dim, self.config.lora_slot_dim),
            dim=-1,
        )
        lora_mapped = self.embedding_lora_projection(lora)
        adjusted = (base.detach() if self.config.detach_query_base else base) + lora_mapped
        return torch.stack(
            [normalizer(adjusted[:, index]) for index, normalizer in enumerate(self.query_slot_norms)],
            dim=1,
        )

    def _prepare_timestamps(
        self,
        timestamps: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        invalid = torch.full_like(timestamps, float(self.config.timestamp_invalid_value))
        values = torch.where(history_valid, timestamps, invalid)
        values = values.clamp(min=0.0, max=float(self.config.timestamp_invalid_value)).floor()
        return torch.cat((torch.zeros_like(values[:, :1]), values), dim=1)

def _ntp_metric_components(prefix: str, result: NTPResult) -> dict[str, torch.Tensor]:
    components = {
        f"{prefix}/positive_score_mean": result.positive_score_mean,
        f"{prefix}/negative_score_mean": result.negative_score_mean,
        f"{prefix}/positive_negative_difference_mean": result.positive_negative_difference_mean,
    }
    components.update({f"{prefix}/recall@{key}": value for key, value in result.recalls.items()})
    return components


def _validate_candidate_mask(
    valid_mask: torch.Tensor | None,
    candidate_slots: torch.Tensor,
) -> None:
    if valid_mask is None:
        return
    if valid_mask.dtype != torch.bool:
        raise TypeError("candidate valid mask must be bool")
    if valid_mask.shape != candidate_slots.shape[:2]:
        raise ValueError("candidate valid mask must match candidate [B,L] shape")
    if valid_mask.device != candidate_slots.device:
        raise ValueError("candidate valid mask and candidate slots must share a device")


def _candidate_activation_loss(
    state: TemporalPerceiverState,
    valid_mask: torch.Tensor | None,
    context: CandidateLossContext | None,
) -> torch.Tensor:
    if valid_mask is None:
        return state.candidate_lora_activation_loss
    values = state.candidate_lora_activation_by_candidate[valid_mask]
    summed = values.sum()
    if context is None:
        return summed / max(values.numel(), 1)
    return summed * (context.world_size / max(context.global_valid_count, 1))


def _acr_cosine_square(delta: torch.Tensor, base: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(delta.float(), base.detach().float(), dim=-1, eps=1e-8)
    values = cosine.square()[valid]
    return values.sum() / max(values.numel(), 1)
