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
    TransformerCrossAttentionBlock,
    TransformerEncoderBlock,
    causal_attention_mask,
)
from .losses import ranking_task_losses


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
class JointCausalInputs:
    """Embedding tensors returned by the joint-causal ``get_slots`` calls.

    ``history_slots``, ``history_action`` and ``history_valid`` use the source
    lookup order.  With the default config the backbone reverses all three
    before causal encoding, mirroring ``tf.reverse(..., axis=[1])``.
    """

    history_slots: torch.Tensor  # [B, S, 27, 32]
    history_action: torch.Tensor  # [B, S, 256]
    history_valid: torch.Tensor  # [B, S], bool
    candidate_slots: torch.Tensor  # [B, L, 110, 32]
    query_slots: torch.Tensor  # [B, 24, 32]
    candidate_valid_mask: torch.Tensor | None = None  # [B, L], bool


@dataclass(frozen=True)
class JointCausalState:
    """Intermediate tensors useful for checkpoint and shape-level auditing."""

    logits: Mapping[str, torch.Tensor]
    history_valid: torch.Tensor
    candidate_view_tokens: torch.Tensor  # [B, task, L, H]
    task_anchors: torch.Tensor  # [B, task, H]


class JointCausalRanker(nn.Module):
    """Joint causal history/candidate ranker after public slot lookup."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        task_count = len(config.task_names)
        self.history_projection = nn.Linear(
            config.history_slot_count * config.slot_dim,
            config.model_dim,
            bias=True,
        )
        self.candidate_projection = nn.Linear(
            config.candidate_slot_count * config.slot_dim,
            config.model_dim,
            bias=True,
        )
        self.anchor_projection = nn.Linear(
            config.query_slot_count * config.slot_dim,
            task_count * config.model_dim,
            bias=True,
        )
        for projection in (
            self.history_projection,
            self.candidate_projection,
            self.anchor_projection,
        ):
            _init_linear(projection)
        self.action_projection: nn.Module
        if config.action_adapter == "linear":
            self.action_projection = nn.Linear(
                config.action_dim,
                config.model_dim,
                bias=True,
            )
            _init_linear(self.action_projection)
        else:
            self.action_projection = nn.Identity()

        self.position_embedding = (
            nn.Embedding(config.history_length + 1, config.model_dim)
            if config.use_position_embedding
            else None
        )
        if self.position_embedding is not None:
            nn.init.trunc_normal_(
                self.position_embedding.weight,
                mean=0.0,
                std=0.02,
                a=-0.04,
                b=0.04,
            )
        self.view_tokens = nn.Parameter(torch.empty(task_count, config.model_dim))
        nn.init.trunc_normal_(self.view_tokens, mean=0.0, std=0.02, a=-0.04, b=0.04)

        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderBlock(
                config.model_dim,
                config.num_heads,
                qkv_layout="fused",
                use_qk_norm=config.use_qk_norm,
                ffn_kind=config.ffn_kind,
                ffn_expansion=config.ffn_expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_encoder_layers)
        )
        self.decoder_blocks = nn.ModuleList(
            TransformerCrossAttentionBlock(
                config.model_dim,
                config.num_heads,
                use_qk_norm=config.use_qk_norm,
                ffn_kind=config.ffn_kind,
                ffn_expansion=config.ffn_expansion,
                rms_norm_eps=config.rms_norm_eps,
            )
            for _ in range(config.num_decoder_layers)
        )
        self.task_attention = TransformerEncoderBlock(
            config.model_dim,
            config.num_heads,
            qkv_layout="fused",
            use_qk_norm=config.use_qk_norm,
            ffn_kind=config.ffn_kind,
            ffn_expansion=config.ffn_expansion,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.score_temperature_init), dtype=torch.float32)
        )

    def forward(self, inputs: JointCausalInputs) -> dict[str, torch.Tensor]:
        return dict(self.forward_with_state(inputs).logits)

    def forward_with_state(self, inputs: JointCausalInputs) -> JointCausalState:
        batch_size = inputs.history_slots.shape[0]
        list_length = inputs.candidate_slots.shape[1]
        history_slots = inputs.history_slots
        history_action = inputs.history_action
        history_valid = inputs.history_valid
        if self.config.reverse_history_after_lookup:
            history_slots = history_slots.flip(dims=(1,))
            history_action = history_action.flip(dims=(1,))
            history_valid = history_valid.flip(dims=(1,))

        history = self.history_projection(history_slots.flatten(start_dim=2))
        if self.position_embedding is not None:
            history = history + self._history_position_embeddings(history_valid)
        history = history + self.action_projection(history_action)
        candidates = self.candidate_projection(inputs.candidate_slots.flatten(start_dim=2))

        task_count = len(self.config.task_names)
        expanded_history = history[:, None].expand(-1, list_length, -1, -1)
        expanded_candidates = candidates[:, :, None]
        expanded_views = self.view_tokens[None, None].expand(
            batch_size,
            list_length,
            -1,
            -1,
        )
        encoded = torch.cat(
            (expanded_history, expanded_candidates, expanded_views),
            dim=2,
        ).reshape(
            batch_size * list_length,
            self.config.history_length + 1 + task_count,
            self.config.model_dim,
        )
        encoder_valid = torch.cat(
            (
                history_valid[:, None].expand(-1, list_length, -1),
                torch.ones(
                    (batch_size, list_length, 1 + task_count),
                    device=encoded.device,
                    dtype=torch.bool,
                ),
            ),
            dim=2,
        ).reshape(batch_size * list_length, -1)
        causal_mask = causal_attention_mask(
            encoded.shape[1], device=encoded.device, dtype=encoded.dtype
        )
        for block in self.encoder_blocks:
            encoded = block(
                encoded,
                key_padding_mask=encoder_valid,
                attn_mask=causal_mask,
            )

        views = encoded[:, -task_count:].reshape(
            batch_size,
            list_length,
            task_count,
            self.config.model_dim,
        ).permute(0, 2, 1, 3).contiguous()
        query_anchors = self.anchor_projection(inputs.query_slots.flatten(start_dim=1)).reshape(
            batch_size,
            task_count,
            self.config.model_dim,
        )
        decoder_query = query_anchors.reshape(
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
            if inputs.candidate_valid_mask is None
            else inputs.candidate_valid_mask[:, None, :]
            .expand(-1, task_count, -1)
            .reshape(batch_size * task_count, list_length)
        )
        for block in self.decoder_blocks:
            decoder_query = block(
                decoder_query,
                decoder_key_value,
                key_padding_mask=decoder_key_padding_mask,
            )
        task_anchors = decoder_query.reshape(batch_size, task_count, self.config.model_dim)
        task_mask = (
            causal_attention_mask(
                task_count,
                device=task_anchors.device,
                dtype=task_anchors.dtype,
            )
            if self.config.task_attention_causal
            else None
        )
        task_anchors = self.task_attention(task_anchors, attn_mask=task_mask)
        temperature = self.log_temperature.exp()
        logits_tensor = torch.einsum(
            "btd,btld->blt",
            F.normalize(task_anchors, dim=-1),
            F.normalize(views, dim=-1),
        ) / temperature
        logits = {
            name: logits_tensor[:, :, index]
            for index, name in enumerate(self.config.task_names)
        }
        return JointCausalState(
            logits=logits,
            history_valid=history_valid,
            candidate_view_tokens=views,
            task_anchors=task_anchors,
        )

    def compute_loss(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: Mapping[str, torch.Tensor],
        *,
        candidate_valid_mask: torch.Tensor | None = None,
        candidate_loss_context: CandidateLossContext | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Combine listwise cross entropy and pointwise BCE for every task."""
        return ranking_task_losses(
            logits,
            labels,
            point_loss_weight=self.config.point_loss_weight,
            list_loss_weight=self.config.list_loss_weight,
            task_loss_weights=self.config.task_loss_weights,
            candidate_valid_mask=candidate_valid_mask,
            candidate_loss_context=candidate_loss_context,
        )

    def _history_position_embeddings(self, history_valid: torch.Tensor) -> torch.Tensor:
        positions = history_valid.to(torch.long).cumsum(dim=1) - 1
        positions = positions.clamp(min=0, max=self.config.history_length)
        return self.position_embedding(positions) * history_valid[..., None].to(
            dtype=self.position_embedding.weight.dtype
        )
