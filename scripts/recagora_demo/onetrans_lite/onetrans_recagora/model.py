"""Minimal public-API wrapper around the plain-PyTorch OneTrans-Lite core."""

from __future__ import annotations

from typing import Any

import torch


#   RecAgoraInputSpec: compile slot info in one
#   RecAgoraRankingModel: base model class from nn.Module
#   RecAgoraSparseBatch: Dataloader
#   RankingOutput:
from recagora import RecAgoraInputSpec, RecAgoraRankingModel, RecAgoraSparseBatch
from recagora import RankingOutput, structured_ranking_output
from recagora import masked_binary_cross_entropy_with_logits
from scripts.recagora_demo.onetrans_lite.onetrans_torch.model import (
    OneTransLite,
    OneTransLiteBatch,
)


class OneTransLiteRecAgoraModel(RecAgoraRankingModel):
    """Translate public RecAgora views into OneTrans-Lite's natural tensors."""

    supports_candidate_mask = True
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        **params: Any,
    ) -> None:
        defaults = {
            "model_dim": 32,
            "num_heads": 4,
            "ffn_dim": 64,
            "history_group_name": "history_group_1",
        }
        unknown = set(params) - set(defaults)
        if unknown:
            raise TypeError(f"unknown OneTrans-Lite params: {sorted(unknown)}")
        config = {**defaults, **params}
        for name in ("model_dim", "num_heads", "ffn_dim"):
            value = config[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if config["model_dim"] % config["num_heads"]:
            raise ValueError("model_dim must be divisible by num_heads")
        history_group_name = config["history_group_name"]
        if not isinstance(history_group_name, str) or not history_group_name.strip():
            raise ValueError("history_group_name must be a non-empty string")
        super().__init__(config, input_spec)
        self.history_group_name = history_group_name
        history = input_spec.histories[history_group_name]
        self.history_slot_ids = history.categorical_slot_ids
        self.history_max_length = history.max_length
        self.lite = OneTransLite(
            query_dim=input_spec.query.total_dim,
            candidate_dim=input_spec.candidate.total_dim,
            history_dim=input_spec.total_dim(self.history_slot_ids),
            model_dim=config["model_dim"],
            num_heads=config["num_heads"],
            ffn_dim=config["ffn_dim"],
            num_tasks=len(input_spec.task_names),
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        history = embedded.histories[self.history_group_name]
        history_tensor = torch.cat(
            [
                history.categorical[slot_id].to_padded(
                    max_length=self.history_max_length,
                    padding_side="left",
                )
                for slot_id in self.history_slot_ids
            ],
            dim=-1,
        )
        positions = torch.arange(
            self.history_max_length,
            device=history.lengths.device,
        ).unsqueeze(0)
        history_valid = positions >= (
            self.history_max_length
            - history.lengths.clamp_max(self.history_max_length)
        ).unsqueeze(1)
        logits = self.lite(
            OneTransLiteBatch(
                query=embedded.query.concat(),
                candidates=embedded.candidate.concat(),
                history=history_tensor,
                history_valid=history_valid,
            )
        )
        return structured_ranking_output(
            {
                name: logits[..., index]
                for index, name in enumerate(self.task_names)
            }
        )

    def compute_loss(
        self,
        model_output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model_output = model_output["logits"]
        losses = {
            name: masked_binary_cross_entropy_with_logits(
                model_output[name],
                batch.labels[name].to(model_output[name].dtype),
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            for name in self.task_names
        }
        total_loss = sum(losses.values())
        return total_loss, losses
