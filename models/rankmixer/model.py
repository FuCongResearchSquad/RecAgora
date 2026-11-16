from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from recagora import (
    RecAgoraInputSpec,
    RecAgoraModelBatch,
    RecAgoraRankingModel,
    RecAgoraSparseBatch,
    RankingOutput,
    masked_binary_cross_entropy_with_logits,
    structured_ranking_output,
)
from models._validation import (
    normalize_output_label_names,
    validate_disjoint_slot_groups,
    validate_known_weights,
)

from .blocks import (
    AdaptiveSparsityCoefficient,
    PFFNSettings,
    RankMixerBlock,
    RankMixerBlockOutput,
    parse_pffn_config,
)
from .sequence import build_sequence_encoder
from ._validation import (
    dropout_probability as _dropout_probability,
    nonempty_string as _nonempty_string,
    nonnegative_float as _nonnegative_float,
    positive_float as _positive_float,
    positive_int as _positive_int,
)


class RankMixerInputs(nn.Module):
    """RankMixer-owned grouping over public slot views."""

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        token_dim: int,
        token_order: Sequence[str],
        tokens_per_group: Mapping[str, int] | None,
        input_groups: Mapping[str, Sequence[int]],
        history_group_name: str,
        query_group_name: str,
        sequence_encoder: Mapping[str, Any] | None,
    ) -> None:
        super().__init__()
        self.input_spec = input_spec
        self.token_dim = _positive_int(token_dim, "token_dim")
        self.token_order = tuple(str(name) for name in token_order)
        self.tokens_per_group = {
            str(name): _positive_int(count, f"tokens_per_group.{name}")
            for name, count in (tokens_per_group or {}).items()
        }
        self.input_groups = {
            str(name): tuple(int(slot_id) for slot_id in slot_ids)
            for name, slot_ids in input_groups.items()
        }
        self.history_group_name = str(history_group_name)
        self.query_group_name = str(query_group_name)
        sequence = dict(sequence_encoder or {})
        unknown_sequence_keys = sorted(set(sequence) - {"type", "group", "params"})
        if unknown_sequence_keys:
            raise ValueError(
                "sequence_encoder has unknown fields: "
                f"{unknown_sequence_keys}"
            )
        self.sequence_type = _nonempty_string(
            sequence.get("type", "masked_mean"),
            "sequence_encoder.type",
        )
        self.sequence_group = _nonempty_string(
            sequence.get("group", "history"),
            "sequence_encoder.group",
        )
        raw_sequence_params = sequence.get("params", {})
        if not isinstance(raw_sequence_params, Mapping):
            raise TypeError("sequence_encoder.params must be a mapping")
        self.sequence_params = dict(raw_sequence_params)
        self._validate_config()

        self.sequence_encoder = build_sequence_encoder(
            sequence_type=self.sequence_type,
            params=self.sequence_params,
            slot_ids=self.input_groups[self.sequence_group],
            input_spec=input_spec,
            history_group_name=self.history_group_name,
        )
        selected = (
            set(input_spec.query.slot_ids)
            | set(input_spec.candidate.slot_ids)
            | {
                slot_id
                for group in input_spec.histories.values()
                for slot_id in group.categorical_slot_ids
            }
        )
        grouped = {
            slot_id
            for slots in self.input_groups.values()
            for slot_id in slots
        }
        if grouped != selected:
            raise ValueError(
                "input_groups must exactly cover selected slots; "
                f"missing={sorted(selected - grouped)}, extra={sorted(grouped - selected)}"
            )

        for group_name in self.token_order:
            self._validate_group(group_name, self.input_groups[group_name])
        self.group_input_dims: dict[str, int] = {}
        self.group_projections = nn.ModuleDict()
        for group_name in self.token_order:
            slot_ids = self.input_groups[group_name]
            input_dim = (
                int(self.sequence_encoder.output_dim)
                if group_name == self.sequence_group
                else input_spec.total_dim(slot_ids)
            )
            self.group_input_dims[group_name] = input_dim
            token_count = int(self.tokens_per_group.get(group_name, 1))
            self.group_projections[group_name] = nn.Linear(
                input_dim,
                self.token_dim * token_count,
            )

    @property
    def total_token_count(self) -> int:
        return sum(
            int(self.tokens_per_group.get(name, 1))
            for name in self.token_order
        )

    def _validate_config(self) -> None:
        if not self.token_order:
            raise ValueError("token_order must not be empty")
        if len(set(self.token_order)) != len(self.token_order):
            raise ValueError("token_order must not contain duplicate groups")
        missing = [name for name in self.token_order if name not in self.input_groups]
        if missing:
            raise ValueError(f"input_groups missing token groups: {missing}")
        unknown = sorted(set(self.input_groups) - set(self.token_order))
        if unknown:
            raise ValueError(f"input_groups has unknown groups: {unknown}")
        if any(not self.input_groups[name] for name in self.token_order):
            raise ValueError("input_groups must not contain empty groups")
        validate_disjoint_slot_groups(
            tuple(self.input_groups[name] for name in self.token_order),
            context="input_groups",
        )
        unknown_counts = set(self.tokens_per_group) - set(self.token_order)
        if unknown_counts:
            raise ValueError(
                f"tokens_per_group contains unknown groups: {sorted(unknown_counts)}"
            )
        if self.sequence_group not in self.input_groups:
            raise ValueError("sequence_encoder.group must name one input group")
        if self.query_group_name not in self.input_groups:
            raise ValueError("query_group_name must name one input group")
        if self.sequence_group == self.query_group_name:
            raise ValueError("sequence and query token groups must be different")

    def _validate_group(self, group_name: str, slot_ids: tuple[int, ...]) -> None:
        expected_view = (
            "history"
            if group_name == self.sequence_group
            else "query" if group_name == self.query_group_name else "candidate"
        )
        expected_role = (
            "sequence"
            if group_name == self.sequence_group
            else "request" if group_name == self.query_group_name else "item"
        )
        for slot_id in slot_ids:
            spec = self.input_spec.slot(slot_id)
            if spec.view != expected_view or spec.role != expected_role:
                raise ValueError(
                    f"group {group_name!r} slot {slot_id} has "
                    f"view/role={spec.view!r}/{spec.role!r}, expected "
                    f"{expected_view!r}/{expected_role!r}"
                )
            if spec.encoding != "embedding":
                raise ValueError(f"RankMixer only supports embedding slot {slot_id}")
            expected_reduction = (
                "none" if group_name == self.sequence_group else "mean"
            )
            if spec.reduction != expected_reduction:
                raise ValueError(
                    f"slot {slot_id} reduction must be {expected_reduction!r}"
                )

    def _project(
        self,
        group_name: str,
        values: torch.Tensor,
        batch_size: int,
        list_len: int,
    ) -> torch.Tensor:
        token_count = int(self.tokens_per_group.get(group_name, 1))
        if values.ndim == 3:
            values = values.reshape(batch_size * list_len, values.shape[-1])
        projected = self.group_projections[group_name](values)
        if values.shape[0] == batch_size * list_len:
            return projected.reshape(
                batch_size,
                list_len,
                token_count,
                self.token_dim,
            )
        return projected.reshape(
            batch_size,
            1,
            token_count,
            self.token_dim,
        ).expand(batch_size, list_len, token_count, self.token_dim)

    def forward(self, batch: RecAgoraModelBatch) -> torch.Tensor:
        batch_size, list_len = batch.batch_size, batch.list_len
        pieces: list[torch.Tensor] = []
        history = batch.histories[self.history_group_name]
        for group_name in self.token_order:
            slot_ids = self.input_groups[group_name]
            if group_name == self.sequence_group:
                group = self.sequence_encoder(history, batch.candidate)
            else:
                view = (
                    batch.query
                    if group_name == self.query_group_name
                    else batch.candidate
                )
                group = view.concat(slot_ids)
                if group_name != self.query_group_name:
                    group = group.reshape(batch_size * list_len, -1)
            pieces.append(self._project(group_name, group, batch_size, list_len))
        return torch.cat(pieces, dim=2).reshape(
            batch_size * list_len,
            self.total_token_count,
            self.token_dim,
        )


class RankMixerTaskTower(nn.Sequential):
    """Task-private MLP tower over one candidate representation."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        dropout = float(dropout)
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        for module in self:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)


class RankMixerModel(RecAgoraRankingModel):
    """Configurable RankMixer with explicit constructor parameters."""

    # Dense RankMixer processes each candidate independently and has no
    # candidate-level auxiliary objective. Sparse-MoE variants override the
    # instance capability below until their routing regularizer is mask-aware.
    candidate_mask_all_empty_policy = "skip_update"
    candidate_mask_loss_reduction = "candidate_mean"

    def __init__(
        self,
        *,
        input_spec: RecAgoraInputSpec,
        token_dim: int = 192,
        num_layers: int = 4,
        expansion_ratio: int = 4,
        head_type: str,
        head_hidden_dim: int,
        head_dropout: float,
        token_order: Sequence[str] = (),
        tokens_per_group: Mapping[str, int] | None = None,
        input_groups: Mapping[str, Sequence[int]] | None = None,
        history_group_name: str = "history_group_1",
        query_group_name: str = "query_token",
        output_label_names: Sequence[str] = ("click",),
        loss_weights: Mapping[str, float] | None = None,
        sequence_encoder: Mapping[str, Any] | None = None,
        pffn: Mapping[str, Any] | None = None,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        token_dim = _positive_int(token_dim, "token_dim")
        num_layers = _positive_int(num_layers, "num_layers")
        expansion_ratio = _positive_int(expansion_ratio, "expansion_ratio")
        head_type = _nonempty_string(head_type, "head_type")
        if head_type not in {"linear", "mlp"}:
            raise ValueError("head_type must be 'linear' or 'mlp'")
        head_hidden_dim = _positive_int(head_hidden_dim, "head_hidden_dim")
        head_dropout = _dropout_probability(head_dropout, "head_dropout")
        layer_norm_eps = _positive_float(layer_norm_eps, "layer_norm_eps")
        output_label_names = normalize_output_label_names(output_label_names)
        loss_weights = {
            str(name): _nonnegative_float(weight, f"loss_weights.{name}")
            for name, weight in (loss_weights or {}).items()
        }
        validate_known_weights(loss_weights, output_label_names)
        if loss_weights and not any(
            loss_weights.get(name, 1.0) > 0 for name in output_label_names
        ):
            raise ValueError("at least one task loss weight must be positive")
        if tuple(input_spec.task_names) != output_label_names:
            raise ValueError(
                "output_label_names must exactly match RecAgoraInputSpec.task_names"
            )
        pffn_settings = parse_pffn_config(pffn)

        model_params = {
            "token_dim": token_dim,
            "num_layers": num_layers,
            "expansion_ratio": expansion_ratio,
            "head_type": head_type,
            "head_hidden_dim": head_hidden_dim,
            "head_dropout": head_dropout,
            "token_order": tuple(token_order),
            "tokens_per_group": dict(tokens_per_group or {}),
            "input_groups": dict(input_groups or {}),
            "history_group_name": history_group_name,
            "query_group_name": query_group_name,
            "output_label_names": output_label_names,
            "loss_weights": loss_weights,
            "sequence_encoder": dict(sequence_encoder or {}),
            "layer_norm_eps": layer_norm_eps,
        }
        if pffn is not None:
            model_params["pffn"] = dict(pffn)
        super().__init__(model_params, input_spec)
        self.supports_candidate_mask = pffn_settings.kind == "dense"
        self.task_names = output_label_names
        self.loss_weights = loss_weights
        self.pffn_settings = pffn_settings
        self.head_type = head_type
        self.inputs = RankMixerInputs(
            input_spec=input_spec,
            token_dim=token_dim,
            token_order=token_order,
            tokens_per_group=tokens_per_group,
            input_groups=input_groups or {},
            history_group_name=history_group_name,
            query_group_name=query_group_name,
            sequence_encoder=sequence_encoder,
        )
        self.blocks = nn.ModuleList(
            RankMixerBlock(
                num_tokens=self.inputs.total_token_count,
                token_dim=token_dim,
                expansion_ratio=expansion_ratio,
                layer_norm_eps=layer_norm_eps,
                pffn_settings=pffn_settings,
            )
            for _ in range(num_layers)
        )
        self.sparsity_controller = (
            AdaptiveSparsityCoefficient(pffn_settings)
            if pffn_settings.kind != "dense"
            else None
        )
        self.output_heads = nn.ModuleDict(
            {
                name: (
                    nn.Linear(token_dim, 1)
                    if head_type == "linear"
                    else RankMixerTaskTower(
                        input_dim=token_dim,
                        hidden_dim=head_hidden_dim,
                        dropout=head_dropout,
                    )
                )
                for name in output_label_names
            }
        )

    def forward(self, batch: RecAgoraSparseBatch) -> RankingOutput:
        embedded = self.get_slots(batch)
        tokens = self.inputs(embedded)
        routing_gates: list[torch.Tensor] = []
        for block in self.blocks:
            if self.pffn_settings.kind == "dense":
                tokens = block(tokens)
                continue
            block_output = block(tokens, return_routing=True)
            tokens = block_output.tokens
            routing_gates.append(block_output.gates)
        representation = tokens.mean(dim=1)
        logits = {
            name: head(representation).squeeze(-1).reshape(
                embedded.batch_size,
                embedded.list_len,
            )
            for name, head in self.output_heads.items()
        }
        if not routing_gates:
            return structured_ranking_output(logits)
        coefficient = self.sparsity_controller.value.detach().clone()
        return structured_ranking_output(
            logits,
            aux={
                "routing_gates": tuple(routing_gates),
                "regularization_coefficient": coefficient,
                "controller_updated": torch.zeros(
                    (),
                    dtype=torch.bool,
                    device=coefficient.device,
                ),
                "routing_metrics": {},
            },
        )

    def compute_loss(
        self,
        output: RankingOutput,
        batch: RecAgoraSparseBatch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_by_task = output["logits"]
        losses: dict[str, torch.Tensor] = {}
        weighted: list[torch.Tensor] = []
        for name, logits in logits_by_task.items():
            label = batch.labels[name]
            loss = masked_binary_cross_entropy_with_logits(
                logits,
                label.to(device=logits.device, dtype=logits.dtype),
                valid_mask=batch.candidate_valid_mask,
                context=batch.candidate_loss_context,
            )
            losses[name] = loss
            weighted.append(loss * float(self.loss_weights.get(name, 1.0)))
        aux = output["aux"]
        if aux:
            raw_regularizer, metrics = self._routing_regularizer(
                aux["routing_gates"]
            )
            coefficient = aux["regularization_coefficient"].to(
                device=raw_regularizer.device,
                dtype=raw_regularizer.dtype,
            )
            auxiliary = raw_regularizer * coefficient
            regularizer_name = (
                "dtsi_gate_l1"
                if self.pffn_settings.kind == "dtsi_moe"
                else "smoe_gate_l1"
            )
            losses[regularizer_name] = auxiliary
            weighted.append(auxiliary)
            aux["routing_metrics"].update(
                {
                    **metrics,
                    "raw_gate_l1": raw_regularizer.detach(),
                    "regularization_coefficient": coefficient.detach(),
                }
            )
            controller_updated = aux["controller_updated"]
            if self.training and not bool(controller_updated.item()):
                self.sparsity_controller.update(
                    metrics["average_active_experts"],
                )
                controller_updated.fill_(True)
        return torch.stack(weighted).sum(), losses

    def _routing_regularizer(
        self,
        gates: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        settings = self.pffn_settings
        stacked = torch.stack(gates, dim=0)
        layer_count, local_rows, token_count, expert_count = stacked.shape

        active_counts = (stacked.detach() > 0).sum(dim=1, dtype=torch.float64)
        global_rows = active_counts.new_tensor(float(local_rows))
        world_size = 1
        if self.training and dist.is_available() and dist.is_initialized():
            packed = torch.cat((active_counts.reshape(-1), global_rows.reshape(1)))
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            active_counts = packed[:-1].reshape_as(active_counts)
            global_rows = packed[-1]
            world_size = dist.get_world_size()

        if settings.load_balance:
            factors = active_counts * (
                float(expert_count)
                / (settings.target_active_experts * global_rows)
            )
        else:
            factors = torch.ones_like(active_counts)
        local_regularizer_sum = (
            stacked
            * factors.to(device=stacked.device, dtype=stacked.dtype).unsqueeze(1)
        ).sum()
        normalization = float(layer_count * token_count) * global_rows
        regularizer = local_regularizer_sum * float(world_size) / normalization.to(
            device=stacked.device,
            dtype=stacked.dtype,
        )
        active_by_layer_token = active_counts.sum(dim=-1) / global_rows
        average_active_experts = active_by_layer_token.mean().to(
            device=stacked.device,
            dtype=stacked.dtype,
        )
        metrics = {
            "average_active_experts": average_active_experts.detach(),
            "active_fraction": (
                average_active_experts / float(expert_count)
            ).detach(),
            "active_experts_by_layer_token": active_by_layer_token.to(
                device=stacked.device,
                dtype=stacked.dtype,
            ).detach(),
        }
        return regularizer, metrics
