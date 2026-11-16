from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class RoutingOutput(NamedTuple):
    values: torch.Tensor
    auxiliary_loss: torch.Tensor
    expert_load: torch.Tensor
    mean_probability: torch.Tensor
    entropy: torch.Tensor


class SORTAttention(nn.Module):
    """Pre-normalized causal attention with SORT candidate isolation."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        local_window: int,
        dropout: float,
        qk_norm_eps: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.local_window = int(local_window)
        self.dropout = float(dropout)
        self.qk_norm_eps = float(qk_norm_eps)
        self.rope_base = float(rope_base)
        if self.head_dim % 2:
            raise ValueError("SORT RoPE requires an even attention head dimension")
        self.qkv = nn.Linear(self.model_dim, 3 * self.model_dim)
        self.gate = nn.Linear(self.model_dim, self.num_heads)
        self.output = nn.Linear(self.model_dim, self.model_dim)
        for layer in (self.qkv, self.gate, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        values: torch.Tensor,
        *,
        valid: torch.Tensor,
        position_ids: torch.Tensor,
        candidate_count: int,
    ) -> torch.Tensor:
        batch_size, token_count, _ = values.shape
        q, k, v = self.qkv(values).chunk(3, dim=-1)
        q = q.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        k = k.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.reshape(batch_size, token_count, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        q = F.rms_norm(q, (self.head_dim,), eps=self.qk_norm_eps)
        k = F.rms_norm(k, (self.head_dim,), eps=self.qk_norm_eps)
        q = _apply_rope(q, position_ids, base=self.rope_base)
        k = _apply_rope(k, position_ids, base=self.rope_base)
        allowed = build_sort_attention_mask(
            position_ids=position_ids,
            valid=valid,
            candidate_count=candidate_count,
            local_window=self.local_window,
        )
        attention_mask = torch.zeros(
            allowed.shape,
            dtype=q.dtype,
            device=q.device,
        ).masked_fill(~allowed, float("-inf"))
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask[:, None],
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(float(self.head_dim)),
        )
        gates = torch.sigmoid(self.gate(values)).transpose(1, 2)[..., None]
        attended = (attended * gates).transpose(1, 2).contiguous().reshape(
            batch_size,
            token_count,
            self.model_dim,
        )
        return self.output(attended) * valid[..., None].to(attended.dtype)


def build_sort_attention_mask(
    *,
    position_ids: torch.Tensor,
    valid: torch.Tensor,
    candidate_count: int,
    local_window: int,
) -> torch.Tensor:
    """Return ``[B,T,T]`` mask with causal context and diagonal candidates."""

    token_count = int(position_ids.numel())
    non_candidate_count = token_count - int(candidate_count)
    if non_candidate_count <= 0 or candidate_count <= 0:
        raise ValueError("SORT attention requires context and candidate tokens")
    query_index = torch.arange(token_count, device=position_ids.device)[:, None]
    key_index = torch.arange(token_count, device=position_ids.device)[None, :]
    non_candidate_query = query_index < non_candidate_count
    non_candidate_key = key_index < non_candidate_count
    causal = key_index <= query_index
    position_distance = position_ids[:, None] - position_ids[None, :]
    local = position_distance.ge(0) & position_distance.lt(local_window)
    # BOS is an attention sink even when it falls outside a future shorter
    # local window; this also makes every padded query row numerically safe.
    local[:, 0] = True
    context_allowed = non_candidate_query & non_candidate_key & causal & local

    candidate_query = query_index >= non_candidate_count
    candidate_context = candidate_query & non_candidate_key & local
    candidate_diagonal = (
        candidate_query
        & (key_index >= non_candidate_count)
        & query_index.eq(key_index)
    )
    structural = context_allowed | candidate_context | candidate_diagonal
    key_valid = valid[:, None, :]
    allowed = structural[None] & key_valid
    # Padded history queries are discarded after attention, but allowing BOS
    # avoids an all-masked SDPA row before that multiplication.
    allowed[:, :, 0] = True
    return allowed


class SwiGLUExpert(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, model_dim, bias=False)
        for layer in (self.gate, self.up, self.down):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(values)) * self.up(values))


class SharedRoutedSwiGLUMoE(nn.Module):
    """DeepSeek-style shared plus top-k routed experts.

    Only valid selected rows execute routed experts.  Parameter-level zero
    graph edges keep unselected experts reachable by DDP without invoking
    their forward paths or materializing dense expert outputs.
    """

    def __init__(
        self,
        *,
        model_dim: int,
        hidden_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        routed_top_k: int,
    ) -> None:
        super().__init__()
        self.num_routed_experts = int(num_routed_experts)
        self.routed_top_k = int(routed_top_k)
        self.shared_experts = nn.ModuleList(
            SwiGLUExpert(model_dim, hidden_dim)
            for _ in range(int(num_shared_experts))
        )
        self.routed_experts = nn.ModuleList(
            SwiGLUExpert(model_dim, hidden_dim)
            for _ in range(self.num_routed_experts)
        )
        self.router = nn.Linear(model_dim, self.num_routed_experts, bias=False)
        nn.init.xavier_uniform_(self.router.weight)

    def forward(
        self,
        values: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> RoutingOutput:
        if values.ndim != 3 or valid.shape != values.shape[:2]:
            raise ValueError("SORT MoE expects values [B,T,D] and valid [B,T]")
        flat_values = values.reshape(-1, values.shape[-1])
        valid_rows = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        zero_graph = sum(
            (
                parameter.sum() * 0.0
                for expert in (*self.shared_experts, *self.routed_experts)
                for parameter in expert.parameters()
            ),
            self.router.weight.sum() * 0.0,
        )
        if valid_rows.numel() == 0:
            zeros = values.new_zeros((self.num_routed_experts,))
            return RoutingOutput(
                values=torch.zeros_like(values) + zero_graph,
                auxiliary_loss=zero_graph,
                expert_load=zeros,
                mean_probability=zeros,
                entropy=zero_graph.detach(),
            )

        valid_values = flat_values.index_select(0, valid_rows)
        probabilities = torch.softmax(
            self.router(valid_values).float(),
            dim=-1,
        ).to(values.dtype)
        top_values, top_indices = probabilities.topk(self.routed_top_k, dim=-1)
        top_values = top_values / top_values.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(top_values.dtype).eps
        )
        valid_output = sum(
            (expert(valid_values) for expert in self.shared_experts),
            torch.zeros_like(valid_values),
        )
        for expert_index, expert in enumerate(self.routed_experts):
            active = top_indices.eq(expert_index)
            rows, selections = active.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            contribution = expert(valid_values.index_select(0, rows))
            weights = top_values[rows, selections]
            valid_output = valid_output.index_add(
                0,
                rows,
                contribution * weights[:, None],
            )
        flat_output = torch.zeros_like(flat_values).index_add(
            0,
            valid_rows,
            valid_output,
        )
        output = flat_output.reshape_as(values) + zero_graph

        assignments = F.one_hot(
            top_indices,
            num_classes=self.num_routed_experts,
        ).to(probabilities.dtype).sum(dim=-2) / float(self.routed_top_k)
        reduce_dims = tuple(range(assignments.ndim - 1))
        expert_load = assignments.mean(dim=reduce_dims)
        mean_probability = probabilities.mean(dim=reduce_dims)
        auxiliary_loss = self.num_routed_experts * torch.sum(
            expert_load.detach() * mean_probability
        )
        entropy = -(
            probabilities.float()
            * probabilities.float().clamp_min(1.0e-9).log()
        ).sum(dim=-1).mean()
        return RoutingOutput(
            values=output,
            auxiliary_loss=auxiliary_loss,
            expert_load=expert_load.detach(),
            mean_probability=mean_probability.detach(),
            entropy=entropy.detach(),
        )


class SORTBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        local_window: int,
        expert_hidden_dim: int,
        num_shared_experts: int,
        num_routed_experts: int,
        routed_top_k: int,
        dropout: float,
        norm_eps: float,
        qk_norm_eps: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.attention = SORTAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            local_window=local_window,
            dropout=dropout,
            qk_norm_eps=qk_norm_eps,
            rope_base=rope_base,
        )
        self.moe_norm = nn.RMSNorm(model_dim, eps=norm_eps)
        self.moe = SharedRoutedSwiGLUMoE(
            model_dim=model_dim,
            hidden_dim=expert_hidden_dim,
            num_shared_experts=num_shared_experts,
            num_routed_experts=num_routed_experts,
            routed_top_k=routed_top_k,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        *,
        valid: torch.Tensor,
        position_ids: torch.Tensor,
        candidate_count: int,
    ) -> RoutingOutput:
        values = values + self.dropout(
            self.attention(
                self.attention_norm(values),
                valid=valid,
                position_ids=position_ids,
                candidate_count=candidate_count,
            )
        )
        routed = self.moe(self.moe_norm(values), valid=valid)
        values = (values + self.dropout(routed.values)) * valid[..., None].to(
            values.dtype
        )
        return RoutingOutput(
            values=values,
            auxiliary_loss=routed.auxiliary_loss,
            expert_load=routed.expert_load,
            mean_probability=routed.mean_probability,
            entropy=routed.entropy,
        )


def _apply_rope(
    values: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    base: float,
) -> torch.Tensor:
    head_dim = values.shape[-1]
    frequency = torch.exp(
        -math.log(base)
        * torch.arange(0, head_dim, 2, device=values.device, dtype=torch.float32)
        / float(head_dim)
    )
    angles = position_ids.to(dtype=torch.float32)[:, None] * frequency[None, :]
    cosine = angles.cos().to(values.dtype)[None, None]
    sine = angles.sin().to(values.dtype)[None, None]
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.stack(
        (even * cosine - odd * sine, odd * cosine + even * sine),
        dim=-1,
    )
    return rotated.flatten(start_dim=-2)
