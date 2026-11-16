from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TokenDomainGroupLayerNorm(nn.Module):
    """Layer-normalize tokens with independent parameters for each MTGR domain."""

    def __init__(self, dim: int, num_domains: int, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_domains = int(num_domains)
        self.norms = nn.ModuleList(
            nn.LayerNorm(dim, eps=eps) for _ in range(num_domains)
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        domain_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(value)
        for domain, norm in enumerate(self.norms):
            selected = domain_ids.eq(domain).unsqueeze(-1)
            result = torch.where(selected, norm(value), result)
        return result.masked_fill(~valid.unsqueeze(-1), 0.0)


class GatedAttentionBlock(nn.Module):
    """Formula-faithful MTGR block with domain GroupLN and SiLU attention."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        value_dim: int,
        query_key_dim: int,
        num_token_domains: int,
        output_hidden_dim: int,
        dropout: float,
        group_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.value_dim = int(value_dim)
        self.query_key_dim = int(query_key_dim)
        uvqk_dim = num_heads * (2 * value_dim + 2 * query_key_dim)
        gated_dim = num_heads * value_dim
        self.input_norm = TokenDomainGroupLayerNorm(
            model_dim,
            num_token_domains,
            eps=group_norm_eps,
        )
        self.uvqk = nn.Linear(model_dim, uvqk_dim)
        self.gated_norm = TokenDomainGroupLayerNorm(
            gated_dim,
            num_token_domains,
            eps=group_norm_eps,
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(gated_dim, output_hidden_dim),
            nn.SiLU(),
            nn.Linear(output_hidden_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        *,
        allowed: torch.Tensor,
        domain_ids: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, model_dim = value.shape
        normalized = self.input_norm(value, domain_ids=domain_ids, valid=valid)
        u, v, q, k = torch.split(
            F.silu(self.uvqk(normalized)),
            [
                self.num_heads * self.value_dim,
                self.num_heads * self.value_dim,
                self.num_heads * self.query_key_dim,
                self.num_heads * self.query_key_dim,
            ],
            dim=-1,
        )
        u = u.view(batch_size, token_count, self.num_heads, self.value_dim)
        v = v.view(batch_size, token_count, self.num_heads, self.value_dim).transpose(1, 2)
        q = q.view(batch_size, token_count, self.num_heads, self.query_key_dim).transpose(1, 2)
        k = k.view(batch_size, token_count, self.num_heads, self.query_key_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        effective_allowed = (
            allowed & valid[:, :, None] & valid[:, None, :]
        )
        weights = F.silu(scores.masked_fill(~effective_allowed[:, None], 0.0))
        denominator = valid.sum(dim=1).clamp_min(1).to(scores.dtype)[:, None, None, None]
        weights = weights * effective_allowed[:, None].to(scores.dtype) / denominator
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous() * u
        attended = attended.reshape(batch_size, token_count, -1)
        attended = self.gated_norm(attended, domain_ids=domain_ids, valid=valid)
        updated = value + self.dropout(self.output_mlp(attended))
        return torch.where(valid.unsqueeze(-1), updated, value)
