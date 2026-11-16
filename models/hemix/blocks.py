from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class CandidateConditionedTokenizer(nn.Module):
    """Generate candidate-local NS tokens without materializing Q+C concat."""

    def __init__(
        self,
        *,
        query_dim: int,
        candidate_dim: int,
        num_tokens: int,
        token_dim: int,
    ) -> None:
        super().__init__()
        if min(query_dim, candidate_dim, num_tokens, token_dim) <= 0:
            raise ValueError("HeMix tokenizer dimensions must be positive")
        self.query_dim = int(query_dim)
        self.candidate_dim = int(candidate_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.query_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.query_dim, self.token_dim)
        )
        self.candidate_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.candidate_dim, self.token_dim)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, self.token_dim))
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.token_dim) for _ in range(self.num_tokens)
        )
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        input_dim = self.query_dim + self.candidate_dim
        bound = math.sqrt(6.0 / float(input_dim + self.token_dim))
        nn.init.uniform_(self.query_weight, -bound, bound)
        nn.init.uniform_(self.candidate_weight, -bound, bound)
        nn.init.zeros_(self.bias)
        for norm in self.norms:
            norm.reset_parameters()

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        query_tokens = torch.einsum("bq,nqd->bnd", query, self.query_weight)
        candidate_tokens = torch.einsum(
            "blc,ncd->blnd",
            candidates,
            self.candidate_weight,
        )
        tokens = candidate_tokens + query_tokens[:, None] + self.bias
        tokens = torch.stack(
            tuple(
                norm(tokens[:, :, index, :])
                for index, norm in enumerate(self.norms)
            ),
            dim=2,
        )
        return self.activation(tokens)


class MixedHeteroAttention(nn.Module):
    """Candidate-conditioned attention with query-private Q and shared K/V."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_heads: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_heads <= 0 or token_dim % num_heads:
            raise ValueError(
                "MixedHeteroAttention token_dim must be positive and divisible by heads"
            )
        if num_queries <= 0:
            raise ValueError("MixedHeteroAttention num_queries must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("MixedHeteroAttention dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.num_queries = int(num_queries)
        self.head_dim = self.token_dim // self.num_heads
        self.dropout = float(dropout)
        self.query_weight = nn.Parameter(
            torch.empty(self.num_queries, self.token_dim, self.token_dim)
        )
        self.key_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.value_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.output_projection = nn.Linear(self.token_dim, self.token_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(
        self,
        queries: torch.Tensor,
        history: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, list_len, query_count, token_dim = queries.shape
        if query_count != self.num_queries or token_dim != self.token_dim:
            raise ValueError("MixedHeteroAttention query shape does not match config")
        if history.shape[:2] != history_valid.shape:
            raise ValueError("MixedHeteroAttention history and mask shapes do not match")
        if history.shape[0] != batch_size or history.shape[-1] != self.token_dim:
            raise ValueError("MixedHeteroAttention history shape does not match queries")

        query = torch.einsum("blqd,qdf->blqf", queries, self.query_weight)
        key = self.key_projection(history)
        value = self.value_projection(history)
        query = query.reshape(
            batch_size,
            list_len,
            query_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 1, 3, 2, 4)
        key = key.reshape(
            batch_size,
            history.shape[1],
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        value = value.reshape(
            batch_size,
            history.shape[1],
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        scores = torch.einsum("blhqd,bhtd->blhqt", query, key)
        scores = scores * (self.head_dim ** -0.5)
        valid = history_valid[:, None, None, None, :]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.einsum("blhqt,bhtd->blhqd", weights, value)
        output = output.permute(0, 1, 3, 2, 4).contiguous().reshape(
            batch_size,
            list_len,
            query_count,
            self.token_dim,
        )
        output = self.output_projection(output)
        has_history = history_valid.any(dim=1).reshape(batch_size, 1, 1, 1)
        return output * has_history.to(output.dtype)


@dataclass(frozen=True)
class QueryMixedInterests:
    global_tokens: torch.Tensor
    realtime_tokens: torch.Tensor


class QueryMixedInterestExtractor(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_global_tokens: int,
        num_realtime_tokens: int,
        num_attention_heads: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        if num_global_tokens <= 0 or num_realtime_tokens <= 0:
            raise ValueError("HeMix global and realtime token counts must be positive")
        self.token_dim = int(token_dim)
        self.num_global_tokens = int(num_global_tokens)
        self.num_realtime_tokens = int(num_realtime_tokens)
        self.num_ns_tokens = self.num_global_tokens + self.num_realtime_tokens
        self.global_fixed_queries = nn.Parameter(
            torch.empty(self.num_global_tokens, self.token_dim)
        )
        self.realtime_fixed_queries = nn.Parameter(
            torch.empty(self.num_realtime_tokens, self.token_dim)
        )
        self.global_attention = MixedHeteroAttention(
            token_dim=self.token_dim,
            num_heads=num_attention_heads,
            num_queries=2 * self.num_global_tokens,
            dropout=attention_dropout,
        )
        self.realtime_attention = MixedHeteroAttention(
            token_dim=self.token_dim,
            num_heads=num_attention_heads,
            num_queries=2 * self.num_realtime_tokens,
            dropout=attention_dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.global_fixed_queries)
        nn.init.xavier_uniform_(self.realtime_fixed_queries)

    def forward(
        self,
        ns_tokens: torch.Tensor,
        *,
        global_history: torch.Tensor,
        realtime_history: torch.Tensor,
        global_valid: torch.Tensor,
        realtime_valid: torch.Tensor,
    ) -> QueryMixedInterests:
        batch_size, list_len, token_count, token_dim = ns_tokens.shape
        if token_count != self.num_ns_tokens or token_dim != self.token_dim:
            raise ValueError("HeMix NS token shape does not match query configuration")
        global_dynamic = ns_tokens[:, :, : self.num_global_tokens]
        realtime_dynamic = ns_tokens[:, :, self.num_global_tokens :]
        global_fixed = self.global_fixed_queries.reshape(
            1,
            1,
            self.num_global_tokens,
            self.token_dim,
        ).expand(batch_size, list_len, -1, -1)
        realtime_fixed = self.realtime_fixed_queries.reshape(
            1,
            1,
            self.num_realtime_tokens,
            self.token_dim,
        ).expand(batch_size, list_len, -1, -1)
        global_queries = torch.cat((global_dynamic, global_fixed), dim=2)
        realtime_queries = torch.cat((realtime_dynamic, realtime_fixed), dim=2)
        return QueryMixedInterests(
            global_tokens=self.global_attention(
                global_queries,
                global_history,
                global_valid,
            ),
            realtime_tokens=self.realtime_attention(
                realtime_queries,
                realtime_history,
                realtime_valid,
            ),
        )


class HeterogeneousTokenMixing(nn.Module):
    """Fuse token subspaces, apply head-private low-rank MLPs, then reconstruct."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_heads: int,
        low_rank_dim: int,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_heads <= 0 or token_dim % num_heads:
            raise ValueError(
                "HeterogeneousTokenMixing token_dim must be divisible by heads"
            )
        if num_tokens <= 0 or low_rank_dim <= 0:
            raise ValueError("HeterogeneousTokenMixing dimensions must be positive")
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.mix_dim = self.num_tokens * self.head_dim
        self.low_rank_dim = int(low_rank_dim)
        self.left_weight = nn.Parameter(
            torch.empty(self.num_heads, self.mix_dim, self.low_rank_dim)
        )
        self.left_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.low_rank_dim)
        )
        self.right_weight = nn.Parameter(
            torch.empty(self.num_heads, self.low_rank_dim, self.mix_dim)
        )
        self.right_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.mix_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.left_weight)
        nn.init.xavier_uniform_(self.right_weight)
        nn.init.zeros_(self.left_bias)
        nn.init.zeros_(self.right_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError("HeterogeneousTokenMixing token shape does not match config")
        leading = tokens.shape[:-2]
        rows = math.prod(leading)
        mixed = tokens.reshape(
            rows,
            self.num_tokens,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        mixed = mixed.reshape(rows, self.num_heads, self.mix_dim)
        hidden = torch.einsum(
            "bhf,hfr->bhr",
            mixed,
            self.left_weight,
        ) + self.left_bias
        hidden = F.relu(hidden)
        output = torch.einsum(
            "bhr,hrf->bhf",
            hidden,
            self.right_weight,
        ) + self.right_bias
        output = output.reshape(
            rows,
            self.num_heads,
            self.num_tokens,
            self.head_dim,
        ).permute(0, 2, 1, 3)
        return output.reshape(*leading, self.num_tokens, self.token_dim)


class TokenSpecificFeedForward(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(token_dim, num_tokens, expansion_factor) <= 0:
            raise ValueError("TokenSpecificFeedForward dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("TokenSpecificFeedForward dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(expansion_factor) * self.token_dim
        self.first_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.hidden_dim)
        )
        self.first_bias = nn.Parameter(
            torch.zeros(self.num_tokens, self.hidden_dim)
        )
        self.second_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_dim, self.token_dim)
        )
        self.second_bias = nn.Parameter(
            torch.zeros(self.num_tokens, self.token_dim)
        )
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.first_weight)
        nn.init.xavier_uniform_(self.second_weight)
        nn.init.zeros_(self.first_bias)
        nn.init.zeros_(self.second_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError("TokenSpecificFeedForward token shape does not match config")
        hidden = torch.einsum(
            "...td,tdh->...th",
            tokens,
            self.first_weight,
        ) + self.first_bias
        hidden = self.dropout(F.gelu(hidden))
        return torch.einsum(
            "...th,thd->...td",
            hidden,
            self.second_weight,
        ) + self.second_bias


class HeteroMixerBlock(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_mix_heads: int,
        low_rank_dim: int,
        expansion_factor: int,
        dropout: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.mixing_norm = nn.LayerNorm(token_dim, eps=float(layer_norm_eps))
        self.ffn_norm = nn.LayerNorm(token_dim, eps=float(layer_norm_eps))
        self.mixing = HeterogeneousTokenMixing(
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_heads=num_mix_heads,
            low_rank_dim=low_rank_dim,
        )
        self.ffn = TokenSpecificFeedForward(
            token_dim=token_dim,
            num_tokens=num_tokens,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.mixing(self.mixing_norm(tokens))
        return tokens + self.ffn(self.ffn_norm(tokens))


class HeteroMixer(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_layers: int,
        num_mix_heads: int,
        low_rank_dim: int,
        expansion_factor: int,
        dropout: float,
        layer_norm_eps: float,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("HeteroMixer num_layers must be positive")
        self.activation_checkpointing = bool(activation_checkpointing)
        self.blocks = nn.ModuleList(
            HeteroMixerBlock(
                token_dim=token_dim,
                num_tokens=num_tokens,
                num_mix_heads=num_mix_heads,
                low_rank_dim=low_rank_dim,
                expansion_factor=expansion_factor,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(int(num_layers))
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if (
                self.activation_checkpointing
                and self.training
                and tokens.requires_grad
                and tokens.device.type == "cuda"
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens
