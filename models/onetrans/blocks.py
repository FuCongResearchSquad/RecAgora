from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _token_specific_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("btd,tdh->bth", inputs, weight) + bias.unsqueeze(0)


class MixedFFN(nn.Module):
    """Shared sequence-token FFN and independent non-sequence-token FFNs."""

    def __init__(self, model_dim: int, ffn_dim: int, num_ns_tokens: int) -> None:
        super().__init__()
        self.num_ns_tokens = num_ns_tokens
        self.shared_up = nn.Linear(model_dim, ffn_dim)
        self.shared_down = nn.Linear(ffn_dim, model_dim)
        self.ns_up_weight = nn.Parameter(
            torch.empty(num_ns_tokens, model_dim, ffn_dim)
        )
        self.ns_up_bias = nn.Parameter(torch.zeros(num_ns_tokens, ffn_dim))
        self.ns_down_weight = nn.Parameter(
            torch.empty(num_ns_tokens, ffn_dim, model_dim)
        )
        self.ns_down_bias = nn.Parameter(torch.zeros(num_ns_tokens, model_dim))
        for weight in self.ns_up_weight:
            nn.init.xavier_uniform_(weight)
        for weight in self.ns_down_weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        sequence_count = tokens.shape[1] - self.num_ns_tokens
        sequence = tokens[:, :sequence_count]
        non_sequence = tokens[:, sequence_count:]
        sequence_out = self.shared_down(F.gelu(self.shared_up(sequence)))
        ns_hidden = F.gelu(
            _token_specific_linear(non_sequence, self.ns_up_weight, self.ns_up_bias)
        )
        ns_out = _token_specific_linear(
            ns_hidden,
            self.ns_down_weight,
            self.ns_down_bias,
        )
        return torch.cat((sequence_out, ns_out), dim=1)

    def forward_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.shared_down(F.gelu(self.shared_up(sequence)))

    def forward_non_sequence(self, non_sequence: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(
            torch.einsum("bnkd,kdh->bnkh", non_sequence, self.ns_up_weight)
            + self.ns_up_bias[None, None]
        )
        return (
            torch.einsum("bnkh,khd->bnkd", hidden, self.ns_down_weight)
            + self.ns_down_bias[None, None]
        )


class MixedCausalAttention(nn.Module):
    """Shared sequence projections plus per-NS-token projections."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        num_ns_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.num_ns_tokens = num_ns_tokens
        self.dropout = dropout
        self.shared_qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.ns_qkv_weight = nn.Parameter(
            torch.empty(num_ns_tokens, model_dim, 3 * model_dim)
        )
        self.output = nn.Linear(model_dim, model_dim, bias=False)
        for weight in self.ns_qkv_weight:
            nn.init.xavier_uniform_(weight)

    def _project(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence_count = tokens.shape[1] - self.num_ns_tokens
        sequence = self.shared_qkv(tokens[:, :sequence_count])
        non_sequence = torch.einsum(
            "btd,tdh->bth",
            tokens[:, sequence_count:],
            self.ns_qkv_weight,
        )
        return torch.cat((sequence, non_sequence), dim=1).chunk(3, dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        query_start: int,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, token_count, _ = tokens.shape
        query, key, value = self._project(tokens)

        def heads(item: torch.Tensor) -> torch.Tensor:
            return item.reshape(
                batch_size,
                item.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = heads(query[:, query_start:])
        key = heads(key)
        value = heads(value)
        query_positions = torch.arange(query_start, token_count, device=tokens.device)
        key_positions = torch.arange(token_count, device=tokens.device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        attention_mask = torch.zeros(
            (query.shape[-2], token_count),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        attention_mask.masked_fill_(~allowed, float("-inf"))
        attention_mask = attention_mask[None, None].expand(batch_size, 1, -1, -1)
        attention_mask = attention_mask.masked_fill(
            ~valid_mask[:, None, None, :],
            float("-inf"),
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            token_count - query_start,
            self.model_dim,
        )
        return self.output(attended)

    def forward_shared_candidates(
        self,
        sequence: torch.Tensor,
        non_sequence: torch.Tensor,
        *,
        retained_sequence_count: int,
        sequence_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse request sequence projections while isolating candidates."""
        batch_size, sequence_count, _ = sequence.shape
        candidate_count, ns_count = non_sequence.shape[1:3]

        sequence_q, sequence_k, sequence_v = self.shared_qkv(sequence).chunk(3, dim=-1)
        ns_q, ns_k, ns_v = torch.einsum(
            "bnkd,kdh->bnkh",
            non_sequence,
            self.ns_qkv_weight,
        ).chunk(3, dim=-1)

        def sequence_heads(item: torch.Tensor) -> torch.Tensor:
            return item.reshape(
                batch_size,
                item.shape[1],
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        def ns_heads(item: torch.Tensor) -> torch.Tensor:
            return item.reshape(
                batch_size,
                candidate_count,
                ns_count,
                self.num_heads,
                self.head_dim,
            ).permute(0, 1, 3, 2, 4)

        sequence_q = sequence_heads(sequence_q)
        sequence_k = sequence_heads(sequence_k)
        sequence_v = sequence_heads(sequence_v)
        ns_q = ns_heads(ns_q)
        ns_k = ns_heads(ns_k)
        ns_v = ns_heads(ns_v)

        if retained_sequence_count:
            query_start = sequence_count - retained_sequence_count
            query_positions = torch.arange(
                query_start,
                sequence_count,
                device=sequence.device,
            )
            key_positions = torch.arange(sequence_count, device=sequence.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            sequence_mask = torch.zeros(
                (retained_sequence_count, sequence_count),
                dtype=sequence.dtype,
                device=sequence.device,
            )
            sequence_mask.masked_fill_(~allowed, float("-inf"))
            sequence_mask = sequence_mask[None, None].expand(
                batch_size,
                1,
                -1,
                -1,
            )
            sequence_mask = sequence_mask.masked_fill(
                ~sequence_valid_mask[:, None, None, :],
                float("-inf"),
            )
            attended_sequence = F.scaled_dot_product_attention(
                sequence_q[:, :, query_start:],
                sequence_k,
                sequence_v,
                attn_mask=sequence_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            attended_sequence = attended_sequence.transpose(1, 2).reshape(
                batch_size,
                retained_sequence_count,
                self.model_dim,
            )
            attended_sequence = self.output(attended_sequence)
        else:
            attended_sequence = sequence.new_empty((batch_size, 0, self.model_dim))

        flat_size = batch_size * candidate_count
        expanded_sequence_k = sequence_k[:, None].expand(
            -1,
            candidate_count,
            -1,
            -1,
            -1,
        ).reshape(flat_size, self.num_heads, sequence_count, self.head_dim)
        expanded_sequence_v = sequence_v[:, None].expand(
            -1,
            candidate_count,
            -1,
            -1,
            -1,
        ).reshape(flat_size, self.num_heads, sequence_count, self.head_dim)
        flat_ns_q = ns_q.reshape(flat_size, self.num_heads, ns_count, self.head_dim)
        flat_ns_k = ns_k.reshape(flat_size, self.num_heads, ns_count, self.head_dim)
        flat_ns_v = ns_v.reshape(flat_size, self.num_heads, ns_count, self.head_dim)
        combined_k = torch.cat((expanded_sequence_k, flat_ns_k), dim=2)
        combined_v = torch.cat((expanded_sequence_v, flat_ns_v), dim=2)

        ns_causal = torch.tril(
            torch.ones((ns_count, ns_count), dtype=torch.bool, device=sequence.device)
        )
        allowed = torch.cat(
            (
                torch.ones(
                    (ns_count, sequence_count),
                    dtype=torch.bool,
                    device=sequence.device,
                ),
                ns_causal,
            ),
            dim=1,
        )
        ns_mask = torch.zeros(
            (ns_count, sequence_count + ns_count),
            dtype=sequence.dtype,
            device=sequence.device,
        )
        ns_mask.masked_fill_(~allowed, float("-inf"))
        key_valid = torch.cat(
            (
                sequence_valid_mask[:, None].expand(-1, candidate_count, -1).reshape(
                    flat_size,
                    sequence_count,
                ),
                torch.ones(
                    (flat_size, ns_count),
                    dtype=torch.bool,
                    device=sequence.device,
                ),
            ),
            dim=1,
        )
        ns_mask = ns_mask[None, None].expand(flat_size, 1, -1, -1)
        ns_mask = ns_mask.masked_fill(~key_valid[:, None, None, :], float("-inf"))
        attended_ns = F.scaled_dot_product_attention(
            flat_ns_q,
            combined_k,
            combined_v,
            attn_mask=ns_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attended_ns = attended_ns.transpose(1, 2).reshape(
            batch_size,
            candidate_count,
            ns_count,
            self.model_dim,
        )
        return attended_sequence, self.output(attended_ns)


class OneTransBlock(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_ns_tokens: int,
        dropout: float,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(model_dim, eps=rms_norm_eps)
        self.attention = MixedCausalAttention(
            model_dim,
            num_heads,
            num_ns_tokens,
            dropout,
        )
        self.ffn_norm = nn.RMSNorm(model_dim, eps=rms_norm_eps)
        self.ffn = MixedFFN(model_dim, ffn_dim, num_ns_tokens)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        query_start: int,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = tokens[:, query_start:]
        attended = self.attention(
            self.attention_norm(tokens),
            query_start=query_start,
            valid_mask=valid_mask,
        )
        tokens = residual + self.dropout(attended)
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))

    def forward_shared_candidates(
        self,
        sequence: torch.Tensor,
        non_sequence: torch.Tensor,
        *,
        retained_sequence_count: int,
        sequence_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_sequence = self.attention_norm(sequence)
        normalized_ns = self.attention_norm(non_sequence)
        attended_sequence, attended_ns = self.attention.forward_shared_candidates(
            normalized_sequence,
            normalized_ns,
            retained_sequence_count=retained_sequence_count,
            sequence_valid_mask=sequence_valid_mask,
        )
        query_start = sequence.shape[1] - retained_sequence_count
        sequence = sequence[:, query_start:] + self.dropout(attended_sequence)
        non_sequence = non_sequence + self.dropout(attended_ns)
        sequence = sequence + self.dropout(
            self.ffn.forward_sequence(self.ffn_norm(sequence))
        )
        non_sequence = non_sequence + self.dropout(
            self.ffn.forward_non_sequence(self.ffn_norm(non_sequence))
        )
        return sequence, non_sequence, sequence_valid_mask[:, query_start:]
