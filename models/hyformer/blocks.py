from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DenseFeedForward(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(model_dim * expansion_factor)
        self.input = nn.Linear(model_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.input, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(F.gelu(self.input(value))))


class PaddedTokenMixer(nn.Module):
    """Parameter-free RankMixer permutation with non-divisible widths."""

    def __init__(self, *, num_tokens: int, model_dim: int) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.model_dim = int(model_dim)
        self.pad_dim = (-self.model_dim) % self.num_tokens
        self.padded_dim = self.model_dim + self.pad_dim
        self.head_dim = self.padded_dim // self.num_tokens
        valid = torch.arange(self.padded_dim) < self.model_dim
        valid = valid.expand(self.num_tokens, -1)
        valid = (
            valid.reshape(self.num_tokens, self.num_tokens, self.head_dim)
            .transpose(0, 1)
            .reshape(-1)
        )
        self.register_buffer(
            "valid_index",
            valid.nonzero(as_tuple=False).flatten(),
            persistent=False,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_tokens, self.model_dim):
            raise ValueError(
                "HyFormer token mixer expected trailing shape "
                f"({self.num_tokens}, {self.model_dim}), got {tokens.shape[-2:]}"
            )
        if self.pad_dim:
            tokens = F.pad(tokens, (0, self.pad_dim))
        prefix = tokens.shape[:-2]
        mixed = (
            tokens.reshape(
                *prefix,
                self.num_tokens,
                self.num_tokens,
                self.head_dim,
            )
            .transpose(-3, -2)
            .contiguous()
            .reshape(*prefix, self.num_tokens, self.padded_dim)
        )
        selected = mixed.reshape(*prefix, self.num_tokens * self.padded_dim)
        selected = selected.index_select(-1, self.valid_index)
        return selected.reshape(*prefix, self.num_tokens, self.model_dim)


class PerTokenFeedForward(nn.Module):
    """Independent dense FFN parameters for every Global/NS token position."""

    def __init__(
        self,
        *,
        num_tokens: int,
        model_dim: int,
        expansion_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        hidden_dim = int(model_dim * expansion_factor)
        self.input_weight = nn.Parameter(
            torch.empty(self.num_tokens, model_dim, hidden_dim)
        )
        self.input_bias = nn.Parameter(torch.zeros(self.num_tokens, hidden_dim))
        self.output_weight = nn.Parameter(
            torch.empty(self.num_tokens, hidden_dim, model_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(self.num_tokens, model_dim))
        self.dropout = nn.Dropout(dropout)
        for weight in self.input_weight:
            nn.init.xavier_uniform_(weight)
        for weight in self.output_weight:
            nn.init.xavier_uniform_(weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum(
            "...td,tdh->...th",
            tokens,
            self.input_weight,
        ) + self.input_bias
        hidden = self.dropout(F.gelu(hidden))
        return torch.einsum(
            "...th,thd->...td",
            hidden,
            self.output_weight,
        ) + self.output_bias


class QueryBoosting(nn.Module):
    def __init__(
        self,
        *,
        num_tokens: int,
        model_dim: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.token_mixer = PaddedTokenMixer(
            num_tokens=num_tokens,
            model_dim=model_dim,
        )
        self.token_norm = nn.LayerNorm(model_dim, eps=norm_eps)
        self.per_token_ffn = PerTokenFeedForward(
            num_tokens=num_tokens,
            model_dim=model_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(model_dim, eps=norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.token_norm(
            tokens + self.dropout(self.token_mixer(tokens))
        )
        return self.ffn_norm(
            tokens + self.dropout(self.per_token_ffn(tokens))
        )


class HistorySequenceEncoderLayer(nn.Module):
    """Request-shared full-attention sequence representation encoder."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.attention_norm = nn.LayerNorm(model_dim, eps=norm_eps)
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.attention_output = nn.Linear(model_dim, model_dim, bias=False)
        self.ffn_norm = nn.LayerNorm(model_dim, eps=norm_eps)
        self.ffn = DenseFeedForward(
            model_dim=model_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = float(dropout)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.attention_output.weight)

    def forward(
        self,
        sequence: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid = history_valid.unsqueeze(-1)
        sequence = sequence.masked_fill(~valid, 0.0)
        normalized = self.attention_norm(sequence)
        batch_size, sequence_length = normalized.shape[:2]
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)

        def heads(item: torch.Tensor) -> torch.Tensor:
            return item.reshape(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        query, key, value = heads(query), heads(key), heads(value)
        safe_valid, empty = _safe_history_mask(history_valid)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=safe_valid[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attention = attention.masked_fill(
            empty[:, None, None, None],
            0.0,
        )
        attention = attention.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.model_dim,
        )
        sequence = sequence + self.dropout(self.attention_output(attention))
        sequence = sequence.masked_fill(~valid, 0.0)
        sequence = sequence + self.dropout(self.ffn(self.ffn_norm(sequence)))
        return sequence.masked_fill(~valid, 0.0)


class GlobalQueryDecoder(nn.Module):
    """Candidate-private Global Tokens decoding request-shared sequence K/V."""

    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.query_norm = nn.LayerNorm(model_dim, eps=norm_eps)
        self.sequence_norm = nn.LayerNorm(model_dim, eps=norm_eps)
        self.query_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.key_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.value_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.output_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = float(dropout)
        for layer in (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        global_tokens: torch.Tensor,
        sequence: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, candidate_count, global_count = global_tokens.shape[:3]
        sequence_length = int(sequence.shape[1])
        query = self.query_projection(self.query_norm(global_tokens))
        normalized_sequence = self.sequence_norm(sequence).masked_fill(
            ~history_valid.unsqueeze(-1),
            0.0,
        )
        key = self.key_projection(normalized_sequence)
        value = self.value_projection(normalized_sequence)
        query = query.reshape(
            batch_size,
            candidate_count,
            global_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 1, 3, 2, 4)
        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)[:, None]
        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)[:, None]
        safe_valid, empty = _safe_history_mask(history_valid)
        decoded = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=safe_valid[:, None, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        decoded = decoded.masked_fill(
            empty[:, None, None, None, None],
            0.0,
        )
        decoded = decoded.permute(0, 1, 3, 2, 4).reshape(
            batch_size,
            candidate_count,
            global_count,
            self.model_dim,
        )
        return global_tokens + self.dropout(self.output_projection(decoded))


class HyFormerLayer(nn.Module):
    def __init__(
        self,
        *,
        model_dim: int,
        num_heads: int,
        num_global_tokens: int,
        num_non_sequence_tokens: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_global_tokens = int(num_global_tokens)
        self.sequence_encoder = HistorySequenceEncoderLayer(
            model_dim=model_dim,
            num_heads=num_heads,
            expansion_factor=expansion_factor,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.query_decoder = GlobalQueryDecoder(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            norm_eps=norm_eps,
        )
        self.query_boosting = QueryBoosting(
            num_tokens=num_global_tokens + num_non_sequence_tokens,
            model_dim=model_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            norm_eps=norm_eps,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        global_tokens: torch.Tensor,
        non_sequence_tokens: torch.Tensor,
        *,
        history_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequence_encoder(sequence, history_valid)
        global_tokens = self.query_decoder(
            global_tokens,
            sequence,
            history_valid,
        )
        boosted = self.query_boosting(
            torch.cat((global_tokens, non_sequence_tokens), dim=2)
        )
        global_tokens = boosted[:, :, : self.num_global_tokens]
        non_sequence_tokens = boosted[:, :, self.num_global_tokens :]
        return sequence, global_tokens, non_sequence_tokens


def _safe_history_mask(
    history_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_valid = history_valid.clone()
    empty = ~safe_valid.any(dim=1)
    safe_valid[:, 0] |= empty
    return safe_valid, empty
