from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ChunkTokenizer(nn.Module):
    """Split a fixed-width vector and tokenize each contiguous chunk privately."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_tokens: int,
        token_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if min(input_dim, num_tokens, token_dim) <= 0:
            raise ValueError("UniMixer chunk tokenizer dimensions must be positive")
        self.input_dim = int(input_dim)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.chunk_dim = math.ceil(self.input_dim / self.num_tokens)
        self.padded_input_dim = self.chunk_dim * self.num_tokens
        self.pad_dim = self.padded_input_dim - self.input_dim
        self.projections = nn.ModuleList(
            nn.Linear(self.chunk_dim, self.token_dim)
            for _ in range(self.num_tokens)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(self.token_dim, eps=float(norm_eps))
            for _ in range(self.num_tokens)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection, norm in zip(self.projections, self.norms, strict=True):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
            norm.reset_parameters()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.input_dim:
            raise ValueError(
                f"ChunkTokenizer expected input_dim={self.input_dim}, "
                f"got {values.shape[-1]}"
            )
        if self.pad_dim:
            values = F.pad(values, (0, self.pad_dim))
        leading = values.shape[:-1]
        chunks = values.reshape(-1, self.num_tokens, self.chunk_dim)
        tokens = torch.stack(
            tuple(
                norm(F.silu(projection(chunks[:, index])))
                for index, (projection, norm) in enumerate(
                    zip(self.projections, self.norms, strict=True)
                )
            ),
            dim=1,
        )
        return tokens.reshape(*leading, self.num_tokens, self.token_dim)


class PerTokenSwiGLU(nn.Module):
    """SwiGLU with independent parameters for every UniMixer token."""

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
            raise ValueError("UniMixer per-token SwiGLU dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("UniMixer per-token SwiGLU dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = self.token_dim * int(expansion_factor)
        self.gate_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.hidden_dim)
        )
        self.value_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim, self.hidden_dim)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.num_tokens, self.hidden_dim, self.token_dim)
        )
        self.gate_bias = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.value_bias = nn.Parameter(torch.zeros(self.num_tokens, self.hidden_dim))
        self.down_bias = nn.Parameter(torch.zeros(self.num_tokens, self.token_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token_index in range(self.num_tokens):
            nn.init.xavier_uniform_(self.gate_weight[token_index])
            nn.init.xavier_uniform_(self.value_weight[token_index])
            nn.init.xavier_uniform_(self.down_weight[token_index])
        nn.init.zeros_(self.gate_bias)
        nn.init.zeros_(self.value_bias)
        nn.init.zeros_(self.down_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError("PerTokenSwiGLU token shape does not match config")
        gate = torch.einsum(
            "...td,tdh->...th",
            tokens,
            self.gate_weight,
        ) + self.gate_bias
        value = torch.einsum(
            "...td,tdh->...th",
            tokens,
            self.value_weight,
        ) + self.value_bias
        hidden = self.dropout(F.silu(gate) * value)
        return torch.einsum(
            "...th,thd->...td",
            hidden,
            self.down_weight,
        ) + self.down_bias


class UniMixingLiteLayer(nn.Module):
    """UniMixing-Lite local feature mixing plus low-rank global token mixing."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_basis: int,
        global_rank: int,
        sinkhorn_iterations: int,
        sinkhorn_eps: float,
        tau: float,
        symmetric: bool,
        local_coefficient_softmax: bool,
        output_projection: bool,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if min(token_dim, num_tokens, num_basis, global_rank, sinkhorn_iterations) <= 0:
            raise ValueError("UniMixing-Lite dimensions and iterations must be positive")
        if float(sinkhorn_eps) <= 0.0 or float(tau) <= 0.0:
            raise ValueError("UniMixing-Lite sinkhorn_eps and tau must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("UniMixing-Lite dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.num_basis = int(num_basis)
        self.global_rank = int(global_rank)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.sinkhorn_eps = float(sinkhorn_eps)
        self.symmetric = bool(symmetric)
        self.local_coefficient_softmax = bool(local_coefficient_softmax)
        self.local_basis = nn.Parameter(
            torch.empty(self.num_basis, self.token_dim, self.token_dim)
        )
        self.local_coefficients = nn.Parameter(
            torch.empty(self.num_tokens, self.num_basis)
        )
        self.global_left = nn.Parameter(
            torch.empty(self.num_tokens, self.global_rank)
        )
        self.global_right = nn.Parameter(
            torch.empty(self.global_rank, self.num_tokens)
        )
        self.norm_x = nn.LayerNorm(self.token_dim, eps=float(norm_eps))
        self.norm_y = nn.LayerNorm(self.token_dim, eps=float(norm_eps))
        self.output_projection = (
            nn.Linear(self.token_dim, self.token_dim, bias=False)
            if output_projection
            else nn.Identity()
        )
        self.dropout = nn.Dropout(float(dropout))
        self.register_buffer("current_tau", torch.tensor(float(tau)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.local_basis)
        nn.init.xavier_uniform_(self.local_coefficients)
        nn.init.xavier_uniform_(self.global_left)
        nn.init.xavier_uniform_(self.global_right)
        if isinstance(self.output_projection, nn.Linear):
            nn.init.xavier_uniform_(self.output_projection.weight)
        self.norm_x.reset_parameters()
        self.norm_y.reset_parameters()

    def set_tau(self, tau: float) -> None:
        if float(tau) <= 0.0 or not math.isfinite(float(tau)):
            raise ValueError("UniMixing-Lite tau must be finite and positive")
        self.current_tau.fill_(float(tau))

    @torch.no_grad()
    def copy_tau(self, tau: torch.Tensor) -> None:
        """Copy a scalar device tensor without synchronizing it to the host."""

        if tau.numel() != 1:
            raise ValueError("UniMixing-Lite tau tensor must be scalar")
        self.current_tau.copy_(
            tau.detach().to(
                device=self.current_tau.device,
                dtype=self.current_tau.dtype,
            )
        )

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        output_dtype = logits.dtype
        values = logits.float() / self.current_tau.float().clamp_min(1.0e-6)
        values = values - values.amax(dim=(-1, -2), keepdim=True)
        matrix = values.exp()
        for _ in range(self.sinkhorn_iterations):
            matrix = matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(
                self.sinkhorn_eps
            )
            matrix = matrix / matrix.sum(dim=-2, keepdim=True).clamp_min(
                self.sinkhorn_eps
            )
        return matrix.to(dtype=output_dtype)

    def local_matrix(self) -> torch.Tensor:
        coefficients = self.local_coefficients
        if self.local_coefficient_softmax:
            coefficients = F.softmax(coefficients, dim=-1)
        logits = torch.einsum(
            "tk,kde->tde",
            coefficients,
            self.local_basis,
        )
        if self.symmetric:
            logits = 0.5 * (logits + logits.transpose(-1, -2))
        return self._sinkhorn(logits)

    def global_matrix(self) -> torch.Tensor:
        logits = torch.matmul(self.global_left, self.global_right)
        logits = logits / math.sqrt(self.global_rank)
        if self.symmetric:
            logits = 0.5 * (logits + logits.transpose(-1, -2))
        return self._sinkhorn(logits)

    def mixing_transform(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError("UniMixingLiteLayer token shape does not match config")
        locally_mixed = torch.einsum(
            "...td,tde->...te",
            tokens,
            self.local_matrix(),
        )
        globally_mixed = torch.einsum(
            "ts,...sd->...td",
            self.global_matrix(),
            locally_mixed,
        )
        return self.dropout(self.output_projection(globally_mixed))

    def forward(
        self,
        x_stream: torch.Tensor,
        y_stream: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update = self.mixing_transform(x_stream + self.norm_y(y_stream))
        return self.norm_x(x_stream + update), y_stream + update


class SiameseFeedForwardLayer(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        expansion_factor: int,
        dropout: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm_x = nn.LayerNorm(token_dim, eps=float(norm_eps))
        self.norm_y = nn.LayerNorm(token_dim, eps=float(norm_eps))
        self.feed_forward = PerTokenSwiGLU(
            token_dim=token_dim,
            num_tokens=num_tokens,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

    def forward(
        self,
        x_stream: torch.Tensor,
        y_stream: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        update = self.feed_forward(x_stream + self.norm_y(y_stream))
        return self.norm_x(x_stream + update), y_stream + update


class UniMixingLiteStack(nn.Module):
    def __init__(
        self,
        *,
        token_dim: int,
        num_tokens: int,
        num_layers: int,
        expansion_factor: int,
        num_basis: int,
        global_rank: int,
        sinkhorn_iterations: int,
        sinkhorn_eps: float,
        tau: float,
        symmetric: bool,
        local_coefficient_softmax: bool,
        output_projection: bool,
        dropout: float,
        norm_eps: float,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("UniMixer num_layers must be positive")
        self.activation_checkpointing = bool(activation_checkpointing)
        self.mixers = nn.ModuleList(
            UniMixingLiteLayer(
                token_dim=token_dim,
                num_tokens=num_tokens,
                num_basis=num_basis,
                global_rank=global_rank,
                sinkhorn_iterations=sinkhorn_iterations,
                sinkhorn_eps=sinkhorn_eps,
                tau=tau,
                symmetric=symmetric,
                local_coefficient_softmax=local_coefficient_softmax,
                output_projection=output_projection,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(int(num_layers))
        )
        self.feed_forwards = nn.ModuleList(
            SiameseFeedForwardLayer(
                token_dim=token_dim,
                num_tokens=num_tokens,
                expansion_factor=expansion_factor,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(int(num_layers))
        )
        self.final_norm = nn.LayerNorm(token_dim, eps=float(norm_eps))

    def set_tau(self, tau: float) -> None:
        for mixer in self.mixers:
            mixer.set_tau(tau)

    @torch.no_grad()
    def copy_tau(self, tau: torch.Tensor) -> None:
        for mixer in self.mixers:
            mixer.copy_tau(tau)

    @staticmethod
    def _run_pair(
        module: nn.Module,
        x_stream: torch.Tensor,
        y_stream: torch.Tensor,
        *,
        checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if checkpointing and x_stream.requires_grad:
            return checkpoint(module, x_stream, y_stream, use_reentrant=False)
        return module(x_stream, y_stream)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x_stream = tokens
        y_stream = tokens
        checkpointing = self.activation_checkpointing and self.training
        for mixer, feed_forward in zip(
            self.mixers,
            self.feed_forwards,
            strict=True,
        ):
            x_stream, y_stream = self._run_pair(
                mixer,
                x_stream,
                y_stream,
                checkpointing=checkpointing,
            )
            x_stream, y_stream = self._run_pair(
                feed_forward,
                x_stream,
                y_stream,
                checkpointing=checkpointing,
            )
        return x_stream + self.final_norm(y_stream)
