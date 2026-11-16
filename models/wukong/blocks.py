from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class OptimizedFactorizationMachine(nn.Module):
    """Compute ``X X^T Y`` as ``X (X^T Y)`` without an ``N x N`` matrix."""

    def __init__(
        self,
        *,
        num_input_tokens: int,
        interaction_dim: int,
        projection_rank: int,
    ) -> None:
        super().__init__()
        if min(num_input_tokens, interaction_dim, projection_rank) <= 0:
            raise ValueError("Wukong optimized-FM dimensions must be positive")
        self.num_input_tokens = int(num_input_tokens)
        self.interaction_dim = int(interaction_dim)
        self.projection_rank = int(projection_rank)
        self.projection = nn.Parameter(
            torch.empty(self.num_input_tokens, self.projection_rank)
        )
        nn.init.xavier_uniform_(self.projection)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_input_tokens,
            self.interaction_dim,
        ):
            raise ValueError(
                "Wukong optimized FM expects "
                f"[rows,{self.num_input_tokens},{self.interaction_dim}]"
            )
        projected = torch.matmul(tokens.transpose(1, 2), self.projection)
        return torch.matmul(tokens, projected)


class FactorizationMachineBlock(nn.Module):
    """Optimized FM followed by an MLP that emits new interaction tokens."""

    def __init__(
        self,
        *,
        num_input_tokens: int,
        interaction_dim: int,
        projection_rank: int,
        output_tokens: int,
        hidden_units: Sequence[int],
    ) -> None:
        super().__init__()
        if output_tokens <= 0:
            raise ValueError("Wukong FMB output_tokens must be positive")
        hidden_units = tuple(int(value) for value in hidden_units)
        if not hidden_units or any(value <= 0 for value in hidden_units):
            raise ValueError("Wukong FMB hidden_units must be non-empty and positive")
        self.num_input_tokens = int(num_input_tokens)
        self.interaction_dim = int(interaction_dim)
        self.output_tokens = int(output_tokens)
        self.fm = OptimizedFactorizationMachine(
            num_input_tokens=self.num_input_tokens,
            interaction_dim=self.interaction_dim,
            projection_rank=int(projection_rank),
        )
        fm_output_dim = self.num_input_tokens * int(projection_rank)
        self.interaction_norm = nn.LayerNorm(fm_output_dim)
        layers: list[nn.Module] = []
        previous = fm_output_dim
        for hidden in hidden_units:
            linear = nn.Linear(previous, hidden)
            _init_linear(linear)
            layers.extend((linear, nn.ReLU()))
            previous = hidden
        output = nn.Linear(previous, self.output_tokens * self.interaction_dim)
        _init_linear(output)
        layers.extend((output, nn.ReLU()))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        interactions = self.fm(tokens).flatten(start_dim=1)
        output = self.mlp(self.interaction_norm(interactions))
        return output.reshape(
            tokens.shape[0],
            self.output_tokens,
            self.interaction_dim,
        )


class LinearCompressionBlock(nn.Module):
    """Recombine whole field embeddings without increasing interaction order."""

    def __init__(
        self,
        *,
        num_input_tokens: int,
        output_tokens: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_input_tokens <= 0 or output_tokens <= 0:
            raise ValueError("Wukong LCB token counts must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Wukong LCB dropout must be in [0,1)")
        self.num_input_tokens = int(num_input_tokens)
        self.output_tokens = int(output_tokens)
        self.weight = nn.Parameter(
            torch.empty(self.output_tokens, self.num_input_tokens)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_tokens))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or int(tokens.shape[1]) != self.num_input_tokens:
            raise ValueError(
                f"Wukong LCB expects {self.num_input_tokens} input tokens"
            )
        output = torch.einsum("oi,bid->bod", self.weight, tokens)
        return self.dropout(output + self.bias[None, :, None])


class TokenResidualProjection(nn.Module):
    def __init__(self, *, input_tokens: int, output_tokens: int) -> None:
        super().__init__()
        self.input_tokens = int(input_tokens)
        self.output_tokens = int(output_tokens)
        if self.input_tokens == self.output_tokens:
            self.weight: nn.Parameter | None = None
        else:
            self.weight = nn.Parameter(
                torch.empty(self.output_tokens, self.input_tokens)
            )
            nn.init.xavier_uniform_(self.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if int(tokens.shape[1]) != self.input_tokens:
            raise ValueError("Wukong residual input token count changed")
        if self.weight is None:
            return tokens
        return torch.einsum("oi,bid->bod", self.weight, tokens)


class WukongLayer(nn.Module):
    def __init__(
        self,
        *,
        num_input_tokens: int,
        interaction_dim: int,
        projection_rank: int,
        fmb_output_tokens: int,
        lcb_output_tokens: int,
        fmb_hidden_units: Sequence[int],
        lcb_dropout: float,
    ) -> None:
        super().__init__()
        self.num_input_tokens = int(num_input_tokens)
        self.interaction_dim = int(interaction_dim)
        self.output_tokens = int(fmb_output_tokens) + int(lcb_output_tokens)
        self.fmb = FactorizationMachineBlock(
            num_input_tokens=self.num_input_tokens,
            interaction_dim=self.interaction_dim,
            projection_rank=int(projection_rank),
            output_tokens=int(fmb_output_tokens),
            hidden_units=fmb_hidden_units,
        )
        self.lcb = LinearCompressionBlock(
            num_input_tokens=self.num_input_tokens,
            output_tokens=int(lcb_output_tokens),
            dropout=float(lcb_dropout),
        )
        self.residual = TokenResidualProjection(
            input_tokens=self.num_input_tokens,
            output_tokens=self.output_tokens,
        )
        self.output_norm = nn.LayerNorm(
            self.output_tokens * self.interaction_dim
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output = torch.cat((self.fmb(tokens), self.lcb(tokens)), dim=1)
        output = output + self.residual(tokens)
        output = self.output_norm(output.flatten(start_dim=1))
        return output.reshape(
            tokens.shape[0],
            self.output_tokens,
            self.interaction_dim,
        )


class WukongBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_input_tokens: int,
        interaction_dim: int,
        num_layers: int,
        projection_rank: int,
        fmb_output_tokens: int,
        lcb_output_tokens: int,
        fmb_hidden_units: Sequence[int],
        lcb_dropout: float,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("Wukong num_layers must be positive")
        self.num_input_tokens = int(num_input_tokens)
        self.interaction_dim = int(interaction_dim)
        self.output_tokens = int(fmb_output_tokens) + int(lcb_output_tokens)
        self.activation_checkpointing = bool(activation_checkpointing)
        layers: list[WukongLayer] = []
        current_tokens = self.num_input_tokens
        for _ in range(int(num_layers)):
            layer = WukongLayer(
                num_input_tokens=current_tokens,
                interaction_dim=self.interaction_dim,
                projection_rank=int(projection_rank),
                fmb_output_tokens=int(fmb_output_tokens),
                lcb_output_tokens=int(lcb_output_tokens),
                fmb_hidden_units=fmb_hidden_units,
                lcb_dropout=float(lcb_dropout),
            )
            layers.append(layer)
            current_tokens = self.output_tokens
        self.layers = nn.ModuleList(layers)

    @property
    def output_dim(self) -> int:
        return self.output_tokens * self.interaction_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (
            self.num_input_tokens,
            self.interaction_dim,
        ):
            raise ValueError("Wukong backbone input shape does not match config")
        for layer in self.layers:
            if (
                self.activation_checkpointing
                and self.training
                and tokens.requires_grad
                and tokens.device.type == "cuda"
            ):
                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        return tokens


def _init_linear(layer: nn.Linear) -> None:
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
