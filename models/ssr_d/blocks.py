from __future__ import annotations

import torch
from torch import nn


class IterativeCompetitiveInhibition(nn.Module):
    """Dynamic ICS from SSR-D with per-iteration thresholds."""

    def __init__(
        self,
        *,
        filter_dim: int,
        iterations: int,
        alpha_init: float,
        scale_init: float,
    ) -> None:
        super().__init__()
        self.iterations = int(iterations)
        self.alpha = nn.Parameter(
            torch.full((self.iterations,), float(alpha_init), dtype=torch.float32)
        )
        self.scale = nn.Parameter(
            torch.full((int(filter_dim),), float(scale_init), dtype=torch.float32)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        inhibited = torch.relu(values)
        for iteration in range(self.iterations):
            mean = inhibited.mean(dim=-1, keepdim=True)
            inhibited = torch.relu(inhibited - self.alpha[iteration] * mean)
        return inhibited * self.scale


class SSRDynamicView(nn.Module):
    """One paper-faithful dynamic structural-sparsity view."""

    def __init__(
        self,
        *,
        input_dim: int,
        filter_dim: int,
        view_dim: int,
        iterations: int,
        alpha_init: float,
        scale_init: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(int(input_dim), int(filter_dim), bias=False)
        self.inhibition = IterativeCompetitiveInhibition(
            filter_dim=int(filter_dim),
            iterations=int(iterations),
            alpha_init=float(alpha_init),
            scale_init=float(scale_init),
        )
        self.fusion = nn.Linear(int(filter_dim), int(view_dim))
        self.normalization = nn.LayerNorm(
            int(view_dim),
            eps=float(layer_norm_eps),
        )
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        filters = self.inhibition(self.projection(inputs))
        return self.normalization(self.fusion(filters))


class SSRDynamicBlock(nn.Module):
    """Independent SSR-D views over the same dense input."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_views: int,
        filter_dim: int,
        view_dim: int,
        iterations: int,
        alpha_init: float,
        scale_init: float,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_views = int(num_views)
        self.filter_dim = int(filter_dim)
        self.view_dim = int(view_dim)
        self.views = nn.ModuleList(
            SSRDynamicView(
                input_dim=self.input_dim,
                filter_dim=self.filter_dim,
                view_dim=self.view_dim,
                iterations=int(iterations),
                alpha_init=float(alpha_init),
                scale_init=float(scale_init),
                layer_norm_eps=float(layer_norm_eps),
            )
            for _ in range(self.num_views)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.stack([view(inputs) for view in self.views], dim=1)
