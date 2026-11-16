from __future__ import annotations

import torch
from torch import nn


class DotInteraction(nn.Module):
    """DLRM field interaction by pairwise dot products."""

    def __init__(self, num_fields: int, *, include_self: bool = False) -> None:
        super().__init__()
        self.num_fields = int(num_fields)
        self.include_self = bool(include_self)
        offset = 0 if self.include_self else -1
        indices = torch.tril_indices(
            self.num_fields,
            self.num_fields,
            offset=offset,
        )
        self.register_buffer("row_indices", indices[0], persistent=False)
        self.register_buffer("column_indices", indices[1], persistent=False)

    @property
    def output_dim(self) -> int:
        return int(self.row_indices.numel())

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        gram = torch.matmul(fields, fields.transpose(-1, -2))
        return gram[..., self.row_indices, self.column_indices]
