from __future__ import annotations

import torch

from utils.compiler.config import DTypeName


TORCH_DTYPES: dict[DTypeName, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def torch_dtype(dtype: DTypeName) -> torch.dtype:
    return TORCH_DTYPES[dtype]
