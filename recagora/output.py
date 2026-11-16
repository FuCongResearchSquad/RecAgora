from __future__ import annotations

from typing import Any, TypedDict

import torch


class RankingOutput(TypedDict):
    """Static typing for a runtime object that remains a plain built-in dict."""

    logits: dict[str, torch.Tensor]
    aux: dict[str, Any]


def structured_ranking_output(
    logits: dict[str, torch.Tensor],
    *,
    aux: dict[str, Any] | None = None,
) -> RankingOutput:
    """Build the public ranking output without introducing custom carriers."""
    if type(logits) is not dict:
        raise TypeError("ranking logits must be a built-in dict")
    if aux is not None and type(aux) is not dict:
        raise TypeError("ranking aux must be a built-in dict")
    return {"logits": logits, "aux": {} if aux is None else aux}
