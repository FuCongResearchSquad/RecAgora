"""Reusable PyTorch building blocks for the OneRank model family.

These layers deliberately consume dense token tensors rather than data-loader
objects. Feature lookup and model-specific token assembly live in the input
adapter; all configured architectures share the attention, masking, and scoring
semantics defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F


_MASK_VALUE = -1.0e9


def _truncated_normal_(tensor: torch.Tensor, std: float) -> None:
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


def _init_linear(linear: nn.Linear, *, std: float) -> None:
    _truncated_normal_(linear.weight, std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class RMSNorm(nn.Module):
    """TensorFlow OneRank-compatible RMSNorm with a learnable scale only."""

    def __init__(self, dimension: int, *, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        return value * rms * self.weight.to(dtype=value.dtype)


def causal_attention_mask(
    sequence_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive lower-triangular attention mask, shape ``[1,1,S,S]``."""
    allowed = torch.ones(
        (sequence_length, sequence_length), device=device, dtype=torch.bool
    ).tril()
    return torch.where(
        allowed,
        torch.zeros((), device=device, dtype=dtype),
        torch.full((), _MASK_VALUE, device=device, dtype=dtype),
    ).reshape(1, 1, sequence_length, sequence_length)


def _additive_attention_mask(
    attn_mask: torch.Tensor | None,
    *,
    key_padding_mask: torch.Tensor | None,
    batch_size: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Combine source-style additive masks with a ``True == valid`` key mask."""
    mask: torch.Tensor | None = None
    if key_padding_mask is not None:
        valid = key_padding_mask.to(device=device, dtype=torch.bool)
        mask = torch.where(
            valid[:, None, None, :],
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), _MASK_VALUE, device=device, dtype=dtype),
        )
    if attn_mask is None:
        return mask

    candidate = attn_mask.to(device=device)
    if candidate.dtype == torch.bool:
        candidate = torch.where(
            candidate,
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), _MASK_VALUE, device=device, dtype=dtype),
        )
    else:
        candidate = candidate.to(dtype=dtype)
    if candidate.ndim == 2:
        candidate = candidate[None, None, :, :]
    elif candidate.ndim == 3:
        candidate = candidate[:, None, :, :]
    return candidate if mask is None else mask + candidate


@dataclass(frozen=True)
class AttentionInfo:
    pre_softmax: torch.Tensor
    post_softmax: torch.Tensor


class TimeOrderRoPE(nn.Module):
    """OneRank split-head, half-rotation time/order RoPE.

    The first ``num_time_heads`` use timestamp gaps; remaining heads use the
    source model's reversed order ids.  This intentionally is not the common
    interleaved RoPE convention.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        num_heads: int,
        num_time_heads: int,
        time_method: Literal["base", "wavelength"] = "base",
        time_theta_base: float = 10000.0,
        time_min_wavelength: float = 1.0,
        time_max_wavelength: float = 1008.0,
        order_method: Literal["base", "wavelength"] = "base",
        order_theta_base: float = 10000.0,
        order_min_wavelength: float = 1.0,
        order_max_wavelength: float = 1008.0,
        time_head_scales: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if num_heads <= 0 or not 0 <= num_time_heads <= num_heads:
            raise ValueError("num_time_heads must be in [0, num_heads]")
        if time_head_scales is not None and len(time_head_scales) != num_time_heads:
            raise ValueError("time_head_scales length must equal num_time_heads")
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.num_time_heads = int(num_time_heads)
        self.register_buffer(
            "time_freqs",
            _rope_frequencies(
                head_dim,
                method=time_method,
                theta_base=time_theta_base,
                min_wavelength=time_min_wavelength,
                max_wavelength=time_max_wavelength,
            ),
            persistent=False,
        )
        self.register_buffer(
            "order_freqs",
            _rope_frequencies(
                head_dim,
                method=order_method,
                theta_base=order_theta_base,
                min_wavelength=order_min_wavelength,
                max_wavelength=order_max_wavelength,
            ),
            persistent=False,
        )
        scales = time_head_scales or (1.0,) * num_time_heads
        self.register_buffer(
            "time_head_scales",
            torch.tensor(scales, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_rope(query, timestamps), self._apply_rope(key, timestamps)

    def _apply_rope(
        self, value: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        batch_size, _, sequence_length, _ = value.shape
        pieces: list[torch.Tensor] = []
        if self.num_time_heads:
            angles = (
                timestamps.to(dtype=value.dtype)[:, None, :, None]
                * self.time_head_scales.to(dtype=value.dtype)[None, :, None, None]
                * self.time_freqs.to(dtype=value.dtype)[None, None, None, :]
            )
            pieces.append(_half_rotate(value[:, : self.num_time_heads], angles))
        if self.num_time_heads < self.num_heads:
            positions = torch.arange(
                sequence_length - 1,
                -1,
                -1,
                device=value.device,
                dtype=value.dtype,
            )
            angles = positions[None, None, :, None] * self.order_freqs.to(
                dtype=value.dtype
            )[None, None, None, :]
            angles = angles.expand(
                batch_size, self.num_heads - self.num_time_heads, -1, -1
            )
            pieces.append(_half_rotate(value[:, self.num_time_heads :], angles))
        return torch.cat(pieces, dim=1)


def _rope_frequencies(
    head_dim: int,
    *,
    method: str,
    theta_base: float,
    min_wavelength: float,
    max_wavelength: float,
) -> torch.Tensor:
    half_dim = head_dim // 2
    if method == "base":
        index = torch.arange(half_dim, dtype=torch.float32)
        return float(theta_base) ** (-2.0 * index / float(head_dim))
    if method == "wavelength":
        if min_wavelength <= 0.0 or max_wavelength <= 0.0:
            raise ValueError("RoPE wavelengths must be positive")
        omega_max = 2.0 * math.pi / float(min_wavelength)
        omega_min = 2.0 * math.pi / float(max_wavelength)
        return torch.exp(
            torch.linspace(math.log(omega_max), math.log(omega_min), half_dim)
        )
    raise ValueError("RoPE method must be 'base' or 'wavelength'")


def _half_rotate(value: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    first, second = value[..., :half], value[..., half:]
    cosine, sine = angles.cos(), angles.sin()
    return torch.cat(
        [first * cosine - second * sine, first * sine + second * cosine], dim=-1
    )


class MultiHeadSelfAttention(nn.Module):
    """Bias-free multi-head self-attention with configurable QKV layout."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        qkv_layout: Literal["fused", "separate"] = "fused",
        use_qk_norm: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.head_dim = dimension // num_heads
        self.qkv_layout = qkv_layout
        self.temperature = float(temperature)
        if qkv_layout == "fused":
            self.qkv_proj = nn.Linear(dimension, 3 * dimension, bias=False)
            _init_linear(self.qkv_proj, std=0.02)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv_proj = None
            self.q_proj = nn.Linear(dimension, dimension, bias=False)
            self.k_proj = nn.Linear(dimension, dimension, bias=False)
            self.v_proj = nn.Linear(dimension, dimension, bias=False)
            for projection in (self.q_proj, self.k_proj, self.v_proj):
                _init_linear(projection, std=0.02)
        self.out_proj = nn.Linear(dimension, dimension, bias=False)
        _init_linear(self.out_proj, std=0.02 / math.sqrt(4.0))
        self.q_norm = RMSNorm(self.head_dim) if use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_qk_norm else None

    def projections(self) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        return self.q_proj, self.k_proj, self.v_proj  # type: ignore[return-value]

    def forward(
        self,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        rope: TimeOrderRoPE | None = None,
        timestamps: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionInfo]:
        batch_size, sequence_length, _ = value.shape
        if self.qkv_proj is not None:
            query, key, item_value = self.qkv_proj(value).chunk(3, dim=-1)
        else:
            q_proj, k_proj, v_proj = self.projections()
            query, key, item_value = q_proj(value), k_proj(value), v_proj(value)
        query = _split_heads(query, self.num_heads)
        key = _split_heads(key, self.num_heads)
        item_value = _split_heads(item_value, self.num_heads)
        if self.q_norm is not None and self.k_norm is not None:
            query, key = self.q_norm(query), self.k_norm(key)
        if rope is not None:
            query, key = rope(query, key, timestamps=timestamps)
        output, info = _attention(
            query,
            key,
            item_value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            temperature=self.temperature,
        )
        output = self.out_proj(_merge_heads(output))
        return (output, info) if return_attention else output


class MultiHeadCrossAttention(nn.Module):
    """Bias-free multi-head cross-attention used by OneRank decoders."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.head_dim = dimension // num_heads
        self.q_proj = nn.Linear(dimension, dimension, bias=False)
        self.k_proj = nn.Linear(dimension, dimension, bias=False)
        self.v_proj = nn.Linear(dimension, dimension, bias=False)
        self.out_proj = nn.Linear(dimension, dimension, bias=False)
        for projection in (self.q_proj, self.k_proj, self.v_proj):
            _init_linear(projection, std=0.02)
        _init_linear(self.out_proj, std=0.02 / math.sqrt(4.0))
        self.q_norm = RMSNorm(self.head_dim) if use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_qk_norm else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionInfo]:
        if value is None:
            value = key
        query_heads = _split_heads(self.q_proj(query), self.num_heads)
        key_heads = _split_heads(self.k_proj(key), self.num_heads)
        value_heads = _split_heads(self.v_proj(value), self.num_heads)
        if self.q_norm is not None and self.k_norm is not None:
            query_heads, key_heads = self.q_norm(query_heads), self.k_norm(key_heads)
        output, info = _attention(
            query_heads,
            key_heads,
            value_heads,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            temperature=1.0,
        )
        output = self.out_proj(_merge_heads(output))
        return (output, info) if return_attention else output


class LoRACrossAttention(nn.Module):
    """Cross-attention with a detached externally-owned base Q/K/V projection.

    The base projections are passed at call time, so encoder modules remain the
    sole owners of their parameters and do not appear twice in ``state_dict``.
    """

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        rank: int,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.num_heads = int(num_heads)
        self.head_dim = dimension // num_heads
        self.q_lora = _LoRAProjection(dimension, rank)
        self.k_lora = _LoRAProjection(dimension, rank)
        self.v_lora = _LoRAProjection(dimension, rank)
        self.out_proj = nn.Linear(dimension, dimension, bias=False)
        _init_linear(self.out_proj, std=0.02 / math.sqrt(4.0))
        self.q_norm = RMSNorm(self.head_dim) if use_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if use_qk_norm else None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None,
        *,
        base_projections: tuple[nn.Linear, nn.Linear, nn.Linear],
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionInfo]:
        if value is None:
            value = key
        base_q, base_k, base_v = base_projections
        query_heads = _split_heads(self.q_lora(query, base_q), self.num_heads)
        key_heads = _split_heads(self.k_lora(key, base_k), self.num_heads)
        value_heads = _split_heads(self.v_lora(value, base_v), self.num_heads)
        if self.q_norm is not None and self.k_norm is not None:
            query_heads, key_heads = self.q_norm(query_heads), self.k_norm(key_heads)
        output, info = _attention(
            query_heads,
            key_heads,
            value_heads,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            temperature=1.0,
        )
        output = self.out_proj(_merge_heads(output))
        return (output, info) if return_attention else output


class _LoRAProjection(nn.Module):
    def __init__(self, dimension: int, rank: int) -> None:
        super().__init__()
        self.a = nn.Linear(dimension, rank, bias=False)
        self.b = nn.Linear(rank, dimension, bias=False)
        _init_linear(self.a, std=0.02)
        nn.init.zeros_(self.b.weight)

    def forward(self, value: torch.Tensor, base: nn.Linear) -> torch.Tensor:
        # TensorFlow source: ``stop_gradient(base_projection(value))``.  Detaching
        # only the weight would still let the base path differentiate with
        # respect to ``value`` in PyTorch, which is not source-equivalent.
        base_value = F.linear(
            value,
            base.weight.detach(),
            None if base.bias is None else base.bias.detach(),
        ).detach()
        return base_value + self.b(self.a(value))


def _split_heads(value: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, sequence_length, dimension = value.shape
    return value.reshape(batch_size, sequence_length, num_heads, dimension // num_heads).transpose(1, 2)


def _merge_heads(value: torch.Tensor) -> torch.Tensor:
    batch_size, _, sequence_length, head_dim = value.shape
    return value.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, -1)


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    key_padding_mask: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    temperature: float,
) -> tuple[torch.Tensor, AttentionInfo]:
    batch_size, _, query_length, head_dim = query.shape
    key_length = key.shape[2]
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    additive_mask = _additive_attention_mask(
        attn_mask,
        key_padding_mask=key_padding_mask,
        batch_size=batch_size,
        query_length=query_length,
        key_length=key_length,
        device=scores.device,
        dtype=scores.dtype,
    )
    if additive_mask is not None:
        scores = scores + additive_mask
    pre_softmax = scores / float(temperature)
    weights = torch.softmax(pre_softmax, dim=-1)
    return torch.matmul(weights, value), AttentionInfo(pre_softmax, weights)


class SwiGLU(nn.Module):
    def __init__(self, dimension: int, *, expansion: float = 4.0) -> None:
        super().__init__()
        self.dimension = int(dimension)
        self.intermediate_dim = int(expansion * dimension * 2.0 / 3.0)
        self.gate_value_proj = nn.Linear(dimension, 2 * self.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(self.intermediate_dim, dimension, bias=False)
        _init_linear(self.gate_value_proj, std=0.02)
        _init_linear(self.down_proj, std=0.02 / math.sqrt(4.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, inner = self.gate_value_proj(value).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * inner)


class ConvSwiGLU(SwiGLU):
    """Optional causal depthwise-convolution SwiGLU variant used by experiments."""

    def __init__(self, dimension: int, *, expansion: float = 4.0, kernel_size: int = 2) -> None:
        super().__init__(dimension, expansion=expansion)
        self.kernel_size = int(kernel_size)
        self.depthwise = nn.Conv1d(
            self.intermediate_dim,
            self.intermediate_dim,
            kernel_size,
            groups=self.intermediate_dim,
            bias=False,
        )
        # ConvSwiGLU has a distinct source initializer from ordinary SwiGLU.
        _truncated_normal_(
            self.gate_value_proj.weight,
            1.0 / math.sqrt(float(dimension)),
        )
        _truncated_normal_(self.depthwise.weight, 1.0 / math.sqrt(kernel_size))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, inner = self.gate_value_proj(value).chunk(2, dim=-1)
        activated = F.silu(gate) * inner
        convolved = self.depthwise(F.pad(activated.transpose(1, 2), (self.kernel_size - 1, 0)))
        return self.down_proj(F.silu(convolved.transpose(1, 2)))


class FeedForward(nn.Module):
    def __init__(
        self,
        dimension: int,
        *,
        expansion: float = 4.0,
        kind: Literal["standard", "swiglu", "convswiglu"] = "swiglu",
        activation: Literal["relu", "gelu"] = "relu",
    ) -> None:
        super().__init__()
        if kind == "swiglu":
            self.impl: nn.Module = SwiGLU(dimension, expansion=expansion)
        elif kind == "convswiglu":
            self.impl = ConvSwiGLU(dimension, expansion=expansion)
        elif kind == "standard":
            hidden = int(dimension * expansion)
            layers: list[nn.Module] = [nn.Linear(dimension, hidden, bias=False)]
            layers.append(nn.ReLU() if activation == "relu" else nn.GELU())
            layers.append(nn.Linear(hidden, dimension, bias=False))
            self.impl = nn.Sequential(*layers)
            _init_linear(self.impl[0], std=0.02)  # type: ignore[index]
            _init_linear(self.impl[-1], std=0.02 / math.sqrt(4.0))  # type: ignore[index]
        else:
            self.impl = ConvSwiGLU(dimension, expansion=expansion)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.impl(value)


class TransformerEncoderBlock(nn.Module):
    """OneRank pre-norm self-attention + FFN residual block."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        qkv_layout: Literal["fused", "separate"] = "fused",
        use_qk_norm: bool = True,
        ffn_kind: Literal["standard", "swiglu", "convswiglu"] = "swiglu",
        ffn_expansion: float = 4.0,
        rms_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dimension, eps=rms_norm_eps)
        self.attention = MultiHeadSelfAttention(
            dimension,
            num_heads,
            qkv_layout=qkv_layout,
            use_qk_norm=use_qk_norm,
        )
        self.norm2 = RMSNorm(dimension, eps=rms_norm_eps)
        self.ffn = FeedForward(dimension, expansion=ffn_expansion, kind=ffn_kind)

    def forward(
        self,
        value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        rope: TimeOrderRoPE | None = None,
        timestamps: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionInfo]:
        attention_result = self.attention(
            self.norm1(value),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
            timestamps=timestamps,
            return_attention=return_attention,
        )
        if return_attention:
            attention_out, info = attention_result  # type: ignore[misc]
        else:
            attention_out = attention_result  # type: ignore[assignment]
        inner = value + attention_out
        result = inner + self.ffn(self.norm2(inner))
        return (result, info) if return_attention else result


class TransformerCrossAttentionBlock(nn.Module):
    """OneRank pre-norm cross-attention + FFN residual block."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        use_qk_norm: bool = True,
        ffn_kind: Literal["standard", "swiglu", "convswiglu"] = "swiglu",
        ffn_expansion: float = 4.0,
        rms_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dimension, eps=rms_norm_eps)
        self.cross_attention = MultiHeadCrossAttention(
            dimension, num_heads, use_qk_norm=use_qk_norm
        )
        self.norm2 = RMSNorm(dimension, eps=rms_norm_eps)
        self.ffn = FeedForward(dimension, expansion=ffn_expansion, kind=ffn_kind)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionInfo]:
        if value is None:
            value = key
        attention_result = self.cross_attention(
            self.norm1(query),
            self.norm1(key),
            self.norm1(value),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            return_attention=return_attention,
        )
        if return_attention:
            attention_out, info = attention_result  # type: ignore[misc]
        else:
            attention_out = attention_result  # type: ignore[assignment]
        inner = query + attention_out
        result = inner + self.ffn(self.norm2(inner))
        return (result, info) if return_attention else result


class PerceiverBlock(nn.Module):
    """OneRank Perceiver block: cross attention then candidate self-attention.

    This intentionally does *not* reuse :class:`TransformerCrossAttentionBlock`:
    the TensorFlow model owns five distinct RMSNorm instances, including a
    shared key/value norm for the cross-attention path.  Keeping those
    parameters separate is necessary for a checkpoint-aligned implementation.
    """

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        use_qk_norm: bool = True,
        ffn_kind: Literal["standard", "swiglu", "convswiglu"] = "swiglu",
        ffn_expansion: float = 4.0,
        rms_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.cross_query_norm = RMSNorm(dimension, eps=rms_norm_eps)
        self.cross_key_value_norm = RMSNorm(dimension, eps=rms_norm_eps)
        self.cross_attention = MultiHeadCrossAttention(
            dimension, num_heads, use_qk_norm=use_qk_norm
        )
        self.cross_ffn_norm = RMSNorm(dimension, eps=rms_norm_eps)
        self.cross_ffn = FeedForward(
            dimension,
            expansion=ffn_expansion,
            kind=ffn_kind,
        )
        self.self_norm = RMSNorm(dimension, eps=rms_norm_eps)
        self.self_attention = MultiHeadSelfAttention(
            dimension,
            num_heads,
            qkv_layout="separate",
            use_qk_norm=use_qk_norm,
        )
        self.self_ffn_norm = RMSNorm(dimension, eps=rms_norm_eps)
        self.self_ffn = FeedForward(
            dimension,
            expansion=ffn_expansion,
            kind=ffn_kind,
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        self_key_padding_mask: torch.Tensor | None = None,
        self_attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, AttentionInfo]]:
        cross_result = self.cross_attention(
            self.cross_query_norm(query),
            self.cross_key_value_norm(key_value),
            self.cross_key_value_norm(key_value),
            key_padding_mask=key_padding_mask,
            return_attention=return_attention,
        )
        if return_attention:
            cross_output, cross_info = cross_result  # type: ignore[misc]
        else:
            cross_output = cross_result  # type: ignore[assignment]
        cross_inner = query + cross_output
        cross_out = cross_inner + self.cross_ffn(self.cross_ffn_norm(cross_inner))

        self_result = self.self_attention(
            self.self_norm(cross_out),
            key_padding_mask=self_key_padding_mask,
            attn_mask=self_attention_mask,
            return_attention=return_attention,
        )
        if return_attention:
            self_output, self_info = self_result  # type: ignore[misc]
        else:
            self_output = self_result  # type: ignore[assignment]
        self_inner = cross_out + self_output
        result = self_inner + self.self_ffn(self.self_ffn_norm(self_inner))
        if return_attention:
            return result, {"cross_attention": cross_info, "self_attention": self_info}
        return result


class LoRATransformerCrossAttentionBlock(nn.Module):
    """Cross-attention block whose Q/K/V base comes from a source encoder layer."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        *,
        lora_rank: int,
        use_qk_norm: bool = True,
        ffn_kind: Literal["standard", "swiglu", "convswiglu"] = "swiglu",
        ffn_expansion: float = 4.0,
        rms_norm_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dimension, eps=rms_norm_eps)
        self.cross_attention = LoRACrossAttention(
            dimension, num_heads, rank=lora_rank, use_qk_norm=use_qk_norm
        )
        self.norm2 = RMSNorm(dimension, eps=rms_norm_eps)
        self.ffn = FeedForward(dimension, expansion=ffn_expansion, kind=ffn_kind)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None,
        *,
        base_projections: tuple[nn.Linear, nn.Linear, nn.Linear],
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionInfo]:
        if value is None:
            value = key
        attention_result = self.cross_attention(
            self.norm1(query),
            self.norm1(key),
            self.norm1(value),
            base_projections=base_projections,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            return_attention=return_attention,
        )
        if return_attention:
            attention_out, info = attention_result  # type: ignore[misc]
        else:
            attention_out = attention_result  # type: ignore[assignment]
        inner = query + attention_out
        result = inner + self.ffn(self.norm2(inner))
        return (result, info) if return_attention else result


def listwise_softmax_cross_entropy(
    labels: torch.Tensor,
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """OneRank's ``-sum(label * log_softmax(logits, list_dim))`` objective."""
    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
    values = -(
        labels.to(dtype=logits.dtype) * F.log_softmax(logits, dim=1)
    ).sum(dim=1)
    return values.sum()
