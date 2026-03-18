"""
CUDA kernel quantized inference modules for Wan2.2-Fun.

Provides real INT8 inference using viditq_extension CUDA kernels,
replacing the fake-quantization simulation with actual INT8 GEMM operations.

Requires:
    cd <ViDiT-Q>/kernels && pip install -e .

Architecture mapping (Wan → CUDA kernel):
    - WanAttentionBlock → WanAttentionBlockWithCudaKernel
    - WanSelfAttention q/k/v/o (nn.Linear) → W8A8OF16LinearDynamicInputScale
    - WanCrossAttention q/o → W8A8, k/v → FP16 nn.Linear (text context)
    - FFN (Sequential) → WanFFNWithCudaKernel (W8A8 + fused GELU)
    - WanLayerNorm + modulation → LayerNormGeneral (fused LN + modulate + quantize)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from viditq_extension.nn.base import QuantParams
from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
from viditq_extension.nn.layernorm import LayerNormGeneral
import viditq_extension.fused as fused_kernels

from qdiff.base.quant_layer import QuantizedLinear

logger = logging.getLogger(__name__)

# W8A8 GEMM tile size — M (total tokens) must be a multiple of this
W8A8_CTA_M = 128


def _qp_slices(quant_params, M):
    """Return (sum_input[:M], scale_input[:M]) views for exact-sized kernel calls."""
    return quant_params.sum_input[:M], quant_params.scale_input[:M]


def _w8a8_with_padding(linear, x, quant_params):
    """Call W8A8 linear, padding M to a multiple of CTA_M if needed.

    The W8A8 GEMM kernel requires:
      1. M (total tokens) is a multiple of W8A8_CTA_M (128)
      2. scale_input.shape == (M,) exactly

    This helper pads input + slices/pads quant_params to satisfy both
    constraints, then slices the output back.  Padding is local to the
    GEMM call and never leaks into attention or RoPE.
    """
    shape = x.shape
    M = x.view(-1, shape[-1]).shape[0]
    M_pad = ((M + W8A8_CTA_M - 1) // W8A8_CTA_M) * W8A8_CTA_M
    pad = M_pad - M

    # Pad input rows if needed: [M, C] → [M_pad, C]
    x_2d = x.view(M, shape[-1])
    if pad > 0:
        x_2d = F.pad(x_2d, (0, 0, 0, pad))

    # Build exact-sized QuantParams for the GEMM (scale_input must be (M_pad,))
    if pad > 0:
        scale = F.pad(quant_params.scale_input[:M], (0, pad))
        sum_inp = F.pad(quant_params.sum_input[:M], (0, pad)) \
            if quant_params.sum_input is not None else None
    else:
        # No row-padding, but still need exact shape
        scale = quant_params.scale_input[:M]
        sum_inp = quant_params.sum_input[:M] \
            if quant_params.sum_input is not None else None

    qp = QuantParams.__new__(QuantParams)
    qp.has_sum_input = quant_params.has_sum_input
    qp.scale_input = scale
    qp.sum_input = sum_inp

    out = linear(x_2d, qp)
    if pad > 0:
        out = out[:M]
    return out.view(*shape[:-1], out.shape[-1])


# ---------------------------------------------------------------------------
# WanRMSNorm fallback (in case the user's package isn't importable here)
# ---------------------------------------------------------------------------
try:
    from videox_fun.models.wan_transformer3d import WanRMSNorm
except ImportError:
    class WanRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Weight export helper
# ---------------------------------------------------------------------------
def quantize_and_save_weight_(submodule, full_name):
    """Convert a QuantizedLinear's weight from fake-quantized FP16 to real INT8."""
    fp_weight = submodule.fp_module.weight.to(torch.float16)

    submodule.w_quantizer.delta = submodule.w_quantizer.delta.view(-1).to(torch.float16)
    submodule.w_quantizer.zero_point = submodule.w_quantizer.zero_point.view(-1).to(torch.float16)
    scale = submodule.w_quantizer.delta
    zero_point = submodule.w_quantizer.zero_point

    int_weight = torch.clamp(
        torch.round(fp_weight / scale.view(-1, 1)) - zero_point.view(-1, 1),
        -128, 127
    ).to(torch.int8)
    submodule.weight.data = int_weight


# ---------------------------------------------------------------------------
# CUDA kernel sub-modules
# ---------------------------------------------------------------------------
class WanSelfAttentionWithCudaKernel(nn.Module):
    """Self-attention with W8A8 INT8 CUDA kernels for Q, K, V, O projections."""

    def __init__(self, dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, eps=1e-6, quant_params=None,
                 has_bias=True, weight_sym=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.quant_params = quant_params

        self.q = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.k = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.v = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.o = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs,
                attention_fn, rope_apply_fn, dtype=torch.bfloat16, t=0):
        """
        Args:
            x: INT8 tensor [B, L, C] from fused LayerNorm kernel.
               quant_params.scale_input/sum_input already filled.
            attention_fn: Wan's attention() function.
            rope_apply_fn: Wan's rope_apply_qk() function.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # W8A8: INT8 input → FP16 output (with GEMM M-padding)
        q = self.norm_q(_w8a8_with_padding(self.q, x, self.quant_params)).view(b, s, n, d)
        k = self.norm_k(_w8a8_with_padding(self.k, x, self.quant_params)).view(b, s, n, d)
        v = _w8a8_with_padding(self.v, x, self.quant_params).view(b, s, n, d)

        # RoPE in FP16, then attention in BF16.
        # BF16 has the same exponent range as FP32 (~3.4e38 vs FP16's ~65504),
        # preventing softmax overflow on large Q*K^T logits that cause the
        # "red tint / high-noise" artifacts observed with FP16 attention.
        q, k = rope_apply_fn(q, k, grid_sizes, freqs)
        x = attention_fn(q.to(torch.bfloat16), k.to(torch.bfloat16),
                         v=v.to(torch.bfloat16),
                         k_lens=seq_lens, window_size=self.window_size)
        # Convert back to FP16: subsequent quant_sum fused kernel requires FP16.
        x = x.to(torch.float16).flatten(2)

        # Quantize attention output for O projection
        M = x.shape[0] * x.shape[1]
        sum_s, scale_s = _qp_slices(self.quant_params, M)
        x = fused_kernels.quant_sum(x.contiguous(), sum_s, scale_s)
        x = _w8a8_with_padding(self.o, x, self.quant_params)
        return x


class WanCrossAttentionWithCudaKernel(nn.Module):
    """Cross-attention: Q and O use W8A8 (image), K and V use FP16 (text context)."""

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6,
                 quant_params=None, has_bias=True, weight_sym=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.quant_params = quant_params

        # Q and O: W8A8 (image features, quantized)
        self.q = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.o = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)

        # K and V: FP16 nn.Linear (text context, not per-token quantized)
        self.k = nn.Linear(dim, dim, bias=has_bias)
        self.v = nn.Linear(dim, dim, bias=has_bias)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens, attention_fn,
                dtype=torch.bfloat16, t=0):
        """
        Args:
            x: INT8 tensor [B, L1, C] (quantized image features).
            context: FP16 tensor [B, L2, C] (text embeddings).
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(_w8a8_with_padding(self.q, x, self.quant_params)).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(torch.float16))).view(b, -1, n, d)
        v = self.v(context.to(torch.float16)).view(b, -1, n, d)

        # Attention in BF16 for numerical stability (avoids FP16 softmax overflow).
        x = attention_fn(q.to(torch.bfloat16), k.to(torch.bfloat16),
                         v.to(torch.bfloat16), k_lens=context_lens)
        # Convert back to FP16: subsequent quant_sum fused kernel requires FP16.
        x = x.to(torch.float16).flatten(2)

        # Quantize attention output for O projection
        M = x.shape[0] * x.shape[1]
        sum_s, scale_s = _qp_slices(self.quant_params, M)
        x = fused_kernels.quant_sum(x.contiguous(), sum_s, scale_s)
        x = _w8a8_with_padding(self.o, x, self.quant_params)
        return x


class WanFFNWithCudaKernel(nn.Module):
    """FFN with W8A8 INT8 kernels and fused GELU+quantize."""

    def __init__(self, dim, ffn_dim, quant_params=None,
                 has_bias=True, weight_sym=False):
        super().__init__()
        # Use fc1/fc2 naming — state dict keys remapped from ffn.0/ffn.2
        self.fc1 = W8A8OF16LinearDynamicInputScale(dim, ffn_dim, has_bias=has_bias, weight_sym=weight_sym)
        self.fc2 = W8A8OF16LinearDynamicInputScale(ffn_dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.ffn_dim = ffn_dim
        self.quant_params = quant_params

    def forward(self, x):
        """x: INT8 input from fused LayerNorm kernel, quant_params already filled."""
        x = _w8a8_with_padding(self.fc1, x, self.quant_params)  # INT8 → FP16
        M = x.view(-1, x.shape[-1]).shape[0]
        sum_s, scale_s = _qp_slices(self.quant_params, M)
        x = fused_kernels.gelu_quant_sum(x, sum_s, scale_s)     # FP16 → GELU → INT8
        x = _w8a8_with_padding(self.fc2, x, self.quant_params)  # INT8 → FP16
        return x


# ---------------------------------------------------------------------------
# Full block with CUDA kernels
# ---------------------------------------------------------------------------
class WanAttentionBlockWithCudaKernel(nn.Module):
    """
    Wan attention block with real INT8 CUDA kernel inference.

    Computation flow per block:
        1. Fused LayerNorm + modulation + quantize → INT8
        2. Self-attention Q/K/V (W8A8) → FP16 → RoPE → attention → quantize → O (W8A8)
        3. Gate + residual (fused kernel)
        4. Cross-attention: quantize image → Q (W8A8); K/V (FP16 on text); → attention → quantize → O (W8A8)
        5. Residual add
        6. Fused LayerNorm + modulation + quantize → INT8
        7. FFN fc1 (W8A8) → fused GELU+quantize → fc2 (W8A8)
        8. Gate + residual (fused kernel)
    """

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6,
                 quant_params=None, has_bias=True, weight_sym=False):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.quant_params = quant_params

        # These are set during hardware_forward_refactor
        self.attention_fn = None
        self.rope_apply_fn = None

        # Fused RMSNorm + T2I modulation + quantization
        # (Wan model uses RMSNorm, not standard LayerNorm)
        self.norm1 = LayerNormGeneral(dim, act_sum=True, eps=eps, use_rmsnorm=True)
        self.norm2 = LayerNormGeneral(dim, act_sum=True, eps=eps, use_rmsnorm=True)

        # Cross-attention norm (applied before quantization, NOT fused)
        self.cross_attn_norm = cross_attn_norm
        if cross_attn_norm:
            self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        else:
            self.norm3 = nn.Identity()

        # Sub-modules with INT8 kernels
        self.self_attn = WanSelfAttentionWithCudaKernel(
            dim, num_heads, window_size, qk_norm, eps, quant_params,
            has_bias, weight_sym)
        self.cross_attn = WanCrossAttentionWithCudaKernel(
            dim, num_heads, qk_norm, eps, quant_params,
            has_bias, weight_sym)
        self.ffn = WanFFNWithCudaKernel(
            dim, ffn_dim, quant_params, has_bias, weight_sym)

        # Modulation parameter (loaded from state dict)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs,
                context, context_lens, dtype=torch.bfloat16, t=0):
        # Fused CUDA kernels require float16; cast from bfloat16 if needed
        input_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
            e = e.to(torch.float16)
            context = context.to(torch.float16)

        B, L, C = x.shape

        # Ensure QuantParams buffers exactly match B * L tokens
        # (fused kernels CHECK_SHAPE for exact match, not just >=)
        total_tokens = B * L
        if self.quant_params.scale_input.numel() != total_tokens:
            qp = QuantParams(total_tokens, has_sum_input=True, device=x.device)
            self.quant_params = qp
            self.self_attn.quant_params = qp
            self.cross_attn.quant_params = qp
            self.ffn.quant_params = qp

        # Compute modulation: 6 vectors of shape [B, 1, C] or [B, L, C]
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [ei.squeeze(2) for ei in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        # Helper: ensure modulation has shape [B, L, C] for fused kernels.
        # mod may be [B, 1, C] (broadcast) or already [B, L, C] (per-token).
        def expand_mod(mod):
            if mod.shape[1] == 1:
                return mod.expand(B, L, C).contiguous()
            return mod.contiguous()

        # ===== Self-Attention =====
        residual = x
        # Fused: LayerNorm(x) * (1 + scale) + shift → INT8, fills quant_params
        x = self.norm1(x.contiguous(), expand_mod(e[0]), expand_mod(e[1]),
                       self.quant_params)
        # Self-attention with INT8 kernels (W8A8 padding handled inside).
        # Note: attention internally uses BF16 to avoid FP16 softmax overflow.
        x = self.self_attn(x, seq_lens, grid_sizes, freqs,
                           self.attention_fn, self.rope_apply_fn, torch.bfloat16, t)
        # Gated residual: residual + attn_out * gate
        x = fused_kernels.gate_residual_fuse(
            x.contiguous().view(-1, C),
            expand_mod(e[2]).view(-1, C),
            residual.contiguous().view(-1, C),
        ).reshape(B, L, C)

        # ===== Cross-Attention =====
        residual = x
        x_norm = self.norm3(x).to(torch.float16)
        # Quantize image features for cross-attention Q
        sum_s, scale_s = _qp_slices(self.quant_params, B * L)
        x_quant = fused_kernels.quant_sum(
            x_norm.contiguous(), sum_s, scale_s)
        x = self.cross_attn(x_quant, context, context_lens,
                            self.attention_fn, torch.bfloat16, t)
        x = residual + x

        # ===== FFN =====
        residual = x
        # Fused: LayerNorm(x) * (1 + scale) + shift → INT8, fills quant_params
        x = self.norm2(x.contiguous(), expand_mod(e[3]), expand_mod(e[4]),
                       self.quant_params)
        x = self.ffn(x)
        # Gated residual: residual + ffn_out * gate
        x = fused_kernels.gate_residual_fuse(
            x.contiguous().view(-1, C),
            expand_mod(e[5]).view(-1, C),
            residual.contiguous().view(-1, C),
        ).reshape(B, L, C)

        # Cast back to original dtype if needed
        if input_dtype != torch.float16:
            x = x.to(input_dtype)

        return x
