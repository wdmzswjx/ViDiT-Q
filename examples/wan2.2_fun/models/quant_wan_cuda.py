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

import torch
import torch.nn as nn
import logging

from viditq_extension.nn.base import QuantParams
from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
from viditq_extension.nn.layernorm import LayerNormGeneral
import viditq_extension.fused as fused_kernels

from qdiff.base.quant_layer import QuantizedLinear

logger = logging.getLogger(__name__)


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

        # W8A8: INT8 input → FP16 output
        q = self.norm_q(self.q(x, self.quant_params)).view(b, s, n, d)
        k = self.norm_k(self.k(x, self.quant_params)).view(b, s, n, d)
        v = self.v(x, self.quant_params).view(b, s, n, d)

        # RoPE + attention in FP16
        q, k = rope_apply_fn(q, k, grid_sizes, freqs)
        x = attention_fn(q.to(dtype), k.to(dtype), v=v.to(dtype),
                         k_lens=seq_lens, window_size=self.window_size)
        x = x.to(dtype).flatten(2)

        # Quantize attention output for O projection
        x = fused_kernels.quant_sum(
            x.contiguous(), self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.o(x, self.quant_params)
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

        q = self.norm_q(self.q(x, self.quant_params)).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)

        x = attention_fn(q.to(dtype), k.to(dtype), v.to(dtype),
                         k_lens=context_lens)
        x = x.to(dtype).flatten(2)

        # Quantize attention output for O projection
        x = fused_kernels.quant_sum(
            x.contiguous(), self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.o(x, self.quant_params)
        return x


class WanFFNWithCudaKernel(nn.Module):
    """FFN with W8A8 INT8 kernels and fused GELU+quantize."""

    def __init__(self, dim, ffn_dim, quant_params=None,
                 has_bias=True, weight_sym=False):
        super().__init__()
        # Use fc1/fc2 naming — state dict keys remapped from ffn.0/ffn.2
        self.fc1 = W8A8OF16LinearDynamicInputScale(dim, ffn_dim, has_bias=has_bias, weight_sym=weight_sym)
        self.fc2 = W8A8OF16LinearDynamicInputScale(ffn_dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.quant_params = quant_params

    def forward(self, x):
        """x: INT8 input from fused LayerNorm kernel, quant_params already filled."""
        x = self.fc1(x, self.quant_params)                  # INT8 → FP16
        x = fused_kernels.gelu_quant_sum(                   # FP16 → GELU → INT8
            x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.fc2(x, self.quant_params)                  # INT8 → FP16
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

        # Fused LayerNorm + T2I modulation + quantization
        self.norm1 = LayerNormGeneral(dim, act_sum=True, eps=eps)
        self.norm2 = LayerNormGeneral(dim, act_sum=True, eps=eps)

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
        B, L, C = x.shape

        # Compute modulation: 6 vectors of shape [B, 1, C]
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [ei.squeeze(2) for ei in e]
        else:
            e = (self.modulation + e).chunk(6, dim=1)

        # Helper: expand [B, 1, C] modulation to [B, L, C] for fused kernels
        # (the fused kernel needs per-token shift/scale matching input shape)
        def expand_mod(mod):
            return mod.expand(B, L, C).contiguous()

        # ===== Self-Attention =====
        residual = x
        # Fused: LayerNorm(x) * (1 + scale) + shift → INT8, fills quant_params
        x = self.norm1(x.contiguous(), expand_mod(e[0]), expand_mod(e[1]),
                       self.quant_params)
        # Self-attention with INT8 kernels
        x = self.self_attn(x, seq_lens, grid_sizes, freqs,
                           self.attention_fn, self.rope_apply_fn, dtype, t)
        # Gated residual: residual + attn_out * gate
        x = fused_kernels.gate_residual_fuse(
            x.contiguous().view(-1, C),
            expand_mod(e[2]).view(-1, C),
            residual.contiguous().view(-1, C),
        ).reshape(B, L, C)

        # ===== Cross-Attention =====
        residual = x
        x_norm = self.norm3(x)
        # Quantize image features for cross-attention Q
        x_quant = fused_kernels.quant_sum(
            x_norm.contiguous(),
            self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.cross_attn(x_quant, context, context_lens,
                            self.attention_fn, dtype, t)
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

        return x
