"""
Diagnostic script to verify fused CUDA kernel accuracy.

Tests:
1. gelu_quant_sum (original path: hidden_size <= 8192)
2. gelu_quant_sum (looped path: hidden_size > 8192, e.g. ffn_dim=12288)
3. quant_sum accuracy
4. Padding impact on block output

Run: python tests/test_fused_kernel_accuracy.py
"""

import torch
import torch.nn.functional as F
import math

def reference_gelu_quant(x_fp16):
    """Pure PyTorch reference: GELU + per-token dynamic INT8 quantization + post-quant sum."""
    # GELU (tanh approximation, same as CUDA kernel's gelu_func)
    x = F.gelu(x_fp16.float(), approximate='tanh').half()

    # Per-token dynamic quantization
    # scale = amax / 127
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    scale = amax / 127.0  # [num_tokens, 1]

    # Quantize to INT8
    tmp_scale = 127.0 / amax.clamp(min=1e-12)
    x_int8 = (x.float() * tmp_scale).round().clamp(-128, 127).to(torch.int8)

    # Post-quant sum: sum(int8_values) / tmp_scale_scalar
    # The kernel computes per-token: sum(int8) / (127/amax) = sum(int8) * amax / 127
    sum_int = x_int8.float().sum(dim=-1)
    sum_output = (sum_int / tmp_scale.squeeze(-1)).half()

    return x_int8, scale.squeeze(-1).half(), sum_output


def reference_quant(x_fp16):
    """Pure PyTorch reference: per-token dynamic INT8 quantization + post-quant sum."""
    x = x_fp16
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    scale = amax / 127.0

    tmp_scale = 127.0 / amax.clamp(min=1e-12)
    x_int8 = (x.float() * tmp_scale).round().clamp(-128, 127).to(torch.int8)

    sum_int = x_int8.float().sum(dim=-1)
    sum_output = (sum_int / tmp_scale.squeeze(-1)).half()

    return x_int8, scale.squeeze(-1).half(), sum_output


def test_gelu_quant_sum(hidden_size, num_tokens=128, label=""):
    """Test gelu_quant_sum CUDA kernel against reference."""
    import viditq_extension.fused as fused_kernels

    print(f"\n{'='*60}")
    print(f"Test gelu_quant_sum: hidden_size={hidden_size}, tokens={num_tokens} {label}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)

    # Reference
    ref_int8, ref_scale, ref_sum = reference_gelu_quant(x)

    # CUDA kernel
    sum_output = torch.empty(num_tokens, device='cuda', dtype=torch.float16)
    scale_output = torch.empty(num_tokens, device='cuda', dtype=torch.float16)
    cuda_int8 = fused_kernels.gelu_quant_sum(x.clone(), sum_output, scale_output)

    # Compare INT8 output
    int8_match = (cuda_int8 == ref_int8).float().mean().item()
    int8_close = ((cuda_int8.float() - ref_int8.float()).abs() <= 1).float().mean().item()
    print(f"  INT8 exact match:  {int8_match*100:.2f}%")
    print(f"  INT8 within +/-1:  {int8_close*100:.2f}%")

    # Compare scales
    scale_diff = (scale_output - ref_scale).abs()
    print(f"  Scale max diff:    {scale_diff.max().item():.6f}")
    print(f"  Scale mean diff:   {scale_diff.mean().item():.6f}")

    # Compare sum
    sum_diff = (sum_output - ref_sum).abs()
    rel_sum_diff = sum_diff / (ref_sum.abs() + 1e-6)
    print(f"  Sum max abs diff:  {sum_diff.max().item():.4f}")
    print(f"  Sum max rel diff:  {rel_sum_diff.max().item():.4f}")

    # Reconstruct FP16 from INT8 for end-to-end comparison
    cuda_fp16 = cuda_int8.float() * (scale_output.unsqueeze(-1).float() / 127.0)
    ref_fp16 = ref_int8.float() * (ref_scale.unsqueeze(-1).float() / 127.0)

    recon_diff = (cuda_fp16 - ref_fp16).abs()
    print(f"  Reconstructed FP16 max diff: {recon_diff.max().item():.6f}")
    print(f"  Reconstructed FP16 mean diff: {recon_diff.mean().item():.6f}")

    return int8_match, scale_diff.max().item()


def test_quant_sum(hidden_size, num_tokens=128, label=""):
    """Test quant_sum CUDA kernel against reference."""
    import viditq_extension.fused as fused_kernels

    print(f"\n{'='*60}")
    print(f"Test quant_sum: hidden_size={hidden_size}, tokens={num_tokens} {label}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)

    # Reference
    ref_int8, ref_scale, ref_sum = reference_quant(x)

    # CUDA kernel
    sum_output = torch.empty(num_tokens, device='cuda', dtype=torch.float16)
    scale_output = torch.empty(num_tokens, device='cuda', dtype=torch.float16)
    cuda_int8 = fused_kernels.quant_sum(x.clone(), sum_output, scale_output)

    int8_match = (cuda_int8 == ref_int8).float().mean().item()
    int8_close = ((cuda_int8.float() - ref_int8.float()).abs() <= 1).float().mean().item()
    print(f"  INT8 exact match:  {int8_match*100:.2f}%")
    print(f"  INT8 within +/-1:  {int8_close*100:.2f}%")

    scale_diff = (scale_output - ref_scale).abs()
    print(f"  Scale max diff:    {scale_diff.max().item():.6f}")

    sum_diff = (sum_output - ref_sum).abs()
    print(f"  Sum max abs diff:  {sum_diff.max().item():.4f}")

    return int8_match, scale_diff.max().item()


def test_dequant_roundtrip(hidden_size, num_tokens=128, label=""):
    """Test full quantize -> dequantize roundtrip error."""
    import viditq_extension.fused as fused_kernels

    print(f"\n{'='*60}")
    print(f"Test dequant roundtrip: hidden_size={hidden_size}, tokens={num_tokens} {label}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)

    # After GELU
    x_gelu = F.gelu(x.float(), approximate='tanh').half()

    # CUDA kernel: GELU + quant
    sum_output = torch.empty(num_tokens, device='cuda', dtype=torch.float16)
    scale_output = torch.empty(num_tokens, device='cuda', dtype=torch.float16)
    cuda_int8 = fused_kernels.gelu_quant_sum(x.clone(), sum_output, scale_output)

    # Dequantize: int8 * scale / 127
    cuda_deq = cuda_int8.float() * (scale_output.unsqueeze(-1).float() / 127.0)

    # Compare against reference GELU
    diff = (cuda_deq - x_gelu.float()).abs()
    rel_diff = diff / (x_gelu.float().abs() + 1e-6)

    print(f"  Dequant vs FP16 GELU:")
    print(f"    Max abs diff:  {diff.max().item():.6f}")
    print(f"    Mean abs diff: {diff.mean().item():.6f}")
    print(f"    Max rel diff:  {rel_diff.max().item():.4f}")
    print(f"    Mean rel diff: {rel_diff.mean().item():.6f}")

    # Expected quantization error: ~amax/254 per element
    expected_err = x_gelu.abs().amax(dim=-1).mean().item() / 254.0
    print(f"    Expected quant error (amax/254): ~{expected_err:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("FUSED KERNEL ACCURACY TESTS")
    print("=" * 60)

    # Test 1: gelu_quant_sum with small hidden_size (original kernel)
    test_gelu_quant_sum(3072, label="[original kernel, float2]")
    test_gelu_quant_sum(4096, label="[original kernel, float2 boundary]")
    test_gelu_quant_sum(8192, label="[original kernel, float4]")

    # Test 2: gelu_quant_sum with large hidden_size (LOOPED kernel)
    test_gelu_quant_sum(12288, label="[LOOPED kernel - Wan 5B ffn_dim]")
    test_gelu_quant_sum(16384, label="[LOOPED kernel]")

    # Test 3: quant_sum with various sizes
    test_quant_sum(3072, label="[original kernel]")
    test_quant_sum(12288, label="[LOOPED kernel]")

    # Test 4: Full dequant roundtrip
    test_dequant_roundtrip(3072, label="[original kernel]")
    test_dequant_roundtrip(12288, label="[LOOPED kernel - Wan 5B]")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
