"""
诊断工具：排查 ViDiT-Q CUDA kernel 量化推理的质量退化问题。

将此脚本的 run_diagnostics() 在 hardware_forward_refactor() 之后调用。
也可独立运行：python diagnose_quant.py --int-weight ./viditq/ckpts/int_weight.pt

检查项：
  [1] 代码文件路径：确认 quant_wan_cuda.py 是修改后的版本
  [2] BF16 attention patch 是否生效
  [3] modulation 参数是否正确加载（非随机初始化）
  [4] norm1/norm2 weight 是否为全 1
  [5] INT8 weight 是否正常加载（非全零）
  [6] attention_fn 调用时 e 的 chunk 结构是否合理
  [7] 推理中各层的张量统计（NaN/Inf/均值/方差）
"""

import sys
import os
import inspect
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# [1] 路径检查
# ─────────────────────────────────────────────────────────────────────────────

def check_file_path():
    print("\n" + "="*60)
    print("[1] 代码文件路径检查")
    print("="*60)
    try:
        import viditq.examples.wan22fun.models.quant_wan_cuda as cuda_mod
        path = inspect.getfile(cuda_mod)
        print(f"  实际使用的 quant_wan_cuda.py: {path}")

        with open(path, 'r') as f:
            content = f.read()
        if 'torch.bfloat16' in content and 'FP16 softmax overflow' in content:
            print("  ✓ BF16 attention patch 已应用于此文件")
        else:
            print("  ✗ 警告: 此文件【未包含】BF16 attention patch！")
            print("    请将修改后的 quant_wan_cuda.py 复制到:", path)
    except ImportError as e:
        print(f"  无法导入 viditq.examples.wan22fun.models.quant_wan_cuda: {e}")
        # 尝试直接路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(script_dir, 'models', 'quant_wan_cuda.py')
        if os.path.exists(local_path):
            print(f"  本地文件: {local_path}")
            with open(local_path, 'r') as f:
                content = f.read()
            if 'torch.bfloat16' in content and 'FP16 softmax overflow' in content:
                print("  ✓ BF16 patch 在本地文件中存在")
            else:
                print("  ✗ 本地文件也未包含 BF16 patch")


# ─────────────────────────────────────────────────────────────────────────────
# [2][3][4][5] 对 QuantWanModel 做静态检查
# ─────────────────────────────────────────────────────────────────────────────

def check_model_state(quant_model):
    """
    quant_model: hardware_forward_refactor() 之后的 QuantWanModel 实例
    """
    print("\n" + "="*60)
    print("[2-5] 模型静态参数检查")
    print("="*60)

    try:
        from viditq.kernels.viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
        from viditq.kernels.viditq_extension.nn.layernorm import LayerNormGeneral
        try_cuda_imports = True
    except ImportError:
        try:
            from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
            from viditq_extension.nn.layernorm import LayerNormGeneral
            try_cuda_imports = True
        except ImportError:
            print("  无法导入 viditq_extension，跳过 [2-5]")
            return

    # 获取内部 model
    inner = quant_model.model if hasattr(quant_model, 'model') else quant_model

    # 找到 blocks
    if hasattr(inner, 'blocks'):
        blocks = inner.blocks
        blocks_attr = 'blocks'
    elif hasattr(inner, 'transformer_blocks'):
        blocks = inner.transformer_blocks
        blocks_attr = 'transformer_blocks'
    else:
        print("  找不到 blocks 或 transformer_blocks 属性")
        return

    print(f"  找到 {len(blocks)} 个 block，属性名: {blocks_attr}")

    cuda_block_count = 0
    issues = []

    for i, block in enumerate(blocks):
        cls_name = type(block).__name__
        if 'CudaKernel' not in cls_name:
            if i == 0:
                print(f"  block[{i}] 类型: {cls_name}（不是 CUDA kernel block，hardware_forward_refactor 可能未生效）")
            continue
        cuda_block_count += 1

        # [2] BF16 patch 检查（通过源码）
        if i == 0:
            try:
                src = inspect.getsource(type(block).forward)
                if 'torch.bfloat16' in src and 'float16).flatten' in src:
                    print(f"  [2] ✓ block[0] forward 包含 BF16 attention 代码")
                else:
                    print(f"  [2] ✗ block[0] forward 未找到 BF16 attention 代码！patch 未生效")
            except Exception as e:
                print(f"  [2] 无法读取 forward 源码: {e}")

        # [3] modulation 参数检查
        if hasattr(block, 'modulation'):
            mod = block.modulation.data
            mod_std = mod.float().std().item()
            mod_mean = mod.float().mean().item()
            # 如果是 randn 初始化且未加载，std ≈ dim^(-0.5) ≈ 0.003 for dim=3072
            # 如果正确加载，值应该比较小但有规律
            if mod_std < 1e-6:
                issues.append(f"block[{i}].modulation 全零（可能未加载）")
            elif mod_std > 1.0:
                issues.append(f"block[{i}].modulation std={mod_std:.4f}（疑似随机初始化未加载）")
            if i == 0 or i == len(blocks)-1:
                print(f"  [3] block[{i}].modulation: mean={mod_mean:.4f}, std={mod_std:.4f}, "
                      f"shape={list(mod.shape)}, dtype={mod.dtype}")
        else:
            issues.append(f"block[{i}] 没有 modulation 属性")

        # [4] norm1/norm2 weight 检查
        for norm_name in ['norm1', 'norm2']:
            norm = getattr(block, norm_name, None)
            if norm is None:
                continue
            if isinstance(norm, LayerNormGeneral) and hasattr(norm, 'weight'):
                w = norm.weight.data
                if i == 0:
                    is_all_ones = (w - 1.0).abs().max().item() < 1e-3
                    print(f"  [4] block[0].{norm_name}.weight: "
                          f"{'全1 (OK)' if is_all_ones else f'非全1, mean={w.float().mean():.4f}'}, "
                          f"shape={list(w.shape)}")

        # [5] INT8 weight 检查（只看第一个块）
        if i == 0:
            for sub_name in ['self_attn', 'cross_attn', 'ffn']:
                sub = getattr(block, sub_name, None)
                if sub is None:
                    continue
                for proj_name in ['q', 'k', 'v', 'o', 'fc1', 'fc2']:
                    proj = getattr(sub, proj_name, None)
                    if proj is None or not isinstance(proj, W8A8OF16LinearDynamicInputScale):
                        continue
                    w = proj.weight.data
                    w_mean = w.float().mean().item()
                    w_std = w.float().std().item()
                    all_zero = (w == 0).all().item()
                    print(f"  [5] block[0].{sub_name}.{proj_name}.weight(INT8): "
                          f"mean={w_mean:.2f}, std={w_std:.2f}, "
                          f"{'⚠ 全零！' if all_zero else 'OK'}")
                    break  # 只看第一个 proj

    if cuda_block_count == 0:
        print("  ✗ 未找到任何 CUDA kernel block，hardware_forward_refactor 可能未生效！")
    else:
        print(f"  共 {cuda_block_count} 个 CUDA kernel block")

    if issues:
        print("\n  发现问题：")
        for iss in issues[:10]:
            print(f"    ✗ {iss}")
    else:
        print("  静态检查通过")


# ─────────────────────────────────────────────────────────────────────────────
# [6] 对第 0 个 block 注入 hook，捕获推理时的 e 结构和张量统计
# ─────────────────────────────────────────────────────────────────────────────

_hooks = []
_stats = {}


def _tensor_stats(name, t):
    if t is None:
        return "None"
    if not isinstance(t, torch.Tensor):
        return str(type(t))
    has_nan = torch.isnan(t.float()).any().item()
    has_inf = torch.isinf(t.float()).any().item()
    mn = t.float().min().item()
    mx = t.float().max().item()
    mean = t.float().mean().item()
    flag = ""
    if has_nan:
        flag += " ⚠NaN"
    if has_inf:
        flag += " ⚠Inf"
    if abs(mx) > 60000:
        flag += " ⚠接近FP16上限"
    return f"shape={list(t.shape)} dtype={t.dtype} min={mn:.3f} max={mx:.3f} mean={mean:.3f}{flag}"


def install_forward_hooks(quant_model, n_blocks_to_watch=2):
    """
    在推理开始前调用。注入 hook 捕获第 0~n_blocks_to_watch 个 block 的中间量。
    """
    global _hooks, _stats
    _stats = {}

    inner = quant_model.model if hasattr(quant_model, 'model') else quant_model

    if hasattr(inner, 'blocks'):
        blocks = inner.blocks
    elif hasattr(inner, 'transformer_blocks'):
        blocks = inner.transformer_blocks
    else:
        print("[hook] 找不到 blocks")
        return

    for i in range(min(n_blocks_to_watch, len(blocks))):
        block = blocks[i]
        if 'CudaKernel' not in type(block).__name__:
            print(f"[hook] block[{i}] 不是 CUDA kernel block，跳过")
            continue
        _install_block_hooks(block, i)

    print(f"[hook] 已在 {min(n_blocks_to_watch, len(blocks))} 个 block 上注入诊断 hook")


def _install_block_hooks(block, block_idx):
    """给一个 WanAttentionBlockWithCudaKernel 注入前向 hook。"""

    original_forward = block.forward

    def hooked_forward(x, e, seq_lens, grid_sizes, freqs,
                       context, context_lens, dtype=torch.bfloat16, t=0):

        print(f"\n{'─'*50}")
        print(f"[hook] block[{block_idx}] forward 被调用")
        print(f"  [6] x 输入: {_tensor_stats('x', x)}")
        print(f"  [6] e 输入: {_tensor_stats('e', e)}")
        print(f"  [6] context: {_tensor_stats('ctx', context)}")

        # 检查 e 的结构（modulation chunk）
        if isinstance(e, torch.Tensor):
            if e.dim() == 3:
                # [B, 6, dim] 格式
                B, six, dim = e.shape
                print(f"  [6] e.shape = {list(e.shape)} → 每个 chunk 形状 [{B}, 1, {dim}]")
                # 模拟 block 内的 modulation 加法
                try:
                    e_mod = block.modulation + e
                    chunks = e_mod.chunk(6, dim=1)
                    for ci, label in enumerate(['shift_msa', 'scale_msa', 'gate_msa',
                                               'shift_mlp', 'scale_mlp', 'gate_mlp']):
                        c = chunks[ci].squeeze(1)  # [B, dim]
                        print(f"    e[{ci}]({label}): min={c.float().min():.3f} "
                              f"max={c.float().max():.3f} mean={c.float().mean():.3f}")
                except Exception as ex:
                    print(f"    chunk 分析失败: {ex}")
            elif e.dim() == 4:
                print(f"  [6] e.shape = {list(e.shape)} (4D, per-token modulation)")
            else:
                print(f"  [6] e.shape = {list(e.shape)} (未知格式)")

        # 执行原始 forward 并捕获输出
        try:
            out = original_forward(x, e, seq_lens, grid_sizes, freqs,
                                   context, context_lens, dtype, t)
            print(f"  [7] block[{block_idx}] 输出: {_tensor_stats('out', out)}")
            return out
        except Exception as ex:
            print(f"  ✗ block[{block_idx}] forward 抛出异常: {ex}")
            raise

    block.forward = hooked_forward
    _hooks.append((block, original_forward))


def remove_hooks(quant_model):
    """推理完成后移除 hook，恢复原始 forward。"""
    inner = quant_model.model if hasattr(quant_model, 'model') else quant_model
    if hasattr(inner, 'blocks'):
        blocks = inner.blocks
    elif hasattr(inner, 'transformer_blocks'):
        blocks = inner.transformer_blocks
    else:
        return
    for block, orig_fwd in _hooks:
        block.forward = orig_fwd
    _hooks.clear()
    print("[hook] 已移除所有诊断 hook")


# ─────────────────────────────────────────────────────────────────────────────
# [8] attention_fn 兼容性检查（独立于推理）
# ─────────────────────────────────────────────────────────────────────────────

def check_attention_fn(quant_model):
    print("\n" + "="*60)
    print("[8] attention_fn BF16 兼容性检查")
    print("="*60)

    inner = quant_model.model if hasattr(quant_model, 'model') else quant_model
    if hasattr(inner, 'blocks'):
        blocks = inner.blocks
    elif hasattr(inner, 'transformer_blocks'):
        blocks = inner.transformer_blocks
    else:
        print("  找不到 blocks")
        return

    for block in blocks:
        if 'CudaKernel' not in type(block).__name__:
            continue
        if block.attention_fn is None:
            print("  ✗ block.attention_fn is None！hardware_forward_refactor 未设置 attention_fn")
            return

        fn = block.attention_fn
        print(f"  attention_fn: {fn}")
        print(f"  attention_fn 模块: {getattr(fn, '__module__', 'unknown')}")

        # 用小 tensor 测试 BF16 调用
        try:
            B, L, n, d = 1, 16, 8, 64
            q = torch.randn(B, L, n, d, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(B, L, n, d, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(B, L, n, d, dtype=torch.bfloat16, device='cuda')
            k_lens = torch.tensor([L], dtype=torch.int32, device='cuda')
            with torch.no_grad():
                out = fn(q, k, v=v, k_lens=k_lens)
            print(f"  ✓ BF16 attention 调用成功: 输出 dtype={out.dtype}, shape={list(out.shape)}")
        except Exception as e:
            print(f"  ✗ BF16 attention 调用失败: {e}")
            print("    这可能是质量退化的原因！attention_fn 不支持 BF16")
        break


# ─────────────────────────────────────────────────────────────────────────────
# 汇总入口
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(quant_model, install_hooks=True):
    """
    在 hardware_forward_refactor() 之后调用此函数。

    用法：
        quant_model.hardware_forward_refactor(load_path=..., max_seq_len=...)
        from diagnose_quant import run_diagnostics
        run_diagnostics(quant_model)
        # ... 然后运行推理（hook 会在推理时自动打印信息）
        # ... 推理完成后：
        from diagnose_quant import remove_hooks
        remove_hooks(quant_model)
    """
    check_file_path()
    check_model_state(quant_model)
    check_attention_fn(quant_model)

    if install_hooks:
        install_forward_hooks(quant_model, n_blocks_to_watch=2)
        print("\n[hook] 推理时将自动打印 block[0], block[1] 的张量统计")
        print("       推理完成后请调用 remove_hooks(quant_model)")


# ─────────────────────────────────────────────────────────────────────────────
# 独立运行：仅检查 INT8 checkpoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--int-weight', type=str, required=True,
                        help='int_weight.pt 路径')
    args = parser.parse_args()

    print(f"\n加载 INT8 checkpoint: {args.int_weight}")
    sd = torch.load(args.int_weight, weights_only=True, map_location='cpu')

    print(f"共 {len(sd)} 个 key\n")

    # 分类统计
    modulation_keys = [k for k in sd if 'modulation' in k]
    norm_keys = [k for k in sd if ('norm1.weight' in k or 'norm2.weight' in k)]
    int8_keys = [k for k in sd if sd[k].dtype == torch.int8]
    nan_keys = [k for k in sd if sd[k].is_floating_point() and torch.isnan(sd[k].float()).any()]
    inf_keys = [k for k in sd if sd[k].is_floating_point() and torch.isinf(sd[k].float()).any()]

    print(f"[3] modulation keys ({len(modulation_keys)} 个):")
    for k in modulation_keys[:5]:
        v = sd[k]
        print(f"    {k}: shape={list(v.shape)}, dtype={v.dtype}, "
              f"std={v.float().std():.4f}, mean={v.float().mean():.4f}")
    if not modulation_keys:
        print("    ✗ 无！CUDA block 的 modulation 将使用随机初始化")

    print(f"\n[4] norm weight keys ({len(norm_keys)} 个):")
    for k in norm_keys[:4]:
        v = sd[k]
        is_ones = (v.float() - 1.0).abs().max().item() < 1e-3
        print(f"    {k}: shape={list(v.shape)}, {'全1 (OK)' if is_ones else f'非全1 max_diff={((v.float()-1.0).abs().max().item()):.4f}'}")

    print(f"\n[5] INT8 weight keys: {len(int8_keys)} 个")
    if int8_keys:
        sample = sd[int8_keys[0]]
        all_zero = (sample == 0).all().item()
        print(f"    示例 {int8_keys[0]}: shape={list(sample.shape)}, "
              f"{'⚠ 全零！' if all_zero else f'mean={sample.float().mean():.2f}'}")

    if nan_keys:
        print(f"\n⚠ 含 NaN 的 key: {nan_keys}")
    if inf_keys:
        print(f"\n⚠ 含 Inf 的 key: {inf_keys}")

    print("\n诊断完成")
