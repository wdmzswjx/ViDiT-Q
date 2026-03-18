# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Quantization module for Wan2.2-Fun based on ViDiT-Q framework.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import logging

from omegaconf import OmegaConf, ListConfig

from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules
from qdiff.base.quant_model import (
    quant_layer_refactor_,
    bitwidth_refactor_,
    load_quant_param_dict_,
    save_quant_param_dict_,
    set_init_done_,
)

logger = logging.getLogger(__name__)


class QuantWanModel(nn.Module):
    """
    Quantized wrapper for WanModel / WanTransformer3DModel.

    This class wraps the original Wan diffusion transformer model and applies
    ViDiT-Q quantization (post-training quantization) to all nn.Linear layers
    according to the provided quant_config.

    The wrapper is agnostic to the specific Wan variant (Wan2.1, Wan2.2, Wan2.2-Fun)
    and works with both the original WanModel and the VideoX-Fun WanTransformer3DModel.

    Usage:
        1. Load the original FP model.
        2. Wrap it: quant_model = QuantWanModel(fp_model, quant_config)
        3. (Optional) Apply SmoothQuant / QuaRot / ViDiT-Q techniques.
        4. Call quant_model.set_init_done() then quant_model.save_quant_param_dict().

    The forward() call delegates to the original model, but all nn.Linear layers
    have been replaced with QuantizedLinear (or ViDiTQuantizedLinear, etc.).
    """

    def __init__(self, model, quant_config):
        """
        Args:
            model: The original Wan model (WanModel / WanTransformer3DModel).
            quant_config: OmegaConf dict with quantization configuration.
        """
        super().__init__()
        self.model = model
        self.quant_config = quant_config
        self.quant_param_dict = {}

        self.quant_layer_refactor()

    def quant_layer_refactor(self):
        """Replace all nn.Linear layers with quantized versions."""
        apply_func_to_submodules(
            self.model,
            class_type=nn.Linear,
            function=quant_layer_refactor_,
            name=None,
            parent_module=None,
            quant_config=self.quant_config,
            full_name=None,
            remain_fp_regex=self.quant_config.remain_fp_regex,
        )

    def save_quant_param_dict(self):
        """Save quantization parameters (delta, zero_point, channel_mask, etc.)."""
        apply_func_to_submodules(
            self.model,
            class_type=BaseQuantizer,
            function=save_quant_param_dict_,
            full_name=None,
            parent_module=None,
            model=self,
        )

    def load_quant_param_dict(self, quant_param_dict):
        """Load saved quantization parameters.

        Handles device mismatch: quant_param_dict from torch.load() is on CPU,
        but model weights may be on CUDA. We move all tensors to the model's
        device before passing to the core load function.
        """
        # Move all tensors in quant_param_dict to the model device
        device = next(self.model.parameters()).device
        quant_param_dict_on_device = {}
        for layer_name, params in quant_param_dict.items():
            quant_param_dict_on_device[layer_name] = {}
            for k, v in params.items():
                if isinstance(v, torch.Tensor):
                    quant_param_dict_on_device[layer_name][k] = v.to(device)
                else:
                    quant_param_dict_on_device[layer_name][k] = v

        apply_func_to_submodules(
            self.model,
            class_type=BaseQuantizer,
            function=load_quant_param_dict_,
            full_name=None,
            parent_module=None,
            quant_param_dict=quant_param_dict_on_device,
            model=self,
        )

    def set_init_done(self):
        """Mark all quantizers as initialization-done."""
        apply_func_to_submodules(
            self.model,
            class_type=BaseQuantizer,
            function=set_init_done_,
        )

    def bitwidth_refactor(self):
        """Apply mixed-precision bitwidth settings per layer."""
        apply_func_to_submodules(
            self.model,
            class_type=QuantizedLinear,
            function=bitwidth_refactor_,
            name=None,
            parent_module=None,
            quant_config=self.quant_config,
            full_name=None,
        )

    # ------ CUDA kernel real INT8 inference ------

    def _get_blocks(self):
        """Find the transformer blocks list (handles both 'blocks' and 'transformer_blocks')."""
        if hasattr(self.model, 'blocks'):
            return self.model.blocks, 'blocks'
        elif hasattr(self.model, 'transformer_blocks'):
            return self.model.transformer_blocks, 'transformer_blocks'
        else:
            raise AttributeError(
                "Cannot find transformer blocks. Expected 'blocks' or 'transformer_blocks' attribute.")

    def _get_attention_fns(self):
        """Extract attention and RoPE functions from the model's module."""
        blocks, _ = self._get_blocks()
        old_self_attn = blocks[0].self_attn
        src_module = __import__(type(old_self_attn).__module__, fromlist=['attention', 'rope_apply_qk'])
        attention_fn = getattr(src_module, 'attention')
        rope_apply_fn = getattr(src_module, 'rope_apply_qk')
        return attention_fn, rope_apply_fn

    def quantize_and_save_weight(self, save_path):
        """Convert all QuantizedLinear weights to real INT8 and save checkpoint.

        This generates the int_weight.pt file used by hardware_forward_refactor().
        Must be called AFTER load_quant_param_dict() and set_init_done().
        """
        from models.quant_wan_cuda import quantize_and_save_weight_

        # Disable gradients (torch requires float for grad, but we store INT8)
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Convert all QuantizedLinear weights to INT8
        apply_func_to_submodules(
            self.model,
            class_type=QuantizedLinear,
            function=quantize_and_save_weight_,
            full_name=None,
        )

        # Process state dict
        sd = self.model.state_dict()

        # Save FP16 weights for cross_attn k and v (they stay as FP16 nn.Linear
        # in the CUDA kernel version since text context is not per-token quantized)
        cross_attn_fp_weights = {}
        for k in list(sd.keys()):
            if ('cross_attn.k.fp_module.weight' in k or
                    'cross_attn.v.fp_module.weight' in k):
                clean_key = k.replace('.fp_module', '')
                cross_attn_fp_weights[clean_key] = sd[k].clone().to(torch.float16)
            # Also save bias
            if ('cross_attn.k.fp_module.bias' in k or
                    'cross_attn.v.fp_module.bias' in k):
                clean_key = k.replace('.fp_module', '')
                cross_attn_fp_weights[clean_key] = sd[k].clone().to(torch.float16)

        # Delete fp_module, fp_weight, a_quantizer keys
        keys_to_delete = ['fp_weight', 'fp_module', 'a_quantizer']
        for k in list(sd.keys()):
            if any(s in k for s in keys_to_delete):
                del sd[k]

        # Rename w_quantizer params → scale_weight / zp_weight
        keys_to_rename = {
            'w_quantizer.delta': 'scale_weight',
            'w_quantizer.zero_point': 'zp_weight',
        }
        for k in list(sd.keys()):
            for old_substr, new_substr in keys_to_rename.items():
                if old_substr in k:
                    new_k = k.replace(old_substr, new_substr)
                    val = sd.pop(k)
                    if 'zp_weight' in new_substr:
                        val = val.to(torch.int16)
                    sd[new_k] = val
                    break

        # Restore FP16 weights for cross_attn k and v, remove their quant params
        for k, v in cross_attn_fp_weights.items():
            sd[k] = v
            # Remove INT8 quant params that are not needed for FP16 path
            sd.pop(k.replace('.weight', '.scale_weight'), None)
            sd.pop(k.replace('.weight', '.zp_weight'), None)
            sd.pop(k.replace('.bias', '.scale_weight'), None)
            sd.pop(k.replace('.bias', '.zp_weight'), None)

        # Remap FFN Sequential keys: ffn.0 → ffn.fc1, ffn.2 → ffn.fc2
        for k in list(sd.keys()):
            if '.ffn.0.' in k:
                sd[k.replace('.ffn.0.', '.ffn.fc1.')] = sd.pop(k)
            elif '.ffn.2.' in k:
                sd[k.replace('.ffn.2.', '.ffn.fc2.')] = sd.pop(k)

        # Add LayerNorm weights (ones) for norm1 and norm2
        # Original WanLayerNorm has no affine params; CUDA LayerNormGeneral needs weight=1
        blocks, blocks_attr = self._get_blocks()
        n_blocks = len(blocks)
        for i in range(n_blocks):
            hidden_size = blocks[i].dim
            sd[f'{blocks_attr}.{i}.norm1.weight'] = torch.ones(
                (hidden_size,), dtype=torch.float16)
            sd[f'{blocks_attr}.{i}.norm2.weight'] = torch.ones(
                (hidden_size,), dtype=torch.float16)

        # Log remaining keys
        logger.info('INT8 checkpoint keys:')
        for k in sorted(sd.keys()):
            logger.info('  %s  %s  %s', k, sd[k].shape, sd[k].dtype)

        torch.save(sd, save_path)
        logger.info("Saved INT8 checkpoint to %s", save_path)

    def hardware_forward_refactor(self, load_path, max_seq_len=None):
        """Replace transformer blocks with CUDA kernel versions and load INT8 weights.

        Args:
            load_path: Path to int_weight.pt (generated by quantize_and_save_weight).
            max_seq_len: Maximum total tokens (B * L). If None, defaults to 300000.
                         Should be >= 2 * num_image_tokens for batch_size=1 with CFG.
        """
        from viditq_extension.nn.base import QuantParams
        from models.quant_wan_cuda import WanAttentionBlockWithCudaKernel

        blocks, blocks_attr = self._get_blocks()
        attention_fn, rope_apply_fn = self._get_attention_fns()

        # Allocate QuantParams buffer
        if max_seq_len is None:
            max_seq_len = 300000  # safe default for most resolutions
        self.quant_params = QuantParams(
            max_seq_len, has_sum_input=True, device=torch.device("cuda"))
        logger.info("QuantParams allocated with max_seq_len=%d", max_seq_len)

        # Determine weight quantization symmetry from config
        weight_sym = self.quant_config.weight.get('sym', False)

        # Replace each block with CUDA kernel version
        n_blocks = len(blocks)
        for i in range(n_blocks):
            old_block = blocks[i]

            new_block = WanAttentionBlockWithCudaKernel(
                dim=old_block.dim,
                ffn_dim=old_block.ffn_dim,
                num_heads=old_block.num_heads,
                window_size=old_block.self_attn.window_size,
                qk_norm=old_block.qk_norm,
                cross_attn_norm=old_block.cross_attn_norm,
                eps=old_block.eps,
                quant_params=self.quant_params,
                has_bias=True,
                weight_sym=weight_sym,
            ).half().to('cuda')

            # Set attention functions
            new_block.attention_fn = attention_fn
            new_block.rope_apply_fn = rope_apply_fn

            blocks[i] = new_block
            logger.info("Replaced block %d with CudaKernel version", i)

        # Load INT8 weights
        quant_sd = torch.load(load_path, weights_only=True, map_location='cuda')
        missing, unexpected = self.model.load_state_dict(quant_sd, strict=False)
        if missing:
            logger.warning("Missing keys when loading INT8 weights: %s",
                           [k for k in missing if 'norm' not in k][:20])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:20])

        logger.info("Hardware forward refactor complete. %d blocks replaced.", n_blocks)

    def forward(self, *args, **kwargs):
        """Delegate forward to the wrapped model."""
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy attribute access to the wrapped model for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class QuantWanTransformer3DModel(nn.Module):
    """
    Quantized wrapper specifically for VideoX-Fun's WanTransformer3DModel.

    This is a thin wrapper that adds ViDiT-Q quantization methods directly
    onto the WanTransformer3DModel, following the same pattern as
    CustomizePixArtTransformer2DModel in the PixArt example.

    The key difference from QuantWanModel is that this class is designed
    to be used as a drop-in replacement within the VideoX-Fun pipeline,
    preserving the original model's interface (config, dtype, device, etc.).
    """

    def __init__(self, model, quant_config=None):
        """
        Args:
            model: The original WanTransformer3DModel instance (already loaded).
            quant_config: OmegaConf dict. If None, quantization is deferred
                          until convert_quant() is called.
        """
        super().__init__()
        self.model = model
        self.quant_config = quant_config
        self.quant_param_dict = {}

        if quant_config is not None:
            self.quant_layer_refactor()

    def convert_quant(self, quant_config):
        """Apply quantization config (can be called after construction)."""
        self.quant_config = quant_config
        self.quant_param_dict = {}
        self.quant_layer_refactor()

    def quant_layer_refactor(self):
        apply_func_to_submodules(
            self.model,
            class_type=nn.Linear,
            function=quant_layer_refactor_,
            name=None,
            parent_module=None,
            quant_config=self.quant_config,
            full_name=None,
            remain_fp_regex=self.quant_config.remain_fp_regex,
        )

    def save_quant_param_dict(self):
        apply_func_to_submodules(
            self.model,
            class_type=BaseQuantizer,
            function=save_quant_param_dict_,
            full_name=None,
            parent_module=None,
            model=self,
        )

    def load_quant_param_dict(self, quant_param_dict):
        device = next(self.model.parameters()).device
        quant_param_dict_on_device = {}
        for layer_name, params in quant_param_dict.items():
            quant_param_dict_on_device[layer_name] = {}
            for k, v in params.items():
                if isinstance(v, torch.Tensor):
                    quant_param_dict_on_device[layer_name][k] = v.to(device)
                else:
                    quant_param_dict_on_device[layer_name][k] = v

        apply_func_to_submodules(
            self.model,
            class_type=BaseQuantizer,
            function=load_quant_param_dict_,
            full_name=None,
            parent_module=None,
            quant_param_dict=quant_param_dict_on_device,
            model=self,
        )

    def set_init_done(self):
        apply_func_to_submodules(
            self.model,
            class_type=BaseQuantizer,
            function=set_init_done_,
        )

    def bitwidth_refactor(self):
        apply_func_to_submodules(
            self.model,
            class_type=QuantizedLinear,
            function=bitwidth_refactor_,
            name=None,
            parent_module=None,
            quant_config=self.quant_config,
            full_name=None,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def config(self):
        return self.model.config

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
