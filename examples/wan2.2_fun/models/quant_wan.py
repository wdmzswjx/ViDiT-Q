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
