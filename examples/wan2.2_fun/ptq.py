"""
Post-Training Quantization (PTQ) pipeline for Wan2.2-Fun with ViDiT-Q.

This script:
1. Loads the FP model.
2. Replaces nn.Linear layers with quantized versions.
3. (Optional) Applies SmoothQuant, QuaRot, or ViDiT-Q techniques.
4. Saves quantization parameters for later inference.

Usage:
    python ptq.py \
        --model-path ./models/Wan2.2-Fun-14B-InP \
        --model-type t2v \
        --quant-config ./configs/w8a8.yaml \
        --log ./ptq_output \
        --seed 42
"""
import torch
import torch.nn as nn
import os
import sys
import argparse
import logging

from omegaconf import OmegaConf, ListConfig
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from qdiff.base.quant_layer import QuantizedLinear
from models.quant_wan import QuantWanModel

logger = logging.getLogger(__name__)


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.log, exist_ok=True)
    log_file = os.path.join(args.log, 'ptq_run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    # Backup configs
    import shutil
    config_dst = os.path.join(args.log, 'configs')
    if os.path.exists(config_dst):
        shutil.rmtree(config_dst)
    if os.path.exists('./configs'):
        shutil.copytree('./configs', config_dst)

    quant_config = OmegaConf.load(args.quant_config)
    logger.info("Quantization config:\n%s", OmegaConf.to_yaml(quant_config))

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # ============================================================
    # Load model
    # ============================================================
    logger.info("Loading model from %s ...", args.model_path)

    fp_model = None

    # Try VideoX-Fun
    try:
        from videox_fun.models.wan_transformer3d import WanTransformer3DModel
        fp_model = WanTransformer3DModel.from_pretrained(
            args.model_path,
            subfolder="transformer" if os.path.isdir(os.path.join(args.model_path, "transformer")) else None,
            torch_dtype=dtype,
        ).to(device).eval()
        logger.info("Loaded as VideoX-Fun WanTransformer3DModel")
    except (ImportError, Exception) as e:
        logger.info("VideoX-Fun not available: %s", e)

    # Try diffusers
    if fp_model is None:
        try:
            from diffusers.models import AutoModel
            fp_model = AutoModel.from_pretrained(
                args.model_path,
                subfolder="transformer" if os.path.isdir(os.path.join(args.model_path, "transformer")) else None,
                torch_dtype=dtype,
            ).to(device).eval()
            logger.info("Loaded as diffusers model")
        except (ImportError, Exception) as e:
            logger.info("diffusers not available: %s", e)

    # Try original WanModel
    if fp_model is None:
        try:
            from wan.modules.model import WanModel
            model_config = OmegaConf.load(os.path.join(args.model_path, "config.yaml"))
            fp_model = WanModel(model_type=args.model_type, **model_config).to(device).eval()
            ckpt_path = os.path.join(args.model_path, "model.safetensors")
            if os.path.exists(ckpt_path):
                from safetensors.torch import load_file
                fp_model.load_state_dict(load_file(ckpt_path))
            logger.info("Loaded as original WanModel")
        except (ImportError, Exception) as e:
            logger.info("WanModel not available: %s", e)

    if fp_model is None:
        raise RuntimeError(f"Could not load model from {args.model_path}")

    # ============================================================
    # Create quantized model
    # ============================================================
    model = QuantWanModel(fp_model, quant_config)
    logger.info("Quantized model created")

    # Check if mixed precision
    if_mixed_precision = (
        isinstance(quant_config.weight.n_bits, ListConfig)
        or isinstance(quant_config.act.n_bits, ListConfig)
    )
    if if_mixed_precision:
        model.bitwidth_refactor()
        logger.info("Applied mixed precision bitwidth refactoring")

    # ============================================================
    # Apply quantization techniques
    # ============================================================

    def init_sq_channel_mask_(module, full_name, calib_data):
        """Initialize SmoothQuant channel masks from calibration data."""
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear
        assert isinstance(module, SQQuantizedLinear)
        weight_device = module.fp_module.weight.device
        act_mask = calib_data[full_name].max(dim=0)[0].to(weight_device)  # [T, C] -> [C]
        zero_mask = act_mask < 1e-3
        act_mask = torch.where(zero_mask, torch.tensor(1e-3, device=weight_device), act_mask)
        module.get_channel_mask(act_mask)
        module.update_quantized_weight_scaled()

    def init_rotation_matrix_(module, full_name):
        """Initialize QuaRot rotation matrices."""
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
        assert isinstance(module, QuarotQuantizedLinear)
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated()

    def init_rotation_and_channel_mask_(module, full_name, calib_data):
        """Initialize ViDiT-Q (rotation + channel scaling) from calibration data."""
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        assert isinstance(module, ViDiTQuantizedLinear)
        weight_device = module.fp_module.weight.device
        act_mask = calib_data[full_name].max(dim=0)[0].to(weight_device)  # [T, C] -> [C]
        zero_mask = act_mask < 1e-3
        act_mask = torch.where(zero_mask, torch.tensor(1e-3, device=weight_device), act_mask)
        module.get_channel_mask(act_mask)
        module.get_rotation_matrix()
        module.update_quantized_weight_rotated_and_scaled()

    # --- SmoothQuant ---
    if quant_config.get("smooth_quant", None) is not None:
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear

        assert quant_config.calib_data.save_path is not None
        calib_path = os.path.join(args.log, quant_config.calib_data.save_path)
        calib_data = torch.load(calib_path, weights_only=True)
        logger.info("Loaded calibration data from %s", calib_path)

        apply_func_to_submodules(
            model.model,
            class_type=SQQuantizedLinear,
            function=init_sq_channel_mask_,
            calib_data=calib_data,
            full_name='',
        )
        logger.info("Applied SmoothQuant channel masks")

    # --- QuaRot ---
    if quant_config.get("quarot", None) is not None:
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear

        apply_func_to_submodules(
            model.model,
            class_type=QuarotQuantizedLinear,
            function=init_rotation_matrix_,
            full_name='',
        )
        logger.info("Applied QuaRot rotation matrices")

    # --- ViDiT-Q (rotation + channel scaling) ---
    if quant_config.get("viditq", None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear

        assert quant_config.calib_data.save_path is not None
        calib_path = os.path.join(args.log, quant_config.calib_data.save_path)
        calib_data = torch.load(calib_path, weights_only=True)
        logger.info("Loaded calibration data from %s", calib_path)

        apply_func_to_submodules(
            model.model,
            class_type=ViDiTQuantizedLinear,
            function=init_rotation_and_channel_mask_,
            full_name='',
            calib_data=calib_data,
        )
        logger.info("Applied ViDiT-Q (rotation + channel scaling)")

    # ============================================================
    # Save quantization parameters
    # ============================================================
    model.set_init_done()
    model.save_quant_param_dict()

    save_path = os.path.join(args.log, 'quant_params.pth')
    torch.save(model.quant_param_dict, save_path)
    logger.info("Saved quantization parameters to %s", save_path)

    # Print summary
    n_quant = 0
    n_fp = 0
    for name, module in model.model.named_modules():
        if isinstance(module, QuantizedLinear):
            if module.quant_mode:
                n_quant += 1
            else:
                n_fp += 1
    logger.info("Quantized layers: %d, FP layers: %d", n_quant, n_fp)
    logger.info("PTQ completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTQ for Wan2.2-Fun with ViDiT-Q")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Wan model")
    parser.add_argument("--model-type", type=str, default="t2v", choices=["t2v", "i2v", "flf2v"])
    parser.add_argument("--quant-config", type=str, required=True, help="Path to quant config YAML")
    parser.add_argument("--log", type=str, required=True, help="Output directory")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
