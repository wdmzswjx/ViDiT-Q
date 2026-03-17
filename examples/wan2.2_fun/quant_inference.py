"""
Quantized inference for Wan2.2-Fun with ViDiT-Q.

Loads the quantized model with saved quantization parameters and runs inference.

Usage:
    python quant_inference.py \
        --model-path ./models/Wan2.2-Fun-14B-InP \
        --model-type t2v \
        --quant-config ./configs/w8a8.yaml \
        --quant-params ./ptq_output/quant_params.pth \
        --prompt "A serene lake surrounded by mountains at sunset." \
        --save-dir ./quant_output \
        --seed 42
"""
import torch
import torch.nn as nn
import os
import sys
import argparse
import logging
import time

from omegaconf import OmegaConf, ListConfig
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from qdiff.base.quant_layer import QuantizedLinear
from models.quant_wan import QuantWanModel

logger = logging.getLogger(__name__)


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)
    log_file = os.path.join(args.save_dir, 'infer_run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    quant_config = OmegaConf.load(args.quant_config)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # ============================================================
    # Load model
    # ============================================================
    logger.info("Loading model from %s ...", args.model_path)

    fp_model = None
    pipe = None

    # Try VideoX-Fun pipeline
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
    # Create quantized model and load params
    # ============================================================
    model = QuantWanModel(fp_model, quant_config)

    # Apply mixed precision if needed
    if_mixed_precision = (
        isinstance(quant_config.weight.n_bits, ListConfig)
        or isinstance(quant_config.act.n_bits, ListConfig)
    )
    if if_mixed_precision:
        model.bitwidth_refactor()

    # Load quantization parameters
    logger.info("Loading quant params from %s ...", args.quant_params)
    quant_param_dict = torch.load(args.quant_params, weights_only=True)
    model.load_quant_param_dict(quant_param_dict)
    model.set_init_done()
    logger.info("Quantized model ready for inference")

    # Print model summary
    n_quant = sum(1 for m in model.model.modules() if isinstance(m, QuantizedLinear) and m.quant_mode)
    n_fp = sum(1 for m in model.model.modules() if isinstance(m, QuantizedLinear) and not m.quant_mode)
    logger.info("Quantized layers: %d, FP layers: %d", n_quant, n_fp)

    # ============================================================
    # Run inference via pipeline
    # ============================================================
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            transformer=model.model,
            torch_dtype=dtype,
        ).to(device)

        prompts = [args.prompt] if args.prompt else [
            "A serene lake surrounded by mountains at sunset."
        ]

        for i, prompt in enumerate(prompts):
            logger.info("Generating video %d: %s", i + 1, prompt[:80])
            start_time = time.time()

            output = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=device).manual_seed(args.seed + i),
            )

            elapsed = time.time() - start_time
            logger.info("Generation took %.2f seconds", elapsed)

            # Save output
            if hasattr(output, 'frames') and output.frames is not None:
                save_path = os.path.join(args.save_dir, f"quant_video_{i}.mp4")
                # Export frames depend on the pipeline output format
                try:
                    from diffusers.utils import export_to_video
                    export_to_video(output.frames[0], save_path, fps=args.fps)
                    logger.info("Saved video to %s", save_path)
                except Exception as e:
                    logger.warning("Could not save video: %s", e)

    except Exception as e:
        logger.warning("Pipeline inference failed: %s", e)
        logger.info("Please integrate the quantized model into your pipeline manually.")
        logger.info("The quantized transformer model is available as `model.model`.")

    logger.info("Inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantized inference for Wan2.2-Fun")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="t2v", choices=["t2v", "i2v", "flf2v"])
    parser.add_argument("--quant-config", type=str, required=True)
    parser.add_argument("--quant-params", type=str, required=True, help="Path to quant_params.pth")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="./quant_output")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
