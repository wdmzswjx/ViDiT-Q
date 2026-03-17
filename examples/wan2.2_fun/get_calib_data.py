"""
Calibration data collection for Wan2.2-Fun ViDiT-Q quantization.

Collects activation statistics (channel-wise max) from the FP model
by hooking into all nn.Linear layers during a few forward passes.

Usage:
    python get_calib_data.py \
        --model-path ./models/Wan2.2-Fun-14B-InP \
        --model-type t2v \
        --quant-config ./configs/w8a8.yaml \
        --prompt prompts.txt \
        --log ./calib_output \
        --num-inference-steps 50 \
        --seed 42
"""
import torch
import torch.nn as nn
import os
import sys
import argparse
import logging

from omegaconf import OmegaConf
from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging


class SaveActivationHook:
    """Hook to collect channel-wise max of activation inputs to nn.Linear layers."""

    def __init__(self):
        self.hook_handle = None
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        """
        Collect channel-wise absolute max of the input.
        Input shape: [BS, N_token, C] or [BS, C]
        Output: [C] (channel-wise max across all tokens and batch)
        """
        C = module_in[0].shape[-1]
        data = module_in[0].reshape([-1, C]).abs().max(dim=0)[0]  # [C]
        self.outputs.append(data.cpu())

    def clear(self):
        self.outputs = []


def add_hook_to_module_(module, hook_cls):
    hook = hook_cls()
    hook.hook_handle = module.register_forward_hook(hook)
    return hook


def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.log is not None:
        os.makedirs(args.log, exist_ok=True)
    log_file = os.path.join(args.log, 'calib_run.log')
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

    # ============================================================
    # Load model - supports multiple loading patterns
    # ============================================================
    logger.info("Loading model from %s ...", args.model_path)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Try loading as VideoX-Fun WanTransformer3DModel
    model = None
    try:
        from videox_fun.models.wan_transformer3d import WanTransformer3DModel
        model = WanTransformer3DModel.from_pretrained(
            args.model_path,
            subfolder="transformer" if os.path.isdir(os.path.join(args.model_path, "transformer")) else None,
            torch_dtype=dtype,
        ).to(device).eval()
        logger.info("Loaded as VideoX-Fun WanTransformer3DModel")
    except (ImportError, Exception) as e:
        logger.info("VideoX-Fun WanTransformer3DModel not available: %s", e)

    # Try loading as diffusers WanTransformer3DModel
    if model is None:
        try:
            from diffusers.models import AutoModel
            model = AutoModel.from_pretrained(
                args.model_path,
                subfolder="transformer" if os.path.isdir(os.path.join(args.model_path, "transformer")) else None,
                torch_dtype=dtype,
            ).to(device).eval()
            logger.info("Loaded as diffusers model")
        except (ImportError, Exception) as e:
            logger.info("diffusers AutoModel not available: %s", e)

    # Try loading as original WanModel
    if model is None:
        try:
            from wan.modules.model import WanModel
            model = WanModel(
                model_type=args.model_type,
                **OmegaConf.load(os.path.join(args.model_path, "config.yaml"))
            ).to(device).eval()
            # Load weights
            ckpt_path = os.path.join(args.model_path, "model.safetensors")
            if os.path.exists(ckpt_path):
                from safetensors.torch import load_file
                state_dict = load_file(ckpt_path)
                model.load_state_dict(state_dict)
            logger.info("Loaded as original WanModel")
        except (ImportError, Exception) as e:
            logger.info("Original WanModel not available: %s", e)

    if model is None:
        raise RuntimeError(
            f"Could not load model from {args.model_path}. "
            "Please ensure either videox_fun, diffusers, or wan package is available."
        )

    logger.info("Model architecture:\n%s", str(model)[:2000])

    # ============================================================
    # Add hooks to all nn.Linear layers
    # ============================================================
    kwargs = {'hook_cls': SaveActivationHook}
    hook_d = apply_func_to_submodules(
        model,
        class_type=nn.Linear,
        function=add_hook_to_module_,
        return_d={},
        **kwargs,
    )
    logger.info("Registered hooks on %d nn.Linear layers", len(hook_d))

    # ============================================================
    # Run inference to collect activations
    # ============================================================
    # Read prompts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    if os.path.exists(prompt_path):
        with open(prompt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    if not prompts:
        prompts = [
            "A serene lake surrounded by mountains at sunset.",
            "A bustling city street at night with neon lights.",
            "A cat playing with a ball of yarn on a wooden floor.",
            "Time lapse of clouds moving over a green valley.",
        ]
    logger.info("Using %d prompts for calibration", len(prompts))

    # Try pipeline-based inference (VideoX-Fun / diffusers)
    try:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            transformer=model,
            torch_dtype=dtype,
        ).to(device)

        for i, prompt in enumerate(prompts):
            logger.info("Calibration forward pass %d/%d: %s", i + 1, len(prompts), prompt[:80])
            _ = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                generator=torch.Generator(device=device).manual_seed(args.seed + i),
            )
    except Exception as e:
        logger.warning("Pipeline inference failed: %s. Using dummy forward pass.", e)
        # Fallback: dummy forward pass for calibration
        _run_dummy_forward(model, device, dtype, prompts, args, logger)

    # ============================================================
    # Save calibration data
    # ============================================================
    save_d = {}
    for k, v in hook_d.items():
        if len(v.outputs) > 0:
            save_d[k] = torch.stack(v.outputs, dim=0)  # [N_timestep, C]
            logger.info(
                'layer: %s, shape: %s, n_samples: %d',
                k, v.outputs[0].shape, len(v.outputs)
            )
        v.hook_handle.remove()

    save_path = os.path.join(args.log, quant_config.calib_data.save_path)
    torch.save(save_d, save_path)
    logger.info("Saved calibration data to %s (%d layers)", save_path, len(save_d))


def _run_dummy_forward(model, device, dtype, prompts, args, logger):
    """
    Run dummy forward passes through the model directly (without pipeline).
    This is a fallback when the full pipeline is not available.
    """
    # Determine model dimensions from config or attributes
    dim = getattr(model, 'dim', 2048)
    text_len = getattr(model, 'text_len', 512)
    text_dim = getattr(model, 'text_dim', 4096)
    patch_size = getattr(model, 'patch_size', (1, 2, 2))
    in_dim = getattr(model, 'in_dim', 16)
    freq_dim = getattr(model, 'freq_dim', 256)

    # Compute latent sizes
    F_lat = (args.num_frames - 1) // patch_size[0] + 1 if args.num_frames > 1 else 1
    H_lat = args.height // 8  # VAE downscale
    W_lat = args.width // 8

    for i in range(min(len(prompts), 4)):
        logger.info("Dummy forward pass %d", i + 1)
        # Create dummy inputs matching WanModel.forward() signature
        x = [torch.randn(in_dim, F_lat, H_lat, W_lat, device=device, dtype=dtype)]
        t = torch.tensor([500.0], device=device)
        context = [torch.randn(text_len, text_dim, device=device, dtype=dtype)]
        seq_len = (F_lat // patch_size[0]) * (H_lat // patch_size[1]) * (W_lat // patch_size[2])

        try:
            _ = model(x=x, t=t, context=context, seq_len=seq_len)
        except Exception as e:
            logger.warning("Dummy forward failed: %s", e)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect calibration data for Wan2.2-Fun ViDiT-Q")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Wan model")
    parser.add_argument("--model-type", type=str, default="t2v", choices=["t2v", "i2v", "flf2v"],
                        help="Model type")
    parser.add_argument("--quant-config", type=str, required=True, help="Path to quant config YAML")
    parser.add_argument("--prompt", type=str, default=None, help="Path to prompts file")
    parser.add_argument("--log", type=str, required=True, help="Output directory")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    main(args)
