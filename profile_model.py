import argparse
import os

import torch
import yaml
from thop import profile

from models.ddm import Net as DDPMNet
from models.ddm2 import Net as DDPM2Net
from models.ddm3 import Net as DDPM3Net


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace


def resolve_config_path(config_name):
    if os.path.isfile(config_name):
        return config_name
    candidate = os.path.join("configs", config_name)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(f"Config file not found: {config_name}")


def select_model(config):
    ckpt_dir_name = os.path.basename(os.path.normpath(config.data.ckpt_dir)).lower()
    model_variant = getattr(config.training, "model_variant", "")
    model_variant = model_variant.lower() if isinstance(model_variant, str) else ""

    if model_variant in {"ddpm", "ddm"}:
        return "ddpm", DDPMNet
    if model_variant in {"ddpm2", "ddm2"}:
        return "ddpm2", DDPM2Net
    if model_variant in {"ddpm3", "ddm3"}:
        return "ddpm3", DDPM3Net
    if ckpt_dir_name == "ckpt2":
        return "ddpm2", DDPM2Net
    if ckpt_dir_name == "chk3":
        return "ddpm3", DDPM3Net
    return "ddpm", DDPMNet


def profile_one_config(config_path, args):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = dict2namespace(yaml.safe_load(handle))

    config.device = torch.device("cpu")
    model_name, model_cls = select_model(config)
    model_args = argparse.Namespace(sampling_timesteps=args.sampling_timesteps)
    model = model_cls(model_args, config).eval()

    channels = args.channels if args.channels is not None else getattr(config.data, "channels", 3)
    dummy_input = torch.randn(args.batch_size, channels, args.height, args.width)

    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    flops = macs * 2.0

    return {
        "config": config_path,
        "model": model_name,
        "params": total_params,
        "m_params": total_params / 1e6,
        "macs": macs,
        "gmacs": macs / 1e9,
        "flops": flops,
        "gflops": flops / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile model size and FLOPs")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more config file names or paths",
    )
    parser.add_argument("--height", type=int, default=256, help="Input height")
    parser.add_argument("--width", type=int, default=256, help="Input width")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for profiling")
    parser.add_argument("--channels", type=int, default=None, help="Override input channel count")
    parser.add_argument(
        "--sampling-timesteps",
        type=int,
        default=10,
        help="Sampling timesteps used by the diffusion sampler",
    )
    args = parser.parse_args()

    for config_name in args.config:
        config_path = resolve_config_path(config_name)
        result = profile_one_config(config_path, args)
        print(f"Config: {result['config']}")
        print(f"Model: {result['model']}")
        print(f"Input: {args.batch_size}x{args.channels or 3}x{args.height}x{args.width}")
        print(f"Model size: {result['m_params']:.3f} M param")
        print(f"MACs: {result['gmacs']:.3f} GMACs")
        print(f"FLOPs: {result['gflops']:.3f} GFLOPs")
        print("Assumption: 1 MAC = 2 FLOPs, FP32 inference")
        print()


if __name__ == "__main__":
    main()