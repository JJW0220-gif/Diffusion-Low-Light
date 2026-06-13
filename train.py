import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import torchvision
import models
import datasets
import utils
from models.ddm import DenoisingDiffusion as DenoisingDiffusionDDPM
from models.ddm2 import DenoisingDiffusion as DenoisingDiffusionDDPM2
from models.ddm3 import DenoisingDiffusion as DenoisingDiffusionDDPM3


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='LOLv1.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=230, type=int, metavar='N',
                        help='Seed for initializing training (default: 230)')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.train_dataset))
    DATASET = datasets.__dict__[config.data.type](config)

    # create model
    print("=> creating denoising-diffusion model...")
    ckpt_dir_name = os.path.basename(os.path.normpath(config.data.ckpt_dir)).lower()
    model_variant = getattr(config.training, "model_variant", "")
    model_variant = model_variant.lower() if isinstance(model_variant, str) else ""

    if model_variant in {"ddpm", "ddm"}:
        diffusion_cls = DenoisingDiffusionDDPM
        selected = "ddpm"
    elif model_variant in {"ddpm2", "ddm2"}:
        diffusion_cls = DenoisingDiffusionDDPM2
        selected = "ddpm2"
    elif model_variant in {"ddpm3", "ddm3"}:
        diffusion_cls = DenoisingDiffusionDDPM3
        selected = "ddpm3"
    elif ckpt_dir_name == "ckpt2":
        diffusion_cls = DenoisingDiffusionDDPM2
        selected = "ddpm2"
    elif ckpt_dir_name == "chk3":
        diffusion_cls = DenoisingDiffusionDDPM3
        selected = "ddpm3"
    else:
        diffusion_cls = DenoisingDiffusionDDPM
        selected = "ddpm"

    print(f"=> model variant: {selected} (ckpt_dir={config.data.ckpt_dir})")
    diffusion = diffusion_cls(args, config)
    diffusion.train(DATASET)


if __name__ == "__main__":
    main()
