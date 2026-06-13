import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml

from datasets.dataset import TestLowLightDataset
from models import DenoisingDiffusion
from utils.logging import save_image


def parse_args_and_config():
    parser = argparse.ArgumentParser(description="Run low-light restoration on the test split")
    parser.add_argument("--config", default="LowLightWebP.yml", type=str,
                        help="Path to the config file")
    parser.add_argument("--resume", default="ckpt/model_latest.pth.tar", type=str,
                        help="Path to the diffusion model checkpoint")
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default="results/submission", type=str,
                        help="Directory to save restored test images")
    parser.add_argument("--output_ext", default="png", choices=["png", "webp"],
                        help="Output image extension for submission files")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation with flips, 90-degree rotations, and small shifts")
    parser.add_argument("--tta_shift_pixels", type=int, default=2,
                        help="Shift magnitude in pixels used by TTA; set to 0 to disable shift variants")
    parser.add_argument("--seed", default=230, type=int, metavar="N",
                        help="Seed for initializing inference")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return args, dict2namespace(config)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def resolve_test_root(config):
    root = Path(config.data.data_dir)
    direct_test = root / "test"
    if direct_test.is_dir():
        return direct_test

    dataset_name = getattr(config.data, "val_dataset", "") or getattr(config.data, "train_dataset", "")
    nested_test = root / dataset_name / "test"
    if nested_test.is_dir():
        return nested_test

    return direct_test


def load_model(args, config):
    diffusion = DenoisingDiffusion(args, config)
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    diffusion.model.eval()
    return diffusion


def _flip_batch(x, flip_h=False, flip_v=False):
    if flip_h:
        x = torch.flip(x, dims=[3])
    if flip_v:
        x = torch.flip(x, dims=[2])
    return x


def _rotate_batch(x, quarter_turns=0):
    quarter_turns = quarter_turns % 4
    if quarter_turns == 0:
        return x
    return torch.rot90(x, k=quarter_turns, dims=[2, 3])


def _shift_batch(x, shift_x=0, shift_y=0):
    if shift_x == 0 and shift_y == 0:
        return x

    pad_left = max(shift_x, 0)
    pad_right = max(-shift_x, 0)
    pad_top = max(shift_y, 0)
    pad_bottom = max(-shift_y, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    start_x = pad_right
    start_y = pad_bottom
    end_y = start_y + x.shape[2]
    end_x = start_x + x.shape[3]
    return padded[:, :, start_y:end_y, start_x:end_x]


def restore_test_image(diffusion, x_cond, use_tta=False):
    batch, channels, height, width = x_cond.shape
    padded_height = int(32 * np.ceil(height / 32.0))
    padded_width = int(32 * np.ceil(width / 32.0))
    x_cond = F.pad(x_cond, (0, padded_width - width, 0, padded_height - height), "reflect")
    if use_tta:
        shift_pixels = max(0, int(getattr(diffusion.args, "tta_shift_pixels", 2)))
        shift_variants = [(0, 0)]
        if shift_pixels > 0:
            shift_variants.extend([
                (shift_pixels, 0),
                (-shift_pixels, 0),
                (0, shift_pixels),
                (0, -shift_pixels),
            ])

        preds = []
        for quarter_turns in range(4):
            rotated = _rotate_batch(x_cond, quarter_turns=quarter_turns)
            for flip_h, flip_v in ((False, False), (True, False), (False, True), (True, True)):
                flipped = _flip_batch(rotated, flip_h=flip_h, flip_v=flip_v)
                for shift_x, shift_y in shift_variants:
                    augmented = _shift_batch(flipped, shift_x=shift_x, shift_y=shift_y)
                    x_output = diffusion.model(augmented)
                    pred = x_output["pred_x"]
                    pred = _shift_batch(pred, shift_x=-shift_x, shift_y=-shift_y)
                    pred = _flip_batch(pred, flip_h=flip_h, flip_v=flip_v)
                    pred = _rotate_batch(pred, quarter_turns=-quarter_turns)
                    preds.append(pred)
        pred = torch.stack(preds, dim=0).mean(dim=0)
    else:
        x_output = diffusion.model(x_cond)
        pred = x_output["pred_x"]
    return pred[:, :, :height, :width]


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    test_root = resolve_test_root(config)
    print(f"=> using test split '{test_root}'")
    test_dataset = TestLowLightDataset(test_root, transform=None)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    print("=> creating denoising-diffusion model")
    diffusion = load_model(args, config)

    output_dir = Path(args.image_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for x_cond, stem in test_loader:
            x_cond = x_cond.to(diffusion.device)
            pred = restore_test_image(diffusion, x_cond, use_tta=args.tta)
            output_path = output_dir / f"{stem[0]}.{args.output_ext}"
            save_image(pred, str(output_path))
            print(f"saved {output_path.name}")


if __name__ == "__main__":
    main()