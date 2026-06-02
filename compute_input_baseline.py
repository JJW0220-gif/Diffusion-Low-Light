from pathlib import Path
import argparse

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_msssim import ssim


def compute_metrics(split_root: Path, max_samples=None, device="cpu", progress_every=100, per_image=False):
    stems = sorted(
        p.name[:-8]
        for p in split_root.glob("*-in.webp")
        if (split_root / f"{p.name[:-8]}-gt.webp").exists()
    )
    if not stems:
        raise RuntimeError(f"No paired samples found under {split_root}")
    if max_samples is not None:
        stems = stems[:max_samples]

    psnr_sum = 0.0
    ssim_sum = 0.0
    valid_psnr_count = 0
    identical_pairs = []
    device = torch.device(device)
    total = len(stems)
    for index, stem in enumerate(stems, start=1):
        input_tensor = TF.to_tensor(Image.open(split_root / f"{stem}-in.webp").convert("RGB")).unsqueeze(0).to(device)
        gt_tensor = TF.to_tensor(Image.open(split_root / f"{stem}-gt.webp").convert("RGB")).unsqueeze(0).to(device)
        mse = torch.mean((input_tensor - gt_tensor) ** 2)
        mse_value = float(mse.item())
        if mse_value == 0.0:
            psnr_value = float("inf")
            identical_pairs.append(stem)
        else:
            psnr_value = float((10.0 * torch.log10(1.0 / mse)).item())
            psnr_sum += psnr_value
            valid_psnr_count += 1
        ssim_value = float(ssim(input_tensor, gt_tensor, data_range=1.0).item())
        ssim_sum += ssim_value
        if per_image:
            print(f"image {stem} | PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")
        elif index == 1 or index % progress_every == 0 or index == total:
            print(f"processed {index}/{total}")

    count = total
    return {
        "count": count,
        "mean_psnr": psnr_sum / valid_psnr_count if valid_psnr_count > 0 else float("inf"),
        "mean_ssim": ssim_sum / count,
        "identical_pair_count": len(identical_pairs),
        "identical_pairs": identical_pairs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute raw input-vs-GT PSNR/SSIM baseline")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Paired split to evaluate")
    parser.add_argument("--data_root", default="data", help="Dataset root that contains train/ val/ test")
    parser.add_argument("--max_samples", type=int, default=None, help="Optionally limit the number of paired samples")
    parser.add_argument("--device", default="cpu", help="Device to use, e.g. cpu or cuda")
    parser.add_argument("--progress_every", type=int, default=100, help="Print progress every N samples")
    parser.add_argument("--per_image", action="store_true", help="Print PSNR/SSIM for each image instead of periodic progress")
    args = parser.parse_args()

    split_root = Path(args.data_root) / args.split
    metrics = compute_metrics(
        split_root,
        max_samples=args.max_samples,
        device=args.device,
        progress_every=args.progress_every,
        per_image=args.per_image,
    )
    split_name = args.split.capitalize()
    print(
        f"{split_name} summary ({metrics['count']} images) | "
        f"PSNR: {metrics['mean_psnr']:.4f}, SSIM: {metrics['mean_ssim']:.4f}"
    )
    if metrics["identical_pair_count"] > 0:
        print(
            f"Skipped {metrics['identical_pair_count']} identical input/GT pairs when averaging PSNR"
        )
