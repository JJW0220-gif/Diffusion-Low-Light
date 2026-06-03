import torch
import numpy as np
import utils
import os
import torch.nn.functional as F
from pytorch_msssim import ssim

try:
    import lpips
except ImportError:
    lpips = None


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion
        self.lpips_model = None

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

        if lpips is not None:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.diffusion.device)
            self.lpips_model.eval()
            print('=> Evaluation metrics: PSNR, SSIM, LPIPS')
        elif getattr(self.args, 'require_lpips', False):
            raise RuntimeError('LPIPS is required for evaluation but the lpips package is not installed.')
        else:
            print('=> Evaluation metrics: PSNR, SSIM (LPIPS disabled; lpips package not installed)')

    @staticmethod
    def _flip_batch(x, flip_h=False, flip_v=False):
        if flip_h:
            x = torch.flip(x, dims=[3])
        if flip_v:
            x = torch.flip(x, dims=[2])
        return x

    def _predict(self, x_cond):
        x_output = self.diffusion.model(x_cond)
        return x_output["pred_x"]

    def _tta_predict(self, x_cond):
        variants = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
        preds = []
        for flip_h, flip_v in variants:
            augmented = self._flip_batch(x_cond, flip_h=flip_h, flip_v=flip_v)
            pred = self._predict(augmented)
            pred = self._flip_batch(pred, flip_h=flip_h, flip_v=flip_v)
            preds.append(pred)
        return torch.stack(preds, dim=0).mean(dim=0)

    @staticmethod
    def _compute_psnr(pred, target):
        mse = torch.mean((pred - target) ** 2)
        if mse <= 0:
            return float('inf')
        return float((10.0 * torch.log10(1.0 / mse)).item())

    def _compute_metrics(self, pred, target):
        metrics = {
            "psnr": self._compute_psnr(pred, target),
            "ssim": float(ssim(pred, target, data_range=1.0).item()),
        }
        if self.lpips_model is not None:
            pred_norm = pred * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            metrics["lpips"] = float(self.lpips_model(pred_norm, target_norm).mean().item())
        return metrics

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        metric_sums = {"psnr": 0.0, "ssim": 0.0}
        if self.lpips_model is not None:
            metric_sums["lpips"] = 0.0
        metric_count = 0

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_gt = x[:, 3:, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')
                x_output = self.diffusive_restoration(x_cond)
                x_output = x_output[:, :, :h, :w]
                metrics = self._compute_metrics(x_output, x_gt)
                for key, value in metrics.items():
                    metric_sums[key] += value
                metric_count += 1
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
                metric_message = ", ".join(f"{key.upper()}: {value:.4f}" for key, value in metrics.items())
                print(f"processing image {y[0]} | {metric_message}")

        if metric_count == 0:
            return

        mean_metrics = {key: value / metric_count for key, value in metric_sums.items()}
        summary_message = ", ".join(f"{key.upper()}: {value:.4f}" for key, value in mean_metrics.items())
        print(f"Validation summary ({metric_count} images) | {summary_message}")

        metrics_path = os.path.join(image_folder, "metrics.txt")
        os.makedirs(image_folder, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as handle:
            handle.write(f"count: {metric_count}\n")
            for key, value in mean_metrics.items():
                handle.write(f"{key}: {value:.6f}\n")

    def diffusive_restoration(self, x_cond):
        if getattr(self.args, 'tta', False):
            return self._tta_predict(x_cond)
        return self._predict(x_cond)

