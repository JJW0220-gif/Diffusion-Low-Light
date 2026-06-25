# [Siggraph Asia 2023]Low-light Image Enhancement with Wavelet-based Diffusion Models [[Paper]](https://arxiv.org/pdf/2306.00306.pdf).
<h4 align="center">Hai Jiang<sup>1,2</sup>, Ao Luo<sup>2</sup>, Haoqiang Fan<sup>2</sup>, Songchen Han<sup>1</sup>, Shuaicheng Liu<sup>3,2</sup></center>
<h4 align="center">1.Sichuan University, 2.Megvii Technology, 
<h4 align="center">3.University of Electronic Science and Technology of China</center></center>

## Pipeline
![](./Figures/pipeline.png)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### Paired datasets 
LOLv1 dataset: Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep Retinex Decomposition for Low-Light Enhancement", BMVC, 2018. [[Baiduyun (extracted code: sdd0)]](https://pan.baidu.com/s/1spt0kYU3OqsQSND-be4UaA) [[Google Drive]](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing)

LOLv2 dataset: Wenhan Yang, Haofeng Huang, Wenjing Wang, Shiqi Wang, and Jiaying Liu. "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light Image Enhancement", TIP, 2021. [[Baiduyun (extracted code: l9xm)]](https://pan.baidu.com/s/1U9ePTfeLlnEbr5dtI1tm5g) [[Google Drive]](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC/view?usp=sharing)

LSRW dataset: Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network", Journal of Visual Communication and Image Representation, 2023. [[Baiduyun (extracted code: wmrr)]](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)

### Unpaired datasets 
Please refer to [[Project Page of RetinexNet.]](https://daooshee.github.io/BMVC2018website/)

## Pre-trained Models 
You can downlaod our pre-trained model from [[Google Drive]](https://drive.google.com/file/d/1f4zDvPsWKrID33OJdeHwc5VOBILkm0KW/view?usp=sharing) and [[Baidu Yun (extracted code:wsw7)]](https://pan.baidu.com/s/1rq8VzdnHeky0iT56coOGog)

## How to train?
For the new paired WebP dataset, place the splits under a single root directory:

```
Diffusion-Low-Light/
├── data/
│   └── low-light/
│       ├── train/
│       ├── val/
│       └── test/
```

Each file should follow `<stem>-in.webp` and `<stem>-gt.webp` for `train/` and `val/`, while `test/` contains only `<stem>-in.webp`.

The provided config `configs/LowLightWebP.yml` assumes the dataset root is `./data/low-light`, so after placing the files there you can run:

```
python train.py --config LowLightWebP.yml
```

## How to test?
```
python evaluate.py --config LowLightWebP.yml
```

## Visual comparison
![](./Figures/comparison.png)

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{jiang2023low,
  title={Low-light image enhancement with wavelet-based diffusion models},
  author={Jiang, Hai and Luo, Ao and Fan, Haoqiang and Han, Songchen and Liu, Shuaicheng},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={6},
  pages={1--14},
  year={2023}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [WeatherDiff](https://github.com/IGITUGraz/WeatherDiffusion), [SDWNet](https://github.com/FlyEgle/SDWNet), and [MIMO-UNet](https://github.com/chosj95/MIMO-UNet). We thank all the authors for their contributions.

---

## Extended Setup and Workflow (Project Updates)

This section is added for the current project branch with WebP paired data, multi-variant training, and enhanced evaluation/inference.

### 1) Environment Setup

Recommended:
- Python 3.10+
- CUDA-capable GPU

Create environment (example with conda):

```bash
conda create -n lowlight python=3.10 -y
conda activate lowlight
pip install -r requirements.txt
```

### 2) Dataset Layout (Current)

```text
Diffusion-Low-Light/
├── data/
│   └── low-light/
│       ├── train/
│       ├── val/
│       └── test/
```

Filename convention:
- train/val: `<id>-in.webp` and `<id>-gt.webp`
- test: `<id>-in.webp` only

### 3) Configs and Model Variants

- `configs/LowLightWebP.yml`: ddpm, checkpoints in `ckpt/`
- `configs/LowLightWebP_ckpt2.yml`: ddpm2, checkpoints in `ckpt2/`
- `configs/LowLightWebP_chk3.yml`: ddpm3, checkpoints in `chk3/`

`train.py` and `evaluate.py` route model variants by `training.model_variant` (with `ckpt_dir` fallback).

### 4) Training

Train from scratch:

```bash
python train.py --config LowLightWebP.yml
python train.py --config LowLightWebP_ckpt2.yml
python train.py --config LowLightWebP_chk3.yml
```

Resume training:

```bash
python train.py --config LowLightWebP.yml --resume ckpt/latest.pth.tar
python train.py --config LowLightWebP_ckpt2.yml --resume ckpt2/latest.pth.tar
python train.py --config LowLightWebP_chk3.yml --resume chk3/latest.pth.tar
```

Notes:
- ddpm3 validates `--resume` path strictly when provided.
- `n_epochs: 0` in ddpm3 config means open-ended training.
- CSV loss logs are saved per variant (for example: `loss_log_ddm.csv`, `loss_log_ddm2.csv`, `loss_log_ddm3.csv`).

### 5) Evaluation

Without TTA:

```bash
python evaluate.py --config LowLightWebP.yml --resume ckpt/bestbest.pth.tar --sampling_timesteps 10 --image_folder results/team_7/no_tta
```

With TTA (rotation + flip + shift):

```bash
python evaluate.py --config LowLightWebP.yml --resume ckpt/bestbest.pth.tar --sampling_timesteps 10 --tta --tta_shift_pixels 2 --image_folder results/team_7/tta_shift2
```

### 6) Test Inference (Submission Images)

Default output naming in `infer_test.py` is `<id>.png`.

```bash
python infer_test.py --config LowLightWebP.yml --resume ckpt/bestbest.pth.tar --sampling_timesteps 10 --image_folder results/submission/base
python infer_test.py --config LowLightWebP.yml --resume ckpt/bestbest.pth.tar --sampling_timesteps 10 --tta --tta_shift_pixels 2 --image_folder results/submission/tta
```

### 7) Model Size and FLOPs

```bash
python profile_model.py --config LowLightWebP_chk3.yml --height 256 --width 256 --batch-size 1 --sampling-timesteps 10
```

This reports parameters (M), MACs (G), and FLOPs (G, with 1 MAC = 2 FLOPs).

### 8) Summary of Added/Modified Features

- Added ddpm3 implementation and config (`models/ddm3.py`, `configs/LowLightWebP_chk3.yml`).
- Added automatic variant routing (ddpm/ddpm2/ddpm3) in training/evaluation entry scripts.
- Updated ddpm loss to weighted `MSE + SSIM + LPIPS`.
- Added high-frequency soft-threshold in ddpm2 (`hf_soft_threshold`).
- Added OHEM-based sample selection and OHEM logging fields in ddpm3.
- Expanded TTA in restoration/inference: rotation, flip, and reflective pixel shifting.
- Added strict checkpoint fail-fast behavior and checkpoint key compatibility handling.
- Added profiling utility (`profile_model.py`) and `thop` dependency.
- Added dataset loader option `training.drop_last`.
