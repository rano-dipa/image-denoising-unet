# Perceptual Image Denoising using Multi-Scale U-Net

## Overview

This project presents a deep learning-based image denoising framework using a U-Net architecture enhanced with perceptual loss, multi-scale training, and model optimization techniques. The objective is to remove real-world noise from images while preserving structural details and perceptual quality.

The model is trained and evaluated on the [Smartphone Image Denoising Dataset (SIDD)](https://abdokamel.github.io/sidd/), which contains real noisy-clean image pairs captured under varying lighting conditions.

---

## Key Features

- U-Net based denoising model
- Perceptual loss using LPIPS
- Multi-scale training for improved generalization
- Mixed precision (AMP) for efficient GPU training
- Model pruning for lightweight deployment
- Tiled inference for handling high-resolution images
- Baseline comparison with classical filtering methods

---

## Results

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| Gaussian Filter | 32.33 | 0.73 | — |
| Median Filter | 30.95 | 0.67 | — |
| Bilateral Filter | 34.36 | 0.82 | — |
| U-Net (Single Scale) | 37.39 | 0.925 | 0.112 |
| **U-Net (Multi-Scale)** | **38.39** | **0.928** | **0.106** |
| Pruned (15%) | 35.83 | 0.926 | 0.111 |

Multi-scale training improves performance across all metrics, while moderate pruning provides a balance between efficiency and accuracy.

---

## Project Structure

```
denoising_project/
│
├── code/
│   ├── train_unet.py
│   ├── train_final.py
│   ├── baseline_filters.py
│   ├── eval_on_test.py
│   ├── prune_and_eval.py
│   └── infer_image.py
│
├── splits.json
|
│── train_unet.slurm
│── train_final.slurm
│── baseline.slurm
│── eval_test.slurm
│── prune.slurm
│
├── logs/
├── checkpoints/
├── outputs/
├── figures/
├── Perceptual_Image_denoising.pdf
└── README.md
```

---

## Setup

### Create environment

```bash
conda create -n denoise python=3.10
conda activate denoise
```

### Install dependencies

```bash
pip install torch torchvision
pip install numpy pillow matplotlib scikit-image
pip install opencv-python lpips
```

---

## Dataset

Download the SIDD dataset from [https://abdokamel.github.io/sidd/](https://abdokamel.github.io/sidd/) and place it as:

```
data/SIDD/
```

---

## Training

### Baseline U-Net (single-scale)

```bash
python code/train_unet.py \
  --data_root data/SIDD \
  --project_root . \
  --epochs 50
```

### Multi-scale training (final model)

```bash
python code/train_final.py \
  --project_root . \
  --epochs 30
```

### SLURM execution

```bash
sbatch slurm/train_final.slurm
```

---

## Evaluation

```bash
python code/eval_on_test.py \
  --project_root . \
  --ckpt checkpoints/final_best.pth
```

---

## Baseline Filters

```bash
python code/baseline_filters.py \
  --data_root data/SIDD \
  --out_dir outputs/baseline
```

---

## Model Pruning

```bash
python code/prune_and_eval.py \
  --ckpt checkpoints/final_best.pth \
  --amount 0.15
```

---

## Inference

Run inference on a single noisy image:

```bash
python code/infer_image.py \
  --input sample_noisy.png \
  --output denoised.png \
  --ckpt checkpoints/final_best.pth
```

The script supports both full and pruned models.

---

## Pipeline

1. Input noisy image
2. Optional tiling for large images
3. U-Net based denoising
4. Reconstruction
5. Final denoised output

---

## Optimization Techniques

- **Multi-scale training** — improves robustness across image resolutions
- **Automatic mixed precision (AMP)** — reduces memory usage and speeds up training
- **Model pruning** — reduces model size for deployment
- **Tiled inference** — enables processing of high-resolution images

---

## Limitations

- Requires paired noisy-clean datasets for supervised training
- Tiling introduces additional inference overhead
- Aggressive pruning leads to performance degradation

---

## Future Work

- Self-supervised denoising approaches
- Transformer-based architectures
- Quantization and structured pruning
- Real-time deployment optimization

---

## Report

Full methodology, experiments, and results available here:  
[Perceptual_Image_Denoising.pdf](./Perceptual_Image_Denoising.pdf)
