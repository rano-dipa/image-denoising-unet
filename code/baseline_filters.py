#!/usr/bin/env python3
"""
baseline_filters.py

Apply simple filters (Gaussian, Median, Bilateral) to the SIDD validation/test images,
save the filtered outputs, and compute PSNR/SSIM for each filter.

Usage:
  python code/baseline_filters.py --data_root data/SIDD --out_dir outputs/baseline
"""
import os
import argparse
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import csv

def apply_filters(img_np):
    # img_np: H x W x 3, float32 in [0,1], RGB
    img_8 = (img_np * 255.0).astype(np.uint8)
    # OpenCV uses BGR
    img_bgr = img_8[:, :, ::-1]
    gauss = cv2.GaussianBlur(img_bgr, (5,5), sigmaX=1.0)
    median = cv2.medianBlur(img_bgr, 3)
    bilateral = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    # convert back to RGB float [0,1]
    res = {
        'gauss': gauss[:, :, ::-1].astype(np.float32) / 255.0,
        'median': median[:, :, ::-1].astype(np.float32) / 255.0,
        'bilateral': bilateral[:, :, ::-1].astype(np.float32) / 255.0
    }
    return res

def safe_ssim(gt, out):
    # gt, out: HxWx3 float [0,1]
    h, w, _ = gt.shape
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    return compare_ssim(gt, out, channel_axis=2, data_range=1.0, win_size=win)

def main(data_root, out_dir):
    data_root = Path(data_root)
    data_dir = data_root / "Data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Expected dataset Data/ under {data_root}. Found: {list(data_root.iterdir())}")

    # create output subfolders
    out_dir = Path(out_dir)
    for name in ['gauss','median','bilateral']:
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    # metrics accumulators
    psnr_vals = { 'gauss': [], 'median': [], 'bilateral': [] }
    ssim_vals = { 'gauss': [], 'median': [], 'bilateral': [] }

    scene_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    total = 0
    for scene in scene_dirs:
        noisy_list = sorted(glob(str(scene / "*NOISY*.PNG")))
        gt_list = sorted(glob(str(scene / "*GT*.PNG")))
        for noisy_path, gt_path in zip(noisy_list, gt_list):
            total += 1
            noisy = np.array(Image.open(noisy_path).convert("RGB")).astype(np.float32)/255.0
            gt = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32)/255.0

            results = apply_filters(noisy)
            for k, out_img in results.items():
                # save image (uint8)
                save_name = out_dir / k / Path(noisy_path).name
                img_uint8 = (out_img * 255.0).round().astype(np.uint8)
                Image.fromarray(img_uint8).save(save_name)

                # compute metrics
                ps = compare_psnr(gt, out_img, data_range=1.0)
                ss = safe_ssim(gt, out_img)
                psnr_vals[k].append(ps)
                ssim_vals[k].append(ss)

            if total % 50 == 0:
                print(f"Processed {total} images...")

    # summarize
    metrics_csv = out_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filter","mean_psnr","mean_ssim","num_images"])
        for k in ['gauss','median','bilateral']:
            mean_ps = float(np.mean(psnr_vals[k])) if len(psnr_vals[k])>0 else 0.0
            mean_ss = float(np.mean(ssim_vals[k])) if len(ssim_vals[k])>0 else 0.0
            writer.writerow([k, f"{mean_ps:.4f}", f"{mean_ss:.4f}", len(psnr_vals[k])])
            print(f"{k:8s}  PSNR: {mean_ps:.4f}  SSIM: {mean_ss:.4f}  (n={len(psnr_vals[k])})")

    print("Done. Saved filtered images and metrics to:", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Path to SIDD root (folder containing Data/)")
    p.add_argument("--out_dir", type=str, required=True, help="Where to save filtered outputs and metrics")
    args = p.parse_args()
    main(args.data_root, args.out_dir)
