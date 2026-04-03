#!/usr/bin/env python3
import os
import json
import argparse
from glob import glob
from PIL import Image

import torch
import torchvision.transforms as T
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

try:
    import lpips
    LPIPS_AVAILABLE = True
except:
    LPIPS_AVAILABLE = False

from train_unet import UNetSimple


# ---------- pad to multiple of 16 ----------
def pad_to_16(x):
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


# ---------- tiled inference ----------
def run_tiled(model, img_tensor, tile=256, overlap=32):
    _, _, H, W = img_tensor.shape

    output = torch.zeros_like(img_tensor)
    weight = torch.zeros_like(img_tensor)

    stride = tile - overlap

    for y in range(0, H, stride):
        for x in range(0, W, stride):

            y1 = y
            x1 = x
            y2 = min(y + tile, H)
            x2 = min(x + tile, W)

            patch = img_tensor[:, :, y1:y2, x1:x2]

            # ---- pad patch to multiple of 16 ----
            patch, pad_h, pad_w = pad_to_16(patch)

            with torch.no_grad():
                pred = model(patch)

            # ---- remove padding ----
            if pad_h > 0:
                pred = pred[:, :, :-pad_h, :]
            if pad_w > 0:
                pred = pred[:, :, :, :-pad_w]

            output[:, :, y1:y2, x1:x2] += pred
            weight[:, :, y1:y2, x1:x2] += 1.0

    return output / weight


def main(args):
    project = args.project_root
    splits_file = os.path.join(project, "splits.json")

    with open(splits_file, "r") as f:
        splits = json.load(f)

    test_scenes = splits["test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading model:", args.ckpt)
    model = UNetSimple(base=48).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if LPIPS_AVAILABLE:
        print("LPIPS enabled")
        lpips_fn = lpips.LPIPS(net="alex").to(device)
    else:
        lpips_fn = None

    to_tensor = T.ToTensor()

    all_psnr = []
    all_ssim = []
    all_lpips = []

    for scene in test_scenes:
        noisy_paths = sorted(glob(os.path.join(scene, "*NOISY*.PNG")))
        gt_paths = sorted(glob(os.path.join(scene, "*GT*.PNG")))

        for npth, gpth in zip(noisy_paths, gt_paths):
            noisy = Image.open(npth).convert("RGB")
            gt = Image.open(gpth).convert("RGB")

            inp = to_tensor(noisy).unsqueeze(0).to(device)

            out = run_tiled(model, inp).clamp(0,1).cpu().squeeze(0)

            out_np = out.permute(1,2,0).numpy()
            gt_np = np.array(gt).astype(np.float32)/255.0

            ps = compare_psnr(gt_np, out_np, data_range=1.0)

            h,w,_ = gt_np.shape
            win = min(7,h,w)
            if win % 2 == 0:
                win -= 1
            win = max(win,3)

            ss = compare_ssim(gt_np, out_np,
                              channel_axis=2,
                              data_range=1.0,
                              win_size=win)

            lp = None
            if lpips_fn is not None:
                a = torch.tensor(out_np).permute(2,0,1).unsqueeze(0).to(device)
                b = torch.tensor(gt_np).permute(2,0,1).unsqueeze(0).to(device)
                lp = float(lpips_fn(a,b).item())

            all_psnr.append(ps)
            all_ssim.append(ss)
            if lp is not None:
                all_lpips.append(lp)

    print("\n===== FINAL TEST RESULTS =====")
    print("Mean PSNR :", np.mean(all_psnr))
    print("Mean SSIM :", np.mean(all_ssim))
    if len(all_lpips)>0:
        print("Mean LPIPS:", np.mean(all_lpips))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str,
        default="/scratch/user/dipanwita22rano/denoising_project")
    parser.add_argument("--ckpt", type=str,
        default="/scratch/user/dipanwita22rano/denoising_project/checkpoints/best_model.pth")

    args = parser.parse_args()
    main(args)
