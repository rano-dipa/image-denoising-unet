#!/usr/bin/env python3
import os, random, argparse, json
from glob import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.optim as optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from train_unet import UNetSimple

# ---------------------------
# Dataset (FIXED)
# ---------------------------
class MultiScaleSIDD(Dataset):
    def __init__(self, scene_dirs, patches_per_image=50, patch_size=128):
        self.pairs = []
        for s in scene_dirs:
            noisy = sorted(glob(os.path.join(s, "*NOISY*.PNG")))
            gt    = sorted(glob(os.path.join(s, "*GT*.PNG")))
            for a,b in zip(noisy, gt):
                self.pairs.append((a,b))

        self.patch_size = patch_size   # ?? FIXED SIZE PER DATASET
        self.patches_per_image = patches_per_image
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, idx):
        npath, gpath = random.choice(self.pairs)

        noisy = Image.open(npath).convert("RGB")
        gt    = Image.open(gpath).convert("RGB")

        ps = self.patch_size

        w,h = noisy.size
        if w <= ps or h <= ps:
            noisy = noisy.resize((ps,ps))
            gt = gt.resize((ps,ps))
        else:
            x = random.randint(0, w-ps)
            y = random.randint(0, h-ps)
            noisy = noisy.crop((x,y,x+ps,y+ps))
            gt    = gt.crop((x,y,x+ps,y+ps))

        return self.to_tensor(noisy), self.to_tensor(gt)

# ---------------------------
# Metrics
# ---------------------------
def psnr_batch(preds, gts):
    psnrs = []
    for i in range(preds.shape[0]):
        p = preds[i].detach().cpu().permute(1,2,0).numpy()
        g = gts[i].detach().cpu().permute(1,2,0).numpy()
        psnrs.append(compare_psnr(g, p, data_range=1.0))
    return np.mean(psnrs)

# ---------------------------
# MAIN TRAIN
# ---------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = json.load(open(os.path.join(args.project_root, "splits.json")))

    model = UNetSimple(base=48).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1 = nn.L1Loss()

    # ? FIXED AMP
    scaler = torch.amp.GradScaler("cuda")

    best_psnr = 0

    patch_sizes = [64, 96, 128, 192]

    for epoch in range(args.epochs):

        # ? MULTI-SCALE (PER EPOCH)
        current_ps = random.choice(patch_sizes)
        print(f"\nEpoch {epoch} using patch size: {current_ps}")

        train_ds = MultiScaleSIDD(splits["train"], patch_size=current_ps)
        val_ds   = MultiScaleSIDD(splits["val"], patch_size=128)  # fixed val

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=6)
        val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

        # ---- TRAIN ----
        model.train()
        for noisy, gt in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                pred = model(noisy)
                loss = l1(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ---- VALIDATION ----
        model.eval()
        val_psnr = 0

        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy)
                val_psnr += psnr_batch(pred, gt)

        val_psnr /= len(val_loader)

        print(f"EPOCH {epoch} VAL_PSNR: {val_psnr:.4f}")

        # ---- SAVE BEST ----
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(args.project_root, "checkpoints/final_best.pth")
            torch.save({"model": model.state_dict()}, save_path)
            print(f"Saved best model ? {save_path}")

    print("\nTraining Done")
    print("Best PSNR:", best_psnr)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    main(args)