#!/usr/bin/env python3
"""
train_unet.py
Train a U-Net denoiser on SIDD-Medium (scene-wise splits).
Saves checkpoints periodically and best model based on val PSNR.
Resume support included.
"""

import os
import random
import argparse
import json
from glob import glob
from PIL import Image
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim

# metrics
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# try optional LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

# ---------------------------
# Utilities & Dataset
# ---------------------------
def list_scene_dirs(data_root):
    d = os.path.join(data_root, "Data")
    scenes = sorted([os.path.join(d, name) for name in os.listdir(d)
                     if os.path.isdir(os.path.join(d, name))])
    return scenes

def make_splits(data_root, out_split_file, seed=42, frac=(0.6,0.2,0.2)):
    if os.path.exists(out_split_file):
        with open(out_split_file, "r") as f:
            splits = json.load(f)
        return splits
    scenes = list_scene_dirs(data_root)
    random.seed(seed)
    random.shuffle(scenes)
    n = len(scenes)
    ntrain = int(frac[0]*n)
    nval = int(frac[1]*n)
    train = scenes[:ntrain]
    val = scenes[ntrain:ntrain+nval]
    test = scenes[ntrain+nval:]
    splits = {"train": train, "val": val, "test": test}
    with open(out_split_file, "w") as f:
        json.dump(splits, f, indent=2)
    return splits

class SIDDPatchDataset(Dataset):
    def __init__(self, scene_dirs, patch_size=128, augment=True, patches_per_image=50):
        self.pairs = []
        for s in scene_dirs:
            noisy = sorted(glob(os.path.join(s, "*NOISY*.PNG")))
            gt    = sorted(glob(os.path.join(s, "*GT*.PNG")))
            # pair length should match; if duplicates exist, zip safely
            for a,b in zip(noisy, gt):
                self.pairs.append((a,b))
        self.patch_size = patch_size
        self.augment = augment
        self.patches_per_image = patches_per_image
        self.to_tensor = T.ToTensor()

    def __len__(self):
        # logical epoch length
        return max(1, len(self.pairs) * self.patches_per_image)

    def __getitem__(self, idx):
        # pick a random pair
        i = random.randrange(len(self.pairs))
        npath, gpath = self.pairs[i]
        noisy = Image.open(npath).convert("RGB")
        gt    = Image.open(gpath).convert("RGB")
        w,h = noisy.size
        ps = self.patch_size
        if w <= ps or h <= ps:
            noisy = noisy.resize((ps,ps))
            gt = gt.resize((ps,ps))
        else:
            x = random.randint(0, w-ps)
            y = random.randint(0, h-ps)
            noisy = noisy.crop((x,y,x+ps,y+ps))
            gt    = gt.crop((x,y,x+ps,y+ps))
        if self.augment:
            if random.random() < 0.5:
                noisy = noisy.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                noisy = noisy.transpose(Image.FLIP_TOP_BOTTOM)
                gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
        noisy = self.to_tensor(noisy)
        gt = self.to_tensor(gt)
        return noisy, gt

# ---------------------------
# UNet model (working)
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class UNetSimple(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*8))
        self.up1 = nn.ConvTranspose2d(base*8, base*8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base*16, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base*8, base*2)
        self.up3 = nn.ConvTranspose2d(base*2, base*2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base*4, base)
        self.up4 = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)        # base
        x2 = self.down1(x1)     # base*2
        x3 = self.down2(x2)     # base*4
        x4 = self.down3(x3)     # base*8
        x5 = self.down4(x4)     # base*8
        u1 = self.up1(x5)
        u1 = torch.cat([u1, x4], dim=1)
        u1 = self.conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x3], dim=1)
        u2 = self.conv2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x2], dim=1)
        u3 = self.conv3(u3)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, x1], dim=1)
        u4 = self.conv4(u4)
        out = self.outc(u4)
        return torch.sigmoid(out)  # images in [0,1]

# ---------------------------
# VGG perceptual loss
# ---------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.eval().to(device)
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layers = [3,8,15,22]  # indices to use
        self.criterion = nn.L1Loss()
        self.device = device
        self.mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)

    def forward(self, pred, target):
        p = (pred - self.mean) / self.std
        t = (target - self.mean) / self.std
        loss = 0.0
        x = p; y = t
        for i, layer in enumerate(self.vgg):
            x = layer(x); y = layer(y)
            if i in self.layers:
                loss = loss + self.criterion(x, y)
        return loss

# ---------------------------
# Simple PatchGAN discriminator (optional)
# ---------------------------
class Discriminator(nn.Module):
    def __init__(self, in_ch=6, base=64):
        super().__init__()
        def block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.net = nn.Sequential(
            block(in_ch, base, 2),
            block(base, base*2, 2),
            block(base*2, base*4, 2),
            block(base*4, base*8, 1),
            nn.Conv2d(base*8, 1, kernel_size=4, padding=1)
        )
    def forward(self, noisy, img):
        x = torch.cat([noisy, img], dim=1)
        return self.net(x)

# ---------------------------
# train & validate functions
# ---------------------------
def psnr_batch(preds, gts):
    # preds,gts: tensors [B, C, H, W] in [0,1]
    psnrs = []
    ssims = []

    for i in range(preds.shape[0]):
        p = preds[i].cpu().permute(1,2,0).numpy()
        g = gts[i].cpu().permute(1,2,0).numpy()

        psnrs.append(compare_psnr(g, p, data_range=1.0))

        h, w, _ = g.shape
        win = min(7, h, w)
        if win % 2 == 0:
            win -= 1
        win = max(win, 3)

        ssims.append(
            compare_ssim(g, p, channel_axis=2, data_range=1.0, win_size=win)
        )

    return float(np.mean(psnrs)), float(np.mean(ssims))


def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, device):
    if not os.path.exists(path): return None
    return torch.load(path, map_location=device)

# ---------------------------
# Main train loop
# ---------------------------
def main(args):
    # paths
    project = args.project_root
    data_root = args.data_root
    splits_file = os.path.join(project, "splits.json")
    os.makedirs(os.path.join(project,"checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(project,"logs"), exist_ok=True)
    os.makedirs(os.path.join(project,"outputs"), exist_ok=True)

    splits = make_splits(data_root, splits_file, seed=args.seed)
    train_ds = SIDDPatchDataset(splits["train"], patch_size=args.patch_size, augment=True,
                                patches_per_image=args.patches_per_image)
    val_ds = SIDDPatchDataset(splits["val"], patch_size=args.patch_size, augment=False,
                                patches_per_image=10)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=max(1,args.num_workers//2), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSimple(in_channels=3, out_channels=3, base=args.base).to(device)

    # optionally discriminator
    disc = None
    if args.use_gan:
        disc = Discriminator(in_ch=6).to(device)
        optimD = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9,0.999))
    else:
        optimD = None

    optimG = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimG, mode='max', factor=0.5, patience=5)

    # losses
    l1 = nn.L1Loss()
    vgg_loss = None
    if args.use_perceptual:
        vgg_loss = VGGPerceptualLoss(device=device)

    # optional LPIPS instance
    lpips_fn = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').to(device)

    start_epoch = 0
    best_val = -1.0

    # resume checkpoint if provided or exists
    ckpt_path = args.resume if args.resume else os.path.join(project, "checkpoints", "latest.pth")
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, device)
        if ckpt:
            model.load_state_dict(ckpt['model'])
            optimG.load_state_dict(ckpt['optimG'])
            if scheduler and 'scheduler' in ckpt:
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                except Exception:
                    pass
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val = ckpt.get('best_val', best_val)
            if args.use_gan and 'optimD' in ckpt and disc is not None:
                optimD.load_state_dict(ckpt['optimD'])
            print("Resumed from", ckpt_path, "epoch", start_epoch)

    # training
    print("TRAIN SET SIZE (pairs):", len(train_ds.pairs))
    print("DEVICE:", device)
    criterion_bce = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if disc: disc.train()
        running_loss = 0.0
        it = 0
        for batch_idx, (noisy, gt) in enumerate(train_loader):
            noisy = noisy.to(device); gt = gt.to(device)
            # Gen step
            optimG.zero_grad()
            fake = model(noisy)
            loss_pixel = l1(fake, gt)
            loss_perc = vgg_loss(fake, gt) if vgg_loss is not None else 0.0
            loss_gan = 0.0
            if args.use_gan and disc is not None:
                # update D
                optimD.zero_grad()
                real_pred = disc(noisy, gt)
                fake_pred_detach = disc(noisy, fake.detach())
                real_label = torch.ones_like(real_pred)
                fake_label = torch.zeros_like(fake_pred_detach)
                d_loss = 0.5 * (criterion_bce(real_pred, real_label) + criterion_bce(fake_pred_detach, fake_label))
                d_loss.backward()
                optimD.step()
                # generator adversarial loss
                fake_pred = disc(noisy, fake)
                loss_gan = criterion_bce(fake_pred, real_label)

            # total gen loss
            total_loss = loss_pixel + args.perc_weight * (loss_perc if isinstance(loss_perc, torch.Tensor) else torch.tensor(0.0).to(device)) + args.gan_weight * (loss_gan if isinstance(loss_gan, torch.Tensor) else torch.tensor(0.0).to(device))
            total_loss.backward()
            optimG.step()

            running_loss += total_loss.item()
            it += 1

            # periodic checkpointing (after every N batches)
            if (batch_idx+1) % args.save_every_batches == 0:
                ck = {
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model': model.state_dict(),
                    'optimG': optimG.state_dict(),
                    'best_val': best_val
                }
                if scheduler: ck['scheduler'] = scheduler.state_dict()
                if args.use_gan and disc is not None:
                    ck['optimD'] = optimD.state_dict()
                save_path = os.path.join(project, "checkpoints", f"ckpt_epoch{epoch}_batch{batch_idx}.pth")
                torch.save(ck, save_path)
                torch.save(ck, os.path.join(project, "checkpoints", "latest.pth"))

        avg_train_loss = running_loss / max(1, it)
        # validate
        model.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        nval = 0
        with torch.no_grad():
            for noisy, gt in val_loader:
                noisy = noisy.to(device); gt = gt.to(device)
                fake = model(noisy)
                ps, ss = psnr_batch(fake, gt)
                val_psnr += ps; val_ssim += ss
                nval += 1
        val_psnr /= max(1, nval)
        val_ssim /= max(1, nval)
        # update scheduler
        if scheduler is not None:
            scheduler.step(val_psnr)
        # save epoch checkpoint and best model
        epoch_ck = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimG': optimG.state_dict(),
            'best_val': best_val
        }
        if args.use_gan and disc is not None:
            epoch_ck['optimD'] = optimD.state_dict()
        torch.save(epoch_ck, os.path.join(project, "checkpoints", f"epoch_{epoch}.pth"))
        torch.save(epoch_ck, os.path.join(project, "checkpoints", "latest.pth"))

        # update best
        if val_psnr > best_val:
            best_val = val_psnr
            torch.save(epoch_ck, os.path.join(project, "checkpoints", "best_model.pth"))

        # log
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"{now} EPOCH {epoch} train_loss={avg_train_loss:.4f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f} best_val={best_val:.4f}\n"
        print(log_line, end="")
        with open(os.path.join(project, "logs", "train.log"), "a") as f:
            f.write(log_line)

    print("Training finished. Best val PSNR:", best_val)
    # final save
    final_ck = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimG': optimG.state_dict(),
        'best_val': best_val
    }
    torch.save(final_ck, os.path.join(project, "checkpoints", "final.pth"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="/scratch/user/dipanwita22rano/denoising_project/data/SIDD_Medium_Srgb", help="SIDD root (folder containing Data/)")
    p.add_argument("--project_root", type=str, default="/scratch/user/dipanwita22rano/denoising_project", help="project root for checkpoints/logs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--patches_per_image", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base", type=int, default=48)
    p.add_argument("--use_perceptual", action="store_true")
    p.add_argument("--use_gan", action="store_true")
    p.add_argument("--perc_weight", type=float, default=0.05)
    p.add_argument("--gan_weight", type=float, default=1e-3)
    p.add_argument("--save_every_batches", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")
    args = p.parse_args()
    main(args)
