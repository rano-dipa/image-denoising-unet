#!/usr/bin/env python3
import argparse
from PIL import Image
import torch
import torchvision.transforms as T

from train_unet import UNetSimple

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model:", args.ckpt)

    model = UNetSimple(base=48).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load image
    img = Image.open(args.input).convert("RGB")

    to_tensor = T.ToTensor()
    inp = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp).clamp(0,1).cpu().squeeze(0)

    out_img = T.ToPILImage()(out)

    out_img.save(args.output)

    print("Saved:", args.output)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default="denoised.png")
    p.add_argument("--ckpt", type=str, default="checkpoints/final_best.pth")
    args = p.parse_args()
    main(args)
