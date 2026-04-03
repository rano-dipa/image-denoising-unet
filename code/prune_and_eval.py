import torch
import torch.nn.utils.prune as prune
from train_unet import UNetSimple

model = UNetSimple(base=48)
ckpt = torch.load("checkpoints/final_best.pth", map_location="cpu")
model.load_state_dict(ckpt["model"])

# prune 40%
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        prune.l1_unstructured(m, name="weight", amount=0.35)
        prune.remove(m, "weight")

torch.save({"model": model.state_dict()}, "checkpoints/pruned.pth")

print("Pruned model saved")