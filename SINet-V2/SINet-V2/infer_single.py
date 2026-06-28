"""
Single-image inference for SINet-V2 (Camouflaged Object Detection).

Runs on Apple Silicon (MPS) or CPU -- no NVIDIA GPU required.
Replaces the repo's MyTesting.py, which relied on the removed scipy.misc.imsave
and was hardwired to full benchmark dataset folders.

Usage:
    python infer_single.py --image path/to/photo.jpg
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from lib.Network_Res2Net_GRA_NCD import Network


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--pth_path", default="./snapshot/SINet_V2/Net_epoch_best.pth")
    ap.add_argument("--testsize", type=int, default=352)
    ap.add_argument("--outdir", default="./res/single")
    args = ap.parse_args()

    device = pick_device()
    print(f"[device] {device}")

    # --- model ---
    model = Network(imagenet_pretrained=False)
    # This checkpoint is a plain state_dict (tensors only), so weights_only=True
    # is safe and avoids unpickling arbitrary objects.
    state = torch.load(args.pth_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # --- preprocessing (same as repo's test_dataset) ---
    tf = transforms.Compose([
        transforms.Resize((args.testsize, args.testsize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    pil = Image.open(args.image).convert("RGB")
    orig_w, orig_h = pil.size
    x = tf(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        res5, res4, res3, res2 = model(x)
        pred = res2  # final refined map

    # upsample to original size, sigmoid, min-max normalize
    pred = F.interpolate(pred, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred = pred.sigmoid().squeeze().cpu().numpy()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.image))[0]

    # 1) grayscale probability mask
    mask_u8 = (pred * 255).astype(np.uint8)
    mask_path = os.path.join(args.outdir, f"{base}_mask.png")
    Image.fromarray(mask_u8, mode="L").save(mask_path)

    # 2) red overlay on the original for an at-a-glance result
    rgb = np.asarray(pil).astype(np.float32)
    heat = np.zeros_like(rgb)
    heat[..., 0] = mask_u8  # red channel
    overlay = (0.55 * rgb + 0.45 * heat).clip(0, 255).astype(np.uint8)
    overlay_path = os.path.join(args.outdir, f"{base}_overlay.png")
    Image.fromarray(overlay).save(overlay_path)

    print(f"[saved] {mask_path}")
    print(f"[saved] {overlay_path}")
    print(f"[stats] predicted camouflaged-region coverage: {(pred > 0.5).mean() * 100:.1f}% of pixels")


if __name__ == "__main__":
    main()
