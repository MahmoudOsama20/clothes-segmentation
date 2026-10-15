"""Inference pipeline: take a personal photo and segment the worn clothes.

    python src/infer.py --ckpt results/exp1/best.pt --input samples/input --output samples/output

For every image it writes:
    <name>_mask.png     color-coded class mask (original resolution)
    <name>_overlay.png  mask blended over the photo
    <name>_clothes.png  RGBA cutout with only clothing kept (transparent elsewhere)
    <name>_panel.jpg    original | overlay | cutout, side by side (handy for the report)

The photo is letterboxed to the model size (aspect ratio preserved), the padding is removed from
the prediction, and the mask is resized back to the original photo resolution.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import MEAN, NON_CLOTHES, STD
from model import flip_channel_perm, load_checkpoint, predict_logits
from viz import make_palette

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess(img_rgb, size):
    """Same pipeline as eval: aspect-preserving resize, centered zero-pad, ImageNet normalize."""
    h, w = img_rgb.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    x = (canvas.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
    return x, (top, left, nh, nw)


@torch.no_grad()
def predict_mask(model, img_rgb, size, device, use_amp, tta=False, flip_ch=None):
    h, w = img_rgb.shape[:2]
    x, (top, left, nh, nw) = preprocess(img_rgb, size)
    with torch.autocast("cuda", enabled=use_amp):
        logits = predict_logits(model, x.to(device), tta, flip_ch)
    logits = logits[:, :, top:top + nh, left:left + nw].float()  # drop the padding
    logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def read_rgb(path):
    bgr = cv2.imread(str(path))
    if bgr is None:  # fallback for paths OpenCV cannot open (non-ASCII characters on Windows)
        return np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/exp1/best.pt")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output", default="samples/output")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    ap.add_argument("--tta", action="store_true", help="average with the flipped prediction (slower, ~+0.002 mIoU)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model, ck = load_checkpoint(args.ckpt, device)
    size, names = ck["size"], ck["class_names"]
    flip_ch = flip_channel_perm(ck["merge_classes"], ck["num_classes"], device)
    palette = make_palette(len(names))
    clothes_ids = [i for i, n in enumerate(names) if n not in NON_CLOTHES]

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            rgb = read_rgb(f)
        except Exception:
            print(f"skip (unreadable): {f}")
            continue
        mask = predict_mask(model, rgb, size, device, use_amp, args.tta, flip_ch)

        color = palette[mask]
        overlay = np.where((mask > 0)[..., None], (rgb * (1 - args.alpha) + color * args.alpha).astype(np.uint8), rgb)
        clothes = np.isin(mask, clothes_ids)
        cutout = np.dstack([rgb, (clothes * 255).astype(np.uint8)])

        stem = f.stem
        cv2.imwrite(str(out / f"{stem}_mask.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{stem}_clothes.png"), cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))
        panel = np.hstack([rgb, overlay, np.where(clothes[..., None], rgb, 255).astype(np.uint8)])
        cv2.imwrite(str(out / f"{stem}_panel.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

        found = [names[i] for i in np.unique(mask) if i != 0]
        print(f"{f.name}: {', '.join(found) if found else 'nothing detected'}")


if __name__ == "__main__":
    main()
