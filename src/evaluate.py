"""Evaluate the best checkpoint on val + test and produce the report artifacts.

    python src/evaluate.py --out results/exp1

Writes to <out>/: per_class_val_test.csv, test_metrics.json, confusion_matrix.png,
                  worst_cases.png, best_cases.png
Add --tta to also report horizontal-flip test-time augmentation (gain was only about +0.0015 mIoU).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import get_eval_loader
from metrics import confusion, evaluate, miou_present
from model import flip_channel_perm, load_checkpoint
from viz import make_palette, plot_confusion, show_cases


@torch.no_grad()
def per_image_scores(model, ds, device, C, n=600, bs=16, amp=True):
    """Per-image mIoU over classes present in the ground truth or the prediction."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    n = min(n, len(ds))
    scores = []
    for x, y in tqdm(DataLoader(Subset(ds, range(n)), bs, shuffle=False, num_workers=0), desc="per-image"):
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", enabled=use_amp):
            pred = model(x).argmax(1)
        for p, t in zip(pred, y):
            c = confusion(p, t, C).double().cpu().numpy()
            tp = np.diag(c)
            den = c.sum(0) + c.sum(1) - tp
            scores.append(float(np.nanmean(np.where(den > 0, tp / np.maximum(den, 1), np.nan))))
    return np.array(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ATR_extracted")
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--ckpt", default=None, help="default: <out>/best.pt")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tta", action="store_true", help="also evaluate test with flip TTA")
    args = ap.parse_args()

    out = Path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(args.ckpt or out / "best.pt", device)
    names, C, merge, size = ck["class_names"], ck["num_classes"], ck["merge_classes"], ck["size"]
    flip_ch = flip_channel_perm(merge, C, device)
    print(f"checkpoint: epoch {ck['epoch']}, val mIoU {ck['val_mIoU']:.4f} ({ck.get('weights', 'raw')} weights)")

    val_dl = get_eval_loader(args.root, "val", args.bs, size, merge, args.workers)
    test_dl = get_eval_loader(args.root, "test", args.bs, size, merge, args.workers)

    val_m, val_conf = evaluate(model, val_dl, device, C)
    test_m, test_conf = evaluate(model, test_dl, device, C)
    rows = [("val", val_m, val_conf), ("test", test_m, test_conf)]
    tta_m = None
    if args.tta:
        tta_m, tta_conf = evaluate(model, test_dl, device, C, tta=True, flip_ch=flip_ch)
        rows.append(("test (flip TTA)", tta_m, tta_conf))

    print(f"{'split':16s} {'mIoU':>7s} {'mIoU(present)':>14s} {'mDice':>7s} {'pixAcc':>7s}")
    for name, m, c in rows:
        mp, absent = miou_present(c, m, names)
        print(f"{name:16s} {m['mIoU']:7.4f} {mp:14.4f} {m['mDice']:7.4f} {m['pixel_acc']:7.4f}"
              + (f"   (no GT pixels: {absent})" if absent else ""))

    per_class = pd.DataFrame({"class": names,
                              "val_IoU": np.round(val_m["iou"], 4),
                              "test_IoU": np.round(test_m["iou"], 4),
                              "test_Dice": np.round(test_m["dice"], 4)})
    if tta_m:
        per_class["test_IoU_TTA"] = np.round(tta_m["iou"], 4)
    freq_path = out / "class_freq.npy"
    if freq_path.exists():
        per_class["train_pixel_share_%"] = (np.load(freq_path) * 100).round(2)
    per_class["test_GT_share_%"] = (test_conf.sum(1).double() / test_conf.sum().double() * 100).cpu().numpy().round(3)
    print(per_class.to_string(index=False))
    per_class.to_csv(out / "per_class_val_test.csv", index=False)

    mp_val, _ = miou_present(val_conf, val_m, names)
    mp_test, absent_test = miou_present(test_conf, test_m, names)
    json.dump({"test_mIoU": test_m["mIoU"], "test_mDice": test_m["mDice"], "test_pixel_acc": test_m["pixel_acc"],
               "test_mIoU_present_classes": mp_test, "classes_without_gt": absent_test,
               "val_mIoU": val_m["mIoU"], "val_mDice": val_m["mDice"], "val_pixel_acc": val_m["pixel_acc"],
               "val_mIoU_present_classes": mp_val,
               "test_mIoU_flip_tta": tta_m["mIoU"] if tta_m else None,
               "class_names": names, "test_iou": test_m["iou"], "test_dice": test_m["dice"],
               "val_iou": val_m["iou"], "val_dice": val_m["dice"]},
              open(out / "test_metrics.json", "w"), indent=2)

    plot_confusion(test_conf, names, out / "confusion_matrix.png")

    # failure analysis: best / worst test images
    palette = make_palette(C)
    test_ds = test_dl.dataset
    scores = per_image_scores(model, test_ds, device, C)
    print(f"per-image mIoU: mean {scores.mean():.3f} | median {np.median(scores):.3f} | "
          f"10th pct {np.percentile(scores, 10):.3f} | min {scores.min():.3f}")
    order = np.argsort(scores)
    show_cases(model, test_ds, order[:6], scores, names, palette, device, out / "worst_cases.png", "Worst test cases")
    show_cases(model, test_ds, order[-4:][::-1], scores, names, palette, device, out / "best_cases.png", "Best test cases")
    print(f"saved metrics and figures to {out}/")


if __name__ == "__main__":
    main()
