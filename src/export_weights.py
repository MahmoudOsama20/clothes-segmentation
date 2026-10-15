"""Shrink best.pt (~98 MB) to an fp16 inference-only file (~half the size) that infer.py / evaluate.py can load.

    python src/export_weights.py --ckpt results/exp1/best.pt --out results/exp1/model_fp16.pt
"""
import argparse

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/exp1/best.pt")
ap.add_argument("--out", default="results/exp1/model_fp16.pt")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
ck["model"] = {k: v.half() if v.is_floating_point() else v for k, v in ck["model"].items()}
torch.save(ck, a.out)
print("saved", a.out)
