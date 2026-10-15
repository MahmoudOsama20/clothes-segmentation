"""Train the clothes-segmentation model (single GPU).

Reproduces the final Kaggle run (2 x T4, DDP, 16 images/GPU = effective batch 32):

    python src/train.py --bs 16 --accum 2 --size 512 --epochs 25

Low-VRAM GPU (e.g. 4 GB):

    python src/train.py --size 384 --bs 4 --accum 8 --epochs 25

Training is resumable: re-run the same command and it continues from <out>/last.pt.
Evaluate afterwards with:  python src/evaluate.py --out results/exp1
"""
import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import IMG_SIZE, class_pixel_frequencies, get_class_names, get_loaders, gpu_normalize, MEAN, STD
from losses import ComboLoss, make_class_weights
from metrics import evaluate
from model import build_model


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """EMA of the weights; BatchNorm statistics are copied, not averaged."""
    ep, mp = list(ema_model.parameters()), list(model.parameters())
    torch._foreach_mul_(ep, decay)
    torch._foreach_add_(ep, mp, alpha=1.0 - decay)
    for be, bm in zip(ema_model.buffers(), model.buffers()):
        be.copy_(bm)


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ATR_extracted")
    ap.add_argument("--out", default="results/exp1")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--size", type=int, default=IMG_SIZE)
    ap.add_argument("--bs", type=int, default=16, help="images per forward pass")
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (effective batch = bs * accum)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline / smoke tests)")
    ap.add_argument("--full18", action="store_true", help="use all 18 ATR classes instead of the 11 merged ones")
    ap.add_argument("--time-budget-hours", type=float, default=0, help="stop cleanly after this long (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True  # fixed input size -> let cuDNN pick the fastest kernels

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    use_cl = device.type == "cuda"  # channels_last: faster conv kernels on tensor cores
    use_ema = not args.no_ema
    merge = not args.full18
    names = get_class_names(merge)
    C = len(names)
    print(f"device: {device} | classes: {C} | effective batch {args.bs * args.accum} | AMP {use_amp} | EMA {use_ema}")

    train_dl, val_dl, _ = get_loaders(args.root, args.bs, args.size, merge, args.workers)

    # ---- class weights from the train split (cached) -------------------- #
    freq_path = out / "class_freq.npy"
    if freq_path.exists():
        freq = np.load(freq_path)
    else:
        freq = class_pixel_frequencies(args.root, "train", merge)
        np.save(freq_path, freq)
    weights = make_class_weights(freq).to(device)
    print("class weights:", {n: round(float(w), 2) for n, w in zip(names, weights.cpu())})
    criterion = ComboLoss(weights)

    # ---- model / optimizer ----------------------------------------------- #
    model = build_model(args.encoder, C, pretrained=not args.no_pretrained).to(device)
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best, start_epoch, ema_state = [], -1.0, 1, None
    last_path, best_path = out / "last.pt", out / "best.pt"
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, best, start_epoch, ema_state = ck["history"], ck["best"], ck["epoch"] + 1, ck.get("ema")
        print(f"Resumed from epoch {ck['epoch']} (best val mIoU so far {best:.4f})")
        del ck

    ema_model = None
    if use_ema:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        if ema_state is not None:
            ema_model.load_state_dict(ema_state)

    updates_per_epoch = len(train_dl) // args.accum
    n_upd = (start_epoch - 1) * updates_per_epoch
    t_start = time.time()

    # ---- training loop ----------------------------------------------------- #
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        run = 0.0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")
        for step, (x, y) in enumerate(pbar, 1):
            x = gpu_normalize(x.to(device, non_blocking=True))
            if use_cl:
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss / args.accum).backward()
            run += loss.item()
            if step % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                n_upd += 1
                if use_ema:  # warm-up: short horizon early, args.ema_decay later
                    ema_update(ema_model, model, min(args.ema_decay, (1 + n_upd) / (10 + n_upd)))
            pbar.set_postfix(loss=f"{run / step:.4f}")
        sched.step()

        # validate raw weights and EMA weights; keep whichever is better
        m_raw, _ = evaluate(model, val_dl, device, C, use_amp, criterion)
        m, pick, m_ema = m_raw, "raw", None
        if use_ema:
            m_ema, _ = evaluate(ema_model, val_dl, device, C, use_amp, criterion)
            if m_ema["mIoU"] >= m_raw["mIoU"]:
                m, pick = m_ema, "ema"

        rec = dict(epoch=epoch, train_loss=run / len(train_dl), val_loss=m["loss"], val_mIoU=m["mIoU"],
                   val_mDice=m["mDice"], val_acc=m["pixel_acc"], lr=opt.param_groups[0]["lr"],
                   sec=round(time.time() - t0), val_mIoU_raw=m_raw["mIoU"],
                   val_mIoU_ema=m_ema["mIoU"] if m_ema else None, weights=pick)
        history.append(rec)
        print(f"epoch {epoch}: train {rec['train_loss']:.4f} | val {rec['val_loss']:.4f} | "
              f"mIoU {rec['val_mIoU']:.4f} ({pick}; raw {m_raw['mIoU']:.4f}"
              + (f" / ema {m_ema['mIoU']:.4f}" if m_ema else "")
              + f") | mDice {rec['val_mDice']:.4f} | acc {rec['val_acc']:.4f} | {rec['sec']}s")

        if m["mIoU"] > best:
            best = m["mIoU"]
            sd = (ema_model if pick == "ema" else model).state_dict()
            save_atomic(dict(model=sd, weights=pick, encoder=args.encoder, num_classes=C, size=args.size,
                             merge_classes=merge, class_names=names, mean=MEAN, std=STD,
                             epoch=epoch, val_mIoU=best), best_path)
            print(f"  saved best (val mIoU {best:.4f}, {pick} weights)")
        save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                         scaler=scaler.state_dict(), history=history, best=best, epoch=epoch,
                         ema=ema_model.state_dict() if use_ema else None), last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        if args.time_budget_hours and (time.time() - t_start) / 3600 > args.time_budget_hours:
            print("Time budget reached - stopping early; re-run the same command to resume.")
            break

    print(f"Done. Best val mIoU: {best:.4f}. Next: python src/evaluate.py --out {out}")


if __name__ == "__main__":
    main()
