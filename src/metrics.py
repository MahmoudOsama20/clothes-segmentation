"""Segmentation metrics from a confusion matrix (rows = ground truth, cols = prediction)."""
import numpy as np
import torch

from model import predict_logits


def confusion(pred, target, num_classes):
    idx = target.reshape(-1) * num_classes + pred.reshape(-1)
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def metrics_from_conf(conf):
    conf = conf.double().cpu().numpy()
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        dice = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return dict(mIoU=float(np.nanmean(iou)), mDice=float(np.nanmean(dice)),
                pixel_acc=float(tp.sum() / conf.sum()), iou=iou.tolist(), dice=dice.tolist())


def miou_present(conf, metrics, class_names):
    """mIoU over classes that actually occur in the ground truth.

    A class with no GT pixels but a few false positives would otherwise count as IoU 0.
    Returns (value, classes absent from the ground truth).
    """
    keep = conf.sum(1).cpu().numpy() > 0
    absent = [n for n, k in zip(class_names, keep) if not k]
    return float(np.nanmean(np.array(metrics["iou"])[keep])), absent


@torch.no_grad()
def evaluate(model, loader, device, num_classes, amp=True, criterion=None, tta=False, flip_ch=None):
    """Returns (metrics dict, confusion matrix). Loader must yield normalized float images."""
    from tqdm import tqdm
    model.eval()
    use_amp = amp and device.type == "cuda"
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=use_amp):
            logits = predict_logits(model, x, tta, flip_ch)
            if criterion is not None:
                total += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        conf += confusion(logits.argmax(1), y, num_classes)
    m = metrics_from_conf(conf)
    m["loss"] = total / n if criterion is not None else None
    return m, conf
