"""Plot helpers: color palette, confusion matrix, best/worst case panels."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from dataset import MEAN, STD


def make_palette(n_classes: int):
    pal = np.random.RandomState(3).randint(60, 256, (max(n_classes, 18), 3)).astype(np.uint8)
    pal[0] = 0  # background = black
    return pal


def denorm(x):
    """Normalized CHW tensor -> HWC float image in [0, 1]."""
    x = x.permute(1, 2, 0).cpu().numpy() * np.array(STD) + np.array(MEAN)
    return np.clip(x, 0, 1)


def _legend(fig, palette, names, fontsize=8):
    fig.legend(handles=[Patch(color=palette[i] / 255, label=n) for i, n in enumerate(names)],
               loc="lower center", ncol=min(len(names), 11), fontsize=fontsize)


def plot_confusion(conf, names, path, title="Normalized confusion matrix (test)"):
    cm = conf.double().cpu().numpy()
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)  # rows = ground truth
    n = len(names)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=60, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0.02:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="ground truth", title=title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def show_cases(model, ds, idxs, scores, names, palette, device, path, title, amp=True):
    """Rows of: image | ground truth | prediction (with per-image mIoU in the title)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    fig, axes = plt.subplots(len(idxs), 3, figsize=(10, 3.3 * len(idxs)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idxs):
        x, y = ds[int(i)]
        with torch.autocast("cuda", enabled=use_amp):
            p = model(x.unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
        panels = [(denorm(x), f"image #{i}"), (palette[y.numpy()], "ground truth"),
                  (palette[p], f"prediction (mIoU {scores[i]:.2f})")]
        for c, (im, t) in enumerate(panels):
            axes[r, c].imshow(im); axes[r, c].set_title(t, fontsize=9); axes[r, c].axis("off")
    _legend(fig, palette, names)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
