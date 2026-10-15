"""Plot training curves from results/<exp>/history.csv -> results/<exp>/curves.png

    python src/plot_curves.py --exp results/exp1
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="results/exp1")
exp = Path(ap.parse_args().exp)

h = pd.read_csv(exp / "history.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h["epoch"], h["train_loss"], label="train")
ax[0].plot(h["epoch"], h["val_loss"], label="val")
ax[0].set(title="Loss (weighted CE + Dice)", xlabel="epoch")
ax[0].legend()
ax[1].plot(h["epoch"], h["val_mIoU"], label="val mIoU")
ax[1].plot(h["epoch"], h["val_mDice"], label="val mDice")
ax[1].set(title="Validation metrics", xlabel="epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig(exp / "curves.png", dpi=150)
print("saved", exp / "curves.png")
