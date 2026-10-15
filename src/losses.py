"""Loss: class-weighted cross-entropy + Dice."""
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def make_class_weights(freq, max_weight: float = 10.0):
    """Inverse-sqrt pixel frequency, scaled so the most common class = 1, capped at max_weight."""
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    return torch.tensor(np.clip(w / w.min(), 1.0, max_weight), dtype=torch.float32)


class ComboLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass")  # softmax is applied internally

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice(logits, target)
