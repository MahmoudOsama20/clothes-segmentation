"""Model: U-Net with a pretrained ImageNet encoder (segmentation_models_pytorch) + checkpoint helpers."""
import numpy as np
import segmentation_models_pytorch as smp
import torch

from dataset import FLIP_LUT


def build_model(encoder: str = "resnet34", num_classes: int = 11, pretrained: bool = True):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,  # raw logits out, shape (N, C, H, W)
    )


def load_checkpoint(path, device):
    """Rebuild the model from a checkpoint written by train.py. Returns (model in eval mode, checkpoint dict)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["encoder"], ck["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])  # also accepts fp16 weights from export_weights.py
    model.eval()
    return model, ck


def flip_channel_perm(merge_classes: bool, num_classes: int, device):
    """Channel permutation that undoes a horizontal flip (identity for merged classes)."""
    idx = np.arange(num_classes) if merge_classes else FLIP_LUT[:num_classes].astype(np.int64)
    return torch.as_tensor(idx, device=device)


def predict_logits(model, x, tta=False, flip_ch=None):
    """Logits for a normalized batch. With tta=True, averaged with the flipped-input prediction."""
    logits = model(x)
    if tta:
        logits = (logits + model(x.flip(3)).flip(3)[:, flip_ch]) / 2
    return logits
