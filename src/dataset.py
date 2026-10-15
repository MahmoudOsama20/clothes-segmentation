"""ATR human-parsing dataset: class remapping, preprocessing and augmentation.

Expected layout (produced by convert_parquet.py):
    ROOT/{train,val,test}/images/*.jpg
    ROOT/{train,val,test}/masks/*.png   (single-channel, pixel value = class id 0..17)

Requires: albumentations>=2.0, opencv-python, torch, pillow, numpy, tqdm
"""
import os
import random
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = 512
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet stats (pretrained encoder)

# --------------------------------------------------------------------------- #
# Original ATR labels (18)
# --------------------------------------------------------------------------- #
ATR_NAMES = [
    "background", "hat", "hair", "sunglasses", "upper-clothes", "skirt", "pants",
    "dress", "belt", "left-shoe", "right-shoe", "face", "left-leg", "right-leg",
    "left-arm", "right-arm", "bag", "scarf",
]

# Clothes-focused label set (11). Body parts are merged into one "body" class so the
# model still learns skin-vs-cloth boundaries, and left/right pairs disappear.
CLOTHES_NAMES = [
    "background", "body", "hat", "upper-clothes", "skirt", "pants",
    "dress", "belt", "shoes", "bag", "scarf",
]
_CLOTHES_MAP = {
    0: 0, 1: 2, 2: 1, 3: 1, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7,
    9: 8, 10: 8, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 9, 17: 10,
}
CLOTHES_LUT = np.array([_CLOTHES_MAP[i] for i in range(18)], dtype=np.uint8)
FULL_LUT = np.arange(18, dtype=np.uint8)

# Horizontal flip swaps left/right labels: shoes 9<->10, legs 12<->13, arms 14<->15.
# Albumentations' HorizontalFlip does NOT do this, so we flip manually before it.
FLIP_LUT = np.arange(18, dtype=np.uint8)
for _a, _b in [(9, 10), (12, 13), (14, 15)]:
    FLIP_LUT[_a], FLIP_LUT[_b] = _b, _a

# Classes that are NOT clothing (used to build the "clothes only" cutout)
NON_CLOTHES = {"background", "body", "hair", "face", "sunglasses",
               "left-leg", "right-leg", "left-arm", "right-arm"}


def get_class_names(merge_classes: bool):
    return CLOTHES_NAMES if merge_classes else ATR_NAMES


def get_clothes_ids(merge_classes: bool):
    return [i for i, n in enumerate(get_class_names(merge_classes)) if n not in NON_CLOTHES]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_train_transform(size: int = IMG_SIZE):
    """Aspect-preserving resize + zero-pad, then geometric / photometric augmentation.

    Output image stays uint8: normalization is done on the GPU with `gpu_normalize`
    (4x less data through the DataLoader workers).
    """
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Affine(
            scale=(0.75, 1.25), translate_percent=(-0.1, 0.1), rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7,
        ),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03, p=0.6),
        A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.GaussNoise()], p=0.2),
        ToTensorV2(),
    ])


def get_eval_transform(size: int = IMG_SIZE):
    """Same resize/pad as training, no augmentation, normalized on the CPU."""
    return A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def gpu_normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW batch (already on the target device) -> ImageNet-normalized float32."""
    mean = torch.tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1) * 255
    std = torch.tensor(STD, device=x_uint8.device).view(1, 3, 1, 1) * 255
    return (x_uint8.float() - mean) / std


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ATRDataset(Dataset):
    def __init__(self, root, split, transform=None, merge_classes=True, hflip_p=0.0):
        root = Path(root)
        self.images = sorted((root / split / "images").glob("*.jpg"))
        self.mask_dir = root / split / "masks"
        assert self.images, f"No images found in {root / split / 'images'}"
        self.transform = transform
        self.lut = CLOTHES_LUT if merge_classes else FULL_LUT
        self.hflip_p = hflip_p
        self.num_classes = 11 if merge_classes else 18

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / f"{img_path.stem}.png"))
        if mask.ndim == 3:  # safety: some exports store masks as RGB
            mask = mask[..., 0]
        mask = mask.astype(np.uint8)

        # Flip image + mask together and swap left/right labels on the RAW mask.
        if self.hflip_p and random.random() < self.hflip_p:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = FLIP_LUT[np.ascontiguousarray(mask[:, ::-1])]

        mask = self.lut[mask]  # remap to the chosen label set

        if self.transform:
            out = self.transform(image=img, mask=mask)
            return out["image"], out["mask"].long()
        return img, mask


def _worker_init(_):
    """Give every DataLoader worker its own augmentation RNG stream."""
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s)
    np.random.seed(s)
    t = torch.utils.data.get_worker_info().dataset.transform
    if t is not None and hasattr(t, "set_random_seed"):
        try:
            t.set_random_seed(s)
        except Exception:
            pass


def _loader_kwargs(num_workers):
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False)
    if num_workers > 0:
        kw.update(prefetch_factor=2, worker_init_fn=_worker_init)
    return kw


def get_loaders(root, batch_size=16, size=IMG_SIZE, merge_classes=True, num_workers=4):
    """Train loader yields uint8 images (use gpu_normalize); val/test yield normalized floats."""
    train_ds = ATRDataset(root, "train", get_train_transform(size), merge_classes, hflip_p=0.5)
    val_ds = ATRDataset(root, "val", get_eval_transform(size), merge_classes)
    test_ds = ATRDataset(root, "test", get_eval_transform(size), merge_classes)
    kw = _loader_kwargs(num_workers)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size, shuffle=False, **kw),
    )


def get_eval_loader(root, split, batch_size=16, size=IMG_SIZE, merge_classes=True, num_workers=4):
    ds = ATRDataset(root, split, get_eval_transform(size), merge_classes)
    return DataLoader(ds, batch_size, shuffle=False, **_loader_kwargs(num_workers))


# --------------------------------------------------------------------------- #
# Helpers: class balance + sanity check
# --------------------------------------------------------------------------- #
def class_pixel_frequencies(root, split="train", merge_classes=True, max_items=2000):
    """Pixel share per class (used for loss weights and the report's dataset analysis)."""
    from tqdm import tqdm
    ds = ATRDataset(root, split, None, merge_classes)
    counts = np.zeros(ds.num_classes, dtype=np.int64)
    for i in tqdm(range(min(len(ds), max_items)), desc="class freq"):
        _, m = ds[i]
        counts += np.bincount(m.ravel(), minlength=ds.num_classes)
    return counts / counts.sum()


if __name__ == "__main__":
    ROOT = "data/ATR_extracted"

    raw = ATRDataset(ROOT, "train", None, merge_classes=False)
    seen = set()
    for i in range(0, len(raw), max(1, len(raw) // 200)):
        seen |= set(np.unique(raw[i][1]).tolist())
    print("raw label ids found:", sorted(seen))

    freq = class_pixel_frequencies(ROOT, "train", merge_classes=True)
    for n, f in zip(CLOTHES_NAMES, freq):
        print(f"{n:>16s}: {f:6.2%}")

    tr, _, _ = get_loaders(ROOT, batch_size=4, num_workers=0)
    x, y = next(iter(tr))
    print(x.shape, x.dtype, y.shape, y.unique().tolist())
