"""Convert the ATR parquet files into plain image / mask folders.

    python src/convert_parquet.py --data-dir data/ATRDataset/data --out-dir data/ATR_extracted

Output layout:
    OUT/{train,val,test}/images/00000.jpg
    OUT/{train,val,test}/masks/00000.png   (single-channel, pixel value = class id 0..17)
"""
import argparse
import io
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _to_pil(raw):
    if isinstance(raw, dict) and "bytes" in raw:
        return Image.open(io.BytesIO(raw["bytes"]))
    if isinstance(raw, bytes):
        return Image.open(io.BytesIO(raw))
    return raw


def extract_split(parquet_files, split_name, out_dir):
    img_dir = Path(out_dir) / split_name / "images"
    mask_dir = Path(out_dir) / split_name / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for p_file in parquet_files:
        print(f"Reading {p_file}...")
        df = pd.read_parquet(p_file)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            _to_pil(row["pixel_values"]).convert("RGB").save(img_dir / f"{idx:05d}.jpg")
            _to_pil(row["label"]).save(mask_dir / f"{idx:05d}.png")
            idx += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing the *.parquet files")
    ap.add_argument("--out-dir", default="data/ATR_extracted")
    a = ap.parse_args()

    d = Path(a.data_dir)
    extract_split(sorted(d.glob("train-*.parquet")), "train", a.out_dir)
    extract_split(sorted(d.glob("validation-*.parquet")), "val", a.out_dir)
    extract_split(sorted(d.glob("test-*.parquet")), "test", a.out_dir)
