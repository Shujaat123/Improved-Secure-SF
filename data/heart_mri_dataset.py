# ============================================================
# FILE: psfh_universal_training/data/heart_mri_dataset.py
#  - Expected structure (same as your TF code):
#       root/
#         train_slices/   (images)
#         train_labels/   (masks)
#         test_slices/
#         test_labels/
#  - Skips:
#       * missing masks
#       * empty masks (all black)
#  - Outputs:
#       image: (1,H,W) float32 in [0,1]
#       mask : (H,W) int64 in {0,1}
# ============================================================
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .filters import is_empty_mask, ensure_chw_image, normalize01


IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _resize_pil(img: Image.Image, size_hw: int, is_mask: bool) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((size_hw, size_hw), resample=resample)


def _read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


@dataclass
class HeartMRISplit:
    img_dir: Path
    msk_dir: Path


def _collect_pairs(split: HeartMRISplit, img_size: int, drop_empty: bool) -> List[Tuple[Path, Path]]:
    img_paths = sorted([p for p in split.img_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {split.img_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for ip in img_paths:
        mp = split.msk_dir / ip.name
        if not mp.exists():
            continue

        # quick empty-mask check (before heavy training)
        m = _read_gray(mp)
        if drop_empty and is_empty_mask(m):
            continue

        pairs.append((ip, mp))

    if not pairs:
        raise RuntimeError(f"No valid (img,mask) pairs after filtering in: {split.img_dir}")
    return pairs


class HeartMRIDataset(Dataset):
    def __init__(self, root: str, split: str, img_size: int = 256, drop_empty_masks: bool = True):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.drop_empty_masks = bool(drop_empty_masks)

        split = split.lower().strip()
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        if split == "train":
            sp = HeartMRISplit(self.root / "train_slices", self.root / "train_labels")
        else:
            sp = HeartMRISplit(self.root / "test_slices", self.root / "test_labels")

        if not sp.img_dir.exists() or not sp.msk_dir.exists():
            raise FileNotFoundError(f"Missing dirs: {sp.img_dir} / {sp.msk_dir}")

        self.pairs = _collect_pairs(sp, self.img_size, self.drop_empty_masks)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, mp = self.pairs[idx]
        x = _read_gray(ip)
        y = _read_gray(mp)

        x = _resize_pil(Image.fromarray(x), self.img_size, is_mask=False)
        y = _resize_pil(Image.fromarray(y), self.img_size, is_mask=True)

        x = normalize01(np.array(x, dtype=np.float32))
        y = (np.array(y, dtype=np.uint8) > 127).astype(np.int64)

        # safety (post-resize) empty-mask check
        if self.drop_empty_masks and is_empty_mask(y):
            # if someone indexes this, redirect to next valid sample deterministically
            j = (idx + 1) % len(self.pairs)
            return self.__getitem__(j)

        x = ensure_chw_image(x)  # (1,H,W)

        return {
            "image": torch.from_numpy(x).float(),     # (1,H,W)
            "mask": torch.from_numpy(y).long(),       # (H,W) in {0,1}
            "case": ip.stem,
        }
