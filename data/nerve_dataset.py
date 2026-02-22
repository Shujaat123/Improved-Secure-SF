# ============================================================
# FILE: psfh_universal_training/data/nerve_dataset.py
#  - Expected structure (Kaggle ultrasound-nerve-segmentation style):
#       root/
#         *.tif
#         *_mask.tif
#  - Skips:
#       * missing masks
#       * empty masks
#  - Includes optional padding like your TF code (pad=14)
# ============================================================
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .filters import is_empty_mask, ensure_chw_image, normalize01


def _resize_pil(img: Image.Image, size_hw: int, is_mask: bool) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((size_hw, size_hw), resample=resample)


def _read_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def _collect_pairs(root: Path, drop_empty: bool) -> List[Tuple[Path, Path]]:
    files = sorted([p.name for p in root.iterdir() if p.is_file()])
    pairs: List[Tuple[Path, Path]] = []

    for fn in files:
        low = fn.lower()
        if low.endswith(".tif") and (not low.endswith("_mask.tif")):
            mask_fn = fn[:-4] + "_mask.tif"
            if mask_fn not in files:
                continue

            ip = root / fn
            mp = root / mask_fn

            if drop_empty:
                m = _read_gray(mp)
                m = (m > 127).astype(np.uint8)
                if is_empty_mask(m):
                    continue

            pairs.append((ip, mp))

    if not pairs:
        raise RuntimeError(f"NERVE: no valid pairs found in {root} (after filtering).")
    return pairs


class NerveDataset(Dataset):
    def __init__(
        self,
        root: str,
        img_size: int = 256,
        pad: int = 14,
        drop_empty_masks: bool = True,
        seed: int = 42,
        split: str = "train",
        test_ratio: float = 0.1,
    ):
        """
        This dataset does a RANDOM split internally (like your TF loader).
        If you already have a separate test folder, you can ignore split/test_ratio
        and just point root appropriately.
        """
        self.root = Path(root)
        self.img_size = int(img_size)
        self.pad = int(pad)
        self.drop_empty_masks = bool(drop_empty_masks)

        pairs = _collect_pairs(self.root, drop_empty=self.drop_empty_masks)

        # random split
        rng = np.random.default_rng(int(seed))
        idx = np.arange(len(pairs))
        rng.shuffle(idx)
        n_test = int(round(float(test_ratio) * len(pairs)))
        test_idx = set(idx[:n_test].tolist())
        keep = []
        for k, p in enumerate(pairs):
            if split == "test":
                if k in test_idx:
                    keep.append(p)
            else:
                if k not in test_idx:
                    keep.append(p)

        if not keep:
            raise RuntimeError(f"NERVE: split='{split}' ended up empty. Adjust test_ratio.")
        self.pairs = keep

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, mp = self.pairs[idx]
        x = _read_gray(ip)
        y = _read_gray(mp)

        # pad image only (as your TF code did)
        if self.pad > 0:
            x = np.pad(x, ((self.pad, self.pad), (self.pad, self.pad)), mode="constant", constant_values=0)

        x = _resize_pil(Image.fromarray(x), self.img_size, is_mask=False)
        y = _resize_pil(Image.fromarray(y), self.img_size, is_mask=True)

        x = normalize01(np.array(x, dtype=np.float32))
        y = (np.array(y, dtype=np.uint8) > 127).astype(np.int64)

        # safety empty check
        if self.drop_empty_masks and is_empty_mask(y):
            j = (idx + 1) % len(self.pairs)
            return self.__getitem__(j)

        x = ensure_chw_image(x)  # (1,H,W)

        return {
            "image": torch.from_numpy(x).float(),
            "mask": torch.from_numpy(y).long(),
            "case": ip.stem,
        }
