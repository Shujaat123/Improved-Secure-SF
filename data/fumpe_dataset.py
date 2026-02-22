# ============================================================
# FILE: psfh_universal_training/data/fumpe_dataset.py
#  - Expected structure (same as your TF loader):
#       root/
#         CT_scans/PAT*/ *.dcm
#         GroundTruth/PAT*.mat
#
#  - Produces a SLICE dataset:
#       each valid slice with non-empty GT mask becomes a sample
#  - Skips:
#       * patient without .mat
#       * shape mismatch
#       * empty mask slices
# ============================================================
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .filters import is_empty_mask, ensure_chw_image, normalize01

# external deps
import pydicom
from scipy.io import loadmat


def _resize_pil(img: Image.Image, size_hw: int, is_mask: bool) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((size_hw, size_hw), resample=resample)


def _dicom_sort_key(ds, path: Path) -> int:
    # best-effort sorting by InstanceNumber
    try:
        return int(getattr(ds, "InstanceNumber"))
    except Exception:
        # fallback to filename sorting
        try:
            return int(path.stem)
        except Exception:
            return 0


def _load_dicom_volume(dcm_paths: List[Path]) -> np.ndarray:
    slices = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p))
            slices.append((ds, p))
        except Exception:
            continue

    if not slices:
        raise RuntimeError("No readable DICOM slices")

    slices.sort(key=lambda t: _dicom_sort_key(t[0], t[1]))

    vol = []
    for ds, _p in slices:
        img = ds.pixel_array.astype(np.int16)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img * slope + intercept
        vol.append(img)

    return np.stack(vol, axis=0)  # (D,H,W)


def _load_mat_mask(mat_path: Path, vol_shape: Tuple[int, int, int]) -> np.ndarray:
    mat = loadmat(str(mat_path))
    # try common keys
    for key in ("GT", "Mask", "seg", "volume"):
        if key in mat:
            m = np.asarray(mat[key])
            break
    else:
        # fallback: first ndarray-like
        m = None
        for v in mat.values():
            if isinstance(v, np.ndarray):
                m = np.asarray(v)
                break
        if m is None:
            raise RuntimeError(f"No ndarray mask found in: {mat_path}")

    m = m.astype(np.float32)

    # Try to coerce to (D,H,W)
    # If (H,W,D), transpose
    if m.shape != vol_shape and sorted(m.shape) == sorted(vol_shape):
        # assume (H,W,D) -> (D,H,W)
        m = np.transpose(m, (2, 0, 1))

    if m.shape != vol_shape:
        raise RuntimeError(f"Mask shape {m.shape} != volume shape {vol_shape}")

    return m


class FUMPEBinarySliceDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 256,
        drop_empty_masks: bool = True,
        seed: int = 42,
        split_strategy: str = "patient",
        test_ratio: float = 0.1,
        hu_clip: Tuple[int, int] = (-1000, 1000),
    ):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.drop_empty_masks = bool(drop_empty_masks)

        scan_root = self.root / "CT_scans"
        gt_root = self.root / "GroundTruth"
        if not scan_root.exists() or not gt_root.exists():
            raise FileNotFoundError(f"Missing FUMPE folders under: {self.root}")

        patient_dirs = sorted([p for p in scan_root.iterdir() if p.is_dir() and p.name.startswith("PAT")])
        if not patient_dirs:
            raise RuntimeError(f"No PAT* dirs found in: {scan_root}")

        # ---- split patients (patient-based recommended) ----
        rng = np.random.default_rng(int(seed))
        pats = np.array([p.name for p in patient_dirs])
        rng.shuffle(pats)

        if split_strategy.lower().strip() == "patient":
            n_test = int(round(float(test_ratio) * len(pats)))
            test_set = set(pats[:n_test].tolist())
            chosen = [p for p in patient_dirs if ((p.name in test_set) == (split == "test"))]
        else:
            # random over patients anyway (slice-level random is not ideal here)
            n_test = int(round(float(test_ratio) * len(pats)))
            test_set = set(pats[:n_test].tolist())
            chosen = [p for p in patient_dirs if ((p.name in test_set) == (split == "test"))]

        if not chosen:
            raise RuntimeError("FUMPE: split ended empty; adjust test_ratio.")

        # ---- build slice index ----
        items: List[Tuple[str, int]] = []  # (patient_id, slice_index)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # pid -> (vol01, mask01)

        lo, hi = int(hu_clip[0]), int(hu_clip[1])

        for pdir in chosen:
            pid = pdir.name
            mat_path = gt_root / f"{pid}.mat"
            if not mat_path.exists():
                continue

            dcm_paths = sorted([x for x in pdir.glob("*.dcm")])
            if not dcm_paths:
                continue

            try:
                vol = _load_dicom_volume(dcm_paths)  # (D,H,W)
                mask = _load_mat_mask(mat_path, vol.shape)  # (D,H,W)
            except Exception:
                continue

            # HU normalize -> [0,1]
            vol = np.clip(vol, lo, hi).astype(np.float32)
            vol = (vol - lo) / float(hi - lo + 1e-8)

            # binary mask
            mask = (mask > 0).astype(np.uint8)

            # drop empty slices
            for si in range(vol.shape[0]):
                if self.drop_empty_masks and is_empty_mask(mask[si]):
                    continue
                items.append((pid, si))

            # cache volumes for speed
            self._cache[pid] = (vol, mask)

        if not items:
            raise RuntimeError("FUMPE: no slices kept after filtering empty masks.")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid, si = self.items[idx]
        vol, mask = self._cache[pid]

        x = vol[si]  # (H,W) float in [0,1]
        y = mask[si]  # (H,W) uint8 in {0,1}

        # resize
        x_img = _resize_pil(Image.fromarray((x * 255).astype(np.uint8)), self.img_size, is_mask=False)
        y_img = _resize_pil(Image.fromarray((y * 255).astype(np.uint8)), self.img_size, is_mask=True)

        x2 = normalize01(np.array(x_img, dtype=np.float32))
        y2 = (np.array(y_img, dtype=np.uint8) > 127).astype(np.int64)

        # safety empty check
        if self.drop_empty_masks and is_empty_mask(y2):
            j = (idx + 1) % len(self.items)
            return self.__getitem__(j)

        x2 = ensure_chw_image(x2)  # (1,H,W)

        return {
            "image": torch.from_numpy(x2).float(),
            "mask": torch.from_numpy(y2).long(),
            "case": f"{pid}_slice{si:04d}",
        }
