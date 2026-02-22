# ============================================================
# FILE: psfh_universal_training/data/filters.py
# ============================================================
from __future__ import annotations
import numpy as np


def to_hw(mask: np.ndarray) -> np.ndarray:
    """
    Normalize mask to (H,W).
    Accepts: (H,W), (H,W,1), (1,H,W)
    """
    m = np.asarray(mask)
    if m.ndim == 2:
        return m
    if m.ndim == 3:
        if m.shape[0] == 1:
            return m[0]
        if m.shape[-1] == 1:
            return m[..., 0]
    raise ValueError(f"Unsupported mask shape for to_hw(): {m.shape}")


def is_empty_mask(mask: np.ndarray) -> bool:
    """
    True if mask has no foreground pixels.
    Works for {0,1} or {0,255} or any thresholded mask.
    """
    m = to_hw(mask)
    return int(np.sum(m > 0)) == 0


def ensure_chw_image(x: np.ndarray) -> np.ndarray:
    """
    Ensure image is (C,H,W) with C=1.
    Accepts: (H,W) or (H,W,1) or (1,H,W).
    """
    a = np.asarray(x)
    if a.ndim == 2:
        return a[None, ...]
    if a.ndim == 3:
        if a.shape[0] == 1:
            return a
        if a.shape[-1] == 1:
            return np.transpose(a, (2, 0, 1))
    raise ValueError(f"Unsupported image shape for ensure_chw_image(): {a.shape}")


def normalize01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.float32)
    if x.max() > 1.0:
        mn, mx = float(x.min()), float(x.max())
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0).astype(np.float32)
