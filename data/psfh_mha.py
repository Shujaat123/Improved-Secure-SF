import numpy as np
from pathlib import Path
from typing import Dict
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


def _read_mha(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img)


def _resize_image_chw_linear(x: np.ndarray, out_hw: int) -> np.ndarray:
    # x: (C,H,W) float32
    c, _, _ = x.shape
    out = np.zeros((c, out_hw, out_hw), dtype=np.float32)
    for i in range(c):
        img = sitk.GetImageFromArray(x[i].astype(np.float32))
        res = sitk.Resample(
            img,
            size=[out_hw, out_hw],
            transform=sitk.Transform(),
            interpolator=sitk.sitkLinear,
            outputOrigin=img.GetOrigin(),
            outputSpacing=[1.0, 1.0],
            outputDirection=img.GetDirection(),
            defaultPixelValue=0,
        )
        out[i] = sitk.GetArrayFromImage(res).astype(np.float32)
    return out


def _resize_mask_hw_nearest(y: np.ndarray, out_hw: int) -> np.ndarray:
    img = sitk.GetImageFromArray(y.astype(np.uint8))
    res = sitk.Resample(
        img,
        size=[out_hw, out_hw],
        transform=sitk.Transform(),
        interpolator=sitk.sitkNearestNeighbor,
        outputOrigin=img.GetOrigin(),
        outputSpacing=[1.0, 1.0],
        outputDirection=img.GetDirection(),
        defaultPixelValue=0,
    )
    return sitk.GetArrayFromImage(res).astype(np.int64)


class PSFHMhaDataset(Dataset):
    """
    Expects:
      root/
        image_mha/*.mha
        label_mha/*.mha

    Image shapes supported:
      - (H,W) -> (1,H,W)
      - (1,H,W) or (3,H,W) -> treated as channels
      - (Z,H,W) with Z not in {1,3} -> take middle slice -> (1,H,W)

    Mask supported:
      - (H,W) or (1,H,W)
      - (Z,H,W) -> take middle slice

    Normalization: /255.0
    Optional resize to img_size (bilinear for image, nearest for mask).

    NOTE: Set in_ch=3 for your case.
    """
    def __init__(self, root: str, img_size: int = 256, in_ch: int = 3, resize: bool = True):
        self.root = Path(root)
        self.img_dir = self.root / "image_mha"
        self.lab_dir = self.root / "label_mha"
        if not self.img_dir.exists() or not self.lab_dir.exists():
            raise FileNotFoundError(f"Missing image_mha/label_mha under: {self.root}")

        imgs = sorted(self.img_dir.glob("*.mha"))
        labs = sorted(self.lab_dir.glob("*.mha"))

        img_map = {p.stem: p for p in imgs}
        lab_map = {p.stem: p for p in labs}

        self.cases = sorted(set(img_map.keys()).intersection(lab_map.keys()))
        if not self.cases:
            raise RuntimeError("No paired mha found. Ensure same stems in image_mha and label_mha")

        self.img_paths = [img_map[c] for c in self.cases]
        self.lab_paths = [lab_map[c] for c in self.cases]

        self.img_size = int(img_size)
        self.in_ch = int(in_ch)
        self.resize = bool(resize)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx]
        x = _read_mha(self.img_paths[idx])
        y = _read_mha(self.lab_paths[idx])

        # image -> (C,H,W)
        if x.ndim == 2:
            x = x[None, ...]
        elif x.ndim == 3:
            if x.shape[0] not in (1, 3):
                z = x.shape[0] // 2
                x = x[z:z+1, ...]
        else:
            raise ValueError(f"Image {case} unexpected shape {x.shape}")

        # mask -> (H,W)
        if y.ndim == 3 and y.shape[0] == 1:
            y = y[0]
        elif y.ndim == 3 and y.shape[0] != 1:
            y = y[y.shape[0] // 2]
        if y.ndim != 2:
            raise ValueError(f"Mask {case} unexpected shape {y.shape}")

        x = x.astype(np.float32) / 255.0
        y = y.astype(np.int64)

        if self.resize:
            if x.shape[-2:] != (self.img_size, self.img_size):
                x = _resize_image_chw_linear(x, self.img_size)
            if y.shape != (self.img_size, self.img_size):
                y = _resize_mask_hw_nearest(y, self.img_size)

        # channel match
        if x.shape[0] == 1 and self.in_ch == 3:
            x = np.repeat(x, 3, axis=0)
        elif x.shape[0] == 3 and self.in_ch == 1:
            x = x.mean(axis=0, keepdims=True)
        elif x.shape[0] != self.in_ch:
            raise ValueError(f"Image {case} has {x.shape[0]} channels but in_ch={self.in_ch}")

        return {
            "image": torch.from_numpy(x).float(),   # (C,H,W)
            "mask": torch.from_numpy(y).long(),     # (H,W)
            "case": case,
        }
