# ============================================================
# FILE: psfh_universal_training/data/factory.py
#  - Single entry-point for Stage-1 and Stage-2 scripts
#  - Returns a torch.utils.data.Dataset
# ============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset

from .fumpe_dataset import FUMPEBinarySliceDataset
from .heart_mri_dataset import HeartMRIDataset
from .nerve_dataset import NerveDataset


@dataclass
class DatasetArgs:
    dataset: str
    data_root: str
    split: str
    img_size: int = 256
    seed: int = 42
    test_ratio: float = 0.1
    drop_empty_masks: bool = True

    # FUMPE-specific
    fumpe_split_strategy: str = "patient"

    # NERVE-specific
    nerve_pad: int = 14


def build_dataset(args: DatasetArgs) -> Dataset:
    name = args.dataset.lower().strip()

    if name in ("fumpe", "fumpe_new", "fumpe_ct"):
        return FUMPEBinarySliceDataset(
            root=args.data_root,
            split=args.split,
            img_size=args.img_size,
            drop_empty_masks=args.drop_empty_masks,
            seed=args.seed,
            split_strategy=args.fumpe_split_strategy,
            test_ratio=args.test_ratio,
        )

    if name in ("heart_mri", "mri", "heart"):
        # uses predefined split folders inside data_root
        return HeartMRIDataset(
            root=args.data_root,
            split=args.split,
            img_size=args.img_size,
            drop_empty_masks=args.drop_empty_masks,
        )

    if name in ("nerve", "ultrasound_nerve", "ultrasound-nerve-segmentation"):
        return NerveDataset(
            root=args.data_root,
            split=args.split,
            img_size=args.img_size,
            pad=args.nerve_pad,
            drop_empty_masks=args.drop_empty_masks,
            seed=args.seed,
            test_ratio=args.test_ratio,
        )

    raise ValueError(f"Unknown dataset: {args.dataset}")
