#!/usr/bin/env python3
# ============================================================
# FILE: visualize_stage1_reconstructions.py
#
# PURPOSE:
#   Visualize reconstructions from Stage-1 AutoEncoders:
#     - Image AE:  x -> x_hat
#     - Mask  AE:  onehot(y) -> y_hat -> argmax -> y_hat_lbl
#
#   Saves ONLY 5 random samples:
#     outdir/
#       sample_<case>/
#         img_in.png
#         img_rec.png
#         img_diff.png
#         mask_gt.png
#         mask_rec.png
#         mask_err.png
#
# ASSUMPTIONS:
#   - PSFH dataset layout:
#       root/image_mha/*.mha
#       root/label_mha/*.mha
#   - You have checkpoints:
#       ckpt_dir/client{K}_img_ae.pt
#       ckpt_dir/client{K}_mask_ae.pt
#   - Your AutoEncoder forward returns reconstruction.
#     If your AE returns (rec, aux) or dict, edit "ae_forward()" below.
#
# RUN EXAMPLE:
#   python visualize_stage1_reconstructions.py \
#     --data-root "/path/to/trainingdataset" \
#     --ckpt-dir "/path/to/checkpoints" \
#     --client 1 \
#     --outdir "./recon_viz/client1" \
#     --img-size 256 --in-ch 3 --num-classes 3 \
#     --use-skips 1 --use-klt 1 \
#     --num-samples 5 --seed 42 --device cuda:0
# ============================================================

# Example run (5 samples)
# python visualization_stage1_recon.py \
#   --data-root "/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/datasets/testdataset" \
#   --ckpt-dir "/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/runs/checkpoints" \
#   --client 1 \
#   --outdir "/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/recon_viz/client1" \
#   --img-size 256 --in-ch 3 --num-classes 3 \
#   --use-skips 1 --use-klt 1 \
#   --num-samples 5 --seed 42 --device cuda:0


import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# headless-safe saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.psfh_mha import PSFHMhaDataset
from models.autoencoder import AutoEncoder


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_numpy01_img(x: torch.Tensor) -> np.ndarray:
    """
    x: (C,H,W) float in [0,1] (assumed)
    returns HxW grayscale in [0,1] for saving
    """
    x = x.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)}")
    if x.size(0) == 1:
        g = x[0]
    else:
        g = x.mean(dim=0)
    g = g.clamp(0, 1).numpy()
    return g


def color_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Simple palette:
      0 black, 1 red, 2 green, 3 blue, others cyan-ish
    """
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if num_classes >= 2:
        out[mask == 1] = [255, 0, 0]
    if num_classes >= 3:
        out[mask == 2] = [0, 255, 0]
    if num_classes >= 4:
        out[mask == 3] = [0, 0, 255]
    # fallback for any other label
    out[(mask != 0) & (mask != 1) & (mask != 2) & (mask != 3)] = [0, 255, 255]
    return out


def save_gray(path: Path, img01: np.ndarray):
    ensure_dir(path.parent)
    plt.imsave(str(path), img01, cmap="gray", vmin=0.0, vmax=1.0)


def save_rgb(path: Path, rgb: np.ndarray):
    ensure_dir(path.parent)
    plt.imsave(str(path), rgb)


def ae_forward(ae: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Handles different AE forward styles.
    Expected output: reconstructed tensor same shape as input.
    Adjust if your AutoEncoder returns dict/tuple.
    """
    out = ae(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        # common keys: "recon", "x_hat"
        out = out.get("recon", out.get("x_hat", None))
        if out is None:
            raise ValueError("AutoEncoder forward returned dict without recon/x_hat key.")
    if not torch.is_tensor(out):
        raise ValueError("AutoEncoder forward output is not a tensor.")
    return out


# ----------------------------
# Main
# ----------------------------
def get_args():
    p = argparse.ArgumentParser("Visualize Stage-1 reconstructions (Image AE + Mask AE)")

    p.add_argument("--data-root", type=str, required=True,
                   help="PSFH dataset root containing image_mha/ and label_mha/")
    p.add_argument("--ckpt-dir", type=str, required=True,
                   help="Folder containing clientK_img_ae.pt and clientK_mask_ae.pt")
    p.add_argument("--client", type=int, default=1, help="Client id (K)")

    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--in-ch", type=int, default=3)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--no-resize", type=int, default=0, choices=[0, 1])

    # match your stage-1 toggles (use 1/0)
    p.add_argument("--use-skips", type=int, default=1, choices=[0, 1])
    p.add_argument("--use-klt", type=int, default=1, choices=[0, 1])

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # save config for reproducibility
    with open(outdir / "recon_viz_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = args.device
    use_skips = bool(args.use_skips)
    use_klt = bool(args.use_klt)

    # dataset
    ds = PSFHMhaDataset(
        root=args.data_root,
        img_size=int(args.img_size),
        in_ch=int(args.in_ch),
        resize=(not bool(args.no_resize)),
    )

    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset is empty.")

    k = min(int(args.num_samples), n)
    rng = np.random.RandomState(int(args.seed))
    chosen = rng.choice(n, size=k, replace=False).tolist()

    dl = DataLoader(
        Subset(ds, chosen),
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=("cuda" in device),
    )

    # load AEs
    ckpt_dir = Path(args.ckpt_dir)
    img_ckpt = ckpt_dir / f"client{int(args.client)}_img_ae.pt"
    msk_ckpt = ckpt_dir / f"client{int(args.client)}_mask_ae.pt"
    if not img_ckpt.exists():
        raise FileNotFoundError(f"Missing: {img_ckpt}")
    if not msk_ckpt.exists():
        raise FileNotFoundError(f"Missing: {msk_ckpt}")

    img_ae = AutoEncoder(in_ch=int(args.in_ch), use_skips=use_skips, use_klt=use_klt, seed=int(args.seed) + int(args.client))
    img_ae.load_state_dict(torch.load(str(img_ckpt), map_location="cpu"))
    img_ae = img_ae.to(device).eval()

    msk_ae = AutoEncoder(in_ch=int(args.num_classes), use_skips=use_skips, use_klt=use_klt, seed=100 + int(args.seed) + int(args.client))
    msk_ae.load_state_dict(torch.load(str(msk_ckpt), map_location="cpu"))
    msk_ae = msk_ae.to(device).eval()

    # iterate and save
    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device)  # (1,C,H,W)
            y = batch["mask"].to(device)   # (1,H,W)
            case = batch.get("case", ["case"])[0]

            # ---- Image reconstruction ----
            x_hat = ae_forward(img_ae, x)  # (1,C,H,W)
            x_in = x[0]
            x_rec = x_hat[0]

            img_in = to_numpy01_img(x_in)
            img_rec = to_numpy01_img(x_rec)
            img_diff = np.abs(img_in - img_rec).clip(0, 1)

            # ---- Mask reconstruction ----
            y0 = y[0]  # (H,W) long
            y_oh = F.one_hot(y0.long(), num_classes=int(args.num_classes)).permute(2, 0, 1).float()  # (K,H,W)
            y_oh = y_oh.unsqueeze(0).to(device)  # (1,K,H,W)

            y_hat = ae_forward(msk_ae, y_oh)  # (1,K,H,W)

            # convert reconstruction to labels
            # if AE output isn't bounded, argmax still works
            y_rec_lbl = torch.argmax(y_hat[0], dim=0).detach().cpu().numpy().astype(np.int64)
            y_gt_lbl = y0.detach().cpu().numpy().astype(np.int64)

            # error map (1 if wrong)
            y_err = (y_rec_lbl != y_gt_lbl).astype(np.uint8)

            # colorize
            gt_rgb = color_mask(y_gt_lbl, int(args.num_classes))
            rec_rgb = color_mask(y_rec_lbl, int(args.num_classes))

            # error rgb: red where wrong
            err_rgb = np.zeros((*y_err.shape, 3), dtype=np.uint8)
            err_rgb[y_err > 0] = [255, 0, 0]

            # folder per sample
            sdir = outdir / f"sample_{case}"
            ensure_dir(sdir)

            save_gray(sdir / "img_in.png", img_in)
            save_gray(sdir / "img_rec.png", img_rec)
            save_gray(sdir / "img_diff.png", img_diff)

            save_rgb(sdir / "mask_gt.png", gt_rgb)
            save_rgb(sdir / "mask_rec.png", rec_rgb)
            save_rgb(sdir / "mask_err.png", err_rgb)

            # also dump a small summary
            with open(sdir / "info.json", "w") as f:
                json.dump(
                    {
                        "case": case,
                        "img_ckpt": str(img_ckpt),
                        "mask_ckpt": str(msk_ckpt),
                        "use_skips": use_skips,
                        "use_klt": use_klt,
                        "img_size": int(args.img_size),
                        "in_ch": int(args.in_ch),
                        "num_classes": int(args.num_classes),
                    },
                    f,
                    indent=2,
                )

            print(f"[SAVED] {sdir}")

    print(f"\nDONE. Saved {k} samples into: {outdir}")


if __name__ == "__main__":
    main()
