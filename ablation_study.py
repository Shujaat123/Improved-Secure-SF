#!/usr/bin/env python3
# ============================================================
# FILE: ablation_clientsweep_ppms_psfh.py
#
# PPMS-PSFH Ablation: #clients vs training strategy (GLOBAL vs LOCAL UMN)
#
# Strategies:
#   1) global: collate all clients' latent pairs at server -> train ONE UMN
#   2) local : each client trains its OWN UMN using only its own latent pairs
#
# No cross prediction:
#   - For each client i, evaluate using (img_encT_i -> UMN -> mask_dec_i)
#   - For global: UMN is shared across all clients
#   - For local : each client has UMN_i
#
# Metrics:
#   - Reconstruction (Image AE): PSNR + SSIM (if skimage available; else SSIM=nan)
#   - Segmentation: merged-FG Dice + IoU for classes (1,2) on test set
#
# Outputs:
#   results_dir/
#     summary_clientsweep.csv
#     detailed_runs.json
#     per_setting/
#       nc{N}_{variant}_{strategy}_perclient.csv
#     runs/
#       nc{N}_{variant}/
#         client{K}_img_ae.pt
#         client{K}_mask_ae.pt
#         umn_global.pt OR umn_client{K}.pt
#python ablation_study.py \
#   --train-root "/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/datasets/Pubic Symphysis-Fetal Head Segmentation and Angle of Progression" \
#   --test-root  "/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/datasets/testdataset" \
#   --results-dir "/home/saheed_bello/PROJECTS/IMAGE SEGMENTATION/validation_code/ppms_psfh/ablation_clientsweep" \
#   --clients-sweep "1,3,4,5,6,7" \
#   --img-size 256 --in-ch 3 --num-classes 3 \
#   --use-skips 1 --use-klt 1 --use-ppm 1 \
#   --ae-epochs 100 --ae-batch 16 --ae-lr 1e-3 --ae-patience 15 \
#   --umn-epochs 150 --umn-batch 16 --umn-lr 1e-3 --umn-patience 20 \
#   --eval-batch 8 \
#   --device cuda:0 --seed 42


# Requirements:
#   - Uses your existing ppms_psfh modules:
#       data.psfh_mha.PSFHMhaDataset
#       models.autoencoder.AutoEncoder
#       models.umn.UMN
#       train.train_ae.train_ae
#       train.train_umn.train_umn
#       train.checks.extract_latents_server_inverse
#   - Test dataset root must contain image_mha/ and label_mha/
# ============================================================

import os
import csv
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset

from data.psfh_mha import PSFHMhaDataset
from models.autoencoder import AutoEncoder
from models.umn import UMN
from utils.seed import set_seed
from train.train_ae import train_ae
from train.train_umn import train_umn
from train.checks import extract_latents_server_inverse


# ----------------------------
# optional skimage metrics
# ----------------------------
def _try_import_skimage():
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        return ssim, psnr
    except Exception:
        return None, None


_SSIM, _PSNR = _try_import_skimage()


# ----------------------------
# dataset wrappers
# ----------------------------
class ImgOnly(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        d = self.base[i]
        return d["image"], d["mask"], d.get("case", str(i))


class MaskAsX(Dataset):
    """
    For mask AE: use one-hot mask as the "image" to reconstruct.
    Returns (K,H,W) float for AE input/target.
    """
    def __init__(self, base: Dataset, num_classes: int):
        self.base = base
        self.num_classes = int(num_classes)
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        d = self.base[i]
        y = d["mask"]  # (H,W) long
        y_oh = F.one_hot(y, num_classes=self.num_classes).permute(2, 0, 1).float()
        return y_oh, d["image"], d.get("case", str(i))


class LatentTripletDS(Dataset):
    def __init__(self, Xb, Xs3, Xs4, Yb, Ys3, Ys4):
        self.Xb, self.Xs3, self.Xs4 = Xb, Xs3, Xs4
        self.Yb, self.Ys3, self.Ys4 = Yb, Ys3, Ys4
    def __len__(self): return self.Xb.size(0)
    def __getitem__(self, i):
        return self.Xb[i], self.Xs3[i], self.Xs4[i], self.Yb[i], self.Ys3[i], self.Ys4[i]


# ----------------------------
# splits
# ----------------------------
def partition_clients_indices(n: int, num_clients: int, seed: int) -> List[List[int]]:
    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    splits = np.array_split(idx, int(num_clients))
    return [s.tolist() for s in splits]


def split_train_val_indices(indices: List[int], val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(int(seed))
    idx = np.array(indices)
    rng.shuffle(idx)
    n_val = int(round(len(idx) * float(val_frac)))
    val_idx = idx[:n_val].tolist()
    tr_idx = idx[n_val:].tolist()
    return tr_idx, val_idx


# ----------------------------
# metrics
# ----------------------------
def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num / (den + eps))


def merged_fg_dice_iou_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    fg_classes: Tuple[int, ...] = (1, 2),
) -> Tuple[float, float]:
    """
    logits: (B,C,H,W) raw or probs
    y:      (B,H,W) long labels
    returns mean dice, mean iou across batch for merged FG.
    """
    pred = torch.argmax(logits, dim=1)  # (B,H,W)
    fg = torch.tensor(list(fg_classes), device=pred.device, dtype=pred.dtype)
    pr = torch.isin(pred, fg)
    gt = torch.isin(y, fg)

    tp = torch.logical_and(pr, gt).sum().item()
    fp = torch.logical_and(pr, ~gt).sum().item()
    fn = torch.logical_and(~pr, gt).sum().item()

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8) if (2 * tp + fp + fn) > 0 else 1.0
    iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 1.0
    return float(dice), float(iou)


def psnr_ssim_batch(x: torch.Tensor, xhat: torch.Tensor) -> Tuple[float, float]:
    """
    x, xhat: (B,C,H,W) in [0,1]
    returns mean PSNR, mean SSIM (SSIM=nan if skimage missing)
    """
    x = x.detach().cpu().float().clamp(0, 1).numpy()
    xhat = xhat.detach().cpu().float().clamp(0, 1).numpy()

    # PSNR
    mse = np.mean((x - xhat) ** 2, axis=(1, 2, 3))
    psnr_vals = []
    for m in mse:
        if m <= 1e-12:
            psnr_vals.append(99.0)
        else:
            psnr_vals.append(10.0 * np.log10(1.0 / m))
    psnr_mean = float(np.mean(psnr_vals))

    # SSIM (if possible)
    if _SSIM is None:
        return psnr_mean, float("nan")

    ssim_vals = []
    for i in range(x.shape[0]):
        # convert to HWC for skimage
        xi = np.transpose(x[i], (1, 2, 0))
        xhi = np.transpose(xhat[i], (1, 2, 0))
        # if C==1, drop channel to 2D
        if xi.shape[2] == 1:
            xi = xi[..., 0]
            xhi = xhi[..., 0]
            ssim_vals.append(_SSIM(xi, xhi, data_range=1.0))
        else:
            # multichannel
            ssim_vals.append(_SSIM(xi, xhi, channel_axis=2, data_range=1.0))
    ssim_mean = float(np.mean(ssim_vals))
    return psnr_mean, ssim_mean


# ----------------------------
# federated stage-2 inference (no cross)
# ----------------------------
@torch.no_grad()
def stage2_predict_logits(
    img_encT,           # encoderT: forward -> (bT,s3T,s4T), inverse_triplet
    umn,                # UMN: forward(b,s3,s4)->(b_hat,s3_hat,s4_hat)
    mask_encT,          # mask encoderT provides T_b/T_s3/T_s4 if klt else identity behavior
    mask_dec,           # decoder: if skips -> decoder(b, s3=..., s4=...), else decoder(b)
    x: torch.Tensor,    # (B,3,H,W) or (B,1,H,W)
    use_skips: bool,
) -> torch.Tensor:
    bT, s3T, s4T = img_encT(x)
    b, s3, s4 = img_encT.inverse_triplet(bT, s3T, s4T)
    b_hat, s3_hat, s4_hat = umn(b, s3, s4)

    # apply target mask T if present
    if getattr(mask_encT, "use_klt", False):
        pb = mask_encT.T_b(b_hat)
        ps3 = mask_encT.T_s3(s3_hat)
        ps4 = mask_encT.T_s4(s4_hat)
    else:
        pb, ps3, ps4 = b_hat, s3_hat, s4_hat

    b2, s32, s42 = mask_encT.inverse_triplet(pb, ps3, ps4)
    if use_skips:
        out = mask_dec(b2, s3=s32, s4=s42)
    else:
        out = mask_dec(b2)
    return out  # (B,C,H,W)


# ----------------------------
# experiment spec
# ----------------------------
@dataclass
class VariantSpec:
    name: str
    use_skips: bool
    use_klt: bool
    use_ppm: bool


# ----------------------------
# training blocks
# ----------------------------
def train_stage1_aes_for_clients(
    base_ds: Dataset,
    client_splits: List[List[int]],
    num_classes: int,
    in_ch_img: int,
    img_size: int,
    resize: bool,
    ae_epochs: int,
    ae_batch: int,
    ae_lr: float,
    ae_patience: int,
    val_frac_clients: float,
    seed: int,
    device: str,
    num_workers: int,
    use_skips: bool,
    use_klt: bool,
    run_dir: Path,
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    Returns:
      encT_list: [(img_encT, mask_encT)] on CPU
      dec_list : [(img_dec,  mask_dec )] on CPU
      img_ae_list: [AutoEncoder] on CPU
      mask_ae_list:[AutoEncoder] on CPU
    Saves:
      clientK_img_ae.pt, clientK_mask_ae.pt into run_dir
    """
    ds_img = ImgOnly(base_ds)
    ds_msk = MaskAsX(base_ds, num_classes=num_classes)

    encT_list, dec_list = [], []
    img_ae_list, mask_ae_list = [], []

    for ci, indices in enumerate(client_splits, start=1):
        tr_idx, va_idx = split_train_val_indices(indices, val_frac=val_frac_clients, seed=seed + ci)

        img_tr = DataLoader(Subset(ds_img, tr_idx), batch_size=ae_batch, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
        img_va = DataLoader(Subset(ds_img, va_idx), batch_size=ae_batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

        msk_tr = DataLoader(Subset(ds_msk, tr_idx), batch_size=ae_batch, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=False)
        msk_va = DataLoader(Subset(ds_msk, va_idx), batch_size=ae_batch, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

        # Image AE
        img_ae = AutoEncoder(in_ch=in_ch_img, use_skips=use_skips, use_klt=use_klt, seed=seed + ci)
        img_ae = train_ae(img_ae, img_tr, img_va, device=device,
                          epochs=ae_epochs, lr=ae_lr, patience=ae_patience,
                          tag=f"C{ci}-IMG", log_csv_path=str(run_dir / f"client{ci}_img_ae.csv"))

        # Mask AE
        mask_ae = AutoEncoder(in_ch=num_classes, use_skips=use_skips, use_klt=use_klt, seed=100 + seed + ci)
        mask_ae = train_ae(mask_ae, msk_tr, msk_va, device=device,
                           epochs=ae_epochs, lr=ae_lr, patience=ae_patience,
                           tag=f"C{ci}-MSK", log_csv_path=str(run_dir / f"client{ci}_mask_ae.csv"))

        # save whole AE weights (so you can reload later)
        torch.save(img_ae.state_dict(), str(run_dir / f"client{ci}_img_ae.pt"))
        torch.save(mask_ae.state_dict(), str(run_dir / f"client{ci}_mask_ae.pt"))

        # keep handles (CPU)
        img_ae_list.append(img_ae.cpu().eval())
        mask_ae_list.append(mask_ae.cpu().eval())

        encT_list.append((img_ae.encoderT.cpu().eval(), mask_ae.encoderT.cpu().eval()))
        dec_list.append((img_ae.decoder.cpu().eval(), mask_ae.decoder.cpu().eval()))

    return encT_list, dec_list, img_ae_list, mask_ae_list


def build_latent_pairs_per_client(
    base_ds: Dataset,
    client_splits: List[List[int]],
    encT_list: List[Tuple[Any, Any]],
    num_workers: int,
    umn_batch: int,
    device: str,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Returns per-client ORIGINAL (server-recovered) latent pairs:
      (Xb, Xs3, Xs4, Yb, Ys3, Ys4) each is CPU tensor.
    """
    ds_img = ImgOnly(base_ds)
    ds_msk = MaskAsX(base_ds, num_classes=encT_list[0][1].in_ch if hasattr(encT_list[0][1], "in_ch") else 3)

    # NOTE: ds_msk uses num_classes from outer scope in training, but for extraction we only need tensors.
    # We'll instead rebuild MaskAsX with explicit classes in caller if you want strictness.
    # Here, we'll extract directly using the same base wrapper approach as stage1 main.

    pairs = []
    for ci, indices in enumerate(client_splits, start=1):
        img_encT, mask_encT = encT_list[ci - 1]

        loader_img = DataLoader(Subset(ds_img, indices), batch_size=umn_batch, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
        loader_msk = DataLoader(Subset(MaskAsX(base_ds, num_classes=getattr(mask_encT, "in_ch", 3)), indices),
                                batch_size=umn_batch, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

        Xb, Xs3, Xs4 = extract_latents_server_inverse(img_encT, loader_img, device=device)
        Yb, Ys3, Ys4 = extract_latents_server_inverse(mask_encT, loader_msk, device=device)

        pairs.append((Xb.cpu(), Xs3.cpu(), Xs4.cpu(), Yb.cpu(), Ys3.cpu(), Ys4.cpu()))
    return pairs


def train_umn_global_from_pairs(
    per_client_pairs,
    use_ppm: bool,
    umn_epochs: int,
    umn_lr: float,
    umn_patience: int,
    umn_batch: int,
    seed: int,
    device: str,
    run_dir: Path,
) -> Any:
    Xb  = torch.cat([p[0] for p in per_client_pairs], 0)
    Xs3 = torch.cat([p[1] for p in per_client_pairs], 0)
    Xs4 = torch.cat([p[2] for p in per_client_pairs], 0)
    Yb  = torch.cat([p[3] for p in per_client_pairs], 0)
    Ys3 = torch.cat([p[4] for p in per_client_pairs], 0)
    Ys4 = torch.cat([p[5] for p in per_client_pairs], 0)

    N = Xb.size(0)
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(round(N * 0.2))
    val_idx = idx[:n_val].tolist()
    tr_idx  = idx[n_val:].tolist()

    tr_loader = DataLoader(LatentTripletDS(Xb[tr_idx], Xs3[tr_idx], Xs4[tr_idx], Yb[tr_idx], Ys3[tr_idx], Ys4[tr_idx]),
                           batch_size=umn_batch, shuffle=True, num_workers=0, drop_last=False)
    va_loader = DataLoader(LatentTripletDS(Xb[val_idx], Xs3[val_idx], Xs4[val_idx], Yb[val_idx], Ys3[val_idx], Ys4[val_idx]),
                           batch_size=umn_batch, shuffle=False, num_workers=0, drop_last=False)

    umn = UMN(use_ppm=use_ppm)
    umn = train_umn(umn, tr_loader, va_loader, device=device,
                    epochs=umn_epochs, lr=umn_lr, patience=umn_patience,
                    log_csv_path=str(run_dir / "umn_global.csv"))
    torch.save(umn.state_dict(), str(run_dir / "umn_global.pt"))
    return umn.cpu().eval()


def train_umn_local_from_pairs(
    per_client_pairs,
    use_ppm: bool,
    umn_epochs: int,
    umn_lr: float,
    umn_patience: int,
    umn_batch: int,
    seed: int,
    device: str,
    run_dir: Path,
) -> List[Any]:
    umns = []
    for ci, p in enumerate(per_client_pairs, start=1):
        Xb, Xs3, Xs4, Yb, Ys3, Ys4 = p
        N = Xb.size(0)
        idx = np.arange(N)
        rng = np.random.default_rng(seed + 1000 + ci)
        rng.shuffle(idx)
        n_val = int(round(N * 0.2))
        val_idx = idx[:n_val].tolist()
        tr_idx  = idx[n_val:].tolist()

        tr_loader = DataLoader(LatentTripletDS(Xb[tr_idx], Xs3[tr_idx], Xs4[tr_idx], Yb[tr_idx], Ys3[tr_idx], Ys4[tr_idx]),
                               batch_size=umn_batch, shuffle=True, num_workers=0, drop_last=False)
        va_loader = DataLoader(LatentTripletDS(Xb[val_idx], Xs3[val_idx], Xs4[val_idx], Yb[val_idx], Ys3[val_idx], Ys4[val_idx]),
                               batch_size=umn_batch, shuffle=False, num_workers=0, drop_last=False)

        umn = UMN(use_ppm=use_ppm)
        umn = train_umn(umn, tr_loader, va_loader, device=device,
                        epochs=umn_epochs, lr=umn_lr, patience=umn_patience,
                        log_csv_path=str(run_dir / f"umn_client{ci}.csv"))
        torch.save(umn.state_dict(), str(run_dir / f"umn_client{ci}.pt"))
        umns.append(umn.cpu().eval())
    return umns


# ----------------------------
# evaluation blocks
# ----------------------------
@torch.no_grad()
def eval_recon_image_ae_per_client(
    img_ae_list: List[Any],
    test_loader: DataLoader,
    device: str,
) -> List[Dict[str, float]]:
    stats = []
    for ci, img_ae in enumerate(img_ae_list, start=1):
        img_ae = img_ae.to(device).eval()
        psnr_vals, ssim_vals = [], []

        for x, _y, _case in test_loader:
            x = x.to(device)
            # AutoEncoder forward should reconstruct image directly
            xhat = img_ae(x)
            p, s = psnr_ssim_batch(x, xhat)
            psnr_vals.append(p)
            ssim_vals.append(s)

        stats.append({
            "client": ci,
            "psnr": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
            "ssim": float(np.nanmean(ssim_vals)) if ssim_vals else float("nan"),
        })
        img_ae = img_ae.cpu()
    return stats


@torch.no_grad()
def eval_segmentation_per_client_global_umn(
    umn_global: Any,
    encT_list: List[Tuple[Any, Any]],
    dec_list: List[Tuple[Any, Any]],
    test_loader: DataLoader,
    device: str,
    use_skips: bool,
    fg_classes: Tuple[int, ...] = (1, 2),
) -> List[Dict[str, float]]:
    stats = []
    umn_global = umn_global.to(device).eval()

    for ci, ((img_encT, mask_encT), (_img_dec, mask_dec)) in enumerate(zip(encT_list, dec_list), start=1):
        img_encT = img_encT.to(device).eval()
        mask_encT = mask_encT.to(device).eval()
        mask_dec = mask_dec.to(device).eval()

        dices, ious = [], []
        for x, y, _case in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = stage2_predict_logits(img_encT, umn_global, mask_encT, mask_dec, x, use_skips=use_skips)
            d, i = merged_fg_dice_iou_from_logits(out, y, fg_classes=fg_classes)
            dices.append(d); ious.append(i)

        stats.append({
            "client": ci,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "iou": float(np.mean(ious)) if ious else 0.0,
        })

        img_encT = img_encT.cpu()
        mask_encT = mask_encT.cpu()
        mask_dec = mask_dec.cpu()

    umn_global = umn_global.cpu()
    return stats


@torch.no_grad()
def eval_segmentation_per_client_local_umn(
    umn_list: List[Any],
    encT_list: List[Tuple[Any, Any]],
    dec_list: List[Tuple[Any, Any]],
    test_loader: DataLoader,
    device: str,
    use_skips: bool,
    fg_classes: Tuple[int, ...] = (1, 2),
) -> List[Dict[str, float]]:
    stats = []
    for ci, (umn_i, (encs), (_decs)) in enumerate(zip(umn_list, encT_list, dec_list), start=1):
        img_encT, mask_encT = encs
        _img_dec, mask_dec = _decs

        umn_i = umn_i.to(device).eval()
        img_encT = img_encT.to(device).eval()
        mask_encT = mask_encT.to(device).eval()
        mask_dec = mask_dec.to(device).eval()

        dices, ious = [], []
        for x, y, _case in test_loader:
            x = x.to(device)
            y = y.to(device)
            out = stage2_predict_logits(img_encT, umn_i, mask_encT, mask_dec, x, use_skips=use_skips)
            d, i = merged_fg_dice_iou_from_logits(out, y, fg_classes=fg_classes)
            dices.append(d); ious.append(i)

        stats.append({
            "client": ci,
            "dice": float(np.mean(dices)) if dices else 0.0,
            "iou": float(np.mean(ious)) if ious else 0.0,
        })

        umn_i = umn_i.cpu()
        img_encT = img_encT.cpu()
        mask_encT = mask_encT.cpu()
        mask_dec = mask_dec.cpu()
    return stats


def mean_std(stats: List[Dict[str, float]], key: str) -> Tuple[float, float]:
    vals = np.array([float(s[key]) for s in stats], dtype=np.float64)
    return float(np.nanmean(vals)), float(np.nanstd(vals))


# ----------------------------
# CLI
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("PPMS-PSFH Ablation: #clients vs (global vs local UMN)")

    p.add_argument("--train-root", type=str, required=True, help="Train root with image_mha/label_mha")
    p.add_argument("--test-root", type=str, required=True, help="Test root with image_mha/label_mha")
    p.add_argument("--results-dir", type=str, required=True)

    p.add_argument("--clients-sweep", type=str, default="1,3,4,5,6,7", help="Comma list, e.g. 1,3,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)

    # data
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--in-ch", type=int, default=3)
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--no-resize", type=int, default=0, choices=[0, 1])

    # toggles (0/1)
    p.add_argument("--use-skips", type=int, default=1, choices=[0, 1])
    p.add_argument("--use-klt", type=int, default=1, choices=[0, 1])
    p.add_argument("--use-ppm", type=int, default=1, choices=[0, 1])

    # splits
    p.add_argument("--client-val-frac", type=float, default=0.2, help="within-client train/val for AEs")
    p.add_argument("--umn-val-frac", type=float, default=0.2, help="within-umn train/val (global/local)")

    # AE training
    p.add_argument("--ae-epochs", type=int, default=100)
    p.add_argument("--ae-batch", type=int, default=16)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--ae-patience", type=int, default=15)

    # UMN training
    p.add_argument("--umn-epochs", type=int, default=150)
    p.add_argument("--umn-batch", type=int, default=16)
    p.add_argument("--umn-lr", type=float, default=1e-3)
    p.add_argument("--umn-patience", type=int, default=20)

    # eval
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--fg-classes", type=str, default="1,2")

    return p


# ----------------------------
# main sweep
# ----------------------------
def main():
    args = build_parser().parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "per_setting").mkdir(parents=True, exist_ok=True)
    (results_dir / "runs").mkdir(parents=True, exist_ok=True)

    clients_sweep = [int(x.strip()) for x in args.clients_sweep.split(",") if x.strip()]
    fg_classes = tuple(int(x.strip()) for x in args.fg_classes.split(",") if x.strip())

    variant = VariantSpec(
        name=f"{'full' if (args.use_skips and args.use_klt and args.use_ppm) else 'custom'}_sk{args.use_skips}_klt{args.use_klt}_ppm{args.use_ppm}",
        use_skips=bool(args.use_skips),
        use_klt=bool(args.use_klt),
        use_ppm=bool(args.use_ppm),
    )

    # datasets
    train_ds = PSFHMhaDataset(root=args.train_root, img_size=args.img_size, in_ch=args.in_ch, resize=(not bool(args.no_resize)))
    test_ds  = PSFHMhaDataset(root=args.test_root,  img_size=args.img_size, in_ch=args.in_ch, resize=(not bool(args.no_resize)))

    test_loader = DataLoader(
        ImgOnly(test_ds),
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        drop_last=False,
    )

    summary_rows: List[Dict[str, Any]] = []
    detailed_runs: List[Dict[str, Any]] = []

    for n_clients in clients_sweep:
        print("\n" + "=" * 100)
        print(f"RUN: num_clients={n_clients} | variant={variant.name}")
        print("=" * 100)

        run_dir = results_dir / "runs" / f"nc{n_clients}_{variant.name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # split clients
        client_splits = partition_clients_indices(len(train_ds), n_clients, args.seed)

        # train AEs
        encT_list, dec_list, img_ae_list, _mask_ae_list = train_stage1_aes_for_clients(
            base_ds=train_ds,
            client_splits=client_splits,
            num_classes=args.num_classes,
            in_ch_img=args.in_ch,
            img_size=args.img_size,
            resize=(not bool(args.no_resize)),
            ae_epochs=args.ae_epochs,
            ae_batch=args.ae_batch,
            ae_lr=args.ae_lr,
            ae_patience=args.ae_patience,
            val_frac_clients=args.client_val_frac,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            use_skips=variant.use_skips,
            use_klt=variant.use_klt,
            run_dir=run_dir,
        )

        # recon metrics per client (image AE)
        recon_stats = eval_recon_image_ae_per_client(img_ae_list, test_loader, device=args.device)
        recon_psnr_mean, recon_psnr_std = mean_std(recon_stats, "psnr")
        recon_ssim_mean, recon_ssim_std = mean_std(recon_stats, "ssim")

        # latent pairs per client
        # (use same extraction approach as your stage1 script)
        pairs = []
        ds_img_train = ImgOnly(train_ds)
        ds_msk_train = MaskAsX(train_ds, num_classes=args.num_classes)

        for ci, indices in enumerate(client_splits, start=1):
            img_encT, mask_encT = encT_list[ci - 1]

            loader_img = DataLoader(Subset(ds_img_train, indices), batch_size=args.umn_batch, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=False)
            loader_msk = DataLoader(Subset(ds_msk_train, indices), batch_size=args.umn_batch, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=False)

            Xb, Xs3, Xs4 = extract_latents_server_inverse(img_encT, loader_img, device=args.device)
            Yb, Ys3, Ys4 = extract_latents_server_inverse(mask_encT, loader_msk, device=args.device)
            pairs.append((Xb.cpu(), Xs3.cpu(), Xs4.cpu(), Yb.cpu(), Ys3.cpu(), Ys4.cpu()))

        # ------------------------------
        # GLOBAL UMN
        # ------------------------------
        print(f"\n[GLOBAL] Training one UMN with all clients' pairs (n_clients={n_clients})...")
        umn_global = train_umn_global_from_pairs(
            per_client_pairs=pairs,
            use_ppm=variant.use_ppm,
            umn_epochs=args.umn_epochs,
            umn_lr=args.umn_lr,
            umn_patience=args.umn_patience,
            umn_batch=args.umn_batch,
            seed=args.seed,
            device=args.device,
            run_dir=run_dir,
        )

        seg_stats_global = eval_segmentation_per_client_global_umn(
            umn_global=umn_global,
            encT_list=encT_list,
            dec_list=dec_list,
            test_loader=test_loader,
            device=args.device,
            use_skips=variant.use_skips,
            fg_classes=fg_classes,
        )

        dice_mean_g, dice_std_g = mean_std(seg_stats_global, "dice")
        iou_mean_g,  iou_std_g  = mean_std(seg_stats_global, "iou")

        summary_rows.append({
            "variant": variant.name,
            "num_clients": n_clients,
            "strategy": "global",
            "recon_psnr_mean": recon_psnr_mean,
            "recon_psnr_std": recon_psnr_std,
            "recon_ssim_mean": recon_ssim_mean,
            "recon_ssim_std": recon_ssim_std,
            "dice_mean": dice_mean_g,
            "dice_std": dice_std_g,
            "iou_mean": iou_mean_g,
            "iou_std": iou_std_g,
            "seed": args.seed,
        })

        detailed_runs.append({
            "variant": variant.name,
            "num_clients": n_clients,
            "strategy": "global",
            "reconstruction_per_client": recon_stats,
            "segmentation_per_client": seg_stats_global,
        })

        per_csv = results_dir / "per_setting" / f"nc{n_clients}_{variant.name}_global_perclient.csv"
        with open(per_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["client", "recon_psnr", "recon_ssim", "dice", "iou"])
            w.writeheader()
            for r, s in zip(recon_stats, seg_stats_global):
                w.writerow({
                    "client": r["client"],
                    "recon_psnr": r["psnr"],
                    "recon_ssim": r["ssim"],
                    "dice": s["dice"],
                    "iou": s["iou"],
                })

        # ------------------------------
        # LOCAL UMN
        # ------------------------------
        print(f"\n[LOCAL] Training one UMN per client (n_clients={n_clients})...")
        umn_list = train_umn_local_from_pairs(
            per_client_pairs=pairs,
            use_ppm=variant.use_ppm,
            umn_epochs=args.umn_epochs,
            umn_lr=args.umn_lr,
            umn_patience=args.umn_patience,
            umn_batch=args.umn_batch,
            seed=args.seed,
            device=args.device,
            run_dir=run_dir,
        )

        seg_stats_local = eval_segmentation_per_client_local_umn(
            umn_list=umn_list,
            encT_list=encT_list,
            dec_list=dec_list,
            test_loader=test_loader,
            device=args.device,
            use_skips=variant.use_skips,
            fg_classes=fg_classes,
        )

        dice_mean_l, dice_std_l = mean_std(seg_stats_local, "dice")
        iou_mean_l,  iou_std_l  = mean_std(seg_stats_local, "iou")

        summary_rows.append({
            "variant": variant.name,
            "num_clients": n_clients,
            "strategy": "local",
            "recon_psnr_mean": recon_psnr_mean,
            "recon_psnr_std": recon_psnr_std,
            "recon_ssim_mean": recon_ssim_mean,
            "recon_ssim_std": recon_ssim_std,
            "dice_mean": dice_mean_l,
            "dice_std": dice_std_l,
            "iou_mean": iou_mean_l,
            "iou_std": iou_std_l,
            "seed": args.seed,
        })

        detailed_runs.append({
            "variant": variant.name,
            "num_clients": n_clients,
            "strategy": "local",
            "reconstruction_per_client": recon_stats,
            "segmentation_per_client": seg_stats_local,
        })

        per_csv2 = results_dir / "per_setting" / f"nc{n_clients}_{variant.name}_local_perclient.csv"
        with open(per_csv2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["client", "recon_psnr", "recon_ssim", "dice", "iou"])
            w.writeheader()
            for r, s in zip(recon_stats, seg_stats_local):
                w.writerow({
                    "client": r["client"],
                    "recon_psnr": r["psnr"],
                    "recon_ssim": r["ssim"],
                    "dice": s["dice"],
                    "iou": s["iou"],
                })

    # write summary
    summary_csv = results_dir / "summary_clientsweep.csv"
    with open(summary_csv, "w", newline="") as f:
        fieldnames = [
            "variant", "num_clients", "strategy",
            "recon_psnr_mean", "recon_psnr_std",
            "recon_ssim_mean", "recon_ssim_std",
            "dice_mean", "dice_std",
            "iou_mean", "iou_std",
            "seed",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    # write detailed json
    detailed_json = results_dir / "detailed_runs.json"
    with open(detailed_json, "w") as f:
        json.dump(detailed_runs, f, indent=2)

    print("\nDONE.")
    print("Summary:", str(summary_csv))
    print("Detailed:", str(detailed_json))
    if _SSIM is None:
        print("NOTE: SSIM is NaN because skimage is not installed. To enable SSIM:")
        print("  pip install scikit-image")


if __name__ == "__main__":
    main()
