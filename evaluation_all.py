#!/usr/bin/env python3
# ============================================================
# FILE: federated_comprehensive_evaluation_stage2_binary.py
#
# Federated Stage-2 evaluator for BINARY datasets (fumpe | heart_mri | nerve)
# Includes:
#   - Region metrics: Dice, IoU, Sensitivity, Specificity
#   - Surface metrics (optional): HD (max), HD95, ASD   (mm if spacing exists, else pixels)
#   - Profile: params, FLOPs (optional thop), latency p50/p90/p99
#   - Saves random qualitative samples (binary)
#
# NOTE ON SURFACE METRICS:
#   - If dataset provides spacing in batch dict:
#         ("spacing_r","spacing_c") or ("spacing",) we use mm
#   - Otherwise we default spacing=(1.0,1.0) => pixel units
#
# Requires Stage-1 checkpoints in --ckpt-dir:
#   umn.pt, clientK_img_ae.pt, clientK_mask_ae.pt
# ============================================================

import os
import csv
import json
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# surface metrics deps
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

# headless-safe image saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.autoencoder import AutoEncoder
from models.umn import UMN
from data import DatasetArgs, build_dataset


# ----------------------------
# Seeding
# ----------------------------
def set_seed(seed: int):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Region metrics (binary foreground=1)
# ----------------------------
def _safe_div(num: float, den: float, eps: float = 1e-8) -> float:
    return float(num / (den + eps))


@torch.no_grad()
def binary_confusion(pred_fg: torch.Tensor, gt_fg: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    pred_fg, gt_fg: bool tensors (H,W)
    """
    tp = torch.logical_and(pred_fg, gt_fg).sum().item()
    fp = torch.logical_and(pred_fg, ~gt_fg).sum().item()
    fn = torch.logical_and(~pred_fg, gt_fg).sum().item()
    tn = torch.logical_and(~pred_fg, ~gt_fg).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def metrics_from_conf(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    dice = _safe_div(2 * tp, (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    iou  = _safe_div(tp, (tp + fp + fn))         if (tp + fp + fn) > 0 else 1.0
    sens = _safe_div(tp, (tp + fn))              if (tp + fn) > 0 else 1.0
    spec = _safe_div(tn, (tn + fp))              if (tn + fp) > 0 else 1.0
    return {"dice": dice, "iou": iou, "sens": sens, "spec": spec}


# ----------------------------
# Surface metrics (HD / HD95 / ASD) in mm or pixels
# ----------------------------
def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    er = binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    return mask & (~er)


def _directed_surface_dists(A: np.ndarray, B: np.ndarray, spacing_rc: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute directed boundary distances A->B and B->A using spacing.
    """
    if not A.any() and not B.any():
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    if not A.any() or not B.any():
        return np.array([math.inf], dtype=np.float64), np.array([math.inf], dtype=np.float64)

    Ab, Bb = _boundary(A), _boundary(B)
    ya, xa = np.nonzero(Ab)
    yb, xb = np.nonzero(Bb)

    ptsA = np.column_stack([ya * spacing_rc[0], xa * spacing_rc[1]]).astype(np.float64)
    ptsB = np.column_stack([yb * spacing_rc[0], xb * spacing_rc[1]]).astype(np.float64)

    dAB, _ = cKDTree(ptsB).query(ptsA, k=1)
    dBA, _ = cKDTree(ptsA).query(ptsB, k=1)
    return dAB.astype(np.float64), dBA.astype(np.float64)


def surface_metrics(gt_fg: np.ndarray, pr_fg: np.ndarray, spacing_rc: Tuple[float, float]) -> Dict[str, float]:
    """
    Returns HD(max), HD95, ASD
    """
    dAB, dBA = _directed_surface_dists(gt_fg, pr_fg, spacing_rc)

    if np.isinf(dAB).all() and np.isinf(dBA).all():
        return {"hd": float("inf"), "hd95": float("inf"), "asd": float("inf")}

    both = np.concatenate([dAB, dBA], axis=0)
    hd = float(np.max(both))
    hd95 = float(np.percentile(both, 95))
    asd = float(np.mean(both))
    return {"hd": hd, "hd95": hd95, "asd": asd}


# ----------------------------
# Qualitative saves (binary)
# ----------------------------
def save_prediction_triplet_binary(pred_dir: Path, case: str, image: torch.Tensor, gt: np.ndarray, pred: np.ndarray):
    pred_dir.mkdir(parents=True, exist_ok=True)

    img = image.detach().cpu().numpy()  # (C,H,W)
    img = img[0] if img.shape[0] == 1 else img.mean(axis=0)
    img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)

    gt_u8 = (gt.astype(np.uint8) * 255)
    pr_u8 = (pred.astype(np.uint8) * 255)

    plt.imsave(pred_dir / f"{case}_img.png", img_u8, cmap="gray")
    plt.imsave(pred_dir / f"{case}_gt.png", gt_u8, cmap="gray")
    plt.imsave(pred_dir / f"{case}_pred.png", pr_u8, cmap="gray")


# ----------------------------
# Profiling: params / flops / latency
# ----------------------------
def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def estimate_flops_thop(model: torch.nn.Module, inp: torch.Tensor) -> Optional[Dict[str, float]]:
    try:
        from thop import profile
    except Exception:
        return None
    model.eval()
    with torch.no_grad():
        macs, _params = profile(model, inputs=(inp,), verbose=False)
    macs = float(macs)
    return {"macs": macs, "flops_approx": 2.0 * macs}


def measure_latency_ms(model: torch.nn.Module, inp: torch.Tensor, device: str, amp: bool,
                       iters: int = 100, warmup: int = 20) -> Dict[str, float]:
    model.eval()
    inp = inp.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                _ = model(inp)
        if "cuda" in device and torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                _ = model(inp)
            if "cuda" in device and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    return {
        "lat_mean_ms": float(arr.mean()),
        "lat_p50_ms": float(np.percentile(arr, 50)),
        "lat_p90_ms": float(np.percentile(arr, 90)),
        "lat_p99_ms": float(np.percentile(arr, 99)),
        "lat_iters": int(iters),
        "lat_warmup": int(warmup),
    }


# ----------------------------
# Logging
# ----------------------------
def setup_logger(outdir: Path) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("federated_eval_stage2_binary_surface")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh = logging.FileHandler(outdir / "eval.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ----------------------------
# Args
# ----------------------------
def get_args():
    p = argparse.ArgumentParser("Federated Evaluation (Stage-2) - Binary datasets + Surface metrics")

    p.add_argument("--dataset", type=str, required=True, help="fumpe | heart_mri | nerve")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)

    p.add_argument("--src-client", type=int, default=1)
    p.add_argument("--tgt-client", type=int, default=1)

    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--save-csv", type=str, default="eval_metrics.csv")

    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--resize", type=int, default=1)
    p.add_argument("--in-ch", type=int, default=1)
    p.add_argument("--n-classes", type=int, default=2)

    # dataset options
    p.add_argument("--drop-empty-masks", action="store_true")
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--fumpe-split-strategy", type=str, default="patient")
    p.add_argument("--nerve-pad", type=int, default=14)

    # runtime
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # must match training
    p.add_argument("--no-skips", action="store_true")
    p.add_argument("--no-klt", action="store_true")
    p.add_argument("--no-ppm", action="store_true")

    # surface metrics
    p.add_argument("--surface-metrics", type=int, default=1,
                   help="1=compute HD/HD95/ASD, 0=skip")
    p.add_argument("--surface-unit", type=str, default="auto",
                   choices=["auto", "mm", "pixel"],
                   help="auto: use dataset spacing if available else pixels")

    # qualitative samples
    p.add_argument("--save-samples", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=5)

    # latency
    p.add_argument("--lat-iters", type=int, default=100)
    p.add_argument("--lat-warmup", type=int, default=20)

    return p.parse_args()


# ----------------------------
# Load checkpoints
# ----------------------------
def load_stage2_components(
    ckpt_dir: Path,
    src_client: int,
    tgt_client: int,
    in_ch: int,
    n_classes: int,
    use_skips: bool,
    use_klt: bool,
    use_ppm: bool,
    seed_base: int,
    device: str,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    ckpt_dir = Path(ckpt_dir)

    umn = UMN(use_ppm=use_ppm)
    umn.load_state_dict(torch.load(str(ckpt_dir / "umn.pt"), map_location="cpu"))
    umn = umn.to(device).eval()

    img_ae = AutoEncoder(in_ch=in_ch, use_skips=use_skips, use_klt=use_klt, seed=seed_base + int(src_client))
    img_ae.load_state_dict(torch.load(str(ckpt_dir / f"client{int(src_client)}_img_ae.pt"), map_location="cpu"))
    img_encT_src = img_ae.encoderT.to(device).eval()

    mask_ae = AutoEncoder(in_ch=n_classes, use_skips=use_skips, use_klt=use_klt, seed=100 + seed_base + int(tgt_client))
    mask_ae.load_state_dict(torch.load(str(ckpt_dir / f"client{int(tgt_client)}_mask_ae.pt"), map_location="cpu"))
    mask_encT_tgt = mask_ae.encoderT.to(device).eval()
    mask_dec_tgt = mask_ae.decoder.to(device).eval()

    return img_encT_src, umn, mask_encT_tgt, mask_dec_tgt


# ----------------------------
# Pipeline wrapper
# ----------------------------
class FederatedStage2Pipeline(torch.nn.Module):
    def __init__(self, img_encT_src, umn, mask_encT_tgt, mask_dec_tgt, use_skips: bool):
        super().__init__()
        self.img_encT_src = img_encT_src
        self.umn = umn
        self.mask_encT_tgt = mask_encT_tgt
        self.mask_dec_tgt = mask_dec_tgt
        self.use_skips = bool(use_skips)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # client(src) -> protected latents
        bT, s3T, s4T = self.img_encT_src(x)

        # server inverse (src)
        b, s3, s4 = self.img_encT_src.inverse_triplet(bT, s3T, s4T)

        # server UMN
        b_hat, s3_hat, s4_hat = self.umn(b, s3, s4)

        # server apply tgt mask transform
        if getattr(self.mask_encT_tgt, "use_klt", False):
            pb = self.mask_encT_tgt.T_b(b_hat)
            ps3 = self.mask_encT_tgt.T_s3(s3_hat)
            ps4 = self.mask_encT_tgt.T_s4(s4_hat)
        else:
            pb, ps3, ps4 = b_hat, s3_hat, s4_hat

        # client(tgt) inverse + decode
        b2, s32, s42 = self.mask_encT_tgt.inverse_triplet(pb, ps3, ps4)
        if self.use_skips:
            out = self.mask_dec_tgt(b2, s3=s32, s4=s42)
        else:
            out = self.mask_dec_tgt(b2)
        return out


# ----------------------------
# Output -> binary pred helper
# ----------------------------
@torch.no_grad()
def logits_to_binary_pred(out: torch.Tensor) -> torch.Tensor:
    """
    out: (B,C,H,W) or (B,1,H,W)
    returns: pred_fg_bool (B,H,W) bool
    """
    if out.dim() != 4:
        raise ValueError(f"Expected output (B,C,H,W), got {tuple(out.shape)}")

    B, C, H, W = out.shape
    if C == 1:
        p = torch.sigmoid(out[:, 0])
        return (p >= 0.5)
    else:
        pred = torch.argmax(out, dim=1)
        return (pred == 1)


# ----------------------------
# Spacing extraction (mm vs pixel)
# ----------------------------
def get_spacing_from_batch(batch: Dict, i: int, surface_unit: str) -> Tuple[float, float]:
    """
    Returns spacing_rc (row, col).
    - If surface_unit == "pixel": always (1,1)
    - If surface_unit == "mm": use spacing if present else (1,1) with a warning handled in log upstream
    - If "auto": use spacing if present else (1,1)
    """
    if surface_unit == "pixel":
        return (1.0, 1.0)

    # Try common keys returned by datasets
    if "spacing_r" in batch and "spacing_c" in batch:
        sr = float(batch["spacing_r"][i].item() if torch.is_tensor(batch["spacing_r"]) else batch["spacing_r"][i])
        sc = float(batch["spacing_c"][i].item() if torch.is_tensor(batch["spacing_c"]) else batch["spacing_c"][i])
        return (sr, sc)

    if "spacing" in batch:
        sp = batch["spacing"][i]
        if torch.is_tensor(sp):
            sp = sp.detach().cpu().numpy().tolist()
        # accept [row,col] or [x,y]
        if len(sp) >= 2:
            return (float(sp[0]), float(sp[1]))

    # fallback: pixels
    return (1.0, 1.0)


# ----------------------------
# Main
# ----------------------------
def main():
    args = get_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(outdir)

    if int(args.in_ch) != 1:
        raise ValueError("These datasets are 1-channel images. Use --in-ch 1")
    if int(args.n_classes) != 2:
        raise ValueError("Binary masks. Use --n-classes 2")

    set_seed(args.seed)

    use_amp = bool(args.amp) and ("cuda" in args.device) and torch.cuda.is_available()
    use_skips = not args.no_skips
    use_klt = not args.no_klt
    use_ppm = not args.no_ppm
    use_surface = bool(int(args.surface_metrics))

    with open(outdir / "eval_config.json", "w") as f:
        cfg = vars(args).copy()
        cfg["use_skips"] = use_skips
        cfg["use_klt"] = use_klt
        cfg["use_ppm"] = use_ppm
        json.dump(cfg, f, indent=2)

    logger.info("==== FEDERATED EVALUATION START (STAGE-2, BINARY) ====")
    logger.info(f"dataset={args.dataset} data_root={args.data_root}")
    logger.info(f"ckpt_dir={args.ckpt_dir}")
    logger.info(f"src_client={args.src_client} tgt_client={args.tgt_client}")
    logger.info(f"img_size={args.img_size} resize={bool(args.resize)} in_ch={args.in_ch} n_classes={args.n_classes}")
    logger.info(f"device={args.device} amp={use_amp}")
    logger.info(f"use_skips={use_skips} use_klt={use_klt} use_ppm={use_ppm}")
    logger.info(f"drop_empty_masks={bool(args.drop_empty_masks)}")
    logger.info(f"surface_metrics={use_surface} surface_unit={args.surface_unit}")
    logger.info(f"save_samples={bool(args.save_samples)} num_samples={args.num_samples} seed={args.seed}")

    # build TEST dataset via factory
    ds = build_dataset(DatasetArgs(
        dataset=args.dataset,
        data_root=args.data_root,
        split="test",
        img_size=args.img_size,
        seed=args.seed,
        test_ratio=args.test_ratio,
        drop_empty_masks=bool(args.drop_empty_masks),
        fumpe_split_strategy=args.fumpe_split_strategy,
        nerve_pad=args.nerve_pad,
    ))

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=("cuda" in args.device),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    # choose cases to save
    all_cases = [ds[i]["case"] for i in range(len(ds))]
    save_cases = set()
    if bool(args.save_samples) and len(all_cases) > 0 and int(args.num_samples) > 0:
        rng = np.random.RandomState(int(args.seed))
        k = min(int(args.num_samples), len(all_cases))
        chosen = rng.choice(len(all_cases), size=k, replace=False).tolist()
        save_cases = {all_cases[i] for i in chosen}
        logger.info(f"Will save {len(save_cases)} sample predictions into: {outdir / 'predictions'}")
    pred_dir = outdir / "predictions"

    # load components + pipeline
    img_encT_src, umn, mask_encT_tgt, mask_dec_tgt = load_stage2_components(
        ckpt_dir=Path(args.ckpt_dir),
        src_client=int(args.src_client),
        tgt_client=int(args.tgt_client),
        in_ch=int(args.in_ch),
        n_classes=int(args.n_classes),
        use_skips=use_skips,
        use_klt=use_klt,
        use_ppm=use_ppm,
        seed_base=int(args.seed),
        device=args.device,
    )

    pipeline = FederatedStage2Pipeline(
        img_encT_src, umn, mask_encT_tgt, mask_dec_tgt, use_skips=use_skips
    ).to(args.device).eval()

    # profile
    profile: Dict[str, float] = {}
    profile["n_params"] = count_params(pipeline)
    dummy = torch.randn(1, args.in_ch, args.img_size, args.img_size, device=args.device)

    flops_info = estimate_flops_thop(pipeline, dummy)
    if flops_info is None:
        profile["flops_note"] = "Install thop to enable FLOPs/MACs: pip install thop"
    else:
        profile.update(flops_info)

    profile.update(measure_latency_ms(
        pipeline, dummy, device=args.device, amp=use_amp,
        iters=int(args.lat_iters), warmup=int(args.lat_warmup)
    ))

    with open(outdir / "model_profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    logger.info(f"Pipeline params: {int(profile['n_params']):,}")
    if "macs" in profile:
        logger.info(f"MACs: {profile['macs']:.3e} | FLOPs~: {profile['flops_approx']:.3e}")
    logger.info(
        f"Latency(ms): mean={profile['lat_mean_ms']:.3f} p50={profile['lat_p50_ms']:.3f} "
        f"p90={profile['lat_p90_ms']:.3f} p99={profile['lat_p99_ms']:.3f}"
    )

    # eval loop
    results: List[Dict[str, float]] = []
    sums: Dict[str, List[float]] = {}

    def _push(key: str, val: float):
        sums.setdefault(key, []).append(float(val))

    spacing_seen = False

    t0 = time.time()
    saved_count = 0

    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(args.device, non_blocking=True)  # (B,1,H,W)
            y = batch["mask"].to(args.device, non_blocking=True)   # (B,H,W) long 0/1
            cases = batch["case"]

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                out = pipeline(x)

            if out.shape[-2:] != x.shape[-2:]:
                out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)

            pred_fg = logits_to_binary_pred(out)  # (B,H,W) bool
            gt_fg = (y == 1)                      # (B,H,W) bool

            for i in range(pred_fg.shape[0]):
                case = cases[i]

                tp, fp, fn, tn = binary_confusion(pred_fg[i], gt_fg[i])
                m = metrics_from_conf(tp, fp, fn, tn)

                row = {
                    "case_name": case,
                    "dice": m["dice"],
                    "iou": m["iou"],
                    "sens": m["sens"],
                    "spec": m["spec"],
                }

                _push("dice", m["dice"])
                _push("iou", m["iou"])
                _push("sens", m["sens"])
                _push("spec", m["spec"])

                # surface metrics
                if use_surface:
                    spacing_rc = get_spacing_from_batch(batch, i, args.surface_unit)
                    if spacing_rc != (1.0, 1.0):
                        spacing_seen = True

                    gt_np = gt_fg[i].detach().cpu().numpy().astype(bool)
                    pr_np = pred_fg[i].detach().cpu().numpy().astype(bool)

                    sm = surface_metrics(gt_np, pr_np, spacing_rc)
                    row["hd"] = sm["hd"]
                    row["hd95"] = sm["hd95"]
                    row["asd"] = sm["asd"]

                    _push("hd", sm["hd"])
                    _push("hd95", sm["hd95"])
                    _push("asd", sm["asd"])

                results.append(row)

                # save qualitative
                if case in save_cases:
                    img_vis = x[i]
                    gt_vis = gt_fg[i].detach().cpu().numpy().astype(np.uint8)
                    pr_vis = pred_fg[i].detach().cpu().numpy().astype(np.uint8)
                    save_prediction_triplet_binary(pred_dir, case, img_vis, gt_vis, pr_vis)
                    saved_count += 1
                    save_cases.remove(case)

    elapsed = time.time() - t0

    if use_surface and args.surface_unit in ("auto", "mm") and not spacing_seen and args.surface_unit != "pixel":
        logger.info("NOTE: No spacing found in dataset batches; surface metrics are in PIXELS (spacing=(1,1)).")

    # write per-case csv
    csv_path = outdir / args.save_csv
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
    else:
        with open(csv_path, "w", newline="") as f:
            f.write("case_name\n")

    # summary
    def _mean_std(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "std": float("nan")}
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    summary = {}
    base_keys = ["dice", "iou", "sens", "spec"]
    if use_surface:
        base_keys += ["hd", "hd95", "asd"]

    for k in base_keys:
        ms = _mean_std(sums.get(k, []))
        summary[f"{k}_mean"] = ms["mean"]
        summary[f"{k}_std"] = ms["std"]

    with open(outdir / "eval_summary_means.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(outdir / "eval_summary_means.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    logger.info("==== SUMMARY MEANS (BINARY FG=1) ====")
    for k, v in summary.items():
        if np.isfinite(v):
            logger.info(f"{k}: {v:.6f}")
        else:
            logger.info(f"{k}: nan")

    logger.info(f"Saved per-case CSV -> {csv_path}")
    logger.info(f"Saved summary -> {outdir / 'eval_summary_means.csv'} and eval_summary_means.json")
    logger.info(f"Saved model profile -> {outdir / 'model_profile.json'}")
    logger.info(f"Saved samples -> {pred_dir} (count={saved_count})")
    logger.info(f"Done in {elapsed:.1f}s, cases={len(ds)}")
    logger.info("==== FEDERATED EVALUATION END ====")


if __name__ == "__main__":
    main()
