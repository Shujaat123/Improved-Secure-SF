import os
import argparse
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
from train.checks import (
    check_transform_inverse_identity,
    extract_latents_server_inverse,
)


# ----------------------------
# dataset wrappers
# ----------------------------
class ImgOnly(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        d = self.base[i]
        return d["image"], d["mask"], d["case"]


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
        return y_oh, d["image"], d["case"]


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
def partition_clients_indices(n: int, num_clients: int, seed: int):
    idx = np.arange(n)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(idx)
    splits = np.array_split(idx, int(num_clients))
    return [s.tolist() for s in splits]


def split_train_val_indices(indices, val_frac: float, seed: int):
    rng = np.random.default_rng(int(seed))
    idx = np.array(indices)
    rng.shuffle(idx)
    n_val = int(round(len(idx) * float(val_frac)))
    val_idx = idx[:n_val].tolist()
    tr_idx = idx[n_val:].tolist()
    return tr_idx, val_idx


# ----------------------------
# argparse
# ----------------------------
def build_parser():
    p = argparse.ArgumentParser("PSFH Universal Training (Stage1: AEs + UMN)")

    # paths
    p.add_argument("--data-train", type=str, required=True,
                   help="Training dataset root containing image_mha/label_mha")
    p.add_argument("--results-dir", type=str, default="./results/ppms_torch_stage1")
    p.add_argument("--log-dirname", type=str, default="logs",
                   help="Subfolder under results-dir to store CSV logs.")

    # general
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=2)

    # data
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--no-resize", action="store_true")
    p.add_argument("--num-classes", type=int, default=3)

    # clients
    p.add_argument("--num-clients", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.2)

    # AE
    p.add_argument("--ae-epochs", type=int, default=100)
    p.add_argument("--ae-batch", type=int, default=16)
    p.add_argument("--ae-lr", type=float, default=1e-3)
    p.add_argument("--ae-patience", type=int, default=15)
    p.add_argument("--use-skips", type=int, default=1, choices=[0, 1],
               help="1=enable skip connections, 0=disable")

    p.add_argument("--use-klt", type=int, default=1, choices=[0, 1],
                help="1=enable KLT, 0=disable")

    p.add_argument("--use-ppm", type=int, default=1, choices=[0, 1],
               help="1=enable PPM, 0=disable")


    # inverse-check
    p.add_argument("--skip-inverse-check", action="store_true")
    p.add_argument("--inverse-check-batches", type=int, default=3)
    p.add_argument("--inverse-atol", type=float, default=1e-5)
    p.add_argument("--inverse-rtol", type=float, default=1e-5)

    # UMN
    p.add_argument("--skip-umn", action="store_true")
    p.add_argument("--umn-epochs", type=int, default=150)
    p.add_argument("--umn-batch", type=int, default=16)
    p.add_argument("--umn-lr", type=float, default=1e-3)
    p.add_argument("--umn-patience", type=int, default=20)
    p.add_argument("--umn-val-frac", type=float, default=0.2)
    # p.add_argument("--no-ppm", action="store_true")

    return p


def main(args):
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    log_dir = os.path.join(args.results_dir, args.log_dirname)
    os.makedirs(log_dir, exist_ok=True)

    device = args.device
    print("Device:", device)

    # You requested image channels = 3:
    base_ds = PSFHMhaDataset(
        root=args.data_train,
        img_size=args.img_size,
        in_ch=3,
        resize=not args.no_resize
    )

    n = len(base_ds)
    print("Train dataset size:", n)

    ds_img = ImgOnly(base_ds)
    ds_msk = MaskAsX(base_ds, num_classes=args.num_classes)

    client_splits = partition_clients_indices(n, args.num_clients, args.seed)
    print("Client split sizes:", [len(s) for s in client_splits])

    encT_list = []  # (img_encT, mask_encT) CPU
    dec_list  = []  # (img_dec,  mask_dec) CPU

    # ----------------------------
    # Train per-client AEs
    # ----------------------------
    for ci, indices in enumerate(client_splits, start=1):
        tr_idx, va_idx = split_train_val_indices(indices, args.val_frac, args.seed + ci)

        img_tr = DataLoader(Subset(ds_img, tr_idx), batch_size=args.ae_batch, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        img_va = DataLoader(Subset(ds_img, va_idx), batch_size=args.ae_batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

        msk_tr = DataLoader(Subset(ds_msk, tr_idx), batch_size=args.ae_batch, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
        msk_va = DataLoader(Subset(ds_msk, va_idx), batch_size=args.ae_batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

        print(f"\n[Client {ci}] Train Image AE (in_ch=3)")
        img_ae = AutoEncoder(in_ch=3, use_skips= bool(args.use_skips), use_klt= bool(args.use_klt), seed=args.seed + ci)
        img_log = os.path.join(log_dir, f"client{ci}_img_ae.csv")
        img_ae = train_ae(img_ae, img_tr, img_va, device=device,
                          epochs=args.ae_epochs, lr=args.ae_lr, patience=args.ae_patience,
                          tag=f"C{ci}-IMG", log_csv_path=img_log)

        print(f"\n[Client {ci}] Train Mask AE (in_ch=num_classes={args.num_classes})")
        mask_ae = AutoEncoder(in_ch=args.num_classes, use_skips= args.use_skips, use_klt=args.use_klt,
                              seed=100 + args.seed + ci)
        msk_log = os.path.join(log_dir, f"client{ci}_mask_ae.csv")
        mask_ae = train_ae(mask_ae, msk_tr, msk_va, device=device,
                           epochs=args.ae_epochs, lr=args.ae_lr, patience=args.ae_patience,
                           tag=f"C{ci}-MSK", log_csv_path=msk_log)

        # ---- inverse-check per client (both checks: round-trip + federated-path)
        if not args.skip_inverse_check:
            print(f"\n[Client {ci}] Inverse-check (Image encoder)")
            chk_loader = DataLoader(Subset(ds_img, va_idx), batch_size=min(args.ae_batch, 8), shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=False)
            img_chk_csv = os.path.join(log_dir, f"client{ci}_img_inverse_check.csv")
            _ = check_transform_inverse_identity(
                encoderT=img_ae.encoderT,
                loader=chk_loader,
                device=device,
                max_batches=args.inverse_check_batches,
                atol=args.inverse_atol,
                rtol=args.inverse_rtol,
                log_csv_path=img_chk_csv
            )

            print(f"\n[Client {ci}] Inverse-check (Mask encoder)")
            chk_loader_m = DataLoader(Subset(ds_msk, va_idx), batch_size=min(args.ae_batch, 8), shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True, drop_last=False)
            msk_chk_csv = os.path.join(log_dir, f"client{ci}_mask_inverse_check.csv")
            _ = check_transform_inverse_identity(
                encoderT=mask_ae.encoderT,
                loader=chk_loader_m,
                device=device,
                max_batches=args.inverse_check_batches,
                atol=args.inverse_atol,
                rtol=args.inverse_rtol,
                log_csv_path=msk_chk_csv
            )

        # save models
        torch.save(img_ae.state_dict(),  os.path.join(args.results_dir, f"client{ci}_img_ae.pt"))
        torch.save(mask_ae.state_dict(), os.path.join(args.results_dir, f"client{ci}_mask_ae.pt"))

        # keep encoder/decoder handles (CPU)
        encT_list.append((img_ae.encoderT.cpu(), mask_ae.encoderT.cpu()))
        dec_list.append((img_ae.decoder.cpu(),  mask_ae.decoder.cpu()))

    if args.skip_umn:
        print("\nSkipping UMN training. Done.")
        return

    # ----------------------------
    # Federated-consistent UMN data build:
    # client sends PROTECTED latents; server inverses with that client's params; then concatenate.
    # ----------------------------
    all_Xb, all_Xs3, all_Xs4 = [], [], []
    all_Yb, all_Ys3, all_Ys4 = [], [], []

    for ci, indices in enumerate(client_splits, start=1):
        img_encT, mask_encT = encT_list[ci - 1]

        loader_img = DataLoader(Subset(ds_img, indices), batch_size=args.umn_batch, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
        loader_msk = DataLoader(Subset(ds_msk, indices), batch_size=args.umn_batch, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)

        # IMPORTANT: federated-consistent extraction
        Xb, Xs3, Xs4 = extract_latents_server_inverse(img_encT, loader_img, device=device)
        Yb, Ys3, Ys4 = extract_latents_server_inverse(mask_encT, loader_msk, device=device)

        all_Xb.append(Xb); all_Xs3.append(Xs3); all_Xs4.append(Xs4)
        all_Yb.append(Yb); all_Ys3.append(Ys3); all_Ys4.append(Ys4)

    Xb  = torch.cat(all_Xb,  0)
    Xs3 = torch.cat(all_Xs3, 0)
    Xs4 = torch.cat(all_Xs4, 0)
    Yb  = torch.cat(all_Yb,  0)
    Ys3 = torch.cat(all_Ys3, 0)
    Ys4 = torch.cat(all_Ys4, 0)

    print("\nUMN tensors (server-recovered original latents):")
    print("  Xb :", tuple(Xb.shape), "Xs3:", tuple(Xs3.shape), "Xs4:", tuple(Xs4.shape))
    print("  Yb :", tuple(Yb.shape), "Ys3:", tuple(Ys3.shape), "Ys4:", tuple(Ys4.shape))

    # UMN train/val split
    N = Xb.size(0)
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = int(round(N * args.umn_val_frac))
    val_idx = idx[:n_val].tolist()
    tr_idx  = idx[n_val:].tolist()

    tr_loader = DataLoader(LatentTripletDS(Xb[tr_idx], Xs3[tr_idx], Xs4[tr_idx],
                                          Yb[tr_idx], Ys3[tr_idx], Ys4[tr_idx]),
                           batch_size=args.umn_batch, shuffle=True, num_workers=0, drop_last=False)

    va_loader = DataLoader(LatentTripletDS(Xb[val_idx], Xs3[val_idx], Xs4[val_idx],
                                          Yb[val_idx], Ys3[val_idx], Ys4[val_idx]),
                           batch_size=args.umn_batch, shuffle=False, num_workers=0, drop_last=False)

    print("\nTraining UMN (original z recovered at server)...")
    umn = UMN(use_ppm=bool( args.use_ppm))
    umn_log = os.path.join(log_dir, "umn.csv")
    umn = train_umn(umn, tr_loader, va_loader, device=device,
                    epochs=args.umn_epochs, lr=args.umn_lr, patience=args.umn_patience,
                    log_csv_path=umn_log)

    torch.save(umn.state_dict(), os.path.join(args.results_dir, "umn.pt"))
    print("\nSaved to:", args.results_dir)
    print("Logs in:", log_dir)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
