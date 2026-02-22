import time
import torch
import torch.nn as nn

from utils.early_stop import EarlyStopper
from utils.logger import CSVLogger


class UMNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_triplet, target_triplet):
        pb, ps3, ps4 = pred_triplet
        tb, ts3, ts4 = target_triplet
        return self.mse(pb, tb) + self.mse(ps3, ts3) + self.mse(ps4, ts4)


@torch.no_grad()
def eval_umn(model: nn.Module, loader, device: str) -> float:
    model.eval().to(device)
    loss_fn = UMNLoss()
    total = 0.0
    n = 0
    for xb, xs3, xs4, yb, ys3, ys4 in loader:
        xb, xs3, xs4 = xb.to(device), xs3.to(device), xs4.to(device)
        yb, ys3, ys4 = yb.to(device), ys3.to(device), ys4.to(device)
        pred = model(xb, xs3, xs4)
        loss = loss_fn(pred, (yb, ys3, ys4))
        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(1, n)


def train_umn(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    log_csv_path: str,
) -> nn.Module:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = UMNLoss()
    stopper = EarlyStopper(patience=patience, mode="min")

    logger = CSVLogger(
        log_csv_path,
        fieldnames=["epoch", "train_loss", "val_loss", "lr", "epoch_seconds", "elapsed_seconds"],
    )

    try:
        for ep in range(1, int(epochs) + 1):
            model.train()
            t0 = time.time()
            tr_loss = 0.0
            n = 0

            for xb, xs3, xs4, yb, ys3, ys4 in train_loader:
                xb, xs3, xs4 = xb.to(device), xs3.to(device), xs4.to(device)
                yb, ys3, ys4 = yb.to(device), ys3.to(device), ys4.to(device)

                opt.zero_grad(set_to_none=True)
                pred = model(xb, xs3, xs4)
                loss = loss_fn(pred, (yb, ys3, ys4))
                loss.backward()
                opt.step()

                tr_loss += float(loss.item()) * xb.size(0)
                n += xb.size(0)

            tr_loss /= max(1, n)
            val_loss = eval_umn(model, val_loader, device=device)
            dt = time.time() - t0
            lr_now = opt.param_groups[0]["lr"]

            print(f"[UMN] epoch {ep:03d}/{epochs} | train={tr_loss:.4f} | val={val_loss:.4f} | {dt:.1f}s")

            logger.log({
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "lr": lr_now,
                "epoch_seconds": dt,
                "elapsed_seconds": logger.elapsed(),
            })

            if stopper.step(val_loss, model):
                print("[UMN] Early stopping.")
                break

        stopper.restore_best(model)
        return model
    finally:
        logger.close()
