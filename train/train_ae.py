import time
import torch
import torch.nn as nn

from losses.losses import CombinedBCEDice
from utils.early_stop import EarlyStopper
from utils.logger import CSVLogger


@torch.no_grad()
def eval_ae(model: nn.Module, loader, device: str, loss_fn: nn.Module) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, _, _ in loader:
        x = x.to(device)
        pred = model(x)
        loss = loss_fn(x, pred)
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def train_ae(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    tag: str,
    log_csv_path: str,
) -> nn.Module:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = CombinedBCEDice(0.5, 0.5)
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

            for x, _, _ in train_loader:
                x = x.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(x)
                loss = loss_fn(x, pred)
                loss.backward()
                opt.step()

                tr_loss += float(loss.item()) * x.size(0)
                n += x.size(0)

            tr_loss /= max(1, n)
            val_loss = eval_ae(model, val_loader, device=device, loss_fn=loss_fn)
            dt = time.time() - t0
            lr_now = opt.param_groups[0]["lr"]

            print(f"[{tag}] epoch {ep:03d}/{epochs} | train={tr_loss:.4f} | val={val_loss:.4f} | {dt:.1f}s")

            logger.log({
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "lr": lr_now,
                "epoch_seconds": dt,
                "elapsed_seconds": logger.elapsed(),
            })

            if stopper.step(val_loss, model):
                print(f"[{tag}] Early stopping.")
                break

        stopper.restore_best(model)
        return model
    finally:
        logger.close()
