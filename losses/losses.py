import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred.float(), 1e-7, 1.0)
    inter = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    denom = torch.sum(y_true + y_pred, dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


class CombinedBCEDice(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 1e-7, 1.0)
        bce = F.binary_cross_entropy(y_pred, y_true.float())
        d = dice_loss(y_true, y_pred)
        return self.alpha * bce + self.beta * d
