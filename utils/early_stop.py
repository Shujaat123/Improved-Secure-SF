import torch
import torch.nn as nn


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "min"):
        self.patience = int(patience)
        self.mode = mode
        self.best = None
        self.bad = 0
        self.best_state = None

    def step(self, metric: float, model: nn.Module) -> bool:
        metric = float(metric)
        if self.best is None:
            self.best = metric
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False

        improved = (metric < self.best) if self.mode == "min" else (metric > self.best)
        if improved:
            self.best = metric
            self.bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
