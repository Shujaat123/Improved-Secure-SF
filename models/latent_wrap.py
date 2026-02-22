from typing import Optional
import torch
import torch.nn as nn


def sample_orthogonal(d: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(int(seed))
        A = torch.randn(d, d, generator=g)
    else:
        A = torch.randn(d, d)
    Q, R = torch.linalg.qr(A, mode="reduced")
    # stabilize sign
    diag = torch.sign(torch.diag(R))
    diag = torch.where(diag == 0, torch.ones_like(diag), diag)
    Q = Q * diag
    return Q


class LatentWrap(nn.Module):
    """
    z: (B,C,H,W)
      forward : z' = z @ Q^T + b
      inverse : z  = (z' - b) @ Q
    applied per spatial position.
    """
    def __init__(self, channels: int, seed: Optional[int] = None):
        super().__init__()
        C = int(channels)
        Q = sample_orthogonal(C, seed=seed)
        b = torch.randn(C)
        self.register_buffer("Q", Q)  # (C,C)
        self.register_buffer("b", b)  # (C,)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)  # (BHW,C)
        zf = zf @ self.Q.T
        zf = zf + self.b
        return zf.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def inverse(self, z_prime: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_prime.shape
        zf = z_prime.permute(0, 2, 3, 1).reshape(-1, C)
        zf = zf - self.b
        zf = zf @ self.Q
        return zf.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
