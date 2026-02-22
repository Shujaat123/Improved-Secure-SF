from typing import Optional, Dict
import torch
import torch.nn as nn

from models.latent_wrap import LatentWrap


def conv_bn_relu(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),  
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class EncoderT(nn.Module):
    """
    Encoder producing:
      s3: 32x32 (64ch)
      s4: 16x16 (128ch)
      b :  8x8  (256ch)

    forward_original(x) -> (b,s3,s4) ORIGINAL (unprotected)
    forward(x)          -> (bT,s3T,s4T) PROTECTED if use_klt else original

    inverse_triplet(bT,s3T,s4T) uses THIS encoder's own T params.
    """
    def __init__(self, in_ch: int, use_klt: bool, seed: Optional[int]):
        super().__init__()
        self.use_klt = bool(use_klt)

        self.e1 = conv_bn_relu(in_ch, 16, 3, 2, 1)    # 128
        self.e2 = conv_bn_relu(16,   32, 3, 2, 1)     # 64
        self.e3 = conv_bn_relu(32,   64, 3, 2, 1)     # 32 (s3)
        self.e4 = conv_bn_relu(64,  128, 3, 2, 1)     # 16 (s4)
        self.b  = conv_bn_relu(128, 256, 3, 2, 1)     # 8  (b)

        if self.use_klt:
            self.T_b  = LatentWrap(256, seed=None if seed is None else int(seed) + 1)
            self.T_s3 = LatentWrap(64,  seed=None if seed is None else int(seed) + 2)
            self.T_s4 = LatentWrap(128, seed=None if seed is None else int(seed) + 3)
        else:
            self.T_b = self.T_s3 = self.T_s4 = None

    def forward_original(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        s3 = self.e3(e2)
        s4 = self.e4(s3)
        b  = self.b(s4)
        return b, s3, s4

    def forward(self, x):
        b, s3, s4 = self.forward_original(x)
        if self.use_klt:
            return self.T_b(b), self.T_s3(s3), self.T_s4(s4)
        return b, s3, s4

    def inverse_triplet(self, bT, s3T, s4T):
        if not self.use_klt:
            return bT, s3T, s4T
        return self.T_b.inverse(bT), self.T_s3.inverse(s3T), self.T_s4.inverse(s4T)

    def get_Tlayers(self) -> Dict[str, Optional[LatentWrap]]:
        return {"T_b": self.T_b, "T_s3": self.T_s3, "T_s4": self.T_s4}


class Decoder(nn.Module):
    """
    Decoder expects ORIGINAL (b,s3,s4) and reconstructs output (C,H,W).
    """
    def __init__(self, out_ch: int, use_skips: bool):
        super().__init__()
        self.use_skips = bool(use_skips)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_bn_relu(256, 128),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_bn_relu(128 + (128 if self.use_skips else 0), 64),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_bn_relu(64 + (64 if self.use_skips else 0), 32),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv_bn_relu(32, 16),
        )
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out_conv = nn.Conv2d(16, out_ch, kernel_size=3, padding=1)
        self.out_act  = nn.Sigmoid()

    def forward(self, b, s3=None, s4=None):
        x = self.up1(b)
        if self.use_skips:
            if s4 is None or s3 is None:
                raise ValueError("Decoder configured with skips but missing s3/s4.")
            x = torch.cat([x, s4], dim=1)
        x = self.up2(x)
        if self.use_skips:
            x = torch.cat([x, s3], dim=1)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final_up(x)
        return self.out_act(self.out_conv(x))


class AutoEncoder(nn.Module):
    """
    Full AE reconstruction:
      x -> (bT,s3T,s4T) -> inverse(T) -> decoder -> recon
    """
    def __init__(self, in_ch: int, use_skips: bool, use_klt: bool, seed: Optional[int]):
        super().__init__()
        self.encoderT = EncoderT(in_ch=in_ch, use_klt=use_klt, seed=seed)
        self.decoder = Decoder(out_ch=in_ch, use_skips=use_skips)
        self.use_skips = bool(use_skips)

    def forward(self, x):
        bT, s3T, s4T = self.encoderT(x)
        b, s3, s4 = self.encoderT.inverse_triplet(bT, s3T, s4T)
        if self.use_skips:
            return self.decoder(b, s3=s3, s4=s4)
        return self.decoder(b)
