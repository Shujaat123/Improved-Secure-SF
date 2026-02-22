from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, pool_sizes=(1, 3, 5, 7), out_channels: Optional[int] = None):
        super().__init__()
        self.pool_sizes = pool_sizes
        oc = out_channels if out_channels is not None else max(1, in_channels // (len(pool_sizes) * 2))

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, oc, kernel_size=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )
            for _ in pool_sizes
        ])

        self.final = nn.Sequential(
            nn.Conv2d(in_channels + oc * len(pool_sizes), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        outs = [x]
        for s, br in zip(self.pool_sizes, self.branches):
            y = x.mean(dim=(2, 3), keepdim=True)
            y = F.interpolate(y, size=(s, s), mode="bilinear", align_corners=False)
            y = br(y)
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(y)
        return self.final(torch.cat(outs, dim=1))


def conv_lrelu_bn(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.LeakyReLU(0.1, inplace=True),
        nn.BatchNorm2d(out_ch),
    )


class MappingBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_ppm: bool = True):
        super().__init__()
        self.use_ppm = bool(use_ppm)

        self.c1 = conv_lrelu_bn(in_ch, 128)
        self.c2 = conv_lrelu_bn(128, 256)
        self.c3 = conv_lrelu_bn(256, 512)
        self.drop1 = nn.Dropout2d(0.2)
        self.ppm1 = PyramidPoolingModule(512) if self.use_ppm else nn.Identity()

        self.c4 = conv_lrelu_bn(512, 512)
        self.c5 = conv_lrelu_bn(512, 1024)
        self.drop2 = nn.Dropout2d(0.2)

        self.pool = nn.MaxPool2d(2)
        self.d1 = conv_lrelu_bn(1024, 256)
        self.d2 = conv_lrelu_bn(256, 128)
        self.d3 = conv_lrelu_bn(128, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.u1 = conv_lrelu_bn(64 + 128, 128)
        self.drop_u1 = nn.Dropout2d(0.2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.u2 = conv_lrelu_bn(128 + 256, 256)
        self.drop_u2 = nn.Dropout2d(0.2)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.u3 = conv_lrelu_bn(256 + 1024, 512)
        self.ppm2 = PyramidPoolingModule(512) if self.use_ppm else nn.Identity()

        self.out = nn.Conv2d(512, out_ch, kernel_size=1)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.drop1(x)
        x0 = self.ppm1(x)

        x = self.c4(x0)
        x = self.c5(x)
        x = self.drop2(x)

        d1 = self.d1(self.pool(x))
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))

        u1 = self.up1(d3)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.drop_u1(self.u1(u1))

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.drop_u2(self.u2(u2))

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x], dim=1)
        u3 = self.u3(u3)
        u3 = self.ppm2(u3)

        return self.out_act(self.out(u3))


class UMN(nn.Module):
    def __init__(self, use_ppm: bool = True):
        super().__init__()
        self.mb  = MappingBlock(256, 256, use_ppm=use_ppm)
        self.ms3 = MappingBlock(64,   64,  use_ppm=use_ppm)
        self.ms4 = MappingBlock(128, 128, use_ppm=use_ppm)

    def forward(self, b, s3, s4):
        return self.mb(b), self.ms3(s3), self.ms4(s4)
