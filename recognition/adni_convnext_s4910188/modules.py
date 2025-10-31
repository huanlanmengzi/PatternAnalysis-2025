# modules.py
import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # NCHW → NHWC
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return x

# === Basic ConvNeXt block ===
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = x + shortcut
        return x


# === Simplified ConvNeXt Tiny ===
class ConvNeXtTiny(nn.Module):
    def __init__(self, in_chans=3, num_classes=2):
        super().__init__()
        depths = [3, 3, 9, 3]  # Tiny config
        dims = [96, 192, 384, 768]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[ConvNeXtBlock(dims[i]) for _ in range(depths[i])])
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.norm(x)      # [B, C]
        x = self.head(x)
        return x

