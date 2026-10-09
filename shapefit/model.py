"""A compact U-Net with CoordConv input and two heads (clean mask, vertex heatmap)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(8, cout),
        nn.SiLU(inplace=True),
    )


class ShapeFitNet(nn.Module):
    """Input ``(B, 1, S, S)`` binary mask -> ``(B, 2, S, S)`` logits [mask, vertices]."""

    def __init__(self, base: int = 32, depth: int = 4):
        super().__init__()
        self.config = {"base": base, "depth": depth}
        chs = [base * 2**i for i in range(depth + 1)]
        self.stem = _block(3, chs[0])  # mask + (x, y) coordinate channels
        self.down = nn.ModuleList(_block(chs[i], chs[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))
        )
        self.dec = nn.ModuleList(_block(chs[i] * 2, chs[i]) for i in reversed(range(depth)))
        self.head = nn.Conv2d(chs[0], 2, 1)
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():  # vertex heatmap is sparse: start with low prior
            self.head.bias[1] = -4.0

    @staticmethod
    def coords(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)
        xs = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy])[None].expand(b, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x * 2 - 1, self.coords(x)], 1)
        skips = [self.stem(x)]
        for d in self.down:
            skips.append(d(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([up(h), skips.pop()], 1))
        return self.head(h)


def focal_heatmap_loss(logits: torch.Tensor, target: torch.Tensor, alpha=2.0, beta=4.0):
    """CenterNet-style penalty-reduced focal loss for Gaussian peak targets."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.ge(0.99).float()
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * p**alpha * torch.log(1 - p) * (1 - pos)
    npos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / npos


def mask_loss(logits: torch.Tensor, target: torch.Tensor):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()
