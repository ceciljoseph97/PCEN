"""Learnable PCEN front end + small CNN classifier (PyTorch).

LearnablePCEN follows Wang et al. 2017 ("Trainable frontend for robust and far-field keyword
spotting"): per-channel s, alpha, delta, r, parameterised in log space so they stay positive.
Input: (B, C, T) power, C = channels (Doppler bins), T = time (frames). Output same shape.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LearnablePCEN(nn.Module):
    def __init__(self, n_channels: int, s: float = 0.05, alpha: float = 0.98, delta: float = 2.0,
                 r: float = 0.5, eps: float = 1e-6, per_channel: bool = True, learn_s: bool = True):
        super().__init__()
        n = n_channels if per_channel else 1
        self.log_s = nn.Parameter(torch.full((n,), math.log(s)), requires_grad=learn_s)
        self.log_alpha = nn.Parameter(torch.full((n,), math.log(alpha)))
        self.log_delta = nn.Parameter(torch.full((n,), math.log(delta)))
        self.log_r = nn.Parameter(torch.full((n,), math.log(r)))
        self.eps = eps

    def forward(self, E: torch.Tensor) -> torch.Tensor:          # (B, C, T)
        # global scale normalisation per example (radar power is arbitrary)
        ref = E.flatten(1).median(dim=1).values.clamp_min(1e-30)[:, None, None]
        E = E / ref
        s = self.log_s.exp().clamp(1e-4, 1.0)[None, :, None]
        alpha = self.log_alpha.exp().clamp(max=1.0)[None, :, None]
        delta = self.log_delta.exp()[None, :, None]
        r = self.log_r.exp().clamp(max=1.0)[None, :, None]
        n0 = min(10, E.shape[-1])
        m = E[..., :n0].mean(-1)
        Ms = []
        s_ = s[..., 0]
        for t in range(E.shape[-1]):
            m = (1 - s_) * m + s_ * E[..., t]
            Ms.append(m)
        M = torch.stack(Ms, dim=-1)
        return (E / (self.eps + M) ** alpha + delta) ** r - delta ** r


class LogFrontEnd(nn.Module):
    def forward(self, E):
        ref = E.flatten(1).median(dim=1).values.clamp_min(1e-30)[:, None, None]
        return torch.log10(E / ref + 1e-6)


class FixedPCEN(LearnablePCEN):
    def __init__(self, n_channels, **kw):
        super().__init__(n_channels, **kw)
        for p in self.parameters():
            p.requires_grad_(False)


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, frontend: nn.Module):
        super().__init__()
        self.frontend = frontend
        self.norm = nn.InstanceNorm2d(1, affine=True)
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.body = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, E):                     # (B, C, T) power
        x = self.frontend(E).unsqueeze(1)
        return self.body(self.norm(x))
