"""
Splice-site model architectures.

SpliceSitePredictor  — original 4-layer CNN (~189 K params), kept for compatibility.
SpliceSiteResNet     — dilated pre-activation ResNet (~676 K params), recommended.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared building blocks ────────────────────────────────────────────────────

class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Original CNN ──────────────────────────────────────────────────────────────

class SpliceSitePredictor(nn.Module):
    """
    Simple 4-layer 1D CNN.
    Input : (B, 4, 140)  one-hot DNA
    Output: (B, 1)       raw logit
    ~189 K parameters.
    """
    def __init__(self, seq_len: int = 140, dropout: float = 0.3):
        super().__init__()
        self.seq_len = seq_len
        self.features = nn.Sequential(
            _ConvBNReLU(4, 32, 9),   nn.MaxPool1d(2),
            _ConvBNReLU(32, 64, 7),  nn.MaxPool1d(2),
            _ConvBNReLU(64, 128, 5), nn.MaxPool1d(2),
            _ConvBNReLU(128, 256, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Dilated pre-activation ResNet ─────────────────────────────────────────────

class _ResBlock1D(nn.Module):
    """
    Pre-activation residual block with dilated depthwise-separable 1D convolution.

    Pre-activation (BN → act → conv) stabilises training in deeper nets and
    gives better gradient flow.  Dilation lets each block see a wider context
    without increasing parameters or losing resolution.
    """
    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        kernel_size: int = 9,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.GELU(),
            nn.Conv1d(in_ch, out_ch, kernel_size,
                      dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
        )
        # 1×1 projection when channel dimensions differ
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class SpliceSiteResNet(nn.Module):
    """
    Dilated pre-activation ResNet for splice-site classification.

    Architecture rationale
    ----------------------
    The 140 bp HS3D window contains multi-scale signals:
      • Dinucleotide GT/AG at the exact splice position (1–2 bp)
      • Splice-site consensus motif (3–8 bp around the site)
      • Exonic/intronic composition bias (10–30 bp)
      • Polypyrimidine tract and branch-point region (up to 70 bp upstream)

    A stack of dilated convolutions with rates [1, 4, 16] covers all three
    scales in a single pass with the same parameter budget as a single
    large kernel:

      dilation=1  →  receptive field ≈ 9 bp    (local motif)
      dilation=4  →  receptive field ≈ 41 bp   (surrounding context)
      dilation=16 →  receptive field ≈ 137 bp  (nearly the full window)

    Pre-activation (BN→GELU→Conv) improves gradient flow in deep nets and
    matches the approach used in Jaganathan et al. (SpliceAI, 2019).

    Input : (B, 4, 140)  one-hot encoded DNA
    Output: (B, 1)       raw logit; sigmoid → P(splice site)
    ~676 K trainable parameters.
    """

    def __init__(self, dropout_res: float = 0.1, dropout_fc: float = 0.3):
        super().__init__()

        # Stem: project nucleotide channels to feature space
        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Dilated tower: captures multi-scale splice-site signals
        self.dilated_tower = nn.Sequential(
            _ResBlock1D(64,  64,  kernel_size=9, dilation=1,  dropout=dropout_res),
            _ResBlock1D(64,  64,  kernel_size=9, dilation=4,  dropout=dropout_res),
            _ResBlock1D(64,  64,  kernel_size=9, dilation=16, dropout=dropout_res),
            _ResBlock1D(64,  64,  kernel_size=9, dilation=1,  dropout=dropout_res),
        )

        # Downsampling branch 1: 140 → 70
        self.down1 = nn.Sequential(
            nn.MaxPool1d(2),
            _ResBlock1D(64,  128, kernel_size=7, dilation=1,  dropout=dropout_res),
            _ResBlock1D(128, 128, kernel_size=7, dilation=2,  dropout=dropout_res),
        )

        # Downsampling branch 2: 70 → 35
        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            _ResBlock1D(128, 256, kernel_size=5, dilation=1,  dropout=dropout_res),
            _ResBlock1D(256, 256, kernel_size=5, dilation=2,  dropout=dropout_res),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_fc),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.dilated_tower(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.pool(x)
        return self.classifier(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
