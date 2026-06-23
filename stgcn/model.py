"""
ST-GCN — Spatial Temporal Graph Convolutional Network
Yan et al., AAAI 2018.

Input  : (N, C, T, V, M)
           N = batch, C = coords (3), T = frames, V = joints, M = persons
Output : (N, num_classes)

The model is intentionally kept clean so a 2-stage head can be
attached easily: replace or augment self.fc with your own module.
"""

import torch
import torch.nn as nn
import numpy as np
from .graph import ADJACENCY


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SpatialGCN(nn.Module):
    """One spatial graph convolution: A-weighted neighbour aggregation."""

    def __init__(self, in_ch, out_ch, A):
        super().__init__()
        self.K = A.shape[0]          # number of adjacency subsets
        self.A = nn.Parameter(torch.from_numpy(A), requires_grad=False)
        # learnable attention mask on top of fixed A
        self.mask = nn.Parameter(torch.zeros_like(self.A))

        self.conv = nn.Conv2d(in_ch, out_ch * self.K, kernel_size=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # residual
        self.residual = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        A = self.A + self.mask                       # (K, V, V)

        res = self.residual(x)
        x   = self.conv(x)                           # (N, K*out, T, V)
        x   = x.view(N, self.K, -1, T, V)           # (N, K, out, T, V)
        x   = torch.einsum("nkctv,kvw->nctw", x, A) # (N, out, T, V)
        x   = self.bn(x)
        return self.relu(x + res)


class TemporalConv(nn.Module):
    """Temporal convolution block with optional downsampling."""

    def __init__(self, ch, t_kernel=9, stride=1):
        super().__init__()
        pad = (t_kernel - 1) // 2
        self.net = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, (t_kernel, 1), stride=(stride, 1), padding=(pad, 0), bias=False),
            nn.BatchNorm2d(ch),
        )
        self.downsample = (
            nn.Conv2d(ch, ch, 1, stride=(stride, 1), bias=False)
            if stride != 1 else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class STGCNBlock(nn.Module):
    """One ST-GCN block = spatial GCN + temporal conv."""

    def __init__(self, in_ch, out_ch, A, stride=1, dropout=0.5):
        super().__init__()
        self.gcn  = SpatialGCN(in_ch, out_ch, A)
        self.tcn  = TemporalConv(out_ch, stride=stride)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.tcn(self.gcn(x)))


class STGCN(nn.Module):
    """
    Full ST-GCN model.

    Args:
        in_channels : sensor channels (default 3 — x, y, z)
        num_classes : output classes (default 2 — fall / no-fall)
        A           : adjacency tensor (K, V, V); defaults to MediaPipe graph
        dropout     : dropout rate in ST-GCN blocks
    """

    def __init__(self, in_channels=3, num_classes=2, A=None, dropout=0.5):
        super().__init__()
        if A is None:
            A = ADJACENCY

        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])

        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64,  A, dropout=dropout),
            STGCNBlock(64,          64,  A, dropout=dropout),
            STGCNBlock(64,          64,  A, dropout=dropout),
            STGCNBlock(64,          128, A, stride=2, dropout=dropout),
            STGCNBlock(128,         128, A, dropout=dropout),
            STGCNBlock(128,         128, A, dropout=dropout),
            STGCNBlock(128,         256, A, stride=2, dropout=dropout),
            STGCNBlock(256,         256, A, dropout=dropout),
            STGCNBlock(256,         256, A, dropout=dropout),
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        """
        x : (N, C, T, V, M)
        return_features : if True, also return the 256-d embedding before fc
                          (useful for 2-stage head)
        """
        N, C, T, V, M = x.shape

        # merge persons into batch
        x = x.permute(0, 4, 3, 1, 2).contiguous()   # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)                             # (N*M, 256, 1, 1)
        x = x.view(N, M, -1).mean(dim=1)            # (N, 256)  — mean over persons

        if return_features:
            return self.fc(x), x
        return self.fc(x)
