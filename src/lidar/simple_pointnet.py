# Source: Lang et al., "PointPillars: Fast Encoders for Object Detection from Point Clouds," CVPR 2019.
# Encodes each pillar's point set into a single feature vector via a shared MLP + max pool.

import torch
import torch.nn as nn


class SimplifiedPointNet(nn.Module):
    """
    Applies a shared linear layer to every point in every pillar, then
    max-pools across points to produce one feature vector per pillar.

    Input:  (P, N, input_dim)
    Output: (P, output_dim)
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (P, N, C) -> transpose to (P, C, N) for Conv1d
        x = x.transpose(1, 2)
        x = self.relu(self.bn(self.fc(x)))
        x, _ = torch.max(x, dim=2)  # max pool over points -> (P, C)
        return x
