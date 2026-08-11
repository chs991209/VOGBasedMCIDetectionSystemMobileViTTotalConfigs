"""CosineLinear — metric-learning head.

Computes the cosine similarity between the input feature and each class
prototype (a row of `weight`), scaled by `scale`. No bias: the output depends
only on the *angle* between feature and prototype, not their magnitudes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, scale=10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = scale

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x_norm, w_norm)
