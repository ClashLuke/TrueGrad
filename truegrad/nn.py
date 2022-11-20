from typing import List

import numpy as np
import torch
import torch.nn as nn

from truegrad.functional import add, gather, matmul, mul


class Normalization(nn.Module):
    def __init__(self, base_module: nn.Module, normalized_shape: List[int], affine: bool = True):
        super(Normalization, self).__init__()
        self.base_module = base_module
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        x = self.base_module(x)
        if not self.affine:
            return x

        return add(mul(x, self.weight), self.bias)


class BatchNorm1d(Normalization):
    def __init__(self, num_features: int, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        super().__init__(nn.BatchNorm1d(num_features, eps, momentum, False, track_running_stats, device, dtype),
                         [1, num_features, 1], affine)


class BatchNorm2d(Normalization):
    def __init__(self, num_features: int, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        super().__init__(nn.BatchNorm2d(num_features, eps, momentum, False, track_running_stats, device, dtype),
                         [1, num_features, 1, 1], affine)


class BatchNorm3d(Normalization):
    def __init__(self, num_features: int, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None,
                 dtype=None):
        super().__init__(nn.BatchNorm3d(num_features, eps, momentum, False, track_running_stats, device, dtype),
                         [1, num_features, 1, 1, 1], affine)


class InstanceNorm1d(Normalization):
    def __init__(self, num_features: int, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None,
                 dtype=None):
        super().__init__(nn.InstanceNorm1d(num_features, eps, momentum, False, track_running_stats, device, dtype),
                         [1, num_features, 1], affine)


class InstanceNorm2d(Normalization):
    def __init__(self, num_features: int, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None,
                 dtype=None):
        super().__init__(nn.InstanceNorm2d(num_features, eps, momentum, False, track_running_stats, device, dtype),
                         [1, num_features, 1, 1], affine)


class InstanceNorm3d(Normalization):
    def __init__(self, num_features: int, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, device=None,
                 dtype=None):
        super().__init__(nn.InstanceNorm3d(num_features, eps, momentum, False, track_running_stats, device, dtype),
                         [1, num_features, 1, 1, 1], affine)


class _LayerNorm(nn.Module):
    def __init__(self, dims: List[int], eps: float):
        super(_LayerNorm, self).__init__()
        self.dims = dims
        self.eps = eps

    def forward(self, x):
        x = x - x.mean(self.dims, True)
        return x / (self.eps + x.norm(2, self.dims, True)) * (np.prod([x.size(d) for d in self.dims]) - 1) ** 0.5


class LayerNorm(Normalization):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        if device is not None or dtype is not None:
            raise ValueError("device and dtype are not supported. Ensure both are set to None.")
        super(LayerNorm, self).__init__(_LayerNorm([-i - 1 for i, dim in enumerate(normalized_shape) if dim != 1], eps),
                                        normalized_shape, elementwise_affine)


class LayerNorm1d(LayerNorm):
    def __init__(self, num_features: int, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(LayerNorm1d, self).__init__([1, num_features, 1], eps, elementwise_affine, device, dtype)


class LayerNorm2d(LayerNorm):
    def __init__(self, num_features: int, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(LayerNorm2d, self).__init__([1, num_features, 1, 1], eps, elementwise_affine, device, dtype)


class LayerNorm3d(LayerNorm):
    def __init__(self, num_features: int, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(LayerNorm3d, self).__init__([1, num_features, 1, 1, 1], eps, elementwise_affine, device, dtype)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn((in_features, out_features)) / in_features ** 0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = matmul(x, self.weight)
        if self.bias is None:
            return x
        return add(x, self.bias)


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        if kwargs:
            raise ValueError(f"{kwargs} are not supported.")
        super(Embedding, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return gather(input, self.weight)
