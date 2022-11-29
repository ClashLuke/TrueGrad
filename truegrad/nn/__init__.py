from typing import List, Union

import torch
import torch.nn as nn

from truegrad.functional import TrueGradParameter, add, is_tgparam, mul
from truegrad.nn import functional

TrueGradParameter = TrueGradParameter
is_tgparam = is_tgparam
F = functional


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


def _batchnorm_forward(self: nn.modules.batchnorm._BatchNorm, input: torch.Tensor):
    self._check_input_dim(input)

    if self.momentum is not None:
        ema_fac = self.momentum
    else:
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
            self.num_batches_tracked.add_(1)
            ema_fac = 1.0 / float(self.num_batches_tracked)
        else:
            ema_fac = 0.0

    bn_training = self.training or ((self.running_mean is None) and (self.running_var is None))

    return F.batch_norm(input,
                        self.running_mean if not self.training or self.track_running_stats else None,
                        self.running_var if not self.training or self.track_running_stats else None,
                        self.weight, self.bias, bn_training, ema_fac, self.eps)


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _batchnorm_forward(self, input)


class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _batchnorm_forward(self, input)


class BatchNorm3d(nn.BatchNorm3d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _batchnorm_forward(self, input)


class LazyBatchNorm1d(nn.LazyBatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _batchnorm_forward(self, input)


class LazyBatchNorm2d(nn.LazyBatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _batchnorm_forward(self, input)


class LazyBatchNorm3d(nn.LazyBatchNorm3d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _batchnorm_forward(self, input)


class InstanceNorm1d(nn.InstanceNorm1d):
    def _apply_instance_norm(self, input):
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training or not self.track_running_stats, self.momentum, self.eps)


class InstanceNorm2d(nn.InstanceNorm2d):
    def _apply_instance_norm(self, input):
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training or not self.track_running_stats, self.momentum, self.eps)


class InstanceNorm3d(nn.InstanceNorm3d):
    def _apply_instance_norm(self, input):
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training or not self.track_running_stats, self.momentum, self.eps)


class LazyInstanceNorm1d(nn.LazyInstanceNorm1d):
    def _apply_instance_norm(self, input):
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training or not self.track_running_stats, self.momentum, self.eps)


class LazyInstanceNorm2d(nn.LazyInstanceNorm2d):
    def _apply_instance_norm(self, input):
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training or not self.track_running_stats, self.momentum, self.eps)


class LazyInstanceNorm3d(nn.LazyInstanceNorm3d):
    def _apply_instance_norm(self, input):
        return F.instance_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training or not self.track_running_stats, self.momentum, self.eps)


class _LayerNorm(nn.Module):
    def __init__(self, normalized_shape: List[int], eps: float, broadcast: bool):
        super(_LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.broadcast = broadcast

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, eps=self.eps, broadcast=self.broadcast)


class LayerNorm(Normalization):
    def __init__(self, normalized_shape: Union[int, List[int]], eps=1e-05, elementwise_affine=True, device=None,
                 dtype=None, broadcast: bool = False):
        if device is not None or dtype is not None:
            raise ValueError("device and dtype are not supported. Ensure both are set to None.")
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        super(LayerNorm, self).__init__(_LayerNorm(normalized_shape, eps, broadcast),
                                        normalized_shape, elementwise_affine)


class LayerNorm1d(LayerNorm):
    def __init__(self, num_features: int, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(LayerNorm1d, self).__init__([1, num_features, 1], eps, elementwise_affine, device, dtype, True)


class LayerNorm2d(LayerNorm):
    def __init__(self, num_features: int, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(LayerNorm2d, self).__init__([1, num_features, 1, 1], eps, elementwise_affine, device, dtype, True)


class LayerNorm3d(LayerNorm):
    def __init__(self, num_features: int, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super(LayerNorm3d, self).__init__([1, num_features, 1, 1, 1], eps, elementwise_affine, device, dtype, True)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn((out_features, in_features)) / in_features ** 0.5)
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class Embedding(nn.Embedding):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq,
                           self.sparse)


class Conv1d(nn.Conv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            self.padding = (0,)
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            self.padding = (0, 0)
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv3d(nn.Conv3d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            self.padding = (0, 0, 0)
        return F.conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


modules = (Embedding, Linear, LayerNorm, LayerNorm1d, LayerNorm2d, LayerNorm3d, InstanceNorm1d, InstanceNorm2d,
           InstanceNorm3d, BatchNorm1d, BatchNorm2d, BatchNorm3d)
