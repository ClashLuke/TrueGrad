import functools
import math
import typing
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch.autograd
from torch import Tensor, nn
from torch.nn import functional as F, grad

from truegrad.functional import add, einsum, matmul, mul, reshape

_torch_functional = {k: getattr(F, k) for k in dir(F)}
_torch = {k: getattr(torch, k) for k in dir(torch)}
_inside_call = {}


def call_torch(fn: Callable, name: Optional[str] = None):
    if name is None:
        name = fn.__name__
    _inside_call[fn] = 0

    def _fn(*args, **kwargs):
        _inside_call[fn] += 1
        if _inside_call[fn] == 1:
            out = fn(*args, **kwargs)
        elif _inside_call[fn] == 2:
            out = _torch_functional[name](*args, **kwargs)
        elif _inside_call[fn] == 3:
            out = _torch[name](*args, **kwargs)
        else:
            raise ValueError
        _inside_call[fn] -= 1
        return out

    return _fn


def no_parameter(fn: Callable):
    @functools.partial(call_torch, name=fn.__name__)
    def _fn(*args, **kwargs):
        for i, arg in enumerate(args):
            if isinstance(arg, nn.Parameter):
                raise ValueError(f"Function does not support parameters. Offending argument: positional argument {i}")
        for k, v in kwargs.items():
            if isinstance(v, nn.Parameter):
                raise ValueError(f"Function does not support parameters. Offending argument: {k}")
        return fn(*args, **kwargs)

    return _fn


@no_parameter
def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return F.avg_pool1d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)


@no_parameter
def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
               divisor_override=None):
    return F.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


@no_parameter
def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
               divisor_override=None):
    return F.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)


@no_parameter
def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return F.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


@no_parameter
def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return F.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


@no_parameter
def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    return F.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


@no_parameter
def max_pool1d_with_indices(input: Tensor, kernel_size, stride=None, padding=0,
                            dilation=1, ceil_mode: bool = False, return_indices: bool = False):
    return F.max_pool1d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


@no_parameter
def max_pool2d_with_indices(input: Tensor, kernel_size, stride=None, padding=0,
                            dilation=1, ceil_mode: bool = False, return_indices: bool = False):
    return F.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


@no_parameter
def max_pool3d_with_indices(input: Tensor, kernel_size, stride=None, padding=0,
                            dilation=1, ceil_mode: bool = False, return_indices: bool = False):
    return F.max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)


@no_parameter
def max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return F.max_unpool1d(input, indices, kernel_size, stride, padding, output_size)


@no_parameter
def max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return F.max_unpool2d(input, indices, kernel_size, stride, padding, output_size)


@no_parameter
def max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    return F.max_unpool3d(input, indices, kernel_size, stride, padding, output_size)


@no_parameter
def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return F.lp_pool1d(input, norm_type, kernel_size, stride, ceil_mode)


@no_parameter
def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    return F.lp_pool2d(input, norm_type, kernel_size, stride, ceil_mode)


@no_parameter
def adaptive_max_pool1d(input, output_size, return_indices=False):
    return F.adaptive_max_pool1d(input, output_size, return_indices)


@no_parameter
def adaptive_max_pool2d(input, output_size, return_indices=False):
    return F.adaptive_max_pool2d(input, output_size, return_indices)


@no_parameter
def adaptive_max_pool3d(input, output_size, return_indices=False):
    return F.adaptive_max_pool3d(input, output_size, return_indices)


@no_parameter
def adaptive_max_pool1d_with_indices(input: Tensor, output_size, return_indices: bool = False):
    return F.adaptive_max_pool1d_with_indices(input, output_size, return_indices)


@no_parameter
def adaptive_max_pool2d_with_indices(input: Tensor, output_size, return_indices: bool = False):
    return F.adaptive_max_pool2d_with_indices(input, output_size, return_indices)  #


@no_parameter
def adaptive_max_pool3d_with_indices(input: Tensor, output_size, return_indices: bool = False):
    return F.adaptive_max_pool3d_with_indices(input, output_size, return_indices)


@no_parameter
def adaptive_avg_pool1d(input, output_size):
    return F.adaptive_avg_pool1d(input, output_size)


@no_parameter
def adaptive_avg_pool2d(input, output_size):
    return F.adaptive_avg_pool2d(input, output_size)


@no_parameter
def adaptive_avg_pool3d(input, output_size):
    return F.adaptive_avg_pool3d(input, output_size)


@no_parameter
def fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
                          _random_samples=None):
    return F.fractional_max_pool2d(input, kernel_size, output_size, output_ratio, return_indices, _random_samples)


@no_parameter
def fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False,
                          _random_samples=None):
    return F.fractional_max_pool3d(input, kernel_size, output_size, output_ratio, return_indices, _random_samples)


@no_parameter
def fractional_max_pool2d_with_indices(input: Tensor, kernel_size, output_size=None,
                                       output_ratio=None, return_indices: bool = False,
                                       _random_samples: typing.Optional[Tensor] = None):
    return F.fractional_max_pool2d_with_indices(input, kernel_size, output_size, output_ratio, return_indices,
                                                _random_samples)


@no_parameter
def fractional_max_pool3d_with_indices(input: Tensor, kernel_size, output_size=None,
                                       output_ratio=None, return_indices: bool = False,
                                       _random_samples: typing.Optional[Tensor] = None):
    return F.fractional_max_pool3d_with_indices(input, kernel_size, output_size, output_ratio, return_indices,
                                                _random_samples)


@no_parameter
def affine_grid(theta: Tensor, size: typing.List[int], align_corners: typing.Optional[bool] = None):
    return F.affine_grid(theta, size, align_corners)


@no_parameter
def alpha_dropout(input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False):
    return F.alpha_dropout(input, p, training, inplace)


@call_torch
def batch_norm(input: Tensor, running_mean: typing.Optional[Tensor],
               running_var: typing.Optional[Tensor], weight: typing.Optional[Tensor] = None,
               bias: typing.Optional[Tensor] = None, training: bool = False, momentum: float = 0.1,
               eps: float = 1e-05):
    input = F.batch_norm(input, running_mean, running_var, None, None, training, momentum, eps)
    if weight is not None:
        input = mul(input, reshape(weight, (1, -1,) + (1,) * (input.ndim - 2)))
    if bias is not None:
        input = add(input, reshape(bias, (1, -1,) + (1,) * (input.ndim - 2)))
    return input


@call_torch
def bilinear(input1: Tensor, input2: Tensor, weight: Tensor, bias: typing.Optional[Tensor] = None):
    batch_dims = ''.join(chr(ord('a') + i) for i in range(input1.ndim - 1))
    x = einsum(f'{batch_dims}x,{batch_dims}y,zxy->{batch_dims}z', input1, input2, weight)
    if bias is None:
        return x
    return add(x, bias)


@no_parameter
def binary_cross_entropy(input: Tensor, target: Tensor, weight: typing.Optional[Tensor] = None,
                         size_average: typing.Optional[bool] = None, reduce: typing.Optional[bool] = None,
                         reduction: str = "mean"):
    return F.binary_cross_entropy(input, target, weight, size_average, reduce, reduction)


@no_parameter
def binary_cross_entropy_with_logits(input: Tensor, target: Tensor,
                                     weight: typing.Optional[Tensor] = None,
                                     size_average: typing.Optional[bool] = None, reduce: typing.Optional[bool] = None,
                                     reduction: str = "mean", pos_weight: typing.Optional[Tensor] = None):
    return F.binary_cross_entropy_with_logits(input, target, weight, size_average, reduce, reduction, pos_weight)


@no_parameter
def celu(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return F.celu(input, alpha, inplace)


@no_parameter
def celu_(input: Tensor, alpha: float = 1.0):
    return F.celu_(input, alpha)


@no_parameter
def channel_shuffle(input: Tensor, groups: int):
    return F.channel_shuffle(input, groups)


class _ConvNdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor], args) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(input, weight, bias)
            ctx.args = args
        dim = input.dim() - 2  # Batch, Feature, *Data
        return getattr(F, f"conv{dim}d")(input, weight, bias, *args)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None]:
        if not ctx.saved_tensors:
            return None, None, None, None
        inp, wgt, bias = ctx.saved_tensors
        dim = inp.dim() - 2
        summed = [0] + list(range(2, 2 + dim))

        dx = getattr(grad, f"conv{dim}d_input")(inp.size(), wgt, dy, *ctx.args)
        dw = getattr(grad, f"conv{dim}d_weight")(inp, wgt.size(), dy, *ctx.args)
        db = None if bias is None else dy.sum(summed)

        if isinstance(wgt, nn.Parameter) or isinstance(bias, nn.Parameter):
            dy_sq = dy.square() * dy.size(0)
        if isinstance(wgt, nn.Parameter):
            wgt.sum_grad_squared = getattr(grad, f"conv{dim}d_weight")(inp.square(), wgt.size(), dy_sq, *ctx.args)
        if isinstance(bias, nn.Parameter):
            bias.sum_grad_squared = dy_sq.sum(summed)
        return dx, dw, db, None


@call_torch
def _convnd(input: Tensor, weight: Tensor, bias: Optional[Tensor], dim: int, *args):
    if input.dim() != dim + 2:
        raise ValueError(f"Input has {input.dim()} dimensions, but expected {dim + 2} dimensions for conv{dim}d.")
    return _ConvNdFn.apply(input, weight, bias, args)


@call_torch
def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1,
           padding: Union[str, int] = "valid",
           dilation: int = 1, groups: int = 1):
    return _convnd(input, weight, bias, 1, stride, padding, dilation, groups)


@call_torch
def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1,
           padding: Union[str, int] = "valid",
           dilation: int = 1, groups: int = 1):
    return _convnd(input, weight, bias, 2, stride, padding, dilation, groups)


@call_torch
def conv3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1,
           padding: Union[str, int] = "valid",
           dilation: int = 1, groups: int = 1):
    return _convnd(input, weight, bias, 3, stride, padding, dilation, groups)


@no_parameter
def conv_transpose1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0,
                     output_padding: int = 0, groups: int = 1, dilation: int = 1):
    return F.conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation)


@no_parameter
def conv_transpose2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0,
                     output_padding: int = 0, groups: int = 1, dilation: int = 1):
    return F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)


@no_parameter
def conv_transpose3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1, padding: int = 0,
                     output_padding: int = 0, groups: int = 1, dilation: int = 1):
    return F.conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation)


@no_parameter
def cosine_embedding_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0,
                          size_average: typing.Optional[bool] = None, reduce: typing.Optional[bool] = None,
                          reduction: str = "mean"):
    return F.cosine_embedding_loss(input1, input2, target, margin, size_average, reduce, reduction)


@no_parameter
def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-08):
    return F.cosine_similarity(x1, x2, dim, eps)


@no_parameter
def cross_entropy(input: Tensor, target: Tensor, weight: typing.Optional[Tensor] = None,
                  size_average: typing.Optional[bool] = None, ignore_index: int = -100,
                  reduce: typing.Optional[bool] = None, reduction: str = "mean", label_smoothing: float = 0.0):
    return F.cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)


@no_parameter
def ctc_loss(log_probs: Tensor, targets: Tensor, input_lengths: Tensor, target_lengths: Tensor,
             blank: int = 0, reduction: str = "mean", zero_infinity: bool = False):
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity)


@no_parameter
def dropout(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    return F.dropout(input, p, training, inplace)


@no_parameter
def dropout1d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    return F.dropout1d(input, p, training, inplace)


@no_parameter
def dropout2d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    return F.dropout2d(input, p, training, inplace)


@no_parameter
def dropout3d(input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    return F.dropout3d(input, p, training, inplace)


@no_parameter
def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False):
    return F.elu(input, alpha, inplace)


@no_parameter
def elu_(input: Tensor, alpha: float = 1.0):
    return F.elu_(input, alpha)


@no_parameter
def embedding(input: Tensor, weight: Tensor, padding_idx: typing.Optional[int] = None,
              max_norm: typing.Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False,
              sparse: bool = False):
    return F.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)


@no_parameter
def embedding_bag(input: Tensor, weight: Tensor, offsets: typing.Optional[Tensor] = None,
                  max_norm: typing.Optional[float] = None, norm_type: float = 2, scale_grad_by_freq: bool = False,
                  mode: str = "mean", sparse: bool = False, per_sample_weights: typing.Optional[Tensor] = None,
                  include_last_offset: bool = False, padding_idx: typing.Optional[int] = None):
    return F.embedding_bag(input, weight, offsets, max_norm, norm_type, scale_grad_by_freq, mode, sparse,
                           per_sample_weights, include_last_offset, padding_idx)


@no_parameter
def feature_alpha_dropout(input: Tensor, p: float = 0.5, training: bool = False, inplace: bool = False):
    return F.feature_alpha_dropout(input, p, training, inplace)


@no_parameter
def fold(input: Tensor, output_size, kernel_size, dilation=1, padding=0,
         stride=1):
    return F.fold(input, output_size, kernel_size, dilation, padding, stride)


@no_parameter
def gaussian_nll_loss(input: Tensor, target: Tensor, var: Tensor, full: bool = False,
                      eps: float = 1e-06, reduction: str = "mean"):
    return F.gaussian_nll_loss(input, target, var, full, eps, reduction)


@no_parameter
def gelu(input: Tensor, approximate='none'):
    return F.gelu(input, approximate)


@no_parameter
def glu(input: Tensor, dim: int = -1):
    return F.glu(input, dim)


@no_parameter
def grid_sample(input: Tensor, grid: Tensor, mode: str = bilinear, padding_mode: str = "zeros",
                align_corners: typing.Optional[bool] = None):
    return F.grid_sample(input, grid, mode, padding_mode, align_corners)


@call_torch
def group_norm(input: Tensor, num_groups: int, weight: typing.Optional[Tensor] = None,
               bias: typing.Optional[Tensor] = None, eps: float = 1e-05):
    x = F.group_norm(input, num_groups, None, None, eps)
    if weight is not None:
        x = mul(x, weight)
    if bias is not None:
        x = add(x, bias)
    return x


@no_parameter
def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1):
    return F.gumbel_softmax(logits, tau, hard, eps, dim)


@no_parameter
def hardshrink(input, lambd=0.5):
    return F.hardshrink(input, lambd)


@no_parameter
def hardsigmoid(input: Tensor, inplace: bool = False):
    return F.hardsigmoid(input, inplace)


@no_parameter
def hardswish(input: Tensor, inplace: bool = False):
    return F.hardswish(input, inplace)


@no_parameter
def hardtanh(input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False):
    return F.hardtanh(input, min_val, max_val, inplace)


@no_parameter
def hardtanh_(input: Tensor, min_val: float = -1.0, max_val: float = 1.0):
    return F.hardtanh_(input, min_val, max_val)


@no_parameter
def hinge_embedding_loss(input: Tensor, target: Tensor, margin: float = 1.0,
                         size_average: typing.Optional[bool] = None, reduce: typing.Optional[bool] = None,
                         reduction: str = "mean"):
    return F.hinge_embedding_loss(input, target, margin, size_average, reduce, reduction)


@no_parameter
def huber_loss(input: Tensor, target: Tensor, reduction: str = "mean", delta: float = 1.0):
    return F.huber_loss(input, target, reduction, delta)


@call_torch
def instance_norm(input: Tensor, running_mean: typing.Optional[Tensor] = None,
                  running_var: typing.Optional[Tensor] = None, weight: typing.Optional[Tensor] = None,
                  bias: typing.Optional[Tensor] = None, use_input_stats: bool = True, momentum: float = 0.1,
                  eps: float = 1e-05):
    x = F.instance_norm(input, running_mean, running_var, use_input_stats=use_input_stats, momentum=momentum, eps=eps)
    if weight is not None:
        x = mul(x, reshape(weight, (1, -1,) + (1,) * (input.ndim - 2)))
    if bias is not None:
        x = add(x, reshape(bias, (1, -1,) + (1,) * (input.ndim - 2)))
    return x


@no_parameter
def interpolate(input: Tensor, size: typing.Optional[int] = None,
                scale_factor: typing.Optional[typing.List[float]] = None, mode: str = "nearest",
                align_corners: typing.Optional[bool] = None, recompute_scale_factor: typing.Optional[bool] = None,
                antialias: bool = False):
    return F.interpolate(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)


@no_parameter
def kl_div(input: Tensor, target: Tensor, size_average: typing.Optional[bool] = None,
           reduce: typing.Optional[bool] = None, reduction: str = "mean", log_target: bool = False):
    return F.kl_div(input, target, size_average, reduce, reduction, log_target)


@no_parameter
def l1_loss(input: Tensor, target: Tensor, size_average: typing.Optional[bool] = None,
            reduce: typing.Optional[bool] = None, reduction: str = "mean"):
    return F.l1_loss(input, target, size_average, reduce, reduction)


@call_torch
def layer_norm(input: Tensor, normalized_shape: typing.List[int], weight: typing.Optional[Tensor] = None,
               bias: typing.Optional[Tensor] = None, eps: float = 1e-05, broadcast: bool = True):
    if broadcast:
        normalized_shape = [input.size(-1 - i) if d == 1 else d for i, d in enumerate(normalized_shape)]
    x = F.layer_norm(input, normalized_shape, eps=eps)
    if weight is not None:
        x = mul(x, weight)
    if bias is not None:
        x = add(x, bias)
    return x


@no_parameter
def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    return F.leaky_relu(input, negative_slope, inplace)


@no_parameter
def leaky_relu_(input: Tensor, negative_slope: float = 0.01):
    return F.leaky_relu_(input, negative_slope)


@call_torch
def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    input = matmul(input, weight)
    if bias is None:
        return input
    return add(input, bias)


@no_parameter
def local_response_norm(input: Tensor, size: int, alpha: float = 0.0001, beta: float = 0.75, k: float = 1.0):
    return F.local_response_norm(input, size, alpha, beta, k)


@no_parameter
def log_softmax(input: Tensor, dim: typing.Optional[int] = None, _stacklevel: int = 3,
                dtype: typing.Optional[int] = None):
    return F.log_softmax(input, dim, _stacklevel, dtype)


@no_parameter
def logsigmoid(input: Tensor):
    return F.logsigmoid(input)


@no_parameter
def lp_pool1d(input: Tensor, norm_type: typing.Union[int, float], kernel_size: int, stride=None,
              ceil_mode: bool = False):
    return F.lp_pool1d(input, norm_type, kernel_size, stride, ceil_mode)


@no_parameter
def lp_pool2d(input: Tensor, norm_type: typing.Union[int, float], kernel_size, stride=None,
              ceil_mode: bool = False):
    return F.lp_pool2d(input, norm_type, kernel_size, stride, ceil_mode)


@no_parameter
def margin_ranking_loss(input1: Tensor, input2: Tensor, target: Tensor, margin: float = 0,
                        size_average: typing.Optional[bool] = None, reduce: typing.Optional[bool] = None,
                        reduction: str = "mean"):
    return F.margin_ranking_loss(input1, input2, target, margin, size_average, reduce, reduction)


@no_parameter
def mish(input: Tensor, inplace: bool = False):
    return F.mish(input, inplace)


@no_parameter
def mse_loss(input: Tensor, target: Tensor, size_average: typing.Optional[bool] = None,
             reduce: typing.Optional[bool] = None, reduction: str = "mean"):
    return F.mse_loss(input, target, size_average, reduce, reduction)


def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
        ) -> List[Tensor]:
    r"""
    Performs the in-projection step of the attention operation, using packed weights.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected. For self-attention,
            these are typically the same tensor; for encoder-decoder attention,
            k and v are typically the same tensor. (We take advantage of these
            identities for performance if they are present.) Regardless, q, k and v
            must share a common embedding dimension; otherwise their shapes may vary.
        w: projection weights for q, k and v, packed into a single tensor. Weights
            are packed along dimension 0, in q, k, v order.
        b: optional projection biases for q, k and v, packed into a single tensor
            in q, k, v order.

    Shape:
        Inputs:
        - q: :math:`(..., E)` where E is the embedding dimension
        - k: :math:`(..., E)` where E is the embedding dimension
        - v: :math:`(..., E)` where E is the embedding dimension
        - w: :math:`(E * 3, E)` where E is the embedding dimension
        - b: :math:`E * 3` where E is the embedding dimension

        Output:
        - in output list :math:`[q', k', v']`, each output tensor will have the
            same shape as the corresponding input tensor.
    """
    E = q.size(-1)
    if k is v:
        if q is k:
            # self-attention
            return linear(q, w, b).chunk(3, dim=-1)
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (linear(q, w_q, b_q),) + linear(k, w_kv, b_kv).chunk(2, dim=-1)
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _in_projection(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w_q: Tensor,
        w_k: Tensor,
        w_v: Tensor,
        b_q: Optional[Tensor] = None,
        b_k: Optional[Tensor] = None,
        b_v: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Performs the in-projection step of the attention operation. This is simply
    a triple of linear projections, with shape constraints on the weights which
    ensure embedding dimension uniformity in the projected outputs.
    Output is a triple containing projection tensors for query, key and value.

    Args:
        q, k, v: query, key and value tensors to be projected.
        w_q, w_k, w_v: weights for q, k and v, respectively.
        b_q, b_k, b_v: optional biases for q, k and v, respectively.

    Shape:
        Inputs:
        - q: :math:`(Qdims..., Eq)` where Eq is the query embedding dimension and Qdims are any
            number of leading dimensions.
        - k: :math:`(Kdims..., Ek)` where Ek is the key embedding dimension and Kdims are any
            number of leading dimensions.
        - v: :math:`(Vdims..., Ev)` where Ev is the value embedding dimension and Vdims are any
            number of leading dimensions.
        - w_q: :math:`(Eq, Eq)`
        - w_k: :math:`(Eq, Ek)`
        - w_v: :math:`(Eq, Ev)`
        - b_q: :math:`(Eq)`
        - b_k: :math:`(Eq)`
        - b_v: :math:`(Eq)`

        Output: in output triple :math:`(q', k', v')`,
         - q': :math:`[Qdims..., Eq]`
         - k': :math:`[Kdims..., Eq]`
         - v': :math:`[Vdims..., Eq]`

    """
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    assert w_q.shape == (Eq, Eq), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_k.shape == (Eq, Ek), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_v.shape == (Eq, Ev), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


def _mha_shape_check(query: Tensor, key: Tensor, value: Tensor,
                     key_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor], num_heads: int):
    # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
    # and returns if the input is batched or not.
    # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

    # Shape check.
    if query.dim() == 3:
        # Batched Inputs
        is_batched = True
        assert key.dim() == 3 and value.dim() == 3, \
            ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, \
                ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")
        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
    elif query.dim() == 2:
        # Unbatched Inputs
        is_batched = False
        assert key.dim() == 2 and value.dim() == 2, \
            ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
             f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 1, \
                ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                 f" but found {key_padding_mask.dim()}-D tensor instead")

        if attn_mask is not None:
            assert attn_mask.dim() in (2, 3), \
                ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                 f" but found {attn_mask.dim()}-D tensor instead")
            if attn_mask.dim() == 3:
                expected_shape = (num_heads, query.shape[0], key.shape[0])
                assert attn_mask.shape == expected_shape, \
                    (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
    else:
        raise AssertionError(
                f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor")

    return is_batched


def multi_head_attention_forward(query: Tensor, key: Tensor, value: Tensor, embed_dim_to_check: int,
                                 num_heads: int, in_proj_weight: typing.Optional[Tensor],
                                 in_proj_bias: typing.Optional[Tensor], bias_k: typing.Optional[Tensor],
                                 bias_v: typing.Optional[Tensor], add_zero_attn: bool, dropout_p: float,
                                 out_proj_weight: Tensor, out_proj_bias: typing.Optional[Tensor],
                                 training: bool = True, key_padding_mask: typing.Optional[Tensor] = None,
                                 need_weights: bool = True, attn_mask: typing.Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_proj_weight: typing.Optional[Tensor] = None,
                                 k_proj_weight: typing.Optional[Tensor] = None,
                                 v_proj_weight: typing.Optional[Tensor] = None,
                                 static_k: typing.Optional[Tensor] = None,
                                 static_v: typing.Optional[Tensor] = None, average_attn_weights: bool = True):

    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.

    Function from
    https://github.com/pytorch/pytorch/blob/56e40fe054ecb7700142ea9ae7fe37e77800a2da/torch/nn/functional.py#L4716-L5207

    Under the following license:
    Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
    Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
    Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
    Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
    Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
    Copyright (c) 2011-2013 NYU                      (Clement Farabet)
    Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
    Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
    Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

    From Caffe2:

    Copyright (c) 2016-present, Facebook Inc. All rights reserved.

    All contributions by Facebook:
    Copyright (c) 2016 Facebook Inc.

    All contributions by Google:
    Copyright (c) 2015 Google Inc.
    All rights reserved.

    All contributions by Yangqing Jia:
    Copyright (c) 2015 Yangqing Jia
    All rights reserved.

    All contributions by Kakao Brain:
    Copyright 2019-2020 Kakao Brain

    All contributions by Cruise LLC:
    Copyright (c) 2022 Cruise LLC.
    All rights reserved.

    All contributions from Caffe:
    Copyright(c) 2013, 2014, 2015, the respective contributors
    All rights reserved.

    All other contributions:
    Copyright(c) 2015, 2016 the respective contributors
    All rights reserved.

    Caffe2 uses a copyright model similar to Caffe: each contributor holds
    copyright over their contributions to Caffe2. The project versioning records
    all such contribution and copyright details. If a contributor wants to further
    mark their specific copyright on a particular contribution, they should
    indicate their copyright solely in the commit message of the change when it is
    committed.

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

    3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
       and IDIAP Research Institute nor the names of its contributors may be
       used to endorse or promote products derived from this software without
       specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
    LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """
    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
            raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


@no_parameter
def multi_margin_loss(input: Tensor, target: Tensor, p: int = 1, margin: float = 1.0,
                      weight: typing.Optional[Tensor] = None, size_average: typing.Optional[bool] = None,
                      reduce: typing.Optional[bool] = None, reduction: str = "mean"):
    return F.multi_margin_loss(input, target, p, margin, weight, size_average, reduce, reduction)


@no_parameter
def multilabel_margin_loss(input: Tensor, target: Tensor, size_average: typing.Optional[bool] = None,
                           reduce: typing.Optional[bool] = None, reduction: str = "mean"):
    return F.multilabel_margin_loss(input, target, size_average, reduce, reduction)


@no_parameter
def multilabel_soft_margin_loss(input: Tensor, target: Tensor, weight: typing.Optional[Tensor] = None,
                                size_average: typing.Optional[bool] = None, reduce: typing.Optional[bool] = None,
                                reduction: str = "mean"):
    return F.multilabel_soft_margin_loss(input, target, weight, size_average, reduce, reduction)


@no_parameter
def nll_loss(input: Tensor, target: Tensor, weight: typing.Optional[Tensor] = None,
             size_average: typing.Optional[bool] = None, ignore_index: int = -100, reduce: typing.Optional[bool] = None,
             reduction: str = "mean"):
    return F.nll_loss(input, target, weight, size_average, ignore_index, reduce, reduction)


@no_parameter
def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12,
              out: typing.Optional[Tensor] = None):
    return F.normalize(input, p, dim, eps, out)


@no_parameter
def one_hot(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12,
            out: typing.Optional[Tensor] = None):
    return F.one_hot(input, p, dim, eps, out)


@no_parameter
def pad(input, pad, mode="constant", value=None):
    return F.pad(input, pad, mode, value)


@no_parameter
def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    return F.pairwise_distance(x1, x2, p, eps, keepdim)


@no_parameter
def pdist(input, p=2.0):
    return F.pdist(input, p)


@no_parameter
def pixel_shuffle(input: Tensor, upscale_factor: int):
    return F.pixel_shuffle(input, upscale_factor)


@no_parameter
def pixel_unshuffle(input: Tensor, downscale_factor: int):
    return F.pixel_unshuffle(input, downscale_factor)


@no_parameter
def poisson_nll_loss(input: Tensor, target: Tensor, log_input: bool = True, full: bool = False,
                     size_average: typing.Optional[bool] = None, eps: float = 1e-08,
                     reduce: typing.Optional[bool] = None, reduction: str = "mean"):
    return F.poisson_nll_loss(input, target, log_input, full, size_average, eps, reduce, reduction)


@no_parameter
def prelu(input: Tensor, weight: Tensor):
    return F.prelu(input, weight)


@no_parameter
def relu(input: Tensor, inplace: bool = False):
    return F.relu(input, inplace)


@no_parameter
def relu6(input: Tensor, inplace: bool = False):
    return F.relu6(input, inplace)


@no_parameter
def relu_(input: Tensor):
    return F.relu_(input)


@no_parameter
def rrelu(input: Tensor, lower: float = 0.125, upper: float = 1 / 3, training: bool = False,
          inplace: bool = False):
    return F.rrelu(input, lower, upper, training, inplace)


@no_parameter
def rrelu_(input: Tensor, lower: float = 0.125, upper: float = 1 / 3, training: bool = False):
    return F.rrelu_(input, lower, upper, training)


@no_parameter
def selu(input: Tensor, inplace: bool = False):
    return F.selu(input, inplace)


@no_parameter
def selu_(input: Tensor):
    return F.selu_(input)


@no_parameter
def sigmoid(input):
    return F.sigmoid(input)


@no_parameter
def silu(input: Tensor, inplace: bool = False):
    return F.silu(input, inplace)


@no_parameter
def smooth_l1_loss(input: Tensor, target: Tensor, size_average: typing.Optional[bool] = None,
                   reduce: typing.Optional[bool] = None, reduction: str = "mean", beta: float = 1.0):
    return F.smooth_l1_loss(input, target, size_average, reduce, reduction, beta)


@no_parameter
def soft_margin_loss(input: Tensor, target: Tensor, size_average: typing.Optional[bool] = None,
                     reduce: typing.Optional[bool] = None, reduction: str = "mean"):
    return F.soft_margin_loss(input, target, size_average, reduce, reduction)


@no_parameter
def softmax(input: Tensor, dim: typing.Optional[int] = None, _stacklevel: int = 3,
            dtype: typing.Optional[int] = None):
    return F.softmax(input, dim, _stacklevel, dtype)


@no_parameter
def softmin(input: Tensor, dim: typing.Optional[int] = None, _stacklevel: int = 3,
            dtype: typing.Optional[int] = None):
    return F.softmin(input, dim, _stacklevel, dtype)


@no_parameter
def softplus(input: Tensor, dim: typing.Optional[int] = None, _stacklevel: int = 3,
             dtype: typing.Optional[int] = None):
    return F.softplus(input, dim, _stacklevel, dtype)


@no_parameter
def softshrink(input: Tensor, dim: typing.Optional[int] = None, _stacklevel: int = 3,
               dtype: typing.Optional[int] = None):
    return F.softshrink(input, dim, _stacklevel, dtype)


@no_parameter
def softsign(input):
    return F.softsign(input)


@no_parameter
def tanh(input):
    return F.tanh(input)


@no_parameter
def tanhshrink(input):
    return F.tanhshrink(input)


@no_parameter
def threshold(input: Tensor, threshold: float, value: float, inplace: bool = False):
    return F.threshold(input, threshold, value, inplace)


@no_parameter
def threshold_(input: Tensor, threshold: float, value: float):
    return F.threshold_(input, threshold, value)


@no_parameter
def triplet_margin_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0,
                        p: float = 2, eps: float = 1e-06, swap: bool = False,
                        size_average: typing.Optional[bool] = None, reduce: typing.Optional[bool] = None,
                        reduction: str = "mean"):
    return F.triplet_margin_loss(anchor, positive, negative, margin, p, eps, swap, size_average, reduce, reduction)


@no_parameter
def triplet_margin_with_distance_loss(anchor: Tensor, positive: Tensor, negative: Tensor,
                                      distance_function: typing.Union[
                                          typing.Callable[[Tensor, Tensor], Tensor], None] = None,
                                      margin: float = 1.0, swap: bool = False, reduction: str = "mean"):
    return F.triplet_margin_with_distance_loss(anchor, positive, negative, distance_function=distance_function,
                                               margin=margin, swap=swap, reduction=reduction)


@no_parameter
def unfold(input: Tensor, kernel_size, dilation=1, padding=0, stride=1):
    return F.unfold(input, kernel_size, dilation, padding, stride)


@no_parameter
def upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return F.upsample(input, size, scale_factor, mode, align_corners)


@no_parameter
def upsample_bilinear(input, size=None, scale_factor=None):
    return F.upsample_bilinear(input, size, scale_factor)


@no_parameter
def upsample_nearest(input, size=None, scale_factor=None):
    return F.upsample_nearest(input, size, scale_factor)
