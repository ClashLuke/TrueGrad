import inspect
import typing
from typing import Callable, Optional, Union

from torch import Tensor, nn
from torch.nn import functional as F


def no_parameter(fn: Callable):
    signature = inspect.signature(fn)

    def _fn(*args, **kwargs):
        for arg, (param, _) in zip(args, signature.parameters.items()):
            kwargs[param] = arg
        for k, v in kwargs.items():
            if isinstance(v, nn.Parameter):
                raise ValueError(f"Function does not support parameters. Offending argument: {k}")
        fn(**kwargs)

    return _fn


# pooling

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


@no_parameter
def batch_norm(input: Tensor, running_mean: typing.Optional[Tensor],
               running_var: typing.Optional[Tensor], weight: typing.Optional[Tensor] = None,
               bias: typing.Optional[Tensor] = None, training: bool = False, momentum: float = 0.1,
               eps: float = 1e-05):
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)


@no_parameter
def bilinear(input1: Tensor, input2: Tensor, weight: typing.Optional[Tensor] = None,
             bias: typing.Optional[Tensor] = None):
    return F.bilinear(input1, input2, weight, bias)


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


@no_parameter
def conv1d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1,
           padding: Union[str, int] = "valid",
           dilation: int = 1, groups: int = 1):
    return F.conv1d(input, weight, bias, stride, padding, dilation, groups)


@no_parameter
def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1,
           padding: Union[str, int] = "valid",
           dilation: int = 1, groups: int = 1):
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


@no_parameter
def conv3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: int = 1,
           padding: Union[str, int] = "valid",
           dilation: int = 1, groups: int = 1):
    return F.conv3d(input, weight, bias, stride, padding, dilation, groups)


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


@no_parameter
def group_norm(input: Tensor, num_groups: int, weight: typing.Optional[Tensor] = None,
               bias: typing.Optional[Tensor] = None, eps: float = 1e-05):
    return F.group_norm(input, num_groups, weight, bias, eps)


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


@no_parameter
def instance_norm(input: Tensor, running_mean: typing.Optional[Tensor] = None,
                  running_var: typing.Optional[Tensor] = None, weight: typing.Optional[Tensor] = None,
                  bias: typing.Optional[Tensor] = None, use_input_stats: bool = True, momentum: float = 0.1,
                  eps: float = 1e-05):
    return F.instance_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)


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


@no_parameter
def layer_norm(input: Tensor, normalized_shape: typing.List[int], weight: typing.Optional[Tensor] = None,
               bias: typing.Optional[Tensor] = None, eps: float = 1e-05):
    return F.layer_norm(input, normalized_shape, weight, bias, eps)


@no_parameter
def leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False):
    return F.leaky_relu(input, negative_slope, inplace)


@no_parameter
def leaky_relu_(input: Tensor, negative_slope: float = 0.01):
    return F.leaky_relu_(input, negative_slope)


@no_parameter
def linear(input: Tensor, weight: Tensor, bias: Tensor):
    return F.linear(input, weight, bias)


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


@no_parameter
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
    return F.multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight,
                                          in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                                          out_proj_bias, training, key_padding_mask, need_weights, attn_mask,
                                          use_separate_proj_weight, q_proj_weight, k_proj_weight, v_proj_weight,
                                          static_k, static_v, average_attn_weights)


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
