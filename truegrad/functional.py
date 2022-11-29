import contextlib
import typing
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F, grad
from torch.utils._pytree import tree_map


# TrueGradParameter


def is_tgparam(param: nn.Parameter):
    if isinstance(param, TrueGradParameter):
        return True
    if isinstance(param, nn.Parameter) and hasattr(param, "activated"):
        return True
    return False


def unpack_tg_param(x: Any) -> Any:
    if is_tgparam(x) and not x.activated:
        x.activated = True
    return x


@contextlib.contextmanager
def activate_tg_params(*tensors):
    for t in tensors:
        unpack_tg_param(t)
    yield
    for t in tensors:
        if is_tgparam(t):
            t.activated = False


_parameter_function = nn.Parameter.__torch_function__


class TrueGradParameter(nn.Parameter):
    activated: bool

    @staticmethod
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.zeros(())
        out = torch.nn.Parameter._make_subclass(cls, data, requires_grad)
        out.activated = False
        return out

    def __repr__(self):
        return f"TrueGradParameter({self.data})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def base(a, k):
            return _parameter_function(func, types, a, k)

        if all(not is_tgparam(a) or a.activated for a in list(args) + list(kwargs.values())):
            return base(args, kwargs)
        with activate_tg_params(*args, *kwargs.values()):
            out = base(tree_map(unpack_tg_param, args), tree_map(unpack_tg_param, kwargs))
        if not isinstance(out, Tensor):
            return out
        return wrap(base, out, args, kwargs)


# TrueGradTensor

def unpack_tg_tensor(x: Any) -> Any:
    if isinstance(x, TrueGradTensor):
        return x.data
    return x


_tensor_function = Tensor.__torch_function__


class TrueGradTensor(Tensor):
    sum_grad_squared: Tensor
    data: Tensor
    requires_grad: bool

    __slots__ = ['sum_grad_squared', "data", "requires_grad"]

    @staticmethod
    def __new__(cls, data: Tensor):
        meta = data.new_empty((0,))
        meta.set_(meta.storage(), 0, data.size(), data.stride())
        r = Tensor._make_subclass(cls, meta, data.requires_grad)
        r.data = data
        r.sum_grad_squared = None
        r.activated = False
        r.requires_grad = data.requires_grad
        return r

    def __repr__(self):
        return f"TrueGradTensor({self.data})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return _tensor_function(func, [], tree_map(unpack_tg_tensor, args), tree_map(unpack_tg_tensor, kwargs))


# utils

def valid_attr(wgt: nn.Parameter, attr: str = "sum_grad_squared"):
    return hasattr(wgt, attr) and getattr(wgt, attr) is not None


def add_or_set(wgt: nn.Parameter, new: torch.Tensor, attr: str = "sum_grad_squared"):
    if valid_attr(wgt, attr):
        new = getattr(wgt, attr) + new
    setattr(wgt, attr, new)


def contiguous(wgt: Any):
    if isinstance(wgt, Tensor):
        return torch.clone(wgt.contiguous())
    return wgt


# Autograd Functions

class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: Tensor, weight: Tensor):
        with activate_tg_params(inp, weight):
            if weight.requires_grad:
                ctx.save_for_backward(inp)
                ctx.weight = weight
            return inp * weight

    @staticmethod
    def backward(ctx, dy: Tensor):
        if not ctx.saved_tensors:
            return None, None
        inp, = ctx.saved_tensors
        weight = ctx.weight
        with activate_tg_params(inp, weight):
            diff = inp.ndim - weight.ndim
            summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
            weight_grad = dy * inp
            sum_grad_squared = weight_grad.square()
            if summed:
                weight_grad = weight_grad.sum(summed)
                sum_grad_squared = sum_grad_squared.sum(summed)
            add_or_set(weight, sum_grad_squared.reshape(weight.size()) * dy.size(0))
        return dy * weight, weight_grad.reshape(weight.size())


class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: Tensor, weight: Tensor):
        with activate_tg_params(inp, weight):
            if weight.requires_grad:
                diff = inp.ndim - weight.ndim
                ctx.summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
                ctx.batch_size = inp.size(0)
                ctx.weight = weight

        return inp + weight

    @staticmethod
    def backward(ctx, dy: Tensor):
        if not hasattr(ctx, "weight"):
            return None, None
        weight = ctx.weight
        with activate_tg_params(weight):
            weight_grad = dy
            sum_grad_squared = dy.square()
            if ctx.summed:
                weight_grad = weight_grad.sum(ctx.summed)
                sum_grad_squared = sum_grad_squared.sum(ctx.summed)
            add_or_set(weight, sum_grad_squared.reshape(weight.size()) * dy.size(0))
        return dy, weight_grad.reshape(weight.size())


class EinsumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spec: str, inp: Tensor, weight: Tensor) -> Tensor:
        with activate_tg_params(inp, weight):
            if weight.requires_grad:
                ctx.save_for_backward(inp, weight)
                ctx.spec = spec
            return torch.einsum(spec, inp, weight).contiguous()

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[None, Tensor, Tensor]:
        if not ctx.saved_tensors:
            return None, None, None
        inp, wgt = ctx.saved_tensors
        with activate_tg_params(inp, wgt):
            inputs, output = ctx.spec.split('->')
            lhs, rhs = inputs.split(',')

            d_wgt = torch.einsum(f'{lhs},{output}->{rhs}', inp, dy).contiguous()
            add_or_set(wgt, torch.einsum(f'{lhs},{output}->{rhs}', inp.square(), dy.square()).contiguous())
            d_inp = torch.einsum(f"{rhs},{output}->{lhs}", wgt, dy).contiguous()

        return None, d_inp, d_wgt


class GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: Tensor, weight: Tensor) -> Tensor:
        with activate_tg_params(inp, weight):
            if weight.requires_grad:
                ctx.save_for_backward(inp)
                ctx.weight = weight
        return torch.gather(weight, 0, inp)

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[None, Tensor]:
        if not ctx.saved_tensors:
            return None, None
        inp, = ctx.saved_tensors
        wgt = ctx.weight
        with activate_tg_params(inp, wgt):
            wgt_grad = torch.zeros_like(wgt)
            add_or_set(wgt, wgt_grad.scatter_add(0, inp, dy.square()))
            wgt_grad.scatter_add_(0, inp, dy)
        return None, wgt_grad


class ReshapeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, new_shape: List[int]) -> Tensor:
        with activate_tg_params(weight):
            out = TrueGradTensor(weight.reshape(new_shape).detach().requires_grad_(True))
            if weight.requires_grad:
                ctx.save_for_backward(weight)
                ctx.out = out
        return out

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[None, Tensor]:
        if not ctx.saved_tensors:
            return None, None
        wgt, = ctx.saved_tensors
        out = ctx.out
        with activate_tg_params(wgt):
            if valid_attr(out):
                add_or_set(wgt, ctx.out.sum_grad_squared.reshape(wgt.size()))
        return dy.reshape(wgt.size()), None


class TransposeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, dims: typing.List[int]) -> Tensor:
        with activate_tg_params(weight):
            out = TrueGradTensor(weight.transpose(*dims).detach().requires_grad_(True))
            if weight.requires_grad:
                ctx.save_for_backward(weight)
                ctx.out = out
                ctx.dims = dims
        return out

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[None, Tensor]:
        if not ctx.saved_tensors:
            return None, None
        wgt, = ctx.saved_tensors
        out = ctx.out
        with activate_tg_params(wgt):
            if valid_attr(out):
                add_or_set(wgt, out.sum_grad_squared.transpose(*ctx.dims()))
        return dy.transpose(*ctx.dims), None


class ChunkFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, chunks: int, dim: int):
        with activate_tg_params(weight):
            out = tuple(TrueGradTensor(c) for c in weight.chunk(chunks, dim))
            if weight.requires_grad:
                ctx.save_for_backward(weight)
                ctx.dim = dim
                ctx.out = out
        return out

    @staticmethod
    def backward(ctx, *dy: Tensor):
        if not ctx.saved_tensors:
            return None, None, None
        wgt, = ctx.saved_tensors
        out = ctx.out
        with activate_tg_params(wgt):
            if all(valid_attr(o) for o in out):
                add_or_set(wgt, torch.cat([o.sum_grad_squared for o in out], dim=ctx.dim))
        return torch.cat(dy, dim=ctx.dim), None, None


class SplitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, split_size: int, dim: int):
        with activate_tg_params(weight):
            out = tuple(TrueGradTensor(c) for c in weight.split(split_size, dim))
            if weight.requires_grad:
                ctx.save_for_backward(weight)
                ctx.dim = dim
                ctx.out = out
        return out

    @staticmethod
    def backward(ctx, *dy: Tensor):
        if not ctx.saved_tensors:
            return None, None, None
        wgt = ctx.saved_tensors
        out = ctx.out
        with activate_tg_params(wgt):
            if all(valid_attr(o) for o in out):
                add_or_set(wgt, torch.cat([o.sum_grad_squared for o in out], dim=ctx.dim))
        return torch.cat(dy, dim=ctx.dim), None, None


class ExpandFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: Tensor, new_shape: List[int]) -> Tensor:
        with activate_tg_params(weight):
            out = TrueGradTensor(weight.expand(new_shape))
            if weight.requires_grad:
                ctx.save_for_backward(weight)
                ctx.summed = [i for i, d in enumerate(new_shape) if d != -1]
                ctx.out = out
        return out

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[None, Tensor]:
        if not ctx.saved_tensors:
            return None, None
        wgt, = ctx.saved_tensors
        out = ctx.out
        with activate_tg_params(wgt):
            if valid_attr(out):
                sum_grad_squared = out.sum_grad_squared
                if ctx.summed:
                    sum_grad_squared = sum_grad_squared.sum(ctx.summed)
                add_or_set(wgt, sum_grad_squared)
        if ctx.summed:
            return dy.sum(ctx.summed)
        return dy


class WrapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, out, args, kwargs) -> Tensor:
        ctx.fn = fn
        ctx.args = args
        ctx.kwargs = kwargs
        return tree_map(contiguous, out)

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[None, Tensor, None, None]:
        def _fn(x: Tensor):
            if isinstance(x, nn.Parameter):
                x = x.data
            if not isinstance(x, Tensor) or not torch.is_floating_point(x):
                return x
            x = torch.square(x.detach()).detach().contiguous()
            x.requires_grad_(True)
            return x

        with activate_tg_params(*ctx.args, *ctx.kwargs.values()):
            args = tree_map(_fn, ctx.args)
            kwargs = tree_map(_fn, ctx.kwargs)

            with torch.enable_grad():
                out = ctx.fn(args, kwargs)
                torch.autograd.backward(out, tree_map(_fn, dy))

            for p, a in zip(list(ctx.args) + list(ctx.kwargs.values()), list(args) + list(kwargs.values())):
                if valid_attr(a, "grad"):
                    add_or_set(p, a.contiguous())

        return None, dy, None, None


class ConvNdFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor], args) -> Tensor:
        with activate_tg_params(input, weight, bias):
            if weight.requires_grad:
                ctx.save_for_backward(input, weight, bias)
                ctx.args = args
            dim = input.dim() - 2  # Batch, Feature, *Data
            return getattr(F, f"conv{dim}d")(input, weight, bias, *args).contiguous()

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor], None]:
        if not ctx.saved_tensors:
            return None, None, None, None
        inp, wgt, bias = ctx.saved_tensors
        with activate_tg_params(inp, wgt, bias):
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


# "Normal" Functions

mul = MulFn.apply
add = AddFn.apply
einsum = EinsumFn.apply
gather = GatherFn.apply
reshape = ReshapeFn.apply
transpose = TransposeFn.apply
chunk = ChunkFn.apply
split = SplitFn.apply
expand = ExpandFn.apply
wrap = WrapFn.apply
convnd = ConvNdFn.apply


def matmul(inp: Tensor, wgt: Tensor):
    batch_dims = ''.join(chr(ord('a') + i) for i in range(inp.ndim - 1))
    return einsum(f"{batch_dims}y,yz->{batch_dims}z", inp, wgt)


def simple_wrap(function: Callable, out: torch.Tensor, *args, **kwargs):
    def _fn(a, k):
        return function(*a, **k)

    _fn.abc = function.__name__
    return wrap(_fn, out, args, kwargs)
