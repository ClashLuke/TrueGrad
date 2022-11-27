import typing
from typing import Any, Callable, List, Tuple

import torch
from torch.utils._pytree import tree_map


def _unpack(x: Any) -> Any:
    if isinstance(x, TrueGradTensor):
        return x.data
    return x


_base_torch_function = torch.Tensor.__torch_function__


class TrueGradTensor(torch.Tensor):
    sum_grad_squared: torch.Tensor
    data: torch.Tensor
    requires_grad: bool

    __slots__ = ['sum_grad_squared', "data", "requires_grad"]

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        meta = data.new_empty((0,))
        meta.set_(meta.storage(), 0, data.size(), data.stride())
        r = torch.Tensor._make_subclass(cls, meta, data.requires_grad)
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
        out = _base_torch_function(func, [], tree_map(_unpack, args), tree_map(_unpack, kwargs))
        return out


class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor):
        if weight.requires_grad:
            ctx.save_for_backward(inp)
            ctx.weight = weight
        return inp * weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None
        inp, = ctx.saved_tensors
        weight = ctx.weight
        diff = inp.ndim - weight.ndim
        summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
        weight_grad = dy * inp
        weight.sum_grad_squared = weight_grad.square()
        if summed:
            weight_grad = weight_grad.sum(summed)
            weight.sum_grad_squared = weight.sum_grad_squared.sum(summed)
        weight.sum_grad_squared = weight.sum_grad_squared.reshape(weight.size()) * dy.size(0)
        return dy * weight, weight_grad.reshape(weight.size())


class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor):
        if weight.requires_grad:
            diff = inp.ndim - weight.ndim
            ctx.summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
            ctx.batch_size = inp.size(0)
            ctx.weight = weight

        return inp + weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not hasattr(ctx, "weight"):
            return None, None
        weight = ctx.weight
        weight_grad = dy
        weight.sum_grad_squared = dy.square()
        if ctx.summed:
            weight_grad = weight_grad.sum(ctx.summed)
            weight.sum_grad_squared = weight.sum_grad_squared.sum(ctx.summed)
        weight.sum_grad_squared = weight.sum_grad_squared.reshape(weight.size()) * dy.size(0)
        return dy, weight_grad.reshape(weight.size())


class EinsumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spec: str, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(inp)
            ctx.weight = weight
            ctx.spec = spec
        return torch.einsum(spec, inp, weight)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None, None
        inp, = ctx.saved_tensors
        wgt = ctx.weight
        inputs, output = ctx.spec.split('->')
        lhs, rhs = inputs.split(',')

        d_wgt = torch.einsum(f'{lhs},{output}->{rhs}', inp, dy)
        wgt.sum_grad_squared = torch.einsum(f'{lhs},{output}->{rhs}', inp.square(), dy.square() * inp.size(0))
        d_inp = torch.einsum(f"{rhs},{output}->{lhs}", wgt, dy)
        return None, d_inp, d_wgt


class GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(inp)
            ctx.weight = weight
        return torch.gather(weight, 0, inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None
        inp, = ctx.saved_tensors
        wgt = ctx.weight
        wgt_grad = torch.zeros_like(wgt)
        wgt.sum_grad_squared = wgt_grad.scatter_add(0, inp, dy.square())
        wgt_grad.scatter_add_(0, inp, dy)
        return None, wgt_grad


class ReshapeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, new_shape: List[int]) -> torch.Tensor:
        out = TrueGradTensor(weight.reshape(new_shape).detach().requires_grad_(True))
        if weight.requires_grad:
            ctx.save_for_backward(weight)
            ctx.out = out
            ctx.original_shape = weight.size()
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None
        wgt, = ctx.saved_tensors
        if ctx.out.sum_grad_squared is not None:
            wgt.sum_grad_squared = ctx.out.sum_grad_squared.reshape(ctx.original_shape)
        return dy.reshape(ctx.original_shape), None


class TransposeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, dims: typing.List[int]) -> torch.Tensor:
        out = TrueGradTensor(weight.transpose(*dims).detach().requires_grad_(True))
        if weight.requires_grad:
            ctx.save_for_backward(weight)
            ctx.out = out
            ctx.dims = dims
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None
        wgt, = ctx.saved_tensors
        if ctx.out.sum_grad_squared is not None:
            wgt.sum_grad_squared = ctx.out.sum_grad_squared.transpose(*ctx.dims)
        return dy.transpose(*ctx.dims), None


class ChunkFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, chunks: int, dim: int):
        out = tuple(TrueGradTensor(c) for c in weight.chunk(chunks, dim))
        if weight.requires_grad:
            ctx.save_for_backward(weight)
            ctx.out = out
            ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, *dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None, None
        wgt, = ctx.saved_tensors
        wgt.sum_grad_squared = torch.cat([o.sum_grad_squared for o in ctx.out], dim=ctx.dim)
        return torch.cat(dy, dim=ctx.dim), None, None


class SplitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, split_size: int, dim: int):
        out = tuple(TrueGradTensor(c) for c in weight.split(split_size, dim))
        if weight.requires_grad:
            ctx.save_for_backward(weight)
            ctx.out = out
            ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, *dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None, None
        wgt, = ctx.saved_tensors
        wgt.sum_grad_squared = torch.cat([o.sum_grad_squared for o in ctx.out], dim=ctx.dim)
        return torch.cat(dy, dim=ctx.dim), None, None


class ExpandFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, new_shape: List[int]) -> torch.Tensor:
        out = TrueGradTensor(weight.expand(new_shape))
        if weight.requires_grad:
            ctx.save_for_backward(weight)
            ctx.out = out
            ctx.summed = [i for i, d in enumerate(new_shape) if d != -1]
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None
        wgt, = ctx.saved_tensors
        if ctx.out.sum_grad_squared is not None and ctx.summed:
            wgt.sum_grad_squared = ctx.out.sum_grad_squared.sum(ctx.summed)
        return dy.sum(ctx.summed)


class WrapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, args, kwargs) -> torch.Tensor:
        ctx.fn = fn
        ctx.args = args
        ctx.kwargs = kwargs
        return fn(*args, **kwargs)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, None, None, None]:
        def _backward(fn: Callable[[torch.Tensor], torch.Tensor], attr: str):
            def _fn(x: torch.Tensor):
                if isinstance(x, torch.nn.Parameter):
                    x = x.data
                if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
                    return x
                x = fn(x.detach())
                x.requires_grad_(True)
                return x

            args = tree_map(_fn, ctx.args)
            kwargs = tree_map(_fn, ctx.kwargs)

            with torch.enable_grad():
                out = ctx.fn(args, kwargs)
                torch.autograd.backward(out, tree_map(_fn, dy))

            for p, a in zip(list(ctx.args) + list(ctx.kwargs.values()), list(args) + list(kwargs.values())):
                if not isinstance(p, torch.nn.Parameter):
                    continue
                if hasattr(p, attr) and getattr(p, attr) is not None:
                    a.grad = getattr(p, attr) + a.grad
                setattr(p, attr, a.grad)

        _backward(torch.square, "sum_grad_squared")
        _backward(lambda x: x, "grad")

        return None, None, None, None


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


def matmul(inp: torch.Tensor, wgt: torch.Tensor):
    batch_dims = ''.join(chr(ord('a') + i) for i in range(inp.ndim - 1))
    return einsum(f"{batch_dims}y,yz->{batch_dims}z", inp, wgt)
