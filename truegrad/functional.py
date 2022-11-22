from typing import List, Tuple

import torch


class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor):
        if weight.requires_grad:
            ctx.save_for_backward(inp, weight)
        return inp * weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None
        inp, weight = ctx.saved_tensors
        diff = inp.ndim - weight.ndim
        summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
        weight_grad = dy * inp
        weight.square_grad = weight_grad.square()
        if summed:
            weight_grad = weight_grad.sum(summed)
            weight.square_grad = weight.square_grad.sum(summed)
        weight.square_grad = weight.square_grad.reshape(weight.size()) * dy.size(0)
        return dy * weight, weight_grad.reshape(weight.size())


class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor):
        if weight.requires_grad:
            diff = inp.ndim - weight.ndim
            ctx.summed = list(range(diff)) + [i for i, dim in enumerate(weight.shape, diff) if dim == 1]
            ctx.batch_size = inp.size(0)
            ctx.save_for_backward(weight)
        return inp + weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None
        weight, = ctx.saved_tensors
        weight_grad = dy
        weight.square_grad = dy.square()
        if ctx.summed:
            weight_grad = weight_grad.sum(ctx.summed)
            weight.square_grad = weight.square_grad.sum(ctx.summed)
        weight.square_grad = weight.square_grad.reshape(weight.size()) * dy.size(0)
        return dy, weight_grad.reshape(weight.size())


class EinsumFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spec: str, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(inp, weight)
            ctx.spec = spec
        return torch.einsum(spec, inp, weight)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None, None
        inp, wgt = ctx.saved_tensors
        inputs, output = ctx.spec.split('->')
        lhs, rhs = inputs.split(',')

        d_wgt = torch.einsum(f'{lhs},{output}->{rhs}', inp, dy)
        wgt.square_grad = torch.einsum(f'{lhs},{output}->{rhs}', inp.square(), dy.square() * inp.size(0))
        d_inp = torch.einsum(f"{rhs},{output}->{lhs}", wgt, dy)
        return None, d_inp, d_wgt


class GatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(inp, weight)
        return torch.gather(weight, 0, inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None
        inp, wgt = ctx.saved_tensors
        wgt_grad = torch.zeros_like(wgt)
        wgt.square_grad = wgt_grad.scatter_add(0, inp, dy.square())
        wgt_grad.scatter_add_(0, inp, dy)
        return None, wgt_grad


class ReshapeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, new_shape: List[int]) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(weight)
            ctx.original_shape = weight.size()
        return weight.reshape(new_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor]:
        if not ctx.saved_tensors:
            return None
        wgt, = ctx.saved_tensors
        if hasattr(wgt, "square_grad"):
            wgt.square_grad = wgt.square_grad.reshape(ctx.original_shape)
        return dy.reshape(ctx.original_shape)


class ExpandFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, new_shape: List[int]) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(weight)
            ctx.summed = [i for i, d in enumerate(new_shape) if d != -1]
        return weight.reshape(new_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[None, torch.Tensor]:
        if not ctx.saved_tensors:
            return None
        wgt, = ctx.saved_tensors
        if hasattr(wgt, "square_grad") and ctx.summed:
            wgt.square_grad = wgt.square_grad.sum(ctx.summed)
        return dy.sum(ctx.summed)


mul = MulFn.apply
add = AddFn.apply
einsum = EinsumFn.apply
gather = GatherFn.apply
reshape = ReshapeFn.apply
expand = ExpandFn.apply


def matmul(inp: torch.Tensor, wgt: torch.Tensor):
    batch_dims = ''.join(chr(ord('a') + i) for i in range(inp.ndim - 1))
    return einsum(f"{batch_dims}y,yz->{batch_dims}z", inp, wgt)
