from typing import List, Tuple

import torch


class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor):
        if not inp.ndim == weight.ndim:
            raise ValueError(f"{inp.ndim=}  !=  {weight.ndim=}")
        if weight.requires_grad:
            ctx.save_for_backward(inp, weight)
        return inp * weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None
        inp, weight = ctx.saved_tensors
        summed = [1] * (inp.ndim - weight.ndim) + [i for i, dim in enumerate(weight.shape) if dim == 1]
        dy_inp = dy * inp
        weight_grad = dy_inp.sum(summed).reshape(weight.size())
        weight.square_grad = dy_inp.square().sum(summed).reshape(weight.size()) * inp.size(0)
        return dy * weight, weight_grad


class AddFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor):
        if not inp.ndim == weight.ndim:
            raise ValueError(f"{inp.ndim=}  !=  {weight.ndim=}")
        if weight.requires_grad:
            ctx.summed = [1] * (inp.ndim - weight.ndim) + [i for i, dim in enumerate(weight.shape) if dim == 1]
            ctx.batch_size = inp.size(0)
            ctx.save_for_backward(weight)
        return inp + weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None
        weight, = ctx.saved_tensors
        weight_grad = dy.sum(ctx.summed).reshape(weight.size())
        weight.square_grad = dy.square().sum(ctx.summed).reshape(weight.size()) * ctx.batch_size
        return dy, weight_grad


class MatMulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if weight.requires_grad:
            ctx.save_for_backward(inp, weight)
        return inp @ weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not ctx.saved_tensors:
            return None, None
        inp, wgt = ctx.saved_tensors
        lhs = ''.join(chr(ord('a') + i) for i in range(dy.ndim - 1))
        d_wgt = torch.einsum(f"{lhs}y,{lhs}z->yz", inp, dy)
        d_wgt_sq = torch.einsum(f"{lhs}y,{lhs}z->yz", inp.square(), dy.square() * inp.size(0))  # * size since mean
        wgt.square_grad = d_wgt_sq
        d_inp = torch.einsum(f"{lhs}z,yz->{lhs}y", dy, wgt)
        return d_inp, d_wgt


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
        if hasattr(wgt, "square_grad"):
            wgt.square_grad = wgt.square_grad.sum(ctx.summed)
        return dy.sum(ctx.summed)


mul = MulFn.apply
add = AddFn.apply
matmul = MatMulFn.apply
gather = GatherFn.apply
reshape = ReshapeFn.apply
expand = ExpandFn.apply
