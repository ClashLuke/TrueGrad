import torch
from torch.nn import functional as torch_func

from truegrad.nn import TrueGradParameter, functional as tg_func


def patch_model(model: torch.nn.Module, recurse: bool = True):
    def _apply_fn(module: torch.nn.Module):
        for name, param in module.named_parameters(recurse=False):
            setattr(module, name, TrueGradParameter(param.data))

    if recurse:
        model.apply(_apply_fn)
    else:
        for mod in model.children():
            _apply_fn(mod)


def patch_torch():
    tg_dir = dir(tg_func)
    for name in dir(torch_func):
        if name not in tg_dir:
            continue
        setattr(torch, name, getattr(tg_func, name))
