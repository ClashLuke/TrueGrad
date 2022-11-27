import torch

import truegrad
from truegrad.nn import TrueGradParameter


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
    tg_dir = dir(truegrad.nn)
    for name in dir(torch.nn):
        if name not in tg_dir:
            continue
        setattr(torch.nn, name, getattr(truegrad.nn, name))
