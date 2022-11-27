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


def _patch(tg, th):
    tg_dir = dir(tg)
    for name in dir(th):
        if name not in tg_dir:
            continue
        item = getattr(tg, name)
        if not hasattr(item, "__module__"):
            continue
        if item.__module__ != tg.__name__:
            continue
        setattr(th, name, item)


def patch_torch():
    _patch(truegrad.nn.functional, torch.nn.functional)
    _patch(truegrad.nn.functional, torch)
    _patch(truegrad.nn, torch.nn)
