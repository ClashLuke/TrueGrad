import collections
import typing

import torch
from torch import overrides

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


def from_x(name: str, fn: typing.Callable, module):
    calls = [0]
    original = getattr(module, name)

    def _fn(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            try:
                return fn(*args, **kwargs)
            except:
                return original(*args, **kwargs)
            finally:
                calls[0] -= 1
        out = original(*args, **kwargs)
        calls[0] -= 1
        return out

    return _fn


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
        setattr(th, name, from_x(name, item, th))


def patch_torch():
    _patch(truegrad.nn.functional, torch.nn.functional)
    _patch(truegrad.nn.functional, torch)
    _patch(truegrad.nn, torch.nn)
    overrides.has_torch_function_variadic = lambda *x: False
