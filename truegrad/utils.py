import torch

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
