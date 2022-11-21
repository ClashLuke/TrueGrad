import torch

from truegrad.nn import TrueGradParameter


def patch_model(model: torch.nn.Module):
    def _apply_fn(module: torch.nn.Module):
        for name, param in module.named_parameters(recurse=False):
            setattr(module, name, TrueGradParameter(param.data))

    model.apply(_apply_fn)
