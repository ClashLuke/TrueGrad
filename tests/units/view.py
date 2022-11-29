import torch

from truegrad.utils import patch_torch

patch_torch()


def test():
    embd = torch.nn.Embedding(16, 32)
    src = torch.randint(0, 15, (2, 2))
    inp = torch.randn((128,))
    inp += embd(src.view(4)).view(-1)
    inp.mean().backward()
