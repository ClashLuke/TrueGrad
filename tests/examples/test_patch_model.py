import torch
from torchvision.models import alexnet

from truegrad.optim import TGAdamW
from truegrad.utils import patch_model


def test():
    model = alexnet()
    optim = TGAdamW(model.parameters())

    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False

    patch_model(model)

    inp = torch.randn((2, 3, 224, 224))
    tgt = torch.randint(0, 1000, (2,))

    for i in range(10):
        loss = torch.nn.functional.cross_entropy(model(inp), tgt)
        loss.backward()
        optim.step()
        optim.zero_grad()
