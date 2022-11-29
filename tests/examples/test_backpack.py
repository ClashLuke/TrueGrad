import backpack
import torch
from torch.nn import CrossEntropyLoss
from torchvision.models import alexnet

from truegrad.optim import TGAdamW


def test():
    model = alexnet()
    optim = TGAdamW(model.parameters(), lr=1e-7, weight_decay=0)

    for mod in model.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False

    model = backpack.extend(model)
    lossfunc = backpack.extend(CrossEntropyLoss())

    inp = torch.randn((2, 3, 224, 224))
    tgt = torch.randint(0, 1000, (2,))

    for i in range(10):
        with backpack.backpack(backpack.extensions.SumGradSquared()):
            loss = lossfunc(model(inp), tgt)
            loss.backward()
        optim.step()
        optim.zero_grad()
