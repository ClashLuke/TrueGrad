import torch

from truegrad import nn
from truegrad.optim import TGAdamW


def test():
    model = torch.nn.Sequential(nn.Linear(1, 10),
                                nn.LayerNorm(10),
                                torch.nn.ReLU(),
                                torch.nn.Linear(10, 1))

    optim = TGAdamW(model.parameters(), default_to_adam=True)

    input = torch.randn((16, 1))
    for i in range(10):
        loss = model(input).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
