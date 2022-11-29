import torch

from truegrad import nn
from truegrad.optim import TGAdamW


def test():
    model = torch.nn.Sequential(nn.Linear(1, 10),
                                nn.LayerNorm(10),
                                torch.nn.ReLU(),
                                nn.Linear(10, 1))
    optim = TGAdamW(model.parameters())

    torch.autograd.set_detect_anomaly(True)

    for i in range(10):
        input = torch.randn((16, 1))
        model(input).mean().backward()
        optim.step()
        optim.zero_grad()
