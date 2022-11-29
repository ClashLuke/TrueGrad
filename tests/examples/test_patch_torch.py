import torch
import transformers
from torch.nn import functional as F
from torchvision.models import resnet18

from truegrad.optim import TGAdamW
from truegrad.utils import patch_torch

patch_torch()  # call before model creation, otherwise complete freedom


def test_resnet():
    model = resnet18()
    optim = TGAdamW(model.parameters())

    inp = torch.randn((2, 3, 224, 224))
    tgt = torch.randint(0, 1000, (2,))

    torch.autograd.set_detect_anomaly(True)

    for i in range(10):
        loss = F.cross_entropy(model(inp), tgt)
        loss.backward()
        optim.step()
        optim.zero_grad()


def test_transformer():

    model = transformers.BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")  # any existing model
    tokenizer = transformers.BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

    optim = TGAdamW(model.parameters())

    input = tokenizer(["Hello World!"], return_tensors="pt")

    for i in range(10):
        out = model(**input)
        loss = F.l1_loss(out[0], torch.ones_like(out[0]))
        loss.backward()
        optim.step()
        optim.zero_grad()
