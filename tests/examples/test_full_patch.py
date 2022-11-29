import torch
from vit_pytorch.levit import LeViT

from truegrad.optim import TGAdamW
from truegrad.utils import patch_model, patch_torch


def test():
    patch_torch()

    levit = LeViT(
            image_size=224,
            num_classes=1000,
            stages=3,
            dim=(256, 384, 512),
            depth=4,
            heads=(4, 6, 8),
            mlp_mult=2,
            dropout=0.1
            )

    opt = TGAdamW(levit.parameters())

    patch_model(levit)

    img = torch.randn(1, 3, 224, 224)

    for i in range(10):
        loss = levit(img).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
