# TrueGrad

PyTorch interface for TrueGrad-AdamW

## Getting Started

### Installation

```BASH
python3 -m pip install truegrad
```

## Examples

TrueGrad supports various backends, each with their own tradeoffs:

| Name                                               | Advantages                                                                                                                                                                                      | Disadvantages                                                                                                                |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [truegrad.nn](#nn)                                 | * What you see is what you get - Modules not in truegrad.nn and truegrad.nn.functional are not supported<br/>* Custom forward/backward for some fused functions<br/>* Optimized backward passes | * Limited applicability - custom modules can't be used<br/>* Requires code modification                                      |
| [truegrad.utils.patch_torch](#patch-torch)         | * Uses truegrad.nn under the hood<br/>* Works for many (off-the-shelf!) torch models<br/>* No code modification necessary                                                                       | * Uncertainty if model is compatible                                                                                         |
| [backpack](#backpack)                              | * Highest stability<br/>* Loud warnings and errors<br/>* Battle-tested<br/>* Simple to extend further                                                                                           | * High memory usage<br/>* High compute usage<br/>* Sparse support for torch operations                                       |
| [truegrad.utils.patch_model](#patch-custom-models) | * Works with custom models                                                                                                                                                                      | * Fails silently on fused functions<br/>* ~50% to 100% slower than truegrad.nn                                               |
| [patch_torch + patch_model](#Full Patching)        | * Best compatibility<br/>* Reduced overheads compared to `patch_model` (by falling back to faster pre-patched `patch_torch` where available)                                                    | * Fails silently on fused functions outside of torch.nn<br/> * Slower than truegrad.nn when truegrad.nn would've been enough |

Below, you'll find examples for each of these backends, as well as a [general strategy](#partial-truegrad) allowing
partial application of TrueGrad.

### nn

The preferred method of using TrueGrad is by replacing `torch.nn` with performant `truegrad.nn` modules. While other
methods add compute and memory overheads, `truegrad.nn` and `truegrad.nn.functional` have hand-crafted gradients. This
is the most powerful method, although it requires code modifications.

```PYTHON
import torch
from truegrad import nn
from truegrad.optim import TGAdamW

# define model by mixing truegrad.nn and torch.nn
model = torch.nn.Sequential(nn.Linear(1, 10),
                            nn.LayerNorm([1, 10]),
                            torch.nn.ReLU(),
                            nn.Linear(10, 1))
optim = TGAdamW(model.parameters())  # truegrad.optim.TGAdamW instead of torch.optim.AdamW

# standard training loop 
while True:
    input = torch.randn((16, 1))
    model(input).mean().backward()
    optim.step()
    optim.zero_grad()
```

### Patch Torch

In some cases, you can't modify the model's source. For example, when importing models from `torchvision`. If that's the
case, or if you simply want to try out TrueGrad, you can use `truegrad.utils.patch_torch()`, to
replace `torch.nn.Module`'s with `truegrad.nn.Module`'s where possible. For example, the code below can be used to train
a ResNet-18:

```PYTHON
import torch
from torchvision.models import resnet18

from truegrad.optim import TGAdamW
from truegrad.utils import patch_torch

patch_torch()  # call before model creation, otherwise complete freedom
model = resnet18().cuda()
optim = TGAdamW(model.parameters(), lr=1e-7, weight_decay=0)

# constant input/output to overfit
inp = torch.randn((2, 3, 224, 224)).cuda()
tgt = torch.randint(0, 1000, (2,)).cuda()

# standard training loop
i = 0
while True:
    loss = torch.nn.functional.cross_entropy(model(inp), tgt)
    loss.backward()
    optim.step()
    optim.zero_grad()
    i += 1
    if i % 5 == 0:
        print(i, loss.item())
```

Similarly, most huggingface transformers work out of the box:

```PYTHON
import torch
import transformers
from torch.nn import functional as F

from truegrad.optim import TGAdamW
from truegrad.utils import patch_torch

patch_torch()  # only added line to get truegrad statistics for TGAdamW

model = transformers.BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")  # any existing model
tokenizer = transformers.BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

optim = TGAdamW(model.parameters())

# constant input to overfit
input = tokenizer(["Hello World!"], return_tensors="pt")

# training loop as normal
while True:
    out = model(**input)
    loss = F.l1_loss(out[0], torch.ones_like(out[0]))
    loss.backward()
    optim.step()
    optim.zero_grad()
    print(loss.item())
```

Note that this works even though transformers have custom modules, which could cause issues. The key factor is that all
parameters come from `torch.nn.Module`'s, which are patched by `patch_torch()`. Therefore, truegrad handles all
parameter usages. Therefore, any composition of `torch.nn.Module`'s makes for a truegrad-compatible model.

### BackPack

The most stable although also memory hungry method to compute TrueGrad statistics is to use
[BackPack](https://github.com/f-dangel/backpack). BackPack is a third-party library that automatically computes the sum
of gradient squares and works for most models by implementing custom backward rules for many `torch.nn.Module`'s.

```PYTHON
import backpack
import torch
from torch.nn import CrossEntropyLoss
from truegrad.optim import TGAdamW
from torchvision.models import alexnet

model = alexnet()  # BatchNorm and in-place ops (like ResNet's residual path) aren't supported
optim = TGAdamW(model.parameters(), lr=1e-7, weight_decay=0)

# replace inplace ops like nn.ReLU(inplace=True) where possible
for mod in model.modules():
    if hasattr(mod, "inplace"):
        mod.inplace = False

# backpack relies on module-level pytorch hooks
model = backpack.extend(model)
lossfunc = backpack.extend(CrossEntropyLoss())

# constant input/output to overfit
inp = torch.randn((2, 3, 224, 224))
tgt = torch.randint(0, 1000, (2,))

# standard training loop
i = 0
while True:
    # "SumGradSquared" computes the sum of the squared gradient
    with backpack.backpack(backpack.extensions.SumGradSquared()):
        loss = lossfunc(model(inp), tgt)
        loss.backward()
    optim.step()
    optim.zero_grad()
    i += 1
    if i % 5 == 0:
        print(i, loss.item())
```

If you're using custom modules with self-defined parameters, this method will not work. Additionally, note that, if
your model has any layer called `.output` or you're using PyTorch >= 1.13, you will need to install
[BackPack-HF](https://github.com/ClashLuke/backpack-hf) via
`python3 -m pip install git+https://github.com/ClashLuke/backpack-hf`.

### Patch Custom Models

Another option to integrate TrueGrad into existing models is to patch them using `truegrad.utils.patch_model()`.
`patch_model()` will go through all `torch.nn.Module`'s in PyTorch model and convert their `torch.nn.Parameter`'s to
`truegrad.nn.TrueGradParameter`'s. A `TrueGradParameter` acts largely the same as a `torch.nn.Parameter`, but adds
required operations into the model's backward pass. Note that this doesn't give the most effective computation graph,
but works well for many custom models.\
Importantly, be aware that this does not work for fused functions, such as `torch.nn.LayerNorm`
and `torch.nn.MultiheadAttention`. However, unfused functions which directly access a parameter, such as multiplication,
work well. Therefore, torch.nn.Linear and HuggingFace's attention work as expected.

```PYTHON
import torch
from truegrad.optim import TGAdamW
from truegrad.utils import patch_model
from torchvision.models import alexnet

model = alexnet()  # patch_model can't handle fused ops like VGG's and ResNet's BatchNorm
optim = TGAdamW(model.parameters())

# replace inplace ops like nn.ReLU(inplace=True) where possible
for mod in model.modules():
    if hasattr(mod, "inplace"):
        mod.inplace = False

patch_model(model)  # replace torch.nn.Parameter with truegrad.nn.Parameter

# constant input/output to overfit
inp = torch.randn((2, 3, 224, 224))
tgt = torch.randint(0, 1000, (2,))

# standard training loop
i = 0
while True:
    # "SumGradSquared" computes the sum of the squared gradient
    loss = torch.nn.functional.cross_entropy(model(inp), tgt)
    loss.backward()
    optim.step()
    optim.zero_grad()
    i += 1
    if i % 5 == 0:
        print(i, loss.item())
```

### Full Patching

One way of avoiding [truegrad.utils.patch_model](#patch-custom-models)'s downsides when working with off-the-shelf
models containing custom parameters, such as [lucidrains' ViT's](https://github.com/lucidrains/vit-pytorch/) is to also
`patch_torch`. This takes care of many fused functions, such as LayerNorm, while still allowing full flexibility in
model design.

```PYTHON
import torch
from vit_pytorch.levit import LeViT
from truegrad.utils import patch_torch, patch_model
from truegrad.optim import TGAdamW

patch_torch()  # before model instantiation

levit = LeViT(
        image_size=224,
        num_classes=1000,
        stages=3,  # number of stages
        dim=(256, 384, 512),  # dimensions at each stage
        depth=4,  # transformer of depth 4 at each stage
        heads=(4, 6, 8),  # heads at each stage
        mlp_mult=2,
        dropout=0.1
        )

opt = TGAdamW(levit.parameters())

patch_model(levit)  # replace torch.nn.Parameter with truegrad.nn.TrueGradParameter

# constant input to overfit
img = torch.randn(1, 3, 224, 224)

# standard training loop
while True:
    loss = levit(img).square().mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(loss.item())
```

### Partial TrueGrad

Unfortunately, it's not always sensible to apply TrueGrad, as some backward passes are too slow, and sometimes it's
impossible to avoid a fused function.
Therefore, it can be an option to use TGAdamW only on specific subsections of the model. To do so, you can
specify `default_to_adam=True` to TGAdamW. Adding this option allows TGAdamW to fall back to AdamW if there is
no `sum_grad_squared` attribute available.
For example, the code from [#nn](#nn) could be extended in the following way:

```PYTHON
import torch
from truegrad import nn
from truegrad.optim import TGAdamW

model = torch.nn.Sequential(nn.Linear(1, 10),  # Weights coming from truegrad.nn 
                            nn.LayerNorm([1, 10]),
                            torch.nn.ReLU(),
                            torch.nn.Linear(10, 1))  # Weights coming torch.nn

optim = TGAdamW(model.parameters(), default_to_adam=True)

# standard training loop
i = 0
while True:
    input = torch.randn((16, 1))
    loss = model(input).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    i += 1
    if i % 5 == 0:
        print(i, loss.item())
```