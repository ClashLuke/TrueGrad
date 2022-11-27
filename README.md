# TrueGrad

PyTorch interface for TrueGrad-AdamW

## Getting Started

### Installation

```BASH
python3 -m pip install truegrad
```

## Examples

TrueGrad supports various backends, each with their own tradeoffs:

| Name                                               | Advantages                                                                                                                                                                                      | Disadvantages                                                                           |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| [truegrad.nn](#nn)                                 | * What you see is what you get - Modules not in truegrad.nn and truegrad.nn.functional are not supported<br/>* Custom forward/backward for some fused functions<br/>* Optimized backward passes | * Limited applicability - custom modules can't be used<br/>* Requires code modification |
| [truegrad.utils.patch_torch](#patch-torch)         | * Uses truegrad.nn under the hood<br/>* Works for many (off-the-shelf!) torch models<br/>* No code modification necessary                                                                       | * Uncertainty if model is compatible                                                    |
| [backpack](#backpack)                              | * Highest stability<br/>* Loud warnings and errors<br/>* Battle-tested<br/>* Simple to extend further                                                                                           | * High memory usage<br/>* High compute usage<br/>* Sparse support for torch operations  |
| [truegrad.utils.patch_model](#patch-custom-models) | * Best compatibility                                                                                                                                                                            | * Fails silently on fused functions<br/>* More costly than truegrad.nn                  |

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
    i += 1
    if i % 5 == 0:
        print(i, loss.item())
```

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
import transformers
from truegrad.utils import patch_model
from truegrad.optim import TGAdamW

model = transformers.BertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")  # any existing model
tokenizer = transformers.BertTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

patch_model(model)  # replace torch.nn.Parameter with truegrad.nn.Parameter
optim = TGAdamW(model.parameters())  # truegrad.optim.TGAdamW instead of torch.optim.AdamW

# training loop as normal
for sample in ["Hello", "World", "!"]:
    out = model(**tokenizer([sample], return_tensors="pt"))
    out[0].mean().backward()
    optim.step()
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
    i += 1
    if i % 5 == 0:
        print(i, loss.item())
```