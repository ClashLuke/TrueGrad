# TrueGrad

PyTorch interface for TrueGrad-AdamW

## Getting Started

### Installation

```BASH
python3 -m pip install truegrad
```

## Examples

### BackPack

The preferred method to integrate TrueGrad is using [BackPack](https://github.com/f-dangel/backpack). BackPack is a
third-party library that automatically computes the sum of gradient squares and works for most models by implementing
custom backward rules for many `torch.nn.Module`'s.

```PYTHON
import backpack
import torch
from torch.nn import CrossEntropyLoss
from truegrad.optim import TGAdamW
from torchvision.models import alexnet

model = alexnet()
optim = TGAdamW(model.parameters(), lr=1e-7, weight_decay=0)

# backpack can't handle inplace ops like nn.ReLU(inplace=True) and `x += y`
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
`patch_model()` will go through all`torch.nn.Module`'s in PyTorch model and convert their `torch.nn.Parameter`'s to
`truegrad.nn.TrueGradParameter`'s. A `TrueGradParameter` acts largely the same as a `torch.nn.Parameter`, but adds
required operations into the model's backward pass.\
Importantly, be aware that this does not work for fused functions, such as `torch.nn.LayerNorm`
and `torch.nn.MultiheadAttention`. However, unfused functions which directly access a parameter, such as multiplication
and work well. Therefore, torch.nn.Linear and HuggingFace's attention work as expected.

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

### nn

Patching existing PyTorch computation graphs on the fly might add unnecessary memory and computation or even fail
unexpectedly. That's why a pre-patched alternative of `torch.nn` with hand-crafted gradients exists alongside the
`truegrad.utils` module. Compared to `truegrad.utils.patch_model()`, `truegrad.nn` offers higher speeds and lower
memory usage, although it might require code alterations and doesn't support all models. You cannot (currently) use
`truegrad.nn` with `truegrad.utils`, as both use different ways to arrive at the same value. However, you can
combine `torch.nn.Modules` and `truegrad.nn.Modules` and use the truegrad information only where it is available (
see [Partial TrueGrad](#Partial-TrueGrad)).

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
while True:
    input = torch.randn((16, 1))
    model(input).mean().backward()
    optim.step()
```