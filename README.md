# TrueGrad

PyTorch interface for TrueGrad-AdamW

## Getting Started

### Installation

```BASH
python3 -m pip install truegrad
```

## Examples

### Patch Custom Models

The easiest way to integrate TrueGrad into existing models is to patch them using `truegrad.utils.patch_model()`.
`patch_model()` will go through all`torch.nn.Module`'s in PyTorch model and convert their `torch.nn.Parameter`'s to
`truegrad.nn.TrueGradParameter`'s. A `TrueGradParameter` acts largely the same as a `torch.nn.Parameter`, but adds
required operations into the model's backward pass.\
Patching an existing

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

Patching existing PyTorch computation graphs on the fly might add unnecessary memory and computation. That's why a
pre-patched alternative of `torch.nn` with hand-crafted gradients exists alongside the `truegrad.utils` module. Compared
to `truegrad.utils.patch_model()`, `truegrad.nn` offers higher speeds and lower memory usage, although it might require
code alterations and doesn't support all models. You cannot (currently) use `truegrad.nn` with `truegrad.utils`, as both
use different ways to arrive at the same value.

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

# training loop as normal
while True:
    input = torch.randn((16, 1))
    model(input).mean().backward()
    optim.step()
```

### Partial TrueGrad

Unfortunately, it's not always sensible to apply TrueGrad, as some backward passes are too slow to do them twice.
Therefore, it can be an option to use TGAdamW only on specific subsections of the model. To do so, you can either check
which parameters are of type `truegrad.nn.TrueGradParameter` when using `truegrad.utils.patch_model()` or which
parameters belong to a module listed in `truegrad.nn.modules`.
For example, the code from [#nn](#nn) could be extended in the following way:

```PYTHON
import torch
from truegrad import nn
from truegrad.optim import TGAdamW

model = torch.nn.Sequential(nn.Linear(1, 10),  # Weights coming from truegrad.nn 
                            nn.LayerNorm([1, 10]),
                            torch.nn.ReLU(),
                            torch.nn.Linear(10, 1))  # Weights coming torch.nn

truegrad_parameters = []
normal_parameters = []


def get_parameters(mod: torch.nn.Module):
    if isinstance(mod, nn.modules):
        truegrad_parameters.extend(list(mod.parameters(recurse=False)))
    else:
        # you could do truegrad.utils.patch_model(mod, recurse=False) here!
        normal_parameters.extend(list(mod.parameters(recurse=False)))


model = model.apply(get_parameters)

optim0 = TGAdamW(truegrad_parameters)
optim1 = torch.optim.AdamW(normal_parameters)

while True:
    input = torch.randn((16, 1))
    model(input).mean().backward()
    optim0.step()  # update both parameter sets separately
    optim1.step()
```