import warnings
from typing import Tuple, Union, List, Dict, Any, Optional

import torch
from torch import Tensor
from torch.nn import Parameter


def ema_(base: Tensor, update: Tensor, beta: float, step: Optional[int] = None):
    base.mul_(beta).add_(update, alpha=1 - beta)
    if step is None:
        return base
    return base / (1 - beta ** step)


def stable_sqrt(base: Tensor, eps: float):
    return base.sqrt().clamp(min=eps)


def div_ema(base: Tensor, eps: float, base_sq: Tensor, update_sq: Tensor, beta_sq: float, step: Optional[int] = None):
    return base / stable_sqrt(ema_(base_sq, update_sq, beta_sq, step), eps)


def decay_weight_(state: Dict[str, Any], param: torch.nn.Parameter, group: Dict[str, Any]):
    if group["decay_to_init"]:
        if "param_at_init" not in state:
            state["param_at_init"] = torch.clone(param.detach())
        else:
            param.add_(state["param_at_init"] - param, alpha=group["weight_decay"] * group["lr"])
    else:
        param.mul_(1 - group["weight_decay"] * group["lr"])


class OptimizerOptimizer(torch.optim.Optimizer):
    def __init__(self, params, inner_optimizer: torch.optim.Optimizer, learning_rate_learning_rate: float = 1,
                 weight_decay: float = 0, decay_to_init: bool = False):
        self.learning_rate_learning_rate = learning_rate_learning_rate

        self.inner_optimizer = inner_optimizer
        param_groups = self.inner_optimizer.param_groups
        self.inner_optimizer.param_groups = []
        for group in param_groups:
            for param in group["params"]:
                group = {k: v for k, v in group.items() if k != "params"}
                group["params"] = [param]
                self.inner_optimizer.param_groups.append(group)

        super(OptimizerOptimizer, self).__init__(params, {"weight_decay": weight_decay, "decay_to_init": decay_to_init})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if "lr" in state:
                    group["lr"] = state["lr"]
                    decay_weight_(state, p, group)
                state["param"] = torch.clone(p.detach())

        self.inner_optimizer.step()

        for group in self.inner_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "param" in state:
                    neg_update = state["param"].double() - p.double()
                    dims = ''.join(chr(ord('a') + i) for i in range(neg_update.ndim))
                    lr_grad = torch.einsum(f"{dims},{dims}->", neg_update, p.grad.double())
                    state["lr"] = group["lr"] = group["lr"] + lr_grad.item() * self.learning_rate_learning_rate
                state["param"] = None

        return loss


class Sign(torch.optim.Optimizer):
    def __init__(self, params, base: torch.optim.Optimizer, lr: float = 1, weight_decay: float = 0,
                 decay_to_init: bool = False, eps: float = 1e-12, graft_to_self: bool = True):
        super().__init__(params, {"weight_decay": weight_decay, "decay_to_init": decay_to_init, "lr": lr, "eps": eps,
                                  "graft_to_self": graft_to_self})
        self.base = base

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        params_flat = []
        for group in self.param_groups:
            for p in group["params"]:
                params_flat.append(p)
                decay_weight_(self.state[p], p, group)

        params_flat = [torch.clone(p.detach()) for p in params_flat]

        self.base.step()

        for group in self.param_groups:
            for p in group["params"]:
                o = params_flat.pop(0)
                update = p.double() - o.double()
                p.set_(o)
                scale = group["lr"]
                if group["graft_to_self"]:
                    scale = scale * torch.norm(update)
                p.add_(torch.sign(update), alpha=scale)

        return loss


class Graft(torch.optim.Optimizer):
    """
    Learning rate grafting of two optimizers. It'll take the direction of one optimizer, but replace the scale of its
    proposed update with that of another optimizer. The notation of a grafted optimizer combinations is
    Magnitude#Direction, where # is the grafting operator.
    Known-good combinations are:
    * Adam#Lion, which outperforms pure Lion and pure Adam by avoiding vanishing gradients
      Lion can be imported from Lucidrains' repository: https://github.com/lucidrains/lion-pytorch
      For experimental results, see https://twitter.com/dvruette/status/1627663196839370755
    * Adam#Shampoo, which outperforms pure Shampoo and pure Adam, by introducing second-order statistics
      Shampoo can be imported from the official implementation: https://github.com/google-research/google-research/tree/master/scalable_shampoo/pytorch  (DO NOT USE https://github.com/jettify/pytorch-optimizer/blob/master/torch_optimizer/shampoo.py or https://github.com/moskomule/shampoo.pytorch)
      For experimental results, see the paper: https://arxiv.org/abs/2002.09018
    * LaProp#Lion, which works similarly to Adam#Lion, but avoids instabilities at sparse and low-magnitude gradients
      LaProp is part of truegrad and can be used as-is.
      Experimental results will be released soon. Sese the LaProp paper for theoretical justification: https://arxiv.org/abs/2002.04839

    Grafting originates from https://openreview.net/forum?id=FpKgG31Z_i9

    Usage:
    >>> import torch
    >>> model = torch.nn.Linear(10, 2)
    # step sizes comes from magnitude_from, so its LR is actively used
    >>> magnitude_from = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0)
    # step direction comes from direction_from. set the LR to 1 to avoid float accuracy issues
    >>> direction_from = torch.optim.SGD(model.parameters(), lr=1, weight_decay=0)
    # turn off weight decay in both input optimizers, but optionally use weight decay in Graft.
    >>> opt = Graft(model.parameters(), magnitude_from, direction_from, weight_decay=0.1)
    >>> model(torch.randn((16, 10))).mean().backward()  # get some random gradients
    >>> opt.step()  # apply as usual
    >>> opt.zero_grad()
    """

    def __init__(self, params, magnitude: torch.optim.Optimizer, direction: torch.optim.Optimizer,
                 weight_decay: float = 0, decay_to_init: bool = False, eps: float = 1e-12, lr: float = 1):
        super().__init__(params, {"weight_decay": weight_decay, "decay_to_init": decay_to_init, "lr": lr, "eps": eps})
        self.magnitude = magnitude
        self.direction = direction

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        params_flat = []
        for group in self.param_groups:
            for p in group["params"]:
                params_flat.append(p)
                decay_weight_(self.state[p], p, group)

        original_params = [torch.clone(p.detach()) for p in params_flat]

        self.magnitude.step()
        magnitudes_flat = []
        for o, p in zip(original_params, params_flat):
            magnitudes_flat.append(torch.norm(o.double() - p.double()))
            p.copy_(o.data)

        self.direction.step()

        for group in self.param_groups:
            for _ in group["params"]:
                o, p, m = original_params.pop(0), params_flat.pop(0), magnitudes_flat.pop(0)
                o_double = o.double()
                update = p.double() - o_double
                p.copy_(o_double + update * m / torch.norm(update).clamp(min=group["eps"]) * group["lr"])

        return loss


class TrueGrad(torch.optim.Optimizer):
    true_statistics: List[str] = []
    base_statistics: List[str] = []
    shared_statistics: List[str] = []

    def __init__(self, params, lr: float = 1e-3,
                 betas: List[float] = (),
                 eps: float = 1e-12,
                 weight_decay: float = 1e-2,
                 graft: bool = True,
                 decay_to_init: bool = False,
                 default_to_baseline: bool = False,
                 enforce_baseline: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                        decay_to_init=decay_to_init, default_to_baseline=default_to_baseline,
                        enforce_baseline=enforce_baseline)
        super(TrueGrad, self).__init__(params, defaults)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any], **kwargs: Tensor
               ) -> Tuple[Optional[Tensor], Optional[Tensor], float]:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                do_base = not hasattr(p, "sum_grad_squared") or p.sum_grad_squared is None or group["enforce_baseline"]
                if not group["default_to_baseline"] and do_base and not group["enforce_baseline"]:
                    raise ValueError(f"Parameter of shape {list(p.size())} doesn't have `sum_grad_squared` attribute. "
                                     f"Make sure to use backpack.")

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    for s in self.shared_statistics:
                        state[s] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if not do_base:
                        for s in self.true_statistics:
                            state[s] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if do_base or group["graft"]:
                        for s in self.base_statistics:
                            state[s] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["decay_to_init"]:
                        state["init"] = torch.clone(p.detach())

                step_t = state['step']
                step_t += 1

                # Perform stepweight decay
                decay = group['lr'] * group['weight_decay']
                if group["decay_to_init"]:
                    p.add_(state["init"] - p, alpha=decay)
                else:
                    p.mul_(1 - decay)

                step = step_t.item()

                base_update, update, alpha = self._inner(step, p, group,
                                                         **{k: state.get(k) for k in self.shared_statistics},
                                                         **{k: state.get(k) for k in self.base_statistics},
                                                         **{k: state.get(k) for k in self.true_statistics})

                if group["graft"] and not do_base:
                    alpha = alpha * base_update.norm() / update.norm().add_(group['eps'])
                elif do_base:
                    update = base_update

                p.add_(update, alpha=alpha)
        return loss


class TGAdamW(TrueGrad):
    true_statistics: List[str] = ["exp_avg_true_sq"]
    base_statistics: List[str] = ["exp_avg_sq"]
    shared_statistics: List[str] = ["exp_avg"]

    def __init__(self, params, lr: float = 1e-3,
                 betas: Union[Tuple[float, float], Tuple[float, float, float]] = (0.9, 0.999, 0.999),
                 eps: float = 1e-12,
                 weight_decay: float = 1e-2,
                 graft: bool = True,
                 decay_to_init: bool = False,
                 default_to_adam: bool = None,
                 default_to_baseline: bool = None,
                 enforce_baseline: bool = False):
        if default_to_baseline is None:
            default_to_baseline = default_to_adam
        elif default_to_adam is not None:
            raise ValueError("Can't set both default_to_baseline and default_to_adam, as both map to the same argument")
        if default_to_adam is not None:
            warnings.warn("default_to_adam is deprecated and will be replaced by default_to_baseline in April 2023")
        if default_to_baseline is None:
            default_to_baseline = False
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                         decay_to_init=decay_to_init, default_to_baseline=default_to_baseline,
                         enforce_baseline=enforce_baseline)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any], exp_avg: Tensor,
               exp_avg_sq: Optional[Tensor] = None, exp_avg_true_sq: Optional[Tensor] = None
               ) -> Tuple[Optional[Tensor], Optional[Tensor], float]:
        if len(group["betas"]) == 2:
            (beta1, beta2), (_, beta3) = group["betas"], group["betas"]
        else:
            beta1, beta2, beta3 = group['betas']

        update, base_update, eps = None, None, group["eps"]
        ema_(exp_avg, p.grad, beta1)
        if exp_avg_true_sq is not None:
            update = div_ema(exp_avg, group["eps"], exp_avg_true_sq, p.sum_grad_squared, beta3, step)
        if exp_avg_sq is not None:
            base_update = div_ema(exp_avg, group["eps"], exp_avg_sq, p.grad.square(), beta2, step)

        return base_update, update, -group['lr'] / (1 - beta1 ** step)


class TGLaProp(TrueGrad):
    true_statistics: List[str] = ["exp_avg_true", "exp_avg_true_sq"]
    base_statistics: List[str] = ["exp_avg", "exp_avg_sq"]

    def __init__(self, params, lr: float = 1e-3,
                 betas: Union[Tuple[float, float], Tuple[float, float, float, float]] = (0.9, 0.99),
                 eps: float = 1e-12,
                 weight_decay: float = 1e-2,
                 graft: bool = True,
                 decay_to_init: bool = False,
                 default_to_baseline: bool = False,
                 enforce_baseline: bool = False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                         decay_to_init=decay_to_init, default_to_baseline=default_to_baseline,
                         enforce_baseline=enforce_baseline)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any],
               exp_avg: Optional[Tensor] = None, exp_avg_sq: Optional[Tensor] = None,
               exp_avg_true: Optional[Tensor] = None, exp_avg_true_sq: Optional[Tensor] = None
               ) -> Tuple[Optional[Tensor], Optional[Tensor], float]:
        if len(group["betas"]) == 2:
            (beta1, beta2), (beta3, beta4) = group["betas"], group["betas"]
        else:
            beta1, beta2, beta3, beta4 = group['betas']

        update, base_update, alpha, eps = None, None, 1, group["eps"]
        if exp_avg_true_sq is not None:
            update = ema_(exp_avg_true, div_ema(p.grad, eps, exp_avg_true_sq, p.sum_grad_squared, beta4, step), beta3)
            alpha = -group['lr'] / (1 - beta3 ** step)

        if exp_avg_sq is not None:
            base_update = ema_(exp_avg, div_ema(p.grad, eps, exp_avg_sq, p.grad.square(), beta2, step), beta1)
            alpha = -group['lr'] / (1 - beta1 ** step)  # if grafting, beta3 issues are "grafted" away

        return base_update, update, alpha


class TGRMSProp(TrueGrad):
    """
    This is NOT correct RMSProp. Instead, it is debiased RMSProp. Debiased RMSProp is similar to RMSProp, but instead of
    0.9 * 0 + 0.1 * grad = 0.1 * grad
    at the first step, you have
    (0.9 * 0 + 0.1 * grad) / correction = grad
    where correction = 1 / (1 - 0.9 ** step)

    It's fundamentally the same as TGLaProp() with beta1 and beta3 = 0
    """
    true_statistics: List[str] = ["exp_avg_true_sq"]
    base_statistics: List[str] = ["exp_avg_sq"]

    def __init__(self, params, lr: float = 1e-3,
                 betas: Union[float, Tuple[float], Tuple[float, float]] = (0.9,),
                 eps: float = 1e-12,
                 weight_decay: float = 1e-2,
                 graft: bool = True,
                 decay_to_init: bool = False,
                 default_to_baseline: bool = False,
                 enforce_baseline: bool = False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                         decay_to_init=decay_to_init, default_to_baseline=default_to_baseline,
                         enforce_baseline=enforce_baseline)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any],
               exp_avg_sq: Optional[Tensor] = None, exp_avg_true_sq: Optional[Tensor] = None
               ) -> Tuple[Optional[Tensor], Optional[Tensor], float]:
        if isinstance(group["betas"], float):
            beta1 = beta2 = group["betas"]
        elif len(group["betas"]) == 1:
            (beta1,), (beta2,) = group["betas"], group["betas"]
        else:
            beta1, beta2 = group['betas']

        update, base_update, eps = None, None, group["eps"]
        if exp_avg_true_sq is not None:
            update = div_ema(p.grad, eps, exp_avg_true_sq, p.sum_grad_squared, beta2, step)

        if exp_avg_sq is not None:
            base_update = div_ema(p.grad, eps, exp_avg_sq, p.grad.square(), beta1, step)

        return base_update, update, -group['lr']
