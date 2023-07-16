import functools
from typing import Tuple, Union, List, Dict, Any, Optional

import torch
from torch import Tensor
from torch.nn import Parameter


class WeightDecayBase:
    def __init__(self):
        pass

    def __call__(self, mod: torch.optim.Optimizer, p: torch.Tensor, idx: int):
        return p


class WeightDecayChain:
    def __init__(self, *operands: WeightDecayBase):
        self.operands = operands

    def __call__(self, mod: torch.optim.Optimizer):
        idx = 0
        for group in mod.param_groups:
            for p in group["params"]:
                p.data.add_(functools.reduce(lambda x, f: f(mod, x, idx), self.operands, p),
                            alpha=-group["lr"] * group["weight_decay"])
                idx += 1


class LpWeightDecay(WeightDecayBase):
    def __init__(self, power: float):
        super().__init__()
        self.power = power

    def __call__(self, mod: torch.optim.Optimizer, p: Tensor, idx: int):
        return p.abs().pow(self.power) * p.sign()


class L1WeightDecay(LpWeightDecay):
    def __init__(self):
        super().__init__(0)


class L2WeightDecay(LpWeightDecay):
    def __init__(self):
        super().__init__(1)


def _detach(x: Tensor) -> Tensor:
    return x.detach().clone()


def _param_iterator(mod: torch.optim.Optimizer, fn=_detach):
    yield from (fn(p) for group in mod.param_groups for p in group["params"])


class WeightDecayToValue(WeightDecayBase):
    def __init__(self):
        super().__init__()
        self.target_values: List[Tensor] = ...
        self.global_step = 0

    def _on_step_start(self, mod: torch.optim.Optimizer):
        pass

    def _on_global_start(self, mod: torch.optim.Optimizer):
        pass

    def _preprocess(self, target: Tensor):
        return target

    def __call__(self, mod: torch.optim.Optimizer, p: Tensor, idx: int):
        if idx == 0:
            if self.global_step == 0:
                self._on_global_start(mod)
            self._on_step_start(mod)
            self.global_step += 1
        return p - self._preprocess(self.target_values[idx])


class WeightDecayToInit(WeightDecayToValue):
    def _on_global_start(self, mod: torch.optim.Optimizer):
        self.target_values = list(_param_iterator(mod))


class WeightDecayToEMA(WeightDecayToInit):
    def __init__(self, beta: float = 0.999):
        super().__init__()
        self.beta = beta

    def _on_global_start(self, mod: torch.optim.Optimizer):
        self.target_values = [torch.zeros_like(x) for x in _param_iterator(mod)]

    def _on_step_start(self, mod: torch.optim.Optimizer):
        self.global_step += 1
        for v, p in zip(self.target_values, _param_iterator(mod)):
            v.mul_(self.beta).add_(p, alpha=1 - self.beta)

    def _preprocess(self, target: Tensor):
        return target / (1 - self.beta ** self.global_step)


def ema_(base: Tensor, update: Tensor, beta: float, step: Optional[int] = None):
    base.mul_(beta).add_(update, alpha=1 - beta)
    if step is None:
        return base
    return base / (1 - beta ** step)


def stable_sqrt(base: Tensor, eps: float):
    return base.sqrt().clamp(min=eps)


def div_ema(base: Tensor, eps: float, base_sq: Tensor, update_sq: Tensor, beta_sq: float, step: Optional[int] = None):
    return base / stable_sqrt(ema_(base_sq, update_sq, beta_sq, step), eps)


def _default_decay(weight_decay_cls: Optional[WeightDecayChain]) -> WeightDecayChain:
    if weight_decay_cls is None:
        return WeightDecayChain(L2WeightDecay())
    return weight_decay_cls


class OptimizerOptimizer(torch.optim.Optimizer):
    def __init__(self, params, inner_optimizer: torch.optim.Optimizer, learning_rate_learning_rate: float = 1,
                 weight_decay: float = 0, weight_decay_cls: Optional[WeightDecayChain] = None):
        self.inner_optimizer = inner_optimizer
        self.learning_rate_learning_rate = learning_rate_learning_rate
        self.weight_decay_cls = _default_decay(weight_decay_cls)
        param_groups = self.inner_optimizer.param_groups
        self.inner_optimizer.param_groups = []
        for group in param_groups:
            for param in group["params"]:
                group = {k: v for k, v in group.items() if k != "params"}
                group["params"] = [param]
                self.inner_optimizer.param_groups.append(group)

        super(OptimizerOptimizer, self).__init__(params, {"weight_decay": weight_decay})

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
                state["param"] = torch.clone(p.detach())

        self.weight_decay_cls(self)

        self.inner_optimizer.step()

        for group in self.inner_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                update = p.double() - state["param"].double()
                if "old_update" in state:
                    dims = ''.join(chr(ord('a') + i) for i in range(update.ndim))
                    lr_grad = torch.einsum(f"{dims},{dims}->", update, state["old_update"].double())
                    state["lr"] = group["lr"] = group["lr"] + lr_grad.item() * self.learning_rate_learning_rate
                state["old_update"] = torch.clone(update.to(torch.bfloat16).detach())
                state["param"] = None

        return loss


class Sign(torch.optim.Optimizer):
    def __init__(self, params, base: torch.optim.Optimizer, lr: float = 1, weight_decay: float = 0, eps: float = 1e-12,
                 graft_to_self: bool = True, weight_decay_cls: Optional[WeightDecayChain] = None):
        self.weight_decay_cls = _default_decay(weight_decay_cls)

        super().__init__(params, {"weight_decay": weight_decay, "lr": lr, "eps": eps, "graft_to_self": graft_to_self})
        self.base = base

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        self.weight_decay_cls(self)
        params_flat = list(_param_iterator(self))
        self.base.step()

        for group in self.param_groups:
            for p in group["params"]:
                o = params_flat.pop(0)
                update = p.double() - o.double()
                p.set_(o)
                scale = group["lr"]
                sign_update = torch.sign(update)
                if group["graft_to_self"]:
                    scale = scale * update.norm() / sign_update.norm().clamp(min=group["eps"])
                p.add_(sign_update, alpha=scale)

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
                 weight_decay: float = 0, eps: float = 1e-12, lr: float = 1,
                 weight_decay_cls: Optional[WeightDecayChain] = None):
        super().__init__(params, {"weight_decay": weight_decay, "lr": lr, "eps": eps})
        self.magnitude = magnitude
        self.direction = direction
        self.weight_decay_cls = _default_decay(weight_decay_cls)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        self.weight_decay_cls(self)
        original_params = list(_param_iterator(self))

        self.magnitude.step()
        params_flat = list(_param_iterator(self, lambda x: x))

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

    def __init__(self, params, lr: float = 1e-3, betas: List[float] = (), eps: float = 1e-12,
                 weight_decay: float = 1e-2, graft: bool = True, default_to_baseline: bool = False,
                 enforce_baseline: bool = False, weight_decay_cls: Optional[WeightDecayChain] = None):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                        default_to_baseline=default_to_baseline, enforce_baseline=enforce_baseline)
        super(TrueGrad, self).__init__(params, defaults)
        self.weight_decay_cls = _default_decay(weight_decay_cls)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any], **kwargs: Tensor) -> Tuple[
        Optional[Tensor], Optional[Tensor], float]:
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

                step_t = state['step']
                step_t += 1

                self.weight_decay_cls(self)

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
                 eps: float = 1e-12, weight_decay: float = 1e-2, graft: bool = True, default_to_baseline: bool = None,
                 enforce_baseline: bool = False, weight_decay_cls: Optional[WeightDecayChain] = None):
        if default_to_baseline is None:
            default_to_baseline = False
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                         default_to_baseline=default_to_baseline, enforce_baseline=enforce_baseline,
                         weight_decay_cls=weight_decay_cls)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any], exp_avg: Tensor,
               exp_avg_sq: Optional[Tensor] = None, exp_avg_true_sq: Optional[Tensor] = None) -> Tuple[
        Optional[Tensor], Optional[Tensor], float]:
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
                 betas: Union[Tuple[float, float], Tuple[float, float, float, float]] = (0.9, 0.99), eps: float = 1e-12,
                 weight_decay: float = 1e-2, graft: bool = True,
                 default_to_baseline: bool = False, enforce_baseline: bool = False,
                 weight_decay_cls: Optional[WeightDecayChain] = None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                         default_to_baseline=default_to_baseline, enforce_baseline=enforce_baseline,
                         weight_decay_cls=weight_decay_cls)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any], exp_avg: Optional[Tensor] = None,
               exp_avg_sq: Optional[Tensor] = None, exp_avg_true: Optional[Tensor] = None,
               exp_avg_true_sq: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Optional[Tensor], float]:
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

    def __init__(self, params, lr: float = 1e-3, betas: Union[float, Tuple[float], Tuple[float, float]] = (0.9,),
                 eps: float = 1e-12, weight_decay: float = 1e-2, graft: bool = True, default_to_baseline: bool = False,
                 enforce_baseline: bool = False, weight_decay_cls: Optional[WeightDecayChain] = None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                         default_to_baseline=default_to_baseline, enforce_baseline=enforce_baseline,
                         weight_decay_cls=weight_decay_cls)

    def _inner(self, step: int, p: Parameter, group: Dict[str, Any], exp_avg_sq: Optional[Tensor] = None,
               exp_avg_true_sq: Optional[Tensor] = None) -> Tuple[Optional[Tensor], Optional[Tensor], float]:
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
