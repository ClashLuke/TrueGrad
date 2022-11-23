from typing import Tuple, Union

import torch


class TGAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: Union[Tuple[float, float], Tuple[float, float, float]] = (0.9, 0.999, 0.999),
                 eps: float = 1e-12,
                 weight_decay: float = 1e-2,
                 graft: bool = True,
                 decay_to_init: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                        decay_to_init=decay_to_init)
        super(TGAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if len(group["betas"]) == 2:
                beta1, beta2 = group["betas"]
                beta3 = beta2
            else:
                beta1, beta2, beta3 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                if not hasattr(p, "square_grad") or p.square_grad is None:
                    raise ValueError(f"Parameter of shape {list(p.size())} doesn't have `square_grad` attribute. "
                                     f"Make sure to use truegrad.utils.patch_model() or truegrad.nn for all optimized "
                                     f"parameters.")

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_true_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["graft"]:
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["decay_to_init"]:
                        state["init"] = torch.clone(p.detach())

                exp_avg = state['exp_avg']
                exp_avg_true_sq = state['exp_avg_true_sq']
                step_t = state['step']

                # update step
                step_t += 1

                # Perform stepweight decay
                decay = group['lr'] * group['weight_decay']
                if group["decay_to_init"]:
                    p.add_(state["init"] - p, alpha=decay)
                else:
                    p.mul_(1 - decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_true_sq.mul_(beta3).add_(p.square_grad, alpha=1 - beta3)
                p.square_grad = None

                step = step_t.item()

                denom = (exp_avg_true_sq / (1 - beta3 ** step)).sqrt().add_(group['eps'])
                update = exp_avg / denom
                alpha = -group['lr'] / (1 - beta1 ** step)

                if group["graft"]:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(p.grad.square(), alpha=1 - beta2)
                    adam_update = exp_avg / (exp_avg_sq / (1 - beta2 ** step)).sqrt().add_(group['eps'])
                    alpha = alpha * adam_update.norm() / update.norm()

                p.add_(update, alpha=alpha)
        return loss
