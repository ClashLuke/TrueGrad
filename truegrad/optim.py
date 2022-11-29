from typing import Tuple, Union

import torch


class TGAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3,
                 betas: Union[Tuple[float, float], Tuple[float, float, float]] = (0.9, 0.999, 0.999),
                 eps: float = 1e-12,
                 weight_decay: float = 1e-2,
                 graft: bool = True,
                 decay_to_init: bool = False,
                 default_to_adam: bool = False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, graft=graft,
                        decay_to_init=decay_to_init, default_to_adam=default_to_adam)
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
                do_adam = not hasattr(p, "sum_grad_squared") or p.sum_grad_squared is None
                if not group["default_to_adam"] and do_adam:
                    raise ValueError(f"Parameter of shape {list(p.size())} doesn't have `sum_grad_squared` attribute. "
                                     f"Make sure to use backpack.")

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if not do_adam:
                        state['exp_avg_true_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if do_adam or group["graft"]:
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

                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)

                step = step_t.item()
                alpha = -group['lr'] / (1 - beta1 ** step)

                if not do_adam:
                    exp_avg_true_sq.mul_(beta3).add_(p.sum_grad_squared, alpha=1 - beta3)
                    p.sum_grad_squared = None
                    denom = (exp_avg_true_sq / (1 - beta3 ** step)).sqrt().add_(group['eps'])
                    update = exp_avg / denom

                if group["graft"] or do_adam:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(p.grad.square(), alpha=1 - beta2)
                    adam_update = exp_avg / (exp_avg_sq / (1 - beta2 ** step)).sqrt().add_(group['eps'])

                if group["graft"] and not do_adam:
                    alpha = alpha * adam_update.norm() / update.norm().add_(group['eps'])
                elif do_adam:
                    update = adam_update

                p.add_(update, alpha=alpha)
        return loss
