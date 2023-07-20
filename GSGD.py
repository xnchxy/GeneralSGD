import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional, Callable
def mysgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        scaling: float,
        Dphi_map: Callable,
        nesterov: float,
        maximize: bool):
    r"""Functional API that performs Generalized SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            mome_param = max(momentum, 1-lr * scaling * (1-momentum) )
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                # buf.mul_(mome_param).add_(d_p, alpha=1 - mome_param)
                buf.add_(d_p.add_(buf, alpha=-1) , alpha=1 - mome_param)

            if nesterov > 0:
                d_p = Dphi_map(buf).add(d_p, alpha=nesterov)
            else:
                d_p = Dphi_map(buf)

        alpha = lr if maximize else -lr
        param.add_(d_p, alpha=alpha)





class GSGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0.9, scaling = None, Dphi_map = lambda tensor: tensor, 
                 weight_decay=0, nesterov=0, *, maximize=False):
        if lr is not required and lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # if scaling is None:
        #     scaling = 1.0 / lr

        defaults = dict(lr=lr, momentum=momentum, scaling=scaling,Dphi_map = Dphi_map, 
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super(GSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            if group['scaling'] is None:
                group['scaling'] = 1.0 / (group['lr'] + 1e-12)
                if not 0.0 <= group['scaling'] :
                    raise ValueError('Invalid scaling parameter: {}'.format(scaling))

    def __setstate__(self, state):
        super(GSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            scaling = group['scaling']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']
            Dphi_map = group['Dphi_map']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            mysgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  scaling=scaling,
                  Dphi_map = Dphi_map,
                  nesterov=nesterov,
                  maximize=maximize,)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss