"""PyTorch implementation of the Lion optimizer."""
import torch
from torch.optim.optimizer import Optimizer

# Define the Lion Optimizer class
class Lion_singlescale(Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), scaling = None, weight_decay=0.0, nesterov_momentum = 0):
        """
        Initialize the hyperparameters.
        
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float, optional): learning rate (default: 1e-4)
            betas (Tuple[float, float], optional): coefficients used for computing
                running averages of gradients (default: (0.9, 0.99))
            weight_decay (float, optional): weight decay coefficient (default: 0)
        """

        # Validate input hyperparameters
        if not 0.0 <=  lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        
        # if scaling is None:
        #     scaling = 1.0 /lr
        #     if not 0.0 <= scaling:
        #         raise ValueError('Invalid learning rate: {}'.format(scaling))
            
        # Set default values for the optimizer
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, scaling = scaling, nesterov_momentum = nesterov_momentum)
        super().__init__(params, defaults)

        for group in self.param_groups:
            if group['scaling'] is None:
                group['scaling'] = 1.0 / group['lr']
                if not 0.0 <= group['scaling'] :
                    raise ValueError('Invalid scaling parameter: {}'.format(scaling))

    # Define the step function for the optimizer
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate through parameter groups and update each parameter
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                scaling = group['scaling']

                tau1 = (1-beta1) * scaling
                tau2 = (1-beta2) * scaling
                lr = group['lr']
                nesterov_momentum = group['nesterov_momentum']

                momentum_param1 = max(1-lr * tau1, beta1)
                momentum_param2 = max(1-lr * tau2, beta2)

                # Weight update
                update = exp_avg * momentum_param1 + grad * (1 - momentum_param1)
                if nesterov_momentum >1e-5:
                    p.add_(torch.sign(update) + nesterov_momentum * grad, alpha=-group['lr'])
                else:
                    p.add_(torch.sign(update), alpha=-group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(momentum_param2).add_(grad, alpha=1 - momentum_param2)

        return loss
