# Stochastic Subgradient Methods for Nonsmooth Nonconvex Optimization

This repository presents an official implementation of our research paper, which introduces a generalized framework for SGD-variant methods and establishes their convergence properties. Our findings guarantee the convergence of SGD, heavy-ball SGD, Lion, and signSGD when training nonsmooth neural networks, such as those utilizing ReLU activation functions as their activation functions.

The optimizers provided in this repository are designed to be user-friendly and can be easily invoked using the same approach as the SGD optimizers in PyTorch. 

## Usage

### GSGD

```python
class GSGD(params, lr=<required parameter>, momentum=0.9, scaling=None, Dphi_map=lambda tensor: tensor, weight_decay=0, nesterov=0, *, maximize=False)
```

**Parameters:**

- `params` (*iterable*): An iterable of parameters to optimize or dictionaries defining parameter groups.
- `lr` ([*float*](https://docs.python.org/3/library/functions.html#float)): The learning rate.
- `momentum` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): The lower-bound for the momentum factor (default: `0.9`).
- `scaling` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): The ratio between the stepsizes for the momentum terms and parameters. If set to `None`, the scaling is automatically chosen as `1.0/lr` (default: `None`).
- `Dphi_map` ([*callable*](https://docs.python.org/3/library/functions.html#callable), *optional*): A mapping function that determines the regularization techniques in SGD-variant methods. In `GSGD`, the updating direction for each tensor in `params` is given by `Dphi_map(tensor)`. By choosing different `Dphi_map` functions, users can apply different regularization techniques to the SGD-variant methods. Detailed requirements for `Dphi_map` can be found in Section 4 of our research paper. (default: `lambda tensor: tensor`)
    - When `Dphi_map = lambda tensor: tensor`, the `GSGD` optimizer becomes a variant of the heavy-ball SGD method.
    - When `Dphi_map = lambda tensor: torch.sign(tensor)`, the `GSGD` optimizer becomes the signSGD method.
- `weight_decay` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): Weight decay (L2 penalty) (default: `0`).
- `nesterov` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): Enables Nesterov momentum (default: `0`).
- `maximize` ([*bool*](https://docs.python.org/3/library/functions.html#bool), *optional*): Determines whether to maximize the parameters based on the objective, instead of minimizing them (default: `False`).

### Lion

For the Lion method, we provide its implementation based on the codes from [this repository](https://github.com/google/automl/tree/master/lion). Detailed instructions on parameter tuning for the Lion optimizer can be found in [lucidrains' repository](https://github.com/lucidrains/lion-pytorch).

```python
class Lion(params, lr=1e-4, betas=(0.9, 0.99), scaling=None, weight_decay=0.0, nesterov_momentum=0)
```

**Parameters:**

- `params` (*iterable*): An iterable of parameters to optimize or dictionaries defining parameter groups.
- `lr` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): The learning rate (default: `1e-4`).
- `betas` (*Tuple* of [*float*](https://docs.python.org/3/library/functions.html#float), *optional*): Coefficients used for computing running averages of the gradient (default: `(0.9, 0.99)`).
- `scaling` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): The ratio between the stepsizes for the momentum terms and parameters. If set to `None`, the scaling is automatically chosen as `1.0/lr` (default: `None`).
- `weight_decay` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): Weight decay (L2 penalty) (default: `0`).
- `nesterov_momentum` ([*float*](https://docs.python.org/3/library/functions.html#float), *optional*): Enables Nesterov momentum (default: `0`).
