import torch

from torch import Tensor
from torch.nn import Module

from typing import Tuple


class MultiActivation(Module):
    """Applies a multi activation transformation to the incoming data.

    Args:
        activation (tuple or str): Activation functions
        strategy (str, optional): Output tensor strategy. Default: ``mean``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output (concat): :math:`(N, n*)`, `n` times the input shape shape
        - Output (mean): :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = MultiActivation((ReLU(), Sigmoid()), strategy='mean')
        >>> input = torch.randn(128, 32)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 32])

    """

    __constants__ = ['activation', 'strategy']

    activation: Tuple[str, ...]
    strategy: str
    
    def __init__(self, activation: Tuple[str, ...], strategy: str = 'mean', **kwargs) -> None:
        super(MultiActivation, self).__init__()

        valid_activation_functions = [
            'Threshold', 'ReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Tanh', 'Softmax',
            'Softmax2d', 'LogSoftmax', 'ELU', 'SELU', 'CELU', 'GLU', 'GELU', 'Hardshrink',
            'LeakyReLU', 'LogSigmoid', 'Softplus', 'Softshrink', 'PReLU', 'Softsign', 'Softmin',
            'Tanhshrink', 'RReLU', 'Hardsigmoid', 'Hardswish', 'SiLU'
        ]
        if not isinstance(activation, tuple):
            raise TypeError(
                "Invalid activation type {!r}, should be tuple".format(
                    type(activation).__name__
                )
            )
        self.activation = ()
        for a in activation:
            if type(a).__name__ not in valid_activation_functions:
                raise ValueError(
                    "Invalid activation function {!r}, should be one of {}".format(
                        a, valid_activation_functions
                    )
                )
            self.activation += (a,)

        valid_strategy_strings = {'concat', 'mean'}
        if not isinstance(strategy, str):
            raise TypeError(
                "Invalid strategy type {!r}, should be string".format(
                    type(strategy).__name__
                )
            )
        if strategy not in valid_strategy_strings:
            raise ValueError(
                "Invalid strategy string {!r}, should be one of {}".format(
                    strategy, valid_strategy_strings
                )
            )
        self.strategy = strategy

    def _multi_activation_forward(self, input: Tensor) -> Tensor:
        if self.strategy == 'concat':
            return torch.hstack([a(input) for a in self.activation])

        return torch.mean(torch.stack([a(input) for a in self.activation]), dim=0)

    def forward(self, input: Tensor) -> Tensor:
        return self._multi_activation_forward(input)
