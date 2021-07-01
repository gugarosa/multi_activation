from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module


class MultiActivation(Module):
    """Applies a multi-activation transformation to the incoming data.

    Args:
        activation (tuple or str): Activation functions
        strategy (str, optional): Output tensor strategy. Default: ``mean``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output (concat): :math:`(N, n*)`, `n` times the input shape shape
        - Output (mean): :math:`(N, *)`, same shape as the input

    Examples:

        >>> m = MultiActivation((ReLU(), Sigmoid()), strategy='mean')
        >>> input = torch.randn(128, 32)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 32])

    """

    __constants__ = ['activation', 'strategy']

    activation: Tuple[str, ...]
    strategy: str
    
    def __init__(self, activation: Tuple[str, ...], strategy: str = 'mean') -> None:
        super(MultiActivation, self).__init__()

	# Softmax2d, PReLU is not working yet

        valid_activation_functions = [
            'Threshold()', 'ReLU()', 'Hardtanh()', 'ReLU6()', 'Sigmoid()', 'Tanh()', 'Softmax()',
            'Softmax2d()', 'LogSoftmax()', 'ELU()', 'SELU()', 'CELU()', 'GELU()', 'Hardshrink()',
            'LeakyReLU()', 'LogSigmoid()', 'Softplus()', 'Softshrink()', 'PReLU()', 'Softsign()', 'Softmin()',
            'Tanhshrink()', 'RReLU()', 'Hardsigmoid()', 'Hardswish()', 'SiLU()','Mish()'
        ]

        valid_torch_functions = {
            'Threshold()': torch.nn.Threshold(threshold=5, value=0),
            'ReLU()': torch.nn.ReLU(),
            'Hardtanh()': torch.nn.Hardtanh(),
            'ReLU6()': torch.nn.ReLU6(),
            'Sigmoid()': torch.nn.Sigmoid(),
            'Tanh()': torch.nn.Tanh(),
            'Softmax()': torch.nn.Softmax(),
            'Softmax2d()': torch.nn.Softmax2d(),
            'LogSoftmax()': torch.nn.LogSoftmax(),
            'ELU()': torch.nn.ELU(),
            'SELU()': torch.nn.SELU(), 
            'CELU()': torch.nn.CELU(),
            'GELU()': torch.nn.GELU(), 
            'Hardshrink()': torch.nn.Hardshrink(),
            'LeakyReLU()': torch.nn.LeakyReLU(), 
            'LogSigmoid()': torch.nn.LogSigmoid(), 
            'Softplus()': torch.nn.Softplus(), 
            'Softshrink()': torch.nn.Softshrink(), 
            'PReLU()': torch.nn.PReLU(), 
            'Softsign()': torch.nn.Softsign(), 
            'Softmin()': torch.nn.Softmin(),
            'Tanhshrink()': torch.nn.Tanhshrink(), 
            'RReLU()': torch.nn.RReLU(), 
            'Hardsigmoid()': torch.nn.Hardsigmoid(), 
            'Hardswish()': torch.nn.Hardswish(), 
            'SiLU()': torch.nn.SiLU(),
            'Mish()': torch.nn.Mish()
        }

        if not isinstance(activation, tuple):
            raise TypeError(
                'Invalid activation type {!r}, should be tuple'.format(
                    type(activation).__name__
                )
            )
        self.activation = ()
        for a in activation:
            if a not in valid_activation_functions:
            #if type(a).__name__ not in valid_activation_functions:
                raise ValueError(
                    'Invalid activation function {!r}, should be one of {}'.format(
                        a, valid_activation_functions
                    )
                )
            self.activation += (valid_torch_functions[a],)
        print(self.activation)

        valid_strategy_strings = {'concat', 'mean'}
        if not isinstance(strategy, str):
            raise TypeError(
                'Invalid strategy type {!r}, should be string'.format(
                    type(strategy).__name__
                )
            )
        if strategy not in valid_strategy_strings:
            raise ValueError(
                'Invalid strategy string {!r}, should be one of {}'.format(
                    strategy, valid_strategy_strings
                )
            )
        self.strategy = strategy

    def _multi_activation_forward(self, input: Tensor) -> Tensor:
        if self.strategy == 'concat':
            return torch.hstack([a(input) for a in self.activation])
        else:
            return torch.mean(torch.stack([a(input) for a in self.activation]), dim=0)

    def forward(self, input: Tensor) -> Tensor:
        return self._multi_activation_forward(input)
