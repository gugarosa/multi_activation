import torch

from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

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

        >>> m = MultiActivation(('relu', 'sigmoid), strategy='mean')
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

        def _check_activation_string(activation, valid_activation_strings):
            if activation not in valid_activation_strings:
                raise ValueError(
                    "Invalid activation string {!r}, should be one of {}".format(
                        activation, valid_activation_strings
                    )
                )

        valid_activation_dict = {
            'relu': F.relu,
            'sigmoid': torch.sigmoid
        }
        if not isinstance(activation, (str, tuple)):
            raise TypeError(
                "Invalid activation type {!r}, should be string or tuple".format(
                    type(activation).__name__
                )
            )
        if isinstance(activation, str):
            _check_activation_string(activation, valid_activation_dict.keys())
            self.activation = (valid_activation_dict[activation],)
        else:
            self.activation = ()
            for a in activation:
                if isinstance(a, str):
                    _check_activation_string(a, valid_activation_dict.keys())
                    self.activation += (valid_activation_dict[a],)

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
