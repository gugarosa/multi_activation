import torch

from torch import Tensor
from torch.nn import Module

from typing import Optional, List, Tuple, Union

class MultiActivation(Module):
    """
    """

    __constants__ = ['activations', 'strategy']

    activation: Tuple[str, ...]
    strategy: str
    
    def __init__(self, activation: Tuple[str, ...], strategy: str = 'mean') -> None:
        super(MultiActivation, self).__init__()

        def _check_activation_string(activation, valid_activation_strings):
            if isinstance(activation, str):
                if activation not in valid_activation_strings:
                    raise ValueError(
                        "Invalid activation string {!r}, should be one of {}".format(
                            activation, valid_activation_strings
                        )
                    )

        valid_activation_strings = {'relu', 'sigmoid'}
        if isinstance(activation, str):
            _check_activation_string(activation, valid_activation_strings)
        if isinstance(activation, tuple):
            for a in activation:
                _check_activation_string(a, valid_activation_strings)

        valid_strategy_strings = {'concat', 'mean'}
        if isinstance(strategy, str):
            if strategy not in valid_strategy_strings:
                raise ValueError(
                    "Invalid strategy string {!r}, should be one of {}".format(
                        strategy, valid_strategy_strings
                    )
                )
        
        self.activation = activation
        self.strategy = strategy

    def forward(self, input: Tensor) -> Tensor:
        pass


if __name__ == '__main__':
    m = MultiActivation(activation=['relu'])