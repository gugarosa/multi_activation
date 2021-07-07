import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from layers.torch.multi_activation import MultiActivation


class ConvNet(Model):
    """A ConvNet class implements a standard convolutional architecture, usually applied to CIFAR-based datasets.

    """

    def __init__(self, n_channels=3, n_classes=10, activations=('ReLU()', 'ReLU6()', 'SiLU()', 'Mish()'),
                 strategy='mean', init_weights=None, device='cpu'):
        """Initialization method.

        Args:
            n_channels (int): Number of input channels.
            n_classes (int): Number of output classes.
            activations (tuple): Tuple of activation functions.
            strategy (str): Multi-activation strategy.
            init_weights (tuple): Tuple holding the minimum and maximum values for weights initialization.
            device (str): Device that model should be trained on, e.g., `cpu` or `cuda`.

        """

        # Override its parent class
        super(ConvNet, self).__init__(init_weights, device)

        # Convolutional layers
        # n_input, n_output, kernel_size
        self.conv1 = nn.Conv2d(n_channels, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        # Pooling layer
        # kernel_size, stride
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layer
        # probability
        self.dropout1 = nn.Dropout2d(0.25)

	    # Fully-connected layers
        # n_input, n_output
        self.fc1 = nn.Linear(64*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

        # Multi-activation layer
        # activations, strategy
        self.multi = MultiActivation(activations, strategy)

        # Compiles the network
        self._compile()

    def forward(self, x):
        """Performs the forward pass.

        Args:
            x (torch.tensor): Input tensor.

        Returns:
            Output tensor.

        """

        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # Second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Flattens the current tensor
        x = torch.flatten(x, 1)
        
        # Fully-connected block
        x = self.multi(self.fc1(x))
        x = self.multi(self.fc2(x))
        x = self.fc3(x)

        return x
