import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.torch.multi_activation import MultiActivation

from core.model import Model


class ConvNet(Model):
    def __init__(self, chn=3, n_filters=(32, 32), kernel_size=(5, 5), p=(0.25,), init_weights=None, device='cpu'):

        super(ConvNet, self).__init__(init_weights, device)
	# Softmax2d, PReLU are not working yet
        self.conv1 = nn.Conv2d(chn, n_filters[0], kernel_size[0])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_filters[0], n_filters[1], kernel_size[1])
        self.dropout1 = nn.Dropout2d(p[0])
        self.fc1 = nn.Linear(n_filters[1]*kernel_size[1]*kernel_size[1], 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.multi = MultiActivation(activation=('ReLU()', 'ReLU6()', 'SiLU()', 'Mish()',), strategy='mean')

        # Compiles the network
        self._compile()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.multi(self.fc1(x))
        x = self.multi(self.fc2(x))
        x = self.fc3(x)
        return x
