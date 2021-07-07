import torch
import torchvision
from torch.utils.data import DataLoader

from models.conv_net import ConvNet

# Input transform
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# Loads training and testing sets
train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Creates training and testing loaders
train_loader = DataLoader(train, batch_size=100, shuffle=True, num_workers=0)
test_loader = DataLoader(test, batch_size=100, shuffle=False, num_workers=0)

# Instantiates the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet(n_channels=3, n_classes=10, device=device)

# Fits the model
model.fit(train_loader, test_loader, epochs=30)

# Saves the pre-trained model
torch.save(model, 'cifar10.pth')
