import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from convnets.convnet import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])#,
     #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


net = Net()
net = net.to(device)

batch_size = 100
epochs = 30

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(epochs):  # loop over the dataset multiple times
    print(f'Epoch: {epoch} of: {epochs}')
    running_loss, acc = 0.0, 0.0
    i = 0
    for inputs, labels in tqdm(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        # Calculating predictions
        _, preds = torch.max(outputs.detach(), 1)

        # Calculating training accuracy
        acc += torch.mean((torch.sum(preds == labels).float()) / inputs.size(0))

        i+=1
    val_acc = 0
    for inputs, labels in tqdm(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, preds = torch.max(outputs.detach(), 1)
        # Calculating validation set accuracy
        val_acc += torch.mean((torch.sum(preds == labels).float()) / inputs.size(0))

    acc/=len(trainloader)
    val_acc/=len(testloader)

    print(f'Train Accuracy: {acc} | Test Accuracy: {val_acc}')

print('Finished Training')


torch.save(net, 'cifar.pth')
