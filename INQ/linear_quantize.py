import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --------- Quantization Functions (Simple Uniform Quantization) ---------
def quantize_tensor(x, bits=8):
    levels = 2 ** bits
    x = torch.clamp(x, 0, 1)
    x = torch.round(x * (levels - 1)) / (levels - 1)
    return x

# --------- AlexNet for CIFAR-10 (with simple quantization) ---------
class AlexNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = quantize_tensor(x, bits=8)  # Quantize input
        x = self.features(x)
        x = quantize_tensor(x, bits=8)  # Quantize after feature extractor
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------- Dataset Loader (normal 8bit CIFAR) ---------
def get_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),  # Already 8-bit
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

# --------- Train and Test Functions ---------
def train(model, device, trainloader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(trainloader, desc=f"Train Epoch {epoch}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

def test(model, device, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# --------- Main ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    trainloader, testloader = get_cifar10()

    model = AlexNetCIFAR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    total_epochs = 150

    for epoch in range(1, total_epochs+1):
        train(model, device, trainloader, optimizer, epoch)
        test(model, device, testloader)
        scheduler.step()

if __name__ == '__main__':
    main()

