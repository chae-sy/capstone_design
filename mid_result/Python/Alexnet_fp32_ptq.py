import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --------- Dataset: CIFAR-10 ---------
def get_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

# --------- AlexNet for CIFAR-10 ---------
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --------- Training and Testing Functions ---------
def train(model, device, trainloader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total_batches = 0

    pbar = tqdm(trainloader, desc=f"Train Epoch {epoch}")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / total_batches
    return avg_loss

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

# --------- Main Script ---------
def main(device):
    trainloader, testloader = get_cifar10()

    model = AlexNetCIFAR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    total_epochs = 150

    train_losses = []
    test_accuracies = []

    for epoch in range(1, total_epochs+1):
        avg_train_loss = train(model, device, trainloader, optimizer, epoch)
        avg_test_acc = test(model, device, testloader)
        scheduler.step()

        train_losses.append(avg_train_loss)
        test_accuracies.append(avg_test_acc)

    # Save loss and accuracy
    torch.save({
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
    }, 'loss_record.pt')
    print("✅ Saved train losses and test accuracies to 'loss_record.pt'!")

    return model, testloader

# --------- PTQ (Post-Training Quantization) ---------
def quantize_static(x, bits=8):
    """Simple uniform quantization."""
    levels = 2 ** bits
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (levels - 1)
    x_quant = torch.round((x - x_min) / scale) * scale + x_min
    return x_quant

def quantize_model_weights(model):
    """Quantize all weights in the model."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data = quantize_static(param.data, bits=8)
    return model

class QuantizedAlexNetCIFAR(nn.Module):
    def __init__(self, fp32_model):
        super(QuantizedAlexNetCIFAR, self).__init__()
        self.features = fp32_model.features
        self.classifier = fp32_model.classifier

    def forward(self, x):
        x = quantize_static(x, bits=8)  # Quantize input
        x = self.features(x)
        x = quantize_static(x, bits=8)  # Quantize features output
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def ptq(model, device, testloader):
    model = quantize_model_weights(model)
    model = QuantizedAlexNetCIFAR(model)
    model.to(device)

    acc = test(model, device, testloader)
    print(f"Test Accuracy after PTQ (W8A8): {acc:.2f}%")

# --------- Script Entry ---------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, testloader = main(device)
    ptq(model, device, testloader)
