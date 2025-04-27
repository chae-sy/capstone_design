# full_powerof2_training.py (All-in-One Integrated with 8bit pretrain)

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# --------------- Dataset: BitQuantized CIFAR10 ---------------
class BitQuantizedCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, bits=3):
        super(BitQuantizedCIFAR10, self).__init__(root, train=train,
                                                   transform=transform,
                                                   target_transform=target_transform,
                                                   download=download)
        self.bits = bits

    def quantize_image(self, img_tensor):
        levels = 2 ** self.bits
        img_tensor = torch.floor(img_tensor * (levels - 1)) / (levels - 1)
        return img_tensor

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        img = self.quantize_image(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# --------------- Quantization Functions ---------------
def quantize_input(x, bits=3):
    levels = 2 ** bits
    x = torch.floor(x * (levels - 1)) / (levels - 1)
    return x

def quantize_activation(x, bits=6):
    levels = 2 ** bits
    x = torch.clamp(x, 0, 1)
    x = torch.round(x * (levels - 1)) / (levels - 1)
    return x

def quantize_weight_to_power_of_2(w):
    w_abs = w.abs()
    w_sign = w.sign()
    w_log2 = torch.log2(w_abs + 1e-10)
    w_rounded = torch.round(w_log2)
    w_pow2 = 2 ** w_rounded
    wq = w_sign * w_pow2
    wq[w_abs < 2**-7] = 0
    return wq

# --------------- CNN Model ---------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        x = quantize_activation(F.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = quantize_activation(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = quantize_activation(F.relu(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --------------- Training and Testing Functions ---------------
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

# --------------- Full Incremental Network Quantization ---------------
def incremental_quantize_and_retrain(model, train_loader, test_loader, device, num_epochs=5):
    layers = [model.conv1, model.conv2, model.conv3, model.fc]
    schedule = [0.3, 0.6, 0.9, 1.0]

    for ratio in schedule:
        print(f"Quantizing {int(ratio*100)}% of weights...")

        for layer in layers:
            weight = layer.weight.data
            mask = getattr(layer, 'quant_mask', torch.zeros_like(weight).bool())

            unfrozen_indices = (mask == 0)
            unfrozen_weights = weight[unfrozen_indices]

            if unfrozen_weights.numel() == 0:
                continue

            k = int(ratio * unfrozen_weights.numel())
            if k == 0:
                continue

            values, indices = torch.topk(unfrozen_weights.abs().flatten(), k, largest=True)
            full_indices = unfrozen_indices.flatten().nonzero()[indices]

            weight_flat = weight.flatten()
            weight_flat[full_indices] = quantize_weight_to_power_of_2(weight_flat[full_indices])
            layer.weight.data = weight_flat.view_as(weight)

            flat_mask = mask.flatten()
            flat_mask[full_indices] = 1
            layer.quant_mask = flat_mask.view_as(weight)

        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
            if hasattr(layer, 'quant_mask'):
                param = layer.weight
                param.requires_grad = True
                mask = layer.quant_mask
                param.register_hook(lambda grad, mask=mask: grad * (~mask))

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        for epoch in range(num_epochs):
            train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        print(f"Accuracy after {int(ratio*100)}% quantization: {acc:.2f}%")

# --------------- Main Function ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_8bit = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_3bit = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader_full8 = DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_8bit),
        batch_size=128, shuffle=True)

    train_loader_3bit = DataLoader(
        BitQuantizedCIFAR10(root='./data', train=True, download=False, transform=transform_3bit, bits=3),
        batch_size=128, shuffle=True)

    test_loader_8bit = DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_8bit),
        batch_size=128, shuffle=False)

    test_loader_3bit = DataLoader(
        BitQuantizedCIFAR10(root='./data', train=False, download=False, transform=transform_3bit, bits=3),
        batch_size=128, shuffle=False)

    model = SimpleCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        train(model, device, train_loader_full8, optimizer, epoch)
    acc = test(model, device, test_loader_8bit)
    print(f"Accuracy after full-precision 8-bit pretraining: {acc:.2f}%")

    incremental_quantize_and_retrain(model, train_loader_3bit, test_loader_3bit, device, num_epochs=5)

    acc = test(model, device, test_loader_3bit)
    print(f"Final Accuracy after full incremental quantization: {acc:.2f}%")

if __name__ == '__main__':
    main()

