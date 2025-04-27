import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

# --------- Helper: Quantization Functions ---------
def quantize_activation(x, bits=8):
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

# --------- Dataset: Normal 8-bit CIFAR10 ---------
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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 32x32 -> 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 32x32 -> 16x16

            nn.Conv2d(64, 192, kernel_size=3, padding=1),           # 16x16 -> 16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 16x16 -> 8x8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # 8x8 -> 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # 8x8 -> 8x8
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # 8x8 -> 8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # 8x8 -> 4x4
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

        # INQ masks
        self.fixed_masks = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                self.fixed_masks[name] = torch.zeros_like(param.data, dtype=torch.bool)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def apply_inq(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    mask = self.fixed_masks[name].to(param.device)
                    quantized = quantize_weight_to_power_of_2(param.data)
                    param.data = torch.where(mask, quantized, param.data)

    def update_masks(self, prune_percent):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    current_mask = self.fixed_masks[name].to(param.device)
                    unfixed_weights = param.data[current_mask == 0]
                    if unfixed_weights.numel() == 0:
                        continue
                    k = int(prune_percent * unfixed_weights.numel())
                    if k == 0:
                        continue
                    threshold = unfixed_weights.abs().kthvalue(k).values.item()
                    new_fixed = (param.data.abs() >= threshold) & (current_mask == 0)
                    self.fixed_masks[name] = (current_mask | new_fixed).to(param.device)

    def register_hooks(self):
        """Register backward hooks to zero gradients of fixed weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                mask = self.fixed_masks[name].to(param.device)
                def hook_fn(grad, mask=mask):
                    return grad * (~mask)
                param.register_hook(hook_fn)

    def apply_inq(self):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    mask = self.fixed_masks[name].to(param.device)
                    quantized = quantize_weight_to_power_of_2(param.data)
                    param.data = torch.where(mask, quantized, param.data)

    def update_masks(self, prune_percent):
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    current_mask = self.fixed_masks[name].to(param.device)
                    unfixed_weights = param.data[current_mask == 0]
                    if unfixed_weights.numel() == 0:
                        continue
                    k = int(prune_percent * unfixed_weights.numel())
                    if k == 0:
                        continue
                    threshold = unfixed_weights.abs().kthvalue(k).values.item()
                    new_fixed = (param.data.abs() >= threshold) & (current_mask == 0)
                    self.fixed_masks[name] = (current_mask | new_fixed).to(param.device)

    def register_hooks(self):
        """Register backward hooks to zero gradients of fixed weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                mask = self.fixed_masks[name].to(param.device)
                def hook_fn(grad, mask=mask):
                    return grad * (~mask)
                param.register_hook(hook_fn)
def test(model, device, testloader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
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
# --------- Training and Testing Functions ---------
def train(model, device, trainloader, optimizer, epoch, loss_list):
    model.train()
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(trainloader, desc=f"Train Epoch {epoch}")
    running_loss = 0.0
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        model.apply_inq()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(trainloader)
    loss_list.append(avg_loss)

# --------- Main Script ---------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_cifar10()

    model = AlexNetCIFAR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    total_epochs = 150
    prune_schedule = [0.3, 0.3, 0.4]  # 30%, 30%, 40%
    prune_interval = total_epochs // len(prune_schedule)

    train_losses = []  # <--- 추가

    for epoch in range(1, total_epochs+1):
        train(model, device, trainloader, optimizer, epoch, train_losses)
        test(model, device, testloader)
        scheduler.step()

        if epoch % prune_interval == 0 and prune_schedule:
            prune_percent = prune_schedule.pop(0)
            print(f"--> Fixing {prune_percent*100:.1f}% of weights")
            model.update_masks(prune_percent)
            model.register_hooks()  # <-- very important to re-register hooks after updating masks!

    # 모든 학습 끝난 뒤, loss 저장
    torch.save(train_losses, "train_losses.pt")
    print("Training losses saved to train_losses.pt")

if __name__ == '__main__':
    main()

