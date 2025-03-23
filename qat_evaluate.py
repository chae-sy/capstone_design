import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Power-of-2 quantization function
def power_of_two_quantize(x):
    """Quantizes the input tensor to the nearest power-of-2 value."""
    sign = torch.sign(x)
    x = torch.abs(x)
    log2_x = torch.log2(x + 1e-8)
    quantized_x = torch.round(log2_x)
    return sign * (2 ** quantized_x)

# Custom ResNet-50 with quantization
class ResNet50Quantized(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Quantized, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        for name, param in self.resnet.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data = power_of_two_quantize(param.data)
        return self.resnet(x)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Quantized(num_classes=10).to(device)
model.load_state_dict(torch.load("resnet50_quantized.pth", map_location=device))
model.eval()

# Evaluate model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
