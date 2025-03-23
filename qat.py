import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Power-of-2 quantization function
def power_of_two_quantize(x):
    """Quantizes the input tensor to the nearest power-of-2 value."""
    sign = torch.sign(x)
    x = torch.abs(x)  # Work with absolute values
    log2_x = torch.log2(x + 1e-8)  # Log2 to find exponent (avoid log(0) issues)
    quantized_x = torch.round(log2_x)  # Round to nearest integer
    return sign * (2 ** quantized_x)  # Convert back to power-of-2 values

# Custom ResNet-50 with quantization
class ResNet50Quantized(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50Quantized, self).__init__()
        self.resnet = models.resnet50(weights=None)  # Train from scratch
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # Adjust for CIFAR-10

    def forward(self, x):
        # Apply power-of-2 quantization to all layers dynamically
        for name, param in self.resnet.named_parameters():
            if 'weight' in name or 'bias' in name:  # Only quantize weights and biases
                param.data = power_of_two_quantize(param.data)
        return self.resnet(x)

# Hyperparameters
batch_size = 32
epochs = 30
learning_rate = 0.01

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet-50
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Quantized(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR every 10 epochs

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "resnet50_quantized.pth")
print("Training complete with power-of-2 quantization adaptation!")
