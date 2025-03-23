import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, random_split

def load_model():
    # Load ResNet-50 and modify FC layer
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, 10)  # Change output to 10 classes

    # Freeze all layers except fc
    for param in resnet50.parameters():
        param.requires_grad = False  # Freeze all layers
    resnet50.fc.weight.requires_grad = True  # Unfreeze fc
    resnet50.fc.bias.requires_grad = True
    print('model freezed')

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)
    print('model to deivce')
    return resnet50
    
def load_data_and_train(resnet50):

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50.fc.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ResNet-50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

    # Reduce dataset size (use only 5000 images)
    subset_indices = torch.randperm(len(full_train_dataset))[:5000]  # Randomly select 5000 samples
    small_dataset = Subset(full_train_dataset, subset_indices)

    # Split into training (80%) and validation (20%)
    train_size = int(0.8 * len(small_dataset))  # 4000 images
    val_size = len(small_dataset) - train_size  # 1000 images
    train_dataset, val_dataset = random_split(small_dataset, [train_size, val_size])

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    print('train & val data loaded')

    # Train only the fc layer
    epochs =150
    patience=10
    no_improvement_counter=0
    best_train_loss=float('inf')
    resnet50.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = resnet50(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        resnet50.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet50(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
            
        # Adjust learning rate based on validation loss
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            no_improvement_counter=0
        else:
            no_improvement_counter+=1
        if no_improvement_counter>=patience:
            print(f"early stopping after {patience} epochs with no improvement.")
            break
    # Save trained model
    _save_model(resnet50)
    print("Training complete!")



def test(resnet50):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ResNet-50
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    # Perform inference on CIFAR-10 with PoT quantized weights
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = resnet50(images)  # Get model predictions

            predicted = torch.argmax(outputs, dim=1)  # Corrected: Get only the index of the max value

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print the model's decisions and actual labels
            print("Predicted:", predicted.tolist())
            print("Actual   :", labels.tolist())

    # Print accuracy after PoT quantization
    accuracy = 100 * correct / total
    print(f'Accuracy after Power-of-2 Quantization: {accuracy:.2f}%')

def weight_dist ():
    # Visualizing one layer's weight distribution after PoT quantization
    layer_name, sample_weights = next(iter(resnet50.named_parameters()))
    sample_weights = sample_weights.cpu().detach().numpy().flatten()

    plt.figure(figsize=(8, 4))
    plt.hist(sample_weights, bins=100, alpha=0.75, color='blue', edgecolor='black')
    plt.title(f'PoT-Quantized Weight Distribution - {layer_name}')
    plt.xlabel('Weight Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def _save_model(model):
    import torch
    import time
    month=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][time.localtime().tm_mon-1]
    date=time.localtime().tm_mday
    today=f'{month}{date}'
    time=f'{time.localtime().tm_hour}{time.localtime().tm_min}'
    file_name=f'model_{today}_{time}.pt'
    scripted_model = torch.jit.script(model)  # Convert model to TorchScript
    torch.jit.save(scripted_model, file_name)
    print('model saved : ', file_name)


if __name__=="__main__":
    model=load_model()
    load_data_and_train(model)
    test(model)
