import torch
from torch.utils.data import DataLoader
import numpy as np
from model import SPRNN
from train import train_model
from dataset import SubpixelDataset, generate_pixel_masks, DIV2KDataset, SPADataset
from PIL import Image
import torchvision.transforms as transforms

def generate_fake_image_tensor(H, W):
    return torch.rand(1, H, W)

def build_dataset(num_samples=100, H=256, W=256):
    dataset = []
    for _ in range(num_samples):
        Ir = generate_fake_image_tensor(H, W)
        Ig = generate_fake_image_tensor(H, W)
        Ib = generate_fake_image_tensor(H, W)

        Pr, Pg, Pb = generate_pixel_masks(H, W)

        dataset.append((Ir, Ig, Ib, Pr, Pg, Pb))
    return dataset

if __name__ == "__main__":
    # Settings
    batch_size = 8
    num_epochs = 10
    image_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.RandomCrop((256, 256)),  # Crop to 256x256 patches
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([0, 90, 180, 270]),
        transforms.ToTensor(),
    ])


    # Paths to datasets
    div2k_train_dir = '../data/DIV2K/DIV2K_train_HR'
    div2k_valid_dir = '../data/DIV2K/DIV2K_valid_HR'
    spa_dir = '../data/SPA'

    # Initialize datasets
    train_dataset = DIV2KDataset(hr_dir=div2k_train_dir, transform=transform)
    valid_dataset = DIV2KDataset(hr_dir=div2k_valid_dir, transform=transform)
    eval_dataset = SPADataset(spa_dir=spa_dir, transform=transform)

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)


    # Model
    model = SPRNN()

    # Train
    train_model(model, train_loader, val_loader, epochs=num_epochs, device=device)
