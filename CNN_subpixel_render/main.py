import torch
from torch.utils.data import DataLoader
import numpy as np
from model import SPRNN
from train import train_model
from dataset import SubpixelDataset, generate_pixel_masks, DIV2KDataset
from PIL import Image
import torchvision.transforms as transforms
import random

if __name__ == "__main__":
    # Settings
    batch_size = 512
    num_epochs = 10
    image_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    transform = transforms.Compose([
        transforms.RandomCrop((image_size, image_size)),  # Crop to 256x256 patches
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=random.choice([0, 90, 180, 270]))),
        transforms.ToTensor(),
    ])


    # Paths to datasets
    div2k_train_dir = 'data/DIV2K_train_HR'
    div2k_valid_dir = 'data/DIV2K_valid_HR'
    div2k_test_dir = 'data/DIV2K_test_HR'

    # Initialize datasets   
    train_dataset = DIV2KDataset(root_dir=div2k_train_dir, transform=transform)
    valid_dataset = DIV2KDataset(root_dir=div2k_valid_dir, transform=transform)  # You can change this to no augmentation if needed
    eval_dataset = DIV2KDataset(root_dir=div2k_test_dir, transform=transforms.ToTensor())  # Just normalization

    # Initialize dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


    # Model
    model = SPRNN()

    # Train
    train_model(model, train_loader, eval_loader, epochs=num_epochs, device=device)
