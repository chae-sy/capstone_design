import torch
from torch.utils.data import DataLoader
import numpy as np
from model import SPRNN
from train import train_model
from dataset import SubpixelDataset, DIV2KDataset
import torchvision.transforms as transforms
import random

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
    train_dataset = DIV2KDataset(root_dir=div2k_train_dir, img_size=image_size, transform=transform)
    valid_dataset = DIV2KDataset(root_dir=div2k_valid_dir, img_size=image_size, transform=transform)  # You can change this to no augmentation if needed
    eval_dataset = DIV2KDataset(root_dir=div2k_test_dir, img_size=image_size, transform=transforms.ToTensor())  # Just normalization

    # Initialize dataloaders
    train_loader = DataLoader(SubpixelDataset(train_dataset), batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(SubpixelDataset(valid_dataset), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(SubpixelDataset(eval_dataset), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # # Load dataset (replace with real DIV2K data later)
    # train_data = build_dataset(100, image_size, image_size)
    # val_data = build_dataset(20, image_size, image_size)
    # test_data 
    # train_loader = DataLoader(SubpixelDataset(train_data), batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(SubpixelDataset(val_data), batch_size=batch_size)
    # test_loader = 


    # Model
    model = SPRNN()

    # Train
    train_model(model, train_loader, valid_loader, epochs=num_epochs, device=device)