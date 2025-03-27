import torch
from torch.utils.data import DataLoader
import numpy as np
from model import SPRNN
from train import train_model
from dataset import SubpixelDataset, generate_pixel_masks

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

    # Load dataset (replace with real DIV2K data later)
    train_data = build_dataset(100, image_size, image_size)
    val_data = build_dataset(20, image_size, image_size)

    train_loader = DataLoader(SubpixelDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SubpixelDataset(val_data), batch_size=batch_size)

    # Model
    model = SPRNN()

    # Train
    train_model(model, train_loader, val_loader, epochs=num_epochs, device=device)
