import torch
from torch.utils.data import DataLoader
import numpy as np
from model import SPRNN
from train import train_model, evaluate_model
from dataset import SubpixelDataset, DIV2KDataset
import torchvision.transforms as transforms
import random
import argparse


# def generate_fake_image_tensor(H, W):
#     return torch.rand(1, H, W)

# def build_dataset(num_samples=100, H=256, W=256):
#     dataset = []
#     for _ in range(num_samples):
#         Ir = generate_fake_image_tensor(H, W)
#         Ig = generate_fake_image_tensor(H, W)
#         Ib = generate_fake_image_tensor(H, W)

#         Pr, Pg, Pb = generate_pixel_masks(H, W)

#         dataset.append((Ir, Ig, Ib, Pr, Pg, Pb))
#     return dataset
def parse_args():
    parser = argparse.ArgumentParser(description='SPRNN training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
    args = parser.parse_args()
    return args

def save_model(model):
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][time.localtime().tm_mon - 1]
        date = time.localtime().tm_mday
        today = f'{month}{date}'
        now_time = f'{time.localtime().tm_hour}{time.localtime().tm_min}'
        file_name = f'model_{today}_{now_time}.pt'
        torch.save(model.state_dict(), file_name)
        print('Model saved:', file_name)

if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Settings
    b_size = 128
    num_epochs = 30
    image_size = 100
    num_w = 16
    transform = transforms.Compose([
        transforms.RandomCrop((image_size, image_size)),  # Crop to 256x256 patches
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(lambda img: transforms.functional.rotate(img, angle=random.choice([0, 90, 180, 270]))),
        transforms.ToTensor(),
    ])

    # Paths to datasets
    div2k_train_dir = 'data/DIV2K_patches_4800'
    div2k_valid_dir = 'data/DIV2K_valid_HR'
    div2k_test_dir = 'data/DIV2K_test_HR'

    # Initialize datasets   
    train_dataset = DIV2KDataset(root_dir=div2k_train_dir, img_size=image_size, transform=transforms.ToTensor())
    valid_dataset = DIV2KDataset(root_dir=div2k_valid_dir, img_size=image_size, transform=transform)  # You can change this to no augmentation if needed
    eval_dataset = DIV2KDataset(root_dir=div2k_test_dir, img_size=image_size, transform=transforms.ToTensor())  # Just normalization

    # Initialize dataloaders
    train_loader = DataLoader(SubpixelDataset(train_dataset), batch_size=b_size, shuffle=True, num_workers=num_w, pin_memory=True)
    valid_loader = DataLoader(SubpixelDataset(valid_dataset), batch_size=b_size, shuffle=False, num_workers=num_w, pin_memory=True)
    eval_loader = DataLoader(SubpixelDataset(eval_dataset), batch_size=b_size, shuffle=False, num_workers=num_w, pin_memory=True)

    # Model
    model = SPRNN()

    # Train
    train_model(model, train_loader, valid_loader, epochs=num_epochs, device=device)

    # Evaluate
    evaluate_model(model, eval_loader)

    # Save model
    save_model(model)
