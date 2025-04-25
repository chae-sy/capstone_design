from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class SubpixelDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list  # list of (Ir, Ig, Ib, Pr, Pg, Pb)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        Ir, Ig, Ib, Pr, Pg, Pb = self.image_list[idx]
        return Ir, Ig, Ib, Pr, Pg, Pb

def remap_and_convolve(D, P, kernel):
    """
    Maps subpixel pattern D to a zero-padded full-resolution pattern using P, then applies bilinear convolution.
    D: [B, 1, h, w]
    P: [B, 1, H, W]
    """
    B, _, H, W = P.shape
    Z = torch.zeros_like(P)
    mask = (P == 1)
    
    # Map low-res D to high-res Z
    for b in range(B):
        d_idx = 0
        for i in range(H):
            for j in range(W):
                if P[b, 0, i, j] == 1:
                    ni = i // 2
                    nj = j // 4 if D.shape[-1] * 4 == W else j // 2
                    Z[b, 0, i, j] = D[b, 0, ni, nj]
    # Convolve with kernel
    kernel = kernel.to(Z.device).unsqueeze(0).unsqueeze(0)
    Z_blur = F.conv2d(Z, kernel, padding=kernel.shape[-1]//2)
    return Z_blur

def get_hvs_kernels():
    Crb = torch.tensor([
        [0, 0, 0, 0.06, 0, 0, 0],
        [0, 0, 0.19, 0.25, 0.19, 0, 0],
        [0, 0.19, 0.50, 0.56, 0.50, 0.19, 0],
        [0.06, 0.25, 0.56, 1.0, 0.56, 0.25, 0.06],
        [0, 0.19, 0.50, 0.56, 0.50, 0.19, 0],
        [0, 0, 0.19, 0.25, 0.19, 0, 0],
        [0, 0, 0, 0.06, 0, 0, 0]
    ], dtype=torch.float32)

    Cg = torch.tensor([
        [0.25, 0.50, 0.25],
        [0.50, 1.00, 0.50],
        [0.25, 0.50, 0.25]
    ], dtype=torch.float32)

    return Crb, Cg

def generate_pixel_masks(H, W):
    Pr = torch.zeros(1, H, W)
    Pg = torch.zeros(1, H, W)
    Pb = torch.zeros(1, H, W)

    for i in range(H):
        for j in range(W):
            if (i % 4 == 1 and j % 4 == 1) or (i % 4 == 3 and j % 4 == 3):
                Pr[0, i, j] = 1
            if i % 2 == 0 and j % 2 == 0:
                Pg[0, i, j] = 1
            if (i % 4 == 1 and j % 4 == 3) or (i % 4 == 3 and j % 4 == 1):
                Pb[0, i, j] = 1

    return Pr, Pg, Pb

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to DIV2K image folder
            transform (callable, optional): A torchvision-style transform pipeline
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([
            f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img



