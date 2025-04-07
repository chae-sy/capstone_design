from torch.utils.data import Dataset
import torch

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

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, transform=None):
        """
        Args:
            hr_dir (string): Directory with high-resolution images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.hr_dir = hr_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(hr_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.hr_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Generate pixel arrangement maps Pr, Pg, Pb
        Pr, Pg, Pb = generate_pixel_masks(image.size[1], image.size[0])
        
        # Split image into Ir, Ig, Ib
        Ir, Ig, Ib = image.split()
        
        return Ir, Ig, Ib, Pr, Pg, Pb

class SPADataset(Dataset):
    def __init__(self, spa_dir, transform=None):
        """
        Args:
            spa_dir (string): Directory with SPA images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.spa_dir = spa_dir
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(spa_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.spa_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Generate pixel arrangement maps Pr, Pg, Pb
        Pr, Pg, Pb = generate_pixel_masks(image.size[1], image.size[0])
        
        # Split image into Ir, Ig, Ib
        Ir, Ig, Ib = image.split()
        
        return Ir, Ig, Ib, Pr, Pg, Pb


