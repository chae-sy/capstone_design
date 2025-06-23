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
    B, _, H, W = P.shape                # H, W: full-res
    _, _, h, w = D.shape                # h, w: low-res

    scale_h = H // h                    # ex) 4  (R/B), 2 (G)
    scale_w = W // w                    # ex) 2  (R/B), 2 (G)

    Z = torch.zeros_like(P)

    for b in range(B):
        for i in range(H):
            ni = i // scale_h           # 높이 매핑
            for j in range(W):
                nj = j // scale_w       # 너비 매핑
                if P[b, 0, i, j] == 1:
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
            ii, jj = i % 4, j % 4       # 4×4 반복 타일 좌표

            # ── Green (변화 없음) ───────────────────────────
            if ii % 2 == 0 and jj % 2 == 0:      # (0,0) (0,2) (2,0) (2,2)
                Pg[0, i, j] = 1

            # ── Red / Blue (1↔3 위치 뒤집음) ────────────────
            if (ii, jj) in {(1, 3), (3, 1)}:     # 🔴 R ← 가로·세로 미러된 자리
                Pr[0, i, j] = 1
            elif (ii, jj) in {(1, 1), (3, 3)}:   # 🔵 B ← 기존 R 자리가 B 로
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
    def __init__(self, root_dir, transform=None, img_size=256):
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
        self.img_size = img_size
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        #print(img.shape)
        Pr, Pg, Pb = generate_pixel_masks(self.img_size, self.img_size)
        return img[0:1], img[1:2], img[2:3], Pr, Pg, Pb
