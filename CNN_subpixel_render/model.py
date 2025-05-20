import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Shared CNN block for RGB channels
# -----------------------------
class SPRNNBlock(nn.Module):
    def __init__(self, in_channels=2, mid_channels=16, kernel_size=3):
        super(SPRNNBlock, self).__init__()
        padding = kernel_size // 2

        # High-resolution conv layers
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm2d(mid_channels)

        # MaxPool - stride varies based on color
        self.maxpool_g = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_rb = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))

        # Low-resolution conv layers
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm2d(mid_channels)
        self.conv5 = nn.Conv2d(mid_channels, 1, kernel_size, padding=padding)
        self.bn5 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x, color):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        if color == 'G':
            x = self.maxpool_g(x)
        else:
            x = self.maxpool_rb(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))
        return x



# -----------------------------
# Full SPRNN Model
# -----------------------------
class SPRNN(nn.Module):
    def __init__(self):
        super(SPRNN, self).__init__()
        self.shared_block = SPRNNBlock()

    def forward(self, Ir, Ig, Ib, Pr, Pg, Pb):
        # align the dimension to torch.cat
        if Ir.dim() == 3: Ir = Ir.unsqueeze(1)
        if Ig.dim() == 3: Ig = Ig.unsqueeze(1)
        if Ib.dim() == 3: Ib = Ib.unsqueeze(1)
        # Concatenate input image with pixel layout mask for each color
        Dr = self.shared_block(torch.cat([Ir, Pr], dim=1), 'R')
        Dg = self.shared_block(torch.cat([Ig, Pg], dim=1), 'G')
        Db = self.shared_block(torch.cat([Ib, Pb], dim=1), 'B')
        return Dr, Dg, Db
