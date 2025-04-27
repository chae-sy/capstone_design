import torch
import torch.nn as nn
import torch.nn.functional as F
from train import quantize_to_pow2, update_fixed_mask
# -------- CNN Model --------
class SimpleCNN_INQ(nn.Module):
    def __init__(self):
        super(SimpleCNN_INQ, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(8*8*64, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # INQ masks
        self.fixed_masks = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                self.fixed_masks[name] = torch.zeros_like(param.data, dtype=torch.bool)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def apply_inq(self):
        """Quantize fixed weights to power-of-2 during training."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    mask = self.fixed_masks[name].to(param.device)
                    quantized = quantize_to_pow2(param.data)
                    param.data = torch.where(mask, quantized, param.data)

    def update_masks(self, prune_percent):
        """Update masks to fix more weights."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name:
                    current_mask = self.fixed_masks[name].to(param.device)  # <<< FIX
                    updated_mask = update_fixed_mask(param.data, current_mask, prune_percent)
                    self.fixed_masks[name] = updated_mask.to(param.device)  # <<< ensure saved mask is also moved


