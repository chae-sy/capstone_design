from dataset import SubpixelDataset, remap_and_convolve, get_hvs_kernels
from model import SPRNN
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb):
    Crb, Cg = get_hvs_kernels()

    Vr = remap_and_convolve(Dr, Pr, Crb)
    Vg = remap_and_convolve(Dg, Pg, Cg)
    Vb = remap_and_convolve(Db, Pb, Crb)

    V = torch.cat([Vr, Vg, Vb], dim=1)
    I = torch.cat([Ir, Ig, Ib], dim=1)

    loss = F.mse_loss(V, I)
    return loss

from torch.utils.data import DataLoader
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=30, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print("Train dataset size:", len(train_loader.dataset))
        print("Train loader length (number of batches):", len(train_loader))
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for Ir, Ig, Ib, Pr, Pg, Pb in loop:
            Ir, Ig, Ib = Ir.to(device), Ig.to(device), Ib.to(device)
            Pr, Pg, Pb = Pr.to(device), Pg.to(device), Pb.to(device)

            optimizer.zero_grad()
            Dr, Dg, Db = model(Ir, Ig, Ib, Pr, Pg, Pb)

            # print("Dr shape:", Dr.shape)
            # print("Ir shape:", Ir.shape)
            # print("Pr shape:", Pr.shape)
            loss = compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"Epoch {epoch+1} | Train Loss: {running_loss / len(train_loader):.4f}")

        # Validation loss (optional)
        if val_loader is not None:
            validate_model(model, val_loader, device)



# --- Subpixel Resize ---
def resize_subpixels(Dr, Dg, Db):
    # Dr, Db를 Dg 크기에 맞춰서 업샘플링
    target_size = (Dg.shape[2], Dg.shape[3])  # (height, width)
    Dr_resized = F.interpolate(Dr, size=target_size, mode='bilinear', align_corners=False)
    Db_resized = F.interpolate(Db, size=target_size, mode='bilinear', align_corners=False)
    return Dr_resized, Dg, Db_resized

# --- Merge to RGB ---
def merge_rgb(R, G, B):
    return torch.cat([R, G, B], dim=1)  # (B, 3, H, W)

# --- Calculate PSNR and SSIM directly on tensors ---
def calculate_metrics(gt_img, pred_img):
    # Tensor to Numpy
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    # Scale 0-1 → 0-255
    if gt_img.max() <= 1.0:
        gt_img = (gt_img * 255).astype(np.uint8)
        pred_img = (pred_img * 255).astype(np.uint8)
    else:
        gt_img = gt_img.astype(np.uint8)
        pred_img = pred_img.astype(np.uint8)

    # Resize pred_img using pure numpy if needed
    if gt_img.shape != pred_img.shape:
        import scipy.ndimage
        zoom_factors = (
            gt_img.shape[0] / pred_img.shape[0],
            gt_img.shape[1] / pred_img.shape[1],
            1  # channel
        )
        pred_img = scipy.ndimage.zoom(pred_img, zoom_factors, order=1)  # bilinear interpolation

    # Compute PSNR and SSIM
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)
    ssim = structural_similarity(gt_img, pred_img, channel_axis=-1, data_range=255)

    return psnr, ssim

# --- Main validate_model ---
def validate_model(model, val_loader, device='cuda'):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for Ir, Ig, Ib, Pr, Pg, Pb in val_loader:
            Ir, Ig, Ib = Ir.to(device), Ig.to(device), Ib.to(device)
            Pr, Pg, Pb = Pr.to(device), Pg.to(device), Pb.to(device)

            # Model prediction
            Dr, Dg, Db = model(Ir, Ig, Ib, Pr, Pg, Pb)

            # Loss
            loss = compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb)
            total_loss += loss.item()

            # --- Prepare GT RGB ---
            Irgb_gt = merge_rgb(Ir, Ig, Ib)

            # --- Prepare Predicted RGB ---
            Dr_resized, Dg_resized, Db_resized = resize_subpixels(Dr, Dg, Db)
            Irgb_pred = merge_rgb(Dr_resized, Dg_resized, Db_resized)

            # --- Batch-wise evaluation ---
            B = Irgb_gt.shape[0]
            for i in range(B):
                psnr, ssim = calculate_metrics(Irgb_gt[i], Irgb_pred[i])
                total_psnr += psnr
                total_ssim += ssim
                count += 1

    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")
    print(f"Average PSNR: {total_psnr / count:.2f} dB")
    print(f"Average SSIM: {total_ssim / count:.4f}")


def evaluate_model(model, val_loader, device='cuda'):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0

    with torch.no_grad():
        for Ir, Ig, Ib, Pr, Pg, Pb in val_loader:
            Ir, Ig, Ib = Ir.to(device), Ig.to(device), Ib.to(device)
            Pr, Pg, Pb = Pr.to(device), Pg.to(device), Pb.to(device)

            Dr, Dg, Db = model(Ir, Ig, Ib, Pr, Pg, Pb)

            Irgb_gt = merge_rgb(Ir, Ig, Ib)
            Dr_resized, Dg_resized, Db_resized = resize_subpixels(Dr, Dg, Db)
            Irgb_pred = merge_rgb(Dr_resized, Dg_resized, Db_resized)

            B = Irgb_gt.shape[0]
            for i in range(B):
                psnr, ssim = calculate_metrics(Irgb_gt[i], Irgb_pred[i])
                total_psnr += psnr
                total_ssim += ssim
                count += 1

    print(f"Average PSNR: {total_psnr / count:.2f} dB")
    print(f"Average SSIM: {total_ssim / count:.4f}")

