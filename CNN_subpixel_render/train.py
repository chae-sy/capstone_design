import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset import SubpixelDataset, remap_and_convolve, get_hvs_kernels
from model import SPRNN


def quantize_to_pow2(tensor, bits=5):
    sign = tensor.sign()
    tensor = tensor.abs()
    log2_val = torch.clamp(torch.round(torch.log2(tensor + 1e-10)), -2 ** (bits - 1), 2 ** (bits - 1) - 1)
    pow2_val = 2.0 ** log2_val
    return sign * pow2_val


def apply_inq(model, quant_ratio=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            num_weights = weight.numel()
            k = int(num_weights * quant_ratio)
            flat = weight.view(-1).abs()
            _, indices = torch.topk(flat, k, largest=True)

            mask = torch.zeros_like(weight.view(-1))
            mask[indices] = 1.0
            mask = mask.view_as(weight)

            module.register_buffer('weight_mask', mask)
            quantized = quantize_to_pow2(weight)
            module.weight.data = module.weight.data * (1 - mask) + quantized * mask


def incremental_quantize(model, epoch, total_epochs, quant_schedule):
    target_ratio = quant_schedule[min(epoch, len(quant_schedule) - 1)]
    for module in model.modules():
        if hasattr(module, 'weight') and hasattr(module, 'weight_mask'):
            weight = module.weight.data
            mask = module.weight_mask

            new_mask = (mask == 1.0).float()
            if new_mask.sum() < int(weight.numel() * target_ratio):
                remaining = 1.0 - new_mask
                num_to_quant = int(weight.numel() * target_ratio) - int(new_mask.sum())
                flat_weights = (weight * remaining).view(-1).abs()
                _, new_indices = torch.topk(flat_weights, num_to_quant, largest=True)
                new_mask_flat = new_mask.view(-1)
                new_mask_flat[new_indices] = 1.0
                new_mask = new_mask_flat.view_as(weight)

                module.weight_mask.copy_(new_mask)
                module.weight.data = module.weight.data * (1 - new_mask) + quantize_to_pow2(weight) * new_mask


def compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb):
    Crb, Cg = get_hvs_kernels()
    Vr = remap_and_convolve(Dr, Pr, Crb)
    Vg = remap_and_convolve(Dg, Pg, Cg)
    Vb = remap_and_convolve(Db, Pb, Crb)
    V = torch.cat([Vr, Vg, Vb], dim=1)
    I = torch.cat([Ir, Ig, Ib], dim=1)
    return F.mse_loss(V, I)


def resize_subpixels(Dr, Dg, Db):
    target_size = (Dg.shape[2], Dg.shape[3])
    Dr_resized = F.interpolate(Dr, size=target_size, mode='bilinear', align_corners=False)
    Db_resized = F.interpolate(Db, size=target_size, mode='bilinear', align_corners=False)
    return Dr_resized, Dg, Db_resized


def merge_rgb(R, G, B):
    return torch.cat([R, G, B], dim=1)


def calculate_metrics(gt_img, pred_img):
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    if gt_img.max() <= 1.0:
        gt_img = (gt_img * 255).astype(np.uint8)
        pred_img = (pred_img * 255).astype(np.uint8)

    if gt_img.shape != pred_img.shape:
        import scipy.ndimage
        zoom_factors = (
            gt_img.shape[0] / pred_img.shape[0],
            gt_img.shape[1] / pred_img.shape[1],
            1
        )
        pred_img = scipy.ndimage.zoom(pred_img, zoom_factors, order=1)

    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)
    ssim = structural_similarity(gt_img, pred_img, channel_axis=-1, data_range=255)
    return psnr, ssim


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

            Dr, Dg, Db = model(Ir, Ig, Ib, Pr, Pg, Pb)
            loss = compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb)
            total_loss += loss.item()

            Irgb_gt = merge_rgb(Ir, Ig, Ib)
            Dr_resized, Dg_resized, Db_resized = resize_subpixels(Dr, Dg, Db)
            Irgb_pred = merge_rgb(Dr_resized, Dg_resized, Db_resized)

            for i in range(Irgb_gt.shape[0]):
                psnr, ssim = calculate_metrics(Irgb_gt[i], Irgb_pred[i])
                total_psnr += psnr
                total_ssim += ssim
                count += 1

    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")
    print(f"Average PSNR: {total_psnr / count:.2f} dB")
    print(f"Average SSIM: {total_ssim / count:.4f}")
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, epochs=30, device='cuda',
                save_loss_path='train_losses.pt', save_val_loss_path='val_losses.pt',
                quant_schedule=[0.3, 0.6, 0.9]):

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
    scaler = GradScaler()

    train_losses = []
    val_losses = []

    apply_inq(model, quant_ratio=quant_schedule[0])

    for epoch in range(epochs):
        incremental_quantize(model, epoch, epochs, quant_schedule)
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True, ncols=100)
        for Ir, Ig, Ib, Pr, Pg, Pb in pbar:
            Ir, Ig, Ib = Ir.to(device), Ig.to(device), Ib.to(device)
            Pr, Pg, Pb = Pr.to(device), Pg.to(device), Pb.to(device)

            optimizer.zero_grad()
            with autocast():
                Dr, Dg, Db = model(Ir, Ig, Ib, Pr, Pg, Pb)
                loss = compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")

        if val_loader is not None:
            val_loss = validate_model(model, val_loader, device)
            val_losses.append(val_loss)

    os.makedirs(os.path.dirname(save_loss_path), exist_ok=True)
    torch.save(train_losses, save_loss_path)
    torch.save(val_losses, save_val_loss_path)
    print(f"✅ Train/Val losses saved to {save_loss_path} / {save_val_loss_path}")
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
