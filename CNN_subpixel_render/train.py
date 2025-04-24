from dataset import SubpixelDataset, remap_and_convolve, get_hvs_kernels
from model import SPRNN
import torch
import torch.nn.functional as F

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
        running_loss = 0
        for Ir, Ig, Ib, Pr, Pg, Pb in train_loader:
            Ir, Ig, Ib = Ir.to(device), Ig.to(device), Ib.to(device)
            Pr, Pg, Pb = Pr.to(device), Pg.to(device), Pb.to(device)

            optimizer.zero_grad()
            Dr, Dg, Db = model(Ir, Ig, Ib, Pr, Pg, Pb)
            loss = compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} | Train Loss: {running_loss / len(train_loader):.4f}")
        
        # Validation loss (optional)
        if val_loader is not None:
            validate_model(model, val_loader, device)

def validate_model(model, val_loader, device='cuda'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for Ir, Ig, Ib, Pr, Pg, Pb in val_loader:
            Ir, Ig, Ib = Ir.to(device), Ig.to(device), Ib.to(device)
            Pr, Pg, Pb = Pr.to(device), Pg.to(device), Pb.to(device)

            Dr, Dg, Db = model(Ir, Ig, Ib, Pr, Pg, Pb)
            loss = compute_loss(Ir, Ig, Ib, Dr, Dg, Db, Pr, Pg, Pb)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(val_loader):.4f}")


    
