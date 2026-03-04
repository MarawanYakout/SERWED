import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler 

# Imports from your local modules
from src_refactored.diffusion import *
from src_refactored.datasets import CustomDataset, transform, CustomContextDataset
from src_refactored.context_unet import ContextUnet

# --- HELPER: EMA Class ---
class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# --- HYPERPARAMETERS ---
timesteps = 500
beta1 = 1e-3
beta2 = 0.02
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64 
n_cfeat = 10 
height = 16 
save_dir = './weights_context/'
sample_dir = './generated_samples/'

# training hyperparameters
batch_size = 64
n_epoch = 120
lrate = 1e-3
min_snr_gamma = 20.0 # Standard value recommended in Min-SNR paper

# --- DDPM NOISE SCHEDULE ---
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()     
ab_t[0] = 1

# --- MODEL INITIALIZATION ---
model = ContextUnet(in_channels=1, n_feat=n_feat, n_cfeat=n_cfeat, height=height).to(device)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)
ema = EMA(0.995)

def perturb_input(x, t, noise):
    if noise.dim() == 5:
        noise = noise.squeeze(2)
    
    w_x = ab_t.sqrt()[t].view(-1, 1, 1, 1)
    w_n = (1 - ab_t[t]).sqrt().view(-1, 1, 1, 1)
    
    return w_x * x + w_n * noise

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_save_context(model, epoch, n_sample=16):
    model.eval()
    x = torch.randn(n_sample, 1, height, height).to(device)
    
    # Random context labels for testing
    random_labels = torch.randint(0, n_cfeat, (n_sample,)).to(device)
    context = F.one_hot(random_labels, num_classes=n_cfeat).float()
    
    for i in range(timesteps, 0, -1):
        t = torch.tensor([i / timesteps]).to(device)
        z = torch.randn_like(x) if i > 1 else 0
        pred_noise = model(x, t, c=context)
        x = denoise_add_noise(x, i, pred_noise, z)
        
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        
    save_image(x.clamp(0, 1), f"{sample_dir}/epoch_{epoch}_context.png", nrow=4)
    model.train() 

if __name__ == "__main__":
    # 1. Load Data
    full_dataset = CustomContextDataset(
        "./data/context/wind_1D16X16.npy",
        "./pregenerated_noise",
        "./data/context/wind_context_1D16X16.npy",
        transform
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epoch, eta_min=1e-6)
    scaler = GradScaler()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ep in range(n_epoch):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, mininterval=2, desc=f"Train Ep {ep}")
        
        for x, noise, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device).long()
            c = F.one_hot(c, num_classes=n_cfeat).float()
            #noise = noise.to(device)
            noise = torch.randn_like(x)
            
            # --- CONTEXT MASKING ---
            context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.9).to(device)
            c = c * context_mask.unsqueeze(-1)
            
            t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
            x_pert = perturb_input(x, t, noise)
            
            with autocast():
                pred_noise = model(x_pert, t / timesteps, c=c)
                
                # --- MIN-SNR LOSS WEIGHTING ---
                # SNR = alpha_bar / (1 - alpha_bar)
                t_idx = t.long()
                snr = ab_t[t_idx] / (1 - ab_t[t_idx])
                
                # Weight = min(snr, gamma) / snr
                # This caps the loss for low-noise/high-SNR timesteps
                mse_loss_weight = torch.clamp(snr, max=min_snr_gamma) / snr
                
                # Calculate per-sample MSE (mean over H, W, C)
                raw_mse = F.mse_loss(pred_noise, noise, reduction='none')
                raw_mse = raw_mse.flatten(start_dim=1).mean(dim=1)
                #print(f"DEBUG: raw_mse shape: {raw_mse.shape}") 
                #print(f"DEBUG: mse_loss_weight shape: {mse_loss_weight.shape}")
                
                
                # Apply Min-SNR weights
                loss = (raw_mse * mse_loss_weight).mean()
            
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            ema.update_model_average(ema_model, model)
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, noise, c in val_loader:
                x = x.to(device).float()
                c = c.to(device).long()
                #noise = noise.to(device).float()
                noise = torch.randn_like(x)
                c = F.one_hot(c, num_classes=n_cfeat).float()
                
                t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device)
                x_pert = perturb_input(x, t, noise)
                
                # Standard MSE for validation metrics
                pred_noise = model(x_pert, t / timesteps, c=c)
                loss = F.mse_loss(pred_noise, noise)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        #print(f"Epoch {ep} | Train Loss (Weighted): {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- SAVE & LOGGING ---
        if ep % 4 == 0 or ep == (n_epoch - 1):
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{ep}.pth"))
            torch.save(ema_model.state_dict(), os.path.join(save_dir, f"ema_model_{ep}.pth"))
            
            with open(os.path.join(save_dir, "loss_log.txt"), "a") as f:
                f.write(f"Epoch {ep}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n")
            
            print("Generating validation samples...")
            sample_save_context(ema_model, ep)

    print('Training completed.')