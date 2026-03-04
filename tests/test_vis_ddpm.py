import os
import argparse
from pathlib import Path
import joblib 

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Assumes this file exists in your path
from src_refactored.context_unet import ContextUnet

# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------
timesteps = 500
beta1 = 1e-4
beta2 = 0.02

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_feat = 64
n_cfeat = 5
height = 16
channels = 1

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# b_t = cosine_beta_schedule(timesteps)
# b_t = torch.cat([torch.tensor([0.0], device=device), b_t])
# a_t = 1 - b_t
# ab_t = torch.cumprod(a_t, dim=0)
# ab_t[0] = 1

# Diffusion schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()
ab_t[0] = 1

# ---------------------------------------------------------------------
# Load Scaler (Global)
# ---------------------------------------------------------------------
# Ensure wind_scaler.pkl is in the same directory as this script
try:
    SCALER = joblib.load('data/wind_scaler.pkl')
    print("Loaded wind_scaler.pkl successfully.")
except FileNotFoundError:
    print("WARNING: wind_scaler.pkl not found. Please place it in the same directory.")
    SCALER = None

# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(nn_model, n_sample, save_rate=20):
    samples = torch.randn(n_sample, channels, height, height).to(device)
    intermediate = []
    
    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')
        t_float = torch.tensor([i / timesteps]).to(device)
        t_float = t_float.repeat(n_sample, 1, 1, 1)
        z = torch.randn_like(samples) if i > 1 else 0
        eps = nn_model(samples, t_float) 
        samples = denoise_add_noise(samples, i, eps, z)

        if i % save_rate == 0 or i == timesteps or i < 8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate) 
    return samples, intermediate

# ---------------------------------------------------------------------
# Visualization Helpers (UPDATED FOR SCALER)
# ---------------------------------------------------------------------
def denormalize(x: np.ndarray) -> np.ndarray:
    """
    Inverse-scales data using the loaded StandardScaler.
    Input: (N, C, H, W)
    Output: (N, C, H, W) in range [0, 255]
    """
    # If no scaler found, fallback to clipping 0-1 (or assume it's already close)
    if SCALER is None:
        return np.clip(x, 0.0, 1.0)

    N, C, H, W = x.shape
    # 1. Reshape to (N, 256) because scaler expects flattened features
    x_flat = x.reshape(N, -1)
    
    # 2. Inverse Transform: x * std + mean
    x_unscaled = SCALER.inverse_transform(x_flat)
    
    # 3. Reshape back to image
    x_restored = x_unscaled.reshape(N, C, H, W)
    
    # 4. Clip to valid [0, 255] pixel range
    return np.clip(x_restored, 0, 255)

def plot_sample(intermediate, n_sample, nrow, save_dir, name, figsize=None, save=True):
    os.makedirs(save_dir, exist_ok=True)
    T, N, C, H, W = intermediate.shape
    n_sample = min(n_sample, N)
    ncol = n_sample // nrow

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (ncol * 2, nrow * 2))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    def update(frame_idx):
        imgs = denormalize(intermediate[frame_idx]) 
        for i in range(n_sample):
            ax = axes[i]
            img = np.transpose(imgs[i], (1, 2, 0))
            # UPDATED: vmax=255 for standard pixel range
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        return axes

    ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)

    if save:
        out_path = os.path.join(save_dir, f"{name}.gif")
        writer = PillowWriter(fps=10)
        ani.save(out_path, writer=writer)
        print(f"Saved animation to {out_path}")

    plt.close(fig)
    return ani

def save_grid(samples: torch.Tensor, save_path: str, nrow: int = 4):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imgs = samples.detach().cpu().numpy()
    imgs = denormalize(imgs)

    N, C, H, W = imgs.shape
    nrow = min(nrow, N)
    ncol = int(np.ceil(N / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i in range(min(N, len(axes))):
        img = np.transpose(imgs[i], (1, 2, 0))
        # UPDATED: vmax=255 for standard pixel range
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=255)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {save_path}")

# ---------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_XX.pth")
    parser.add_argument("--output_dir", type=str, default="./tests/vis_out")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--save_rate", type=int, default=20)
    args = parser.parse_args()

    # Load Model
    nn_model = ContextUnet(
        in_channels=1,
        n_feat=n_feat,
        n_cfeat=n_cfeat,
        height=height,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    state_dict = torch.load(args.checkpoint, map_location=device)
    nn_model.load_state_dict(state_dict)
    nn_model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # Sample
    samples, intermediate_ddpm = sample_ddpm(
        nn_model, 
        n_sample=args.num_samples, 
        save_rate=args.save_rate
    )

    # Save
    ckpt_name = Path(args.checkpoint).stem
    
    # 1. Animation
    ani_name = f"ani_{ckpt_name}"
    plot_sample(intermediate_ddpm, args.num_samples, nrow=4, 
                save_dir=args.output_dir, name=ani_name)

    # 2. Final Grid
    grid_path = os.path.join(args.output_dir, f"{ckpt_name}_grid.png")
    save_grid(samples, grid_path, nrow=4)

    print("Done.")

if __name__ == "__main__":
    main()
