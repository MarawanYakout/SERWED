# train_cli.py (FIXED)

"""
CLI script for training DDPM with pre-generated noise.
Syncs logic with trainer.py (EMA, Mixed Precision, Validation Split).
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler 
from torchvision.utils import save_image

# Local imports
from src_refactored.context_unet import ContextUnet
from src_refactored.datasets import CustomDataset, transform
from src_refactored.diffusion import make_ddpm_schedule

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' to enable logging.")

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

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train DDPM (EMA + AMP + No Labels)")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--data_np", type=str, help="Path to data numpy file")
    parser.add_argument("--pregenerated_noise_dir", type=str, required=True,
                        help="Directory with pre-generated noise files (REQUIRED)")
    
    # Hyperparameters
    parser.add_argument("--timesteps", type=int, help="Number of diffusion steps")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--save_dir", type=str, help="Checkpoint save directory")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--n_feat", type=int, help="Base feature dimension")
    parser.add_argument("--save_every", type=int, help="Save checkpoint every N epochs")

    args = parser.parse_args()
    
    # Defaults (matching trainer.py where possible)
    config_defaults = {
        'data_np': 'wind_3D16X16.npy',
        'timesteps': 500,
        'beta1': 1e-4, 
        'beta2': 0.02,
        'epochs': 120,
        'batch_size': 64,
        'lr': 1e-4,
        'save_dir': 'weights/',
        'height': 16,
        'n_feat': 64,
        'save_every': 4
    }
    
    # WandB config holder
    use_wandb = False
    wandb_config = {}
    
    # Load config file if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        full_cfg = load_config(args.config)
        
        # Extract train/model configs
        train_cfg = full_cfg.get('train', {})
        diff_cfg = full_cfg.get('diffusion', {})
        model_cfg = full_cfg.get('model', {})
        
        config_defaults['data_np'] = full_cfg.get('dataset', {}).get('npy_images', config_defaults['data_np'])
        config_defaults['epochs'] = int(train_cfg.get('epochs', config_defaults['epochs']))
        config_defaults['batch_size'] = int(train_cfg.get('batch_size', config_defaults['batch_size']))
        config_defaults['lr'] = float(train_cfg.get('lr', config_defaults['lr']))
        config_defaults['save_every'] = int(train_cfg.get('save_every', config_defaults['save_every']))
        config_defaults['save_dir'] = train_cfg.get('save_dir', config_defaults['save_dir'])
        
        config_defaults['timesteps'] = int(diff_cfg.get('timesteps', config_defaults['timesteps']))
        config_defaults['beta1'] = float(diff_cfg.get('beta1', config_defaults['beta1']))
        config_defaults['beta2'] = float(diff_cfg.get('beta2', config_defaults['beta2']))
        
        config_defaults['height'] = int(model_cfg.get('height', config_defaults['height']))
        config_defaults['n_feat'] = int(model_cfg.get('n_feat', config_defaults['n_feat']))
        
        wandb_config = full_cfg.get('wandb', {})
        use_wandb = wandb_config.get('enabled', False) and WANDB_AVAILABLE

    # Override with CLI args
    for key, default_val in config_defaults.items():
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            setattr(args, key, arg_val)
        else:
            setattr(args, key, default_val)
            
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # --- 1. SETUP DATA ---
    print(f"Loading dataset: {args.data_np}")
    full_dataset = CustomDataset(args.data_np, args.pregenerated_noise_dir, transform)
    
    # 90/10 Split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"✓ Data loaded. Train: {len(train_set)}, Val: {len(val_set)}")

    # --- 2. SETUP SCHEDULE ---
    b_t, a_t, ab_t = make_ddpm_schedule(args.timesteps, args.beta1, args.beta2, device)
    
    # Helper for perturbation (Inline to match trainer.py logic)
    def perturb_input(x, t, noise):
        return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

    # --- 3. SETUP MODEL & EMA ---
    print("Initializing models...")
    # NOTE: trainer.py forces in_channels=1. We apply that here.
    model = ContextUnet(
        in_channels=1, 
        n_feat=args.n_feat, 
        n_cfeat=0, # Unconditional
        height=args.height
    ).to(device)
    
    # EMA Model
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = EMA(0.995)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler() # For Mixed Precision
    
    # WandB Init
    if use_wandb:
        wandb.init(
            project=wandb_config.get('project', 'DDPM-Training'),
            group=wandb_config.get('group', 'experiment'),
            name=wandb_config.get('name', 'ddpm-ema-amp'),
            config=vars(args)
        )
    
    # Ensure save dir exists
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --- 4. TRAINING LOOP ---
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs}", mininterval=1)
        
        for x, noise in pbar:
            optimizer.zero_grad()
            x = x.to(device)
            noise = noise.to(device)
            
            # --- Specific Hack from trainer.py ---
            # Slicing 3 channel image to 1 channel if needed
            # x = x[:, 0:1, :, :] 
            
            t = torch.randint(1, args.timesteps + 1, (x.shape[0],)).to(device)
            x_pert = perturb_input(x, t, noise)
            
            # === FIX 1: Create Dummy Context ===
            # We must pass a dummy context with shape (batch_size, 0) for n_cfeat=0
            c = torch.zeros((x.shape[0], 0)).to(device)
            
            # AMP Forward
            with autocast():
                # === FIX 2: Pass c to model ===
                pred_noise = model(x_pert, t / args.timesteps, c)
                loss = F.mse_loss(pred_noise, noise)
            
            # AMP Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            ema.update_model_average(ema_model, model)
            
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader)
        scheduler.step()
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, noise in val_loader:
                x = x.to(device)
                noise = noise.to(device)
                x = x[:, 0:1, :, :] # Slice to 1 channel
                
                t = torch.randint(1, args.timesteps + 1, (x.shape[0],)).to(device)
                x_pert = perturb_input(x, t, noise)
                
                # === FIX 3: Dummy Context for Validation ===
                c = torch.zeros((x.shape[0], 0)).to(device)
                
                pred_noise = model(x_pert, t / args.timesteps, c)
                loss = F.mse_loss(pred_noise, noise)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  --> Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]['lr']
            })

        # --- SAVE CHECKPOINTS & SAMPLES ---
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            # Save Standard
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{epoch}.pth"))
            # Save EMA
            torch.save(ema_model.state_dict(), os.path.join(args.save_dir, f"ema_model_{epoch}.pth"))
            
            # Optional: Simple Sample Generation to Disk (using EMA)
            ema_model.eval()
            with torch.no_grad():
                n_sample = 16
                # Start from noise
                x_gen = torch.randn(n_sample, 1, args.height, args.height).to(device)
                
                # === FIX 4: Dummy Context for Sampling ===
                # Must match n_sample
                c_sample = torch.zeros((n_sample, 0)).to(device)
                
                for i in range(args.timesteps, 0, -1):
                    t_vec = torch.tensor([i / args.timesteps]).to(device)
                    z = torch.randn_like(x_gen) if i > 1 else 0
                    
                    # Pass context to EMA model
                    pred_n = ema_model(x_gen, t_vec, c_sample)
                    
                    # denoise_add_noise logic inline
                    noise_factor = b_t.sqrt()[i] * z
                    mean = (x_gen - pred_n * ((1 - a_t[i]) / (1 - ab_t[i]).sqrt())) / a_t[i].sqrt()
                    x_gen = mean + noise_factor
                
                sample_path = os.path.join(args.save_dir, f"sample_epoch_{epoch}.png")
                save_image(x_gen.clamp(0, 1), sample_path, nrow=4)
                
                if use_wandb:
                    wandb.log({"generated_samples": [wandb.Image(sample_path)]})

    if use_wandb:
        wandb.finish()
        
    print("\nTraining Complete.")

if __name__ == "__main__":
    main()