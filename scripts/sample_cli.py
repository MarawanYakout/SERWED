"""
CLI script for sampling from trained DDPM model.
Usage: python scripts/sample_cli.py --ckpt weights_16/model_100.pth --n 64 --context_idx 2
"""
import argparse
import torch
from torchvision.utils import save_image, make_grid
from src_refactored.context_unet import ContextUnet
from src_refactored.diffusion import make_ddpm_schedule
from src_refactored.sampler import DDPMSampler
from src_refactored.vis import norm_torch

def main():
    parser = argparse.ArgumentParser(description="Sample from trained DDPM model")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--n", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--height", type=int, default=16, help="Image height")
    parser.add_argument("--n_feat", type=int, default=64, help="Base feature dimension")
    parser.add_argument("--n_cfeat", type=int, default=5, help="Context feature dimension")
    parser.add_argument("--timesteps", type=int, default=500, help="Number of diffusion steps")
    parser.add_argument("--beta1", type=float, default=1e-4, help="Starting beta")
    parser.add_argument("--beta2", type=float, default=0.02, help="Ending beta")
    parser.add_argument("--context_idx", type=int, default=None, help="Context class index (0-4 for wind speed)")
    parser.add_argument("--save_path", default="samples.png", help="Output image path")
    parser.add_argument("--nrow", type=int, default=8, help="Number of images per row in grid")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Rebuild schedule
    print("Building diffusion schedule...")
    b_t, a_t, ab_t = make_ddpm_schedule(args.timesteps, args.beta1, args.beta2, device)
    
    # Rebuild model
    print("Building model...")
    model = ContextUnet(
        in_channels=3, 
        n_feat=args.n_feat, 
        n_cfeat=args.n_cfeat, 
        height=args.height
    ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # Build context
    if args.context_idx is not None:
        print(f"Using context class {args.context_idx}")
        context = torch.zeros(args.n, args.n_cfeat).to(device)
        context[:, args.context_idx] = 1
    else:
        print("Using null context (unconditional)")
        context = None
    
    # Sample
    print(f"\nGenerating {args.n} samples...")
    sampler = DDPMSampler(model, args.timesteps, b_t, a_t, ab_t, device)
    samples = sampler.sample(args.n, args.height, 3, context)
    
    # Normalize and save
    samples_norm = norm_torch(samples)
    grid = make_grid(samples_norm, nrow=args.nrow)
    save_image(grid, args.save_path)
    print(f"\nSamples saved to {args.save_path}")

if __name__ == "__main__":
    main()
