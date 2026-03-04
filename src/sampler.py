"""
DDPM sampler: reverse diffusion process for image generation.
"""
import torch
from tqdm import tqdm

class DDPMSampler:
    def __init__(self, model, timesteps, b_t, a_t, ab_t, device):
        """
        DDPM reverse diffusion sampler.
        
        Args:
            model: trained noise prediction model
            timesteps: number of diffusion steps
            b_t: beta schedule
            a_t: alpha schedule  
            ab_t: cumulative alpha product schedule
            device: torch device
        """
        self.model = model
        self.timesteps = timesteps
        self.b_t = b_t
        self.a_t = a_t
        self.ab_t = ab_t
        self.device = device
    
    @torch.no_grad()
    def sample(self, n_samples, height, n_channels, context=None):
        """
        Generate samples via reverse diffusion.
        
        Args:
            n_samples: batch size
            height: image height (assumes square)
            n_channels: number of channels (3 for RGB)
            context: (n_samples, n_cfeat) context vectors or None
        
        Returns:
            Generated images (n_samples, n_channels, height, height)
        """
        self.model.eval()
        x = torch.randn(n_samples, n_channels, height, height).to(self.device)
        
        for t_idx in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.tensor([t_idx], device=self.device).long()
            t_batch = t.repeat(n_samples)
            
            # Predict noise
            pred_noise = self.model(x, t_batch, context)
            
            # DDPM reverse step
            if t_idx > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            # Denoising formula
            x = (
                1 / torch.sqrt(self.a_t[t])
            ) * (
                x - ((1 - self.a_t[t]) / torch.sqrt(1 - self.ab_t[t])) * pred_noise
            ) + torch.sqrt(self.b_t[t]) * noise
        
        return x
