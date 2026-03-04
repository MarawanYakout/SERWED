"""
DDPM diffusion process: noise schedules, forward process, and loss computation.

"""
import torch
import torch.nn.functional as F

def make_ddpm_schedule(timesteps, beta1, beta2, device):
    """
    Construct linear beta schedule and precompute alpha products.
    
    Args:
        timesteps: number of diffusion steps
        beta1: starting beta value
        beta2: ending beta value
        device: torch device
    
    Returns:
        b_t: beta schedule (timesteps+1,)
        a_t: alpha schedule (timesteps+1,)
        ab_t: cumulative alpha product (timesteps+1,)
    """
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()
    ab_t[0] = 1
    return b_t, a_t, ab_t

def perturb_input(x, t, noise, ab_t):
    """
    Forward diffusion q(x_t | x_0).
    
    Args:
        x: clean images (batch, channels, h, w)
        t: timesteps (batch,)
        noise: sampled noise (batch, channels, h, w)
        ab_t: cumulative alpha product schedule
    
    Returns:
        x_t: noisy images at timestep t
    """
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise

def compute_loss(model, x, t, c, ab_t):
    """
    Compute noise prediction loss for DDPM.
    
    Args:
        model: noise prediction model
        x: clean images (batch, channels, h, w)
        t: timesteps (batch,)
        c: context vectors (batch, n_cfeat)
        ab_t: cumulative alpha product schedule
    
    Returns:
        MSE loss between predicted and true noise
    """
    noise = torch.randn_like(x)
    x_noisy = perturb_input(x, t, noise, ab_t)
    pred_noise = model(x_noisy, t, c)
    loss = F.mse_loss(pred_noise, noise)
    return loss
