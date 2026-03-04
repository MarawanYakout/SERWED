# src_refactored/datasets.py

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------
# Image transform used in training
# You can tweak this (e.g. add normalization) as you like.
# For now: convert to float and scale to [-1, 1].
# ---------------------------------------------------------------------



# NORMALIZATION STATISTICS
# Load precomputed statistics for image normalization

def load_norm_stats(stats_path: str = "norm_stats.npz"):
    """
    Load precomputed normalization statistics.
    
    If file doesn't exist, return None (will skip normalization).
    """
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        return {
            "min": torch.tensor(stats["min"], dtype=torch.float32),
            "max": torch.tensor(stats["max"], dtype=torch.float32),
            "mean": torch.tensor(stats["mean"], dtype=torch.float32),
            "std": torch.tensor(stats["std"], dtype=torch.float32),
        }
    else:
        print(f" Warning: {stats_path} not found. Using default transform.")
        return None


# Load stats globally (computed once at import time) ADD THE PATH TO YOUR STATS FILE HERE
_norm_stats = load_norm_stats("norm_stats.npz")


# ============================================================================
# TRANSFORM FUNCTIONS
# ============================================================================

def minmax_normalize(x: torch.Tensor, stats: dict, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-Max Normalization: rescales to [0, 1].
    
    x_norm = (x - x_min) / (x_max - x_min + eps)
    
    Args:
        x: tensor of shape [C, H, W]
        stats: dict with keys 'min', 'max' (precomputed per-channel)
        eps: small constant to avoid division by zero
    
    Returns:
        normalized tensor in [0, 1]
    """
    if stats is None:
        return x
    
    data_min = stats["min"]  # shape [C, 1, 1]
    data_max = stats["max"]  # shape [C, 1, 1]
    
    # Ensure shapes broadcast correctly
    if data_min.dim() == 1:
        data_min = data_min.view(-1, 1, 1)
    if data_max.dim() == 1:
        data_max = data_max.view(-1, 1, 1)
    
    normalized = (x - data_min) / (data_max - data_min + eps)
    
    # Clamp to [0, 1] to handle numerical issues
    normalized = torch.clamp(normalized, 0.0, 1.0)
    
    return normalized


def zscore_normalize(x: torch.Tensor, stats: dict, eps: float = 1e-8) -> torch.Tensor:
    """
    Z-Score Normalization (Standardization): rescales to mean=0, std=1.
    
    x_norm = (x - mean) / (std + eps)
    
    Args:
        x: tensor of shape [C, H, W]
        stats: dict with keys 'mean', 'std' (precomputed per-channel)
        eps: small constant to avoid division by zero
    
    Returns:
        standardized tensor (unbounded, typically ~[-3, 3])
    """
    if stats is None:
        return x
    
    mean = stats["mean"]  # shape [C, 1, 1]
    std = stats["std"]    # shape [C, 1, 1]
    
    # Ensure shapes broadcast correctly
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1)
    
    normalized = (x - mean) / (std + eps)
    
    return normalized


def create_transform(method: str = "minmax", stats: dict = None) -> transforms.Compose:
    """
    Create a transform pipeline for the given normalization method.
    
    Args:
        method: "minmax", "zscore", or "none"
        stats: precomputed normalization statistics (dict)
    
    Returns:
        transforms.Compose object
    """
    if method == "minmax":
        return transforms.Compose([
            transforms.Lambda(
                lambda x: minmax_normalize(x, stats, eps=1e-8)
            )
        ])
    
    elif method == "zscore":
        return transforms.Compose([
            transforms.Lambda(
                lambda x: zscore_normalize(x, stats, eps=1e-8)
            )
        ])
    
    elif method == "none":
        # Original: scale to [-1, 1]
        return transforms.Compose([
            transforms.Lambda(lambda x: x / 127.5 - 1.0)
        ])
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Default transform (can be overridden)
transform = create_transform("minmax", stats=_norm_stats)


## Alternative default

# transform = transforms.Compose([
#     # input is expected to be a torch.Tensor already; normalization only
#     transforms.Lambda(lambda x: x / 127.5 - 1.0)
# ])


class CustomDataset(Dataset):
    """
    Dataset that returns (image, pre_generated_noise_all_timesteps).

    images_npy: path to images .npy, shape (N, H, W, C) or (N, C, H, W)
    noise_dir: directory containing:
        - noise.npy       # shape (N, T, C, H, W)
        - metadata.npy    # dict with keys: n_images, timesteps, height, channels, single_file, ...
    transform: optional transform applied to image tensor only.
    """

    def __init__(self, images_npy: str, noise_dir: str, transform: Optional[transforms.Compose] = None):
        super().__init__()
        self.images_npy = images_npy
        self.noise_dir = noise_dir
        self.transform = transform

        # 1. Load images lazily with memmap
        if not os.path.exists(self.images_npy):
            raise FileNotFoundError(f"Images .npy not found: {self.images_npy}")

        self.imgs = np.load(self.images_npy, mmap_mode="r")
        self.n_images = self.imgs.shape[0]

        # 2. Load metadata
        meta_path = os.path.join(self.noise_dir, "metadata.npy")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.npy not found in noise_dir={self.noise_dir}")

        meta = np.load(meta_path, allow_pickle=True).item()

        self.timesteps = int(meta["timesteps"])
        self.height = int(meta["height"])
        self.channels = int(meta["channels"])

        if not meta.get("single_file", False):
            raise ValueError(
                "metadata.npy indicates chunked noise (single_file=False). "
                "This CustomDataset expects a single noise.npy file."
            )

        # 3. Load noise as memmap
        noise_path = os.path.join(self.noise_dir, "noise.npy")
        if not os.path.exists(noise_path):
            raise FileNotFoundError(f"noise.npy not found in noise_dir={self.noise_dir}")

        # Shape: (N, T, C, H, W)
        self.noise_all = np.load(noise_path, mmap_mode="r")

        # 4. Consistency checks
        if self.noise_all.shape[0] != self.n_images:
            raise ValueError(
                f"Image count ({self.n_images}) != noise count ({self.noise_all.shape[0]}). "
                "Make sure noise was generated with the same images_npy."
            )

        if "n_images" in meta and int(meta["n_images"]) != self.n_images:
            raise ValueError(
                f"Metadata n_images ({meta['n_images']}) != images count ({self.n_images})."
            )

        # if self.noise_all.shape[1] != self.timesteps:
        #     raise ValueError(
        #         f"Noise timesteps ({self.noise_all.shape[1]}) != metadata timesteps ({self.timesteps})."
        #     )

    def __len__(self) -> int:
        return self.n_images

    def __getitem__(self, idx: int):
        # ---- Image ----
        img = self.imgs[idx]  # np.ndarray

        # If image is (H, W, C), convert to (C, H, W)
        if img.ndim == 3 and img.shape[-1] in (1, 3):
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)

        img = torch.from_numpy(img).float()  # to float tensor

        if self.transform is not None:
            img = self.transform(img)

        # ---- Noise for all timesteps ----
        # noise_all_for_idx: (T, C, H, W)
        noise_all_for_idx = self.noise_all[idx]
        noise_all_for_idx = torch.from_numpy(noise_all_for_idx).float()

        # Return exactly what the no-label trainer expects:
        # (image, noise_all_timesteps)
        return img, noise_all_for_idx
