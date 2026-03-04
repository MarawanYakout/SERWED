"""
AugMix augmentation strategy.

The follwoing file includes only the AugMix Implementation.
"""
import random
import numpy as np
import torch
import torchvision.transforms as T
from src_refactored.utils import (
    autocontrast, rotate, shear_x, shear_y, sample_level
)

# ======================================
# AugMix Implementation 
# ======================================

# Augmentation operations list (Line 43)
augmentations = [
    lambda x: x,  # Identity
    autocontrast,
    lambda x: rotate(x, sample_level(3)),
    lambda x: shear_x(x, sample_level(3)),
    lambda x: shear_y(x, sample_level(3)),
]


def augmix(image, severity=3, width=3, depth=-1, alpha=1.):
    """
    Apply AugMix augmentation to image.
    
    Args:
        image: PIL Image
        severity: Severity of augmentations (unused but kept for API compatibility)
        width: Number of augmentation chains to mix
        depth: Number of augmentations per chain (-1 for random 1-3)
        alpha: Dirichlet/Beta distribution parameter
    
    Returns:
        Augmented image as torch.Tensor
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = torch.zeros_like(T.ToTensor()(image))
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = random.choice(augmentations)
            image_aug = op(image_aug)
        mix += ws[i] * T.ToTensor()(image_aug)

    mixed = (1 - m) * T.ToTensor()(image) + m * mix
    return mixed