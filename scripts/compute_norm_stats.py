'''

Compute normalization statistics for a dataset of images stored in a NumPy file.
The output can then be used for normalizing images during training or inference.

After you run this script, you can use the generated 'norm_stats.npz' file during the training ,
the code to load the stats is avaliable in src_refactored/datasets.py

Last updated: 5 December 2025
By: Marawan Yakout

'''

import numpy as np

def compute_norm_stats(images_npy: str, output_path: str = "norm_stats.npz"):
    imgs = np.load(images_npy, mmap_mode="r")  # (N, H, W, C) or (N, C, H, W)

    # Ensure (N, C, H, W)
    if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
        imgs = np.transpose(imgs, (0, 3, 1, 2))  # (N, C, H, W)

    N, C, H, W = imgs.shape
    imgs_flat = imgs.reshape(N * H * W, C)

    min_vals = imgs_flat.min(axis=0)
    max_vals = imgs_flat.max(axis=0)
    mean_vals = imgs_flat.mean(axis=0)
    std_vals = imgs_flat.std(axis=0)

    np.savez(
        output_path,
        min=min_vals,
        max=max_vals,
        mean=mean_vals,
        std=std_vals,
    )

if __name__ == "__main__":
    compute_norm_stats("./training_data/wind_3D16X16.npy", "norm_stats.npz") # Adjust path as needed
