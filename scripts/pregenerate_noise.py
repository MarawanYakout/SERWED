# pregenerate_noise.py (SINGLE FILE VERSION)
"""
Pre-generate all noise for DDPM training as a single file.
WARNING: This will create large amounts of data and take several hours.
"""
import numpy as np
import os
from tqdm import tqdm
import argparse


def pregenerate_noise(n_images, timesteps, height, channels, save_dir):
    """
    Pre-generate noise and save as a SINGLE file.

    Args:
        n_images: number of images in dataset
        timesteps: number of diffusion timesteps
        height: image height/width
        channels: number of channels (3 for RGB)
        save_dir: directory to save noise file
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 60)
    print("Pre-generating Noise for DDPM Training (SINGLE FILE)")
    print("=" * 60)
    print(f"Dataset size: {n_images} images")
    print(f"Timesteps: {timesteps}")
    print(f"Image dimensions: {height}×{height}×{channels}")

    # Calculate storage
    bytes_per_noise = height * height * channels * 4  # float32
    total_noise_values = n_images * timesteps
    total_bytes = total_noise_values * bytes_per_noise
    total_gb = total_bytes / (1024 ** 3)

    print(f"\nEstimated storage: {total_gb:.2f} GB")
    print(f"RAM required: ~{total_gb * 2:.2f} GB (during generation)")
    print("=" * 60)

    if total_gb > 20:
        print("⚠️  WARNING: File will be very large (>20 GB)!")
        print("   Ensure you have sufficient RAM and storage.")

    response = input("\nDo you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.")
        return

    print("\nGenerating all noise at once...")
    print("(This may take a while and use significant RAM)")

    # Generate ALL noise at once: (n_images, timesteps, channels, height, height)
    all_noise = np.random.randn(
        n_images, timesteps, channels, height, height
    ).astype(np.float32)

    print("Noise generation complete. Saving to disk...")

    # Save as single file
    noise_file = os.path.join(save_dir, "noise.npy")
    np.save(noise_file, all_noise)

    file_size_gb = os.path.getsize(noise_file) / (1024 ** 3)

    print(f"\n✓ Noise generation complete!")
    print(f"Total file: 1 noise file + 1 metadata file")
    print(f"Location: {save_dir}")
    print(f"Noise file: {noise_file}")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"Shape: {all_noise.shape}")

    # Save metadata
    metadata = {
        "n_images": n_images,
        "timesteps": timesteps,
        "height": height,
        "channels": channels,
        "single_file": True,
        "noise_shape": all_noise.shape,
        "file_size_gb": file_size_gb,
    }
    metadata_file = os.path.join(save_dir, "metadata.npy")
    np.save(metadata_file, metadata)
    print(f"Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_images", type=int, default=None,
                        help="Number of images (used if --images_np is not provided)")
    parser.add_argument("--images_np", type=str, default=None,
                        help="Path to images .npy file; if set, n_images is inferred from this file")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()

    # Infer n_images from images_np if provided
    if args.images_np is not None:
        print(f"Loading images from: {args.images_np}")
        # Use mmap_mode to avoid loading full array into RAM
        imgs = np.load(args.images_np, mmap_mode="r")
        n_images = imgs.shape[0]
        print(f"Inferred n_images from file: {n_images}")
    else:
        if args.n_images is None:
            raise ValueError(
                "You must provide either --images_np or --n_images."
            )
        n_images = args.n_images
        print(f"Using n_images from argument: {n_images}")

    pregenerate_noise(
        n_images=n_images,
        timesteps=args.timesteps,
        height=args.height,
        channels=args.channels,
        save_dir=args.save_dir,
    )
