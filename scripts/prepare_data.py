"""
Command-line script to prepare wind speed dataset (images only, no labels).
Usage: python scripts/prepare_data.py --folder_path . --sample_fraction 0.1
"""
import argparse
from src_refactored.data import prepare_wind_speed_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare wind speed dataset (images only)")
    parser.add_argument("--folder_path", type=str, default=".", help="Root folder path")
    parser.add_argument("--generate_augmix", action="store_true", help="Generate AugMix images")
    parser.add_argument("--sample_fraction", type=float, default=0.1, help="Fraction of images to process")
    parser.add_argument("--image_size", type=int, default=16, help="Target image size")
    parser.add_argument("--output_images", type=str, default="wind_3D16X16.npy",
                        help="Output images file (no labels will be saved)")
    # removed --output_labels

    args = parser.parse_args()

    prepare_wind_speed_dataset(
        folder_path=args.folder_path,
        generate_augmix_data=args.generate_augmix,
        sample_fraction=args.sample_fraction,
        image_size=args.image_size,
        output_image_file=args.output_images,
        output_label_file=None,   # IMPORTANT: signal “no labels”
    )
