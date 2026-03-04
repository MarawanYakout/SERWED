import argparse
import os
import sys

def main():
    """
    Parses command-line arguments and simulates data preparation.
    """
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Prepares image data for machine learning training.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Define the arguments expected from the command line
    parser.add_argument(
        '--folder_path',
        type=str,
        required=True,
        help="Path to the folder containing raw images (e.g., /data)."
    )
    parser.add_argument(
        '--sample_fraction',
        type=float,
        default=1.0,
        help="Fraction of the dataset to sample (0.0 to 1.0). Default is 1.0."
    )
    parser.add_argument(
        '--image_size',
        type=int,
        required=True,
        help="The target dimension (width/height) for resizing images."
    )
    parser.add_argument(
        '--output_images',
        type=str,
        required=True,
        help="Full path to save the processed NumPy array (e.g., /training_data/wind_3D16X16.npy)."
    )

    # 2. Parse Arguments
    args = parser.parse_args()

    # 3. Validation and Setup (Basic checks)
    if not os.path.isdir(args.folder_path):
        sys.exit(f"Error: Input folder path not found: {args.folder_path}")
    
    if not 0.0 <= args.sample_fraction <= 1.0:
        sys.exit(f"Error: Sample fraction must be between 0.0 and 1.0. Got {args.sample_fraction}")

    # 4. Simulation of Data Processing (Replace this with actual NumPy/Pillow/OpenCV logic)
    
    print("--- Starting Data Preparation ---")
    print(f"Input Folder: {args.folder_path}")
    print(f"Target Image Size: {args.image_size}x{args.image_size}")
    print(f"Sampling Fraction: {args.sample_fraction}")
    
    # In a real scenario, you would:
    # 1. List all images in args.folder_path
    # 2. Loop through a sampled fraction of images
    # 3. Load, resize (to args.image_size), and normalize the images
    # 4. Stack them into a NumPy array (np.save)
    
    print("\n[Simulated] Loading, resizing, and processing images...")
    
    # Simulate the data size result
    simulated_data_size = (int(5000 * args.sample_fraction), args.image_size, args.image_size, 3) 
    print(f"Successfully generated a dataset array of shape: {simulated_data_size}")

    # 5. Output Confirmation
    output_dir = os.path.dirname(args.output_images)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
        
    print(f"Saving final array to: {args.output_images}")
    print("--- Data Preparation Complete! ---")

if __name__ == '__main__':
    main()