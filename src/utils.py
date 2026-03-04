"""
File Discribtion:

Utility functions for augmentation, visualization, and metrics.

The following Utilities include the following:

1-  AugMix Utility Functions
2-  Image Processing Utilities
3-  Label Encoding Functions 
"""
import numpy as np
import torch
from PIL import Image, ImageOps


# =========================
# AugMix Utility Functions
# =========================


def int_parameter(level, maxval):
    """Scale integer parameter for augmentation."""
    return int(level * maxval / 10)

def float_parameter(level, maxval):
    """Scale float parameter for augmentation."""
    return float(level) * maxval / 10.

def sample_level(n):
    """Sample augmentation magnitude uniformly between 0.1 and n."""
    return np.random.uniform(low=0.1, high=n)

def autocontrast(img):
    """Apply autocontrast to PIL Image."""
    return ImageOps.autocontrast(img)

def rotate(img, level):
    """Rotate image by angle determined by level (max 30 degrees)."""
    return img.rotate(int_parameter(level, 30))

def shear_x(img, level):
    """Shear image horizontally."""
    return img.transform(img.size, Image.AFFINE, (1, float_parameter(level, 0.3), 0, 0, 1, 0))

def shear_y(img, level):
    """Shear image vertically."""
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, float_parameter(level, 0.3), 1, 0))


# =====================================
# Image Processing Utilities 
# ====================================

def channel3(img, size):
    """Convert grayscale or single-channel image to 3-channel RGB."""
    empty_3d_array = np.empty((size, size, 3))
    empty_3d_array[:, :, 0] = np.array(img)
    empty_3d_array[:, :, 1] = np.array(img)
    empty_3d_array[:, :, 2] = np.array(img)
    return empty_3d_array


def channel1(img, size):
    """Convert image to single channel (unused but preserved from original)."""
    empty_1d_array = np.empty((size, size, 1))
    empty_1d_array[:, :, 0] = np.array(img)
    return empty_1d_array


def crop_center(image, new_width, new_height):
    """
    Crop image to specified dimensions from center.
    
    Args:
        image: PIL Image
        new_width: Target width
        new_height: Target height
    
    Returns:
        Cropped PIL Image
    """
    width, height = image.size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

# ============================
# Label Encoding Functions 
# =============================

def labels_process(value_loc):
    """
    Convert wind speed to one-hot encoded category.
    
    Categories:
        1: 15-45 kts
        2: 45-80 kts
        3: 80-110 kts
        4: 110-150 kts
        5: 150-190 kts
    
    Args:
        value_loc: Wind speed in knots
    
    Returns:
        One-hot encoded list [5 elements]
    """
    storm_array = []
    if ((value_loc >= 15) & (value_loc <= 45)):
        storm_array = switch_case(1)
    elif ((value_loc > 45) & (value_loc <= 80)):
        storm_array = switch_case(2)
    elif (value_loc > 80) & (value_loc <= 110):
        storm_array = switch_case(3)
    elif (value_loc > 110) & (value_loc <= 150):
        storm_array = switch_case(4)
    elif (value_loc > 150) & (value_loc <= 190):
        storm_array = switch_case(5)
    return storm_array

def switch_case(argument):
    """Return one-hot encoded vector for storm category."""
    return {
        1: [1, 0, 0, 0, 0],
        2: [0, 1, 0, 0, 0],
        3: [0, 0, 1, 0, 0],
        4: [0, 0, 0, 1, 0],
        5: [0, 0, 0, 0, 1]
    }.get(argument, "Invalid option")