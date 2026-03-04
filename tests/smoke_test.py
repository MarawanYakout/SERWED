"""
Testor Created by Marawan Yakout

Objective: ensure imports, function calls, and basic paths work.
"""

import sys, os
sys.path.insert(0, '.')

from src_refactored.data import load_metadata, prepare_wind_speed_dataset
from src_refactored.utils import rotate, labels_process
from src_refactored.augmentation import augmix
from PIL import Image

# 1) Metadata load
labels, meta = load_metadata(".")
assert not labels.empty, "Labels CSV is empty or missing"
print("✓ Metadata load")

# 2) Utils sanity
assert labels_process(60) == [0,1,0,0,0], "Label encoder mismatch"
img = Image.new('RGB', (64,64), color='white')
rot = rotate(img, 5)
assert rot.size == (64,64), "Rotate changed size"
print("✓ Utils ok")

# 3) Augmix tensor shape
aug = augmix(img)
assert tuple(aug.shape) == (3,64,64), "AugMix shape mismatch"
print("✓ AugMix ok")

print("SMOKE TEST PASSED")
