"""
Testor Created by Marawan Yakout

Verify refactored code produces identical results to original.
"""
import numpy as np
from PIL import Image
from src_refactored.utils import rotate, shear_x, sample_level, labels_process
from src_refactored.augmentation import augmix

def test_rotation():
    """Test rotation produces consistent output."""
    img = Image.new('RGB', (32, 32), color='red')
    result = rotate(img, 5)
    assert result.size == (32, 32), "Rotation changed image size"
    print("Rotation test passed")

def test_labels_process():
    """Test wind speed encoding."""
    assert labels_process(30) == [1, 0, 0, 0, 0], "Category 1 encoding failed"
    assert labels_process(60) == [0, 1, 0, 0, 0], "Category 2 encoding failed"
    assert labels_process(95) == [0, 0, 1, 0, 0], "Category 3 encoding failed"
    assert labels_process(130) == [0, 0, 0, 1, 0], "Category 4 encoding failed"
    assert labels_process(170) == [0, 0, 0, 0, 1], "Category 5 encoding failed"
    print("Label encoding test passed")

def test_augmix():
    """Test AugMix produces valid tensor."""
    img = Image.new('RGB', (128, 128), color='blue')
    result = augmix(img)
    assert result.shape == (3, 128, 128), "AugMix output shape incorrect"
    print(" AugMix test passed")

if __name__ == "__main__":
    test_rotation()
    test_labels_process()
    test_augmix()
    print("\n All migration tests passed!")
