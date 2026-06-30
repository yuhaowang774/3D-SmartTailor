import numpy as np
from src.input.preprocess import validate_image, resize_image


def test_validate_image_valid_rgb():
    img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    assert validate_image(img) is True


def test_validate_image_grayscale():
    img = np.random.randint(0, 255, (800, 600), dtype=np.uint8)
    assert validate_image(img) is False


def test_validate_image_too_small():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    assert validate_image(img) is False


def test_resize_image_keeps_aspect():
    img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    target = 1024
    resized = resize_image(img, target_short_side=target)
    h, w = resized.shape[:2]
    assert min(h, w) == target, f"Short side should be {target}, got {min(h,w)}"


def test_validate_rejects_empty():
    assert validate_image(np.array([])) is False
