import numpy as np
from src.keypoint.detect import detect_keypoints


def test_detect_keypoints_on_synthetic():
    img = np.ones((1024, 768, 3), dtype=np.uint8) * 200
    result = detect_keypoints(img)
    assert isinstance(result, np.ndarray)
    assert result.shape == (33, 4), f"Expected (33,4), got {result.shape}"


def test_detect_empty_image():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    result = detect_keypoints(img)
    assert result.shape == (33, 4)
    # 纯黑图像检测不到人体，应全零
    assert result[:, 3].max() == 0.0, "Should detect no person in black image"
