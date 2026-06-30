import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pose_landmarker.task")


def _get_detector():
    base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


_detector = None


def detect_keypoints(img: np.ndarray, static_image_mode: bool = True, model_complexity: int = 1) -> np.ndarray:
    global _detector
    if _detector is None:
        _detector = _get_detector()

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    result = _detector.detect(mp_image)

    if not result.pose_landmarks:
        return np.zeros((33, 4), dtype=np.float32)

    h, w = img.shape[:2]
    keypoints = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(result.pose_landmarks[0]):
        keypoints[i] = [lm.x * w, lm.y * h, lm.z, lm.visibility]
    return keypoints
