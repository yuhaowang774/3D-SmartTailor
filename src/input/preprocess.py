import numpy as np
import cv2


def validate_image(img: np.ndarray, min_side: int = 512) -> bool:
    if img is None or img.size == 0:
        return False
    if len(img.shape) != 3 or img.shape[2] != 3:
        return False
    if min(img.shape[:2]) < min_side:
        return False
    return True


def resize_image(img: np.ndarray, target_short_side: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    short_side = min(h, w)
    scale = target_short_side / short_side
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_images(front_path, side_path, height_cm, config):
    target_short = config['input']['target_short_side']
    min_res = config['input']['min_resolution']

    img_front = cv2.imread(front_path)
    img_side = cv2.imread(side_path)

    if img_front is None:
        raise ValueError(f"Can't load front photo: {front_path}")
    if img_side is None:
        raise ValueError(f"Can't load side photo: {side_path}")

    img_front = cv2.cvtColor(img_front, cv2.COLOR_BGR2RGB)
    img_side = cv2.cvtColor(img_side, cv2.COLOR_BGR2RGB)

    if not validate_image(img_front, min_res):
        raise ValueError(f"Front photo does not meet minimum requirements (>= {min_res}px, RGB)")
    if not validate_image(img_side, min_res):
        raise ValueError(f"Side photo does not meet minimum requirements (>= {min_res}px, RGB)")

    img_front = resize_image(img_front, target_short)
    img_side = resize_image(img_side, target_short)

    return {
        'img_front': img_front,
        'img_side': img_side,
        'height_cm': height_cm,
    }
