import torch
import numpy as np


def estimate_focal_length(keypoints, img_width, img_height):
    valid = keypoints[:, 3] > 0.5
    if valid.sum() < 10:
        return float(max(img_width, img_height) * 1.2)
    return float(max(img_width, img_height) * 1.2)


def build_camera_params(focal_length, img_width, img_height, device):
    return {
        'focal_length': torch.tensor([focal_length, focal_length], dtype=torch.float32, device=device),
        'principal_point': torch.tensor([img_width / 2.0, img_height / 2.0], dtype=torch.float32, device=device),
    }
