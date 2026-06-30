import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from fitting.losses import keypoint_2d_loss, shape_regularization, height_loss


def test_keypoint_2d_loss_perfect_match():
    kp = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
    loss = keypoint_2d_loss(kp, kp)
    assert loss.item() < 1e-6


def test_shape_regularization_zero_betas():
    betas = torch.zeros(1, 10)
    loss = shape_regularization(betas)
    assert loss.item() == 0.0


def test_shape_regularization_nonzero():
    betas = torch.ones(1, 10)
    loss = shape_regularization(betas)
    assert loss.item() > 0.0


def test_height_loss_exact_match():
    vertices = torch.randn(1, 1000, 3)
    vertices[:, :, 1] = torch.linspace(-0.9, 0.9, 1000)
    loss = height_loss(vertices, torch.tensor(1.8))
    assert loss.item() < 1e-4


import numpy as np
from fitting.smplify import _prepare_keypoints, MEDIAPIPE_TO_SMPLX


def test_prepare_keypoints():
    kp = np.random.randn(33, 4).astype(np.float32)
    kp[:, 3] = 1.0
    kp[:, 0] = np.random.rand(33) * 500 + 100
    kp[:, 1] = np.random.rand(33) * 800 + 100
    device = torch.device('cpu')
    obs, weights, idx_list = _prepare_keypoints(kp, kp, device)
    assert obs.shape[1] == 2
    assert len(weights) == obs.shape[0]
    assert obs.shape[0] > 0
    assert len(idx_list) == obs.shape[0]


def test_mediapipe_smplx_mapping():
    for mp_idx, smplx_idx in MEDIAPIPE_TO_SMPLX.items():
        assert 0 <= mp_idx <= 32, f"MediaPipe index {mp_idx} out of range"
        assert 0 <= smplx_idx <= 21, f"SMPL-X index {smplx_idx} out of range"
