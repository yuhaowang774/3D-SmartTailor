import torch


def keypoint_2d_loss(projected, observed, weights=None):
    diff = projected - observed
    squared = (diff ** 2).sum(dim=-1)
    if weights is not None:
        squared = squared * weights
    return squared.mean()


def shape_regularization(betas):
    return (betas ** 2).mean()


def pose_regularization(pose):
    return (pose[:, 3:] ** 2).mean()


def height_loss(vertices, target_height_m):
    max_y = vertices[:, :, 1].max(dim=1).values
    min_y = vertices[:, :, 1].min(dim=1).values
    predicted_height = max_y - min_y
    return ((predicted_height - target_height_m) ** 2).mean()


def total_loss(kp_proj, kp_obs, betas, pose, vertices, target_height_m, kp_weights=None,
               w_kp=1.0, w_shape=0.01, w_pose=0.001, w_height=10.0):
    target_h = torch.tensor(target_height_m, dtype=vertices.dtype, device=vertices.device)
    loss = w_kp * keypoint_2d_loss(kp_proj, kp_obs, kp_weights)
    loss = loss + w_shape * shape_regularization(betas)
    loss = loss + w_pose * pose_regularization(pose)
    loss = loss + w_height * height_loss(vertices, target_h)
    return loss
