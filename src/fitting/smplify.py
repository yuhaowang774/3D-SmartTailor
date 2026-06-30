import torch
import numpy as np
import smplx
from .camera import estimate_focal_length, build_camera_params
from .losses import total_loss


# MediaPipe 33点 → SMPL-X body joint 索引
MEDIAPIPE_TO_SMPLX = {
    0:  15,   # nose → head
    11: 16,   # left_shoulder
    12: 17,   # right_shoulder
    13: 18,   # left_elbow
    14: 19,   # right_elbow
    15: 20,   # left_wrist
    16: 21,   # right_wrist
    23: 1,    # left_hip
    24: 2,    # right_hip
    25: 4,    # left_knee
    26: 5,    # right_knee
    27: 7,    # left_ankle
    28: 8,    # right_ankle
    29: 10,   # left_foot_index → left_foot
    30: 11,   # right_foot_index → right_foot
    31: 10,   # left_heel → left_foot
    32: 11,   # right_heel → right_foot
    5:  16,   # left_shoulder (redundant)
    6:  17,   # right_shoulder (redundant)
}


def _prepare_keypoints(kp_front, kp_side, device):
    mp_indices = list(MEDIAPIPE_TO_SMPLX.keys())
    obs_list = []
    weight_list = []
    idx_list = []

    for kp, view_weight in [(kp_front, 1.0), (kp_side, 0.8)]:
        for mp_i in mp_indices:
            smplx_i = MEDIAPIPE_TO_SMPLX[mp_i]
            if mp_i < len(kp) and kp[mp_i, 3] > 0.3:
                obs_list.append([kp[mp_i, 0], kp[mp_i, 1]])
                w = 2.0 if smplx_i in [16, 17, 1, 2] else 1.0
                weight_list.append(w * view_weight)
                idx_list.append(smplx_i)

    if len(obs_list) == 0:
        raise RuntimeError("No valid keypoints for fitting")

    obs_kp = torch.tensor(obs_list, dtype=torch.float32, device=device)
    weights = torch.tensor(weight_list, dtype=torch.float32, device=device)
    return obs_kp, weights, idx_list


def _project_points(joints_3d, camera_params, smplx_indices):
    fx, fy = camera_params['focal_length']
    cx, cy = camera_params['principal_point']
    kp_3d = joints_3d[0, smplx_indices]
    x = fx * kp_3d[:, 0] / (kp_3d[:, 2] + 1e-8) + cx
    y = fy * kp_3d[:, 1] / (kp_3d[:, 2] + 1e-8) + cy
    return torch.stack([x, y], dim=-1)


def fit_smplx(kp_front, kp_side, height_cm, config):
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    fit_cfg = config['fitting']
    w = fit_cfg['loss_weights']

    model_path = config['model']['smplx_path']
    gender = config['model']['gender']
    body_model = smplx.create(model_path, model_type='smplx', gender=gender,
                              num_betas=10, batch_size=1).to(device)

    obs_kp, kp_weights, proj_indices = _prepare_keypoints(kp_front, kp_side, device)

    img_h, img_w = 1024, 1024
    focal = estimate_focal_length(kp_front, img_w, img_h)
    cam = build_camera_params(focal, img_w, img_h, device)

    num_pca_comps = body_model.num_pca_comps  # 6 for SMPL-X v1.1
    body_pose_dim = body_model.NUM_BODY_JOINTS * 3  # 21 × 3 = 63
    jaw_pose_dim = 3
    hand_pose_dim = num_pca_comps  # 6 per hand (PCA representation)

    # Stage 1: Pose fitting
    betas_init = torch.zeros(1, 10, device=device, requires_grad=False)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    body_pose = torch.zeros(1, body_pose_dim, device=device, requires_grad=True)
    jaw_pose = torch.zeros(1, jaw_pose_dim, device=device, requires_grad=False)
    left_hand = torch.zeros(1, hand_pose_dim, device=device, requires_grad=False)
    right_hand = torch.zeros(1, hand_pose_dim, device=device, requires_grad=False)

    opt_params = [global_orient, body_pose]
    optimizer1 = torch.optim.LBFGS(opt_params, lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure1():
        optimizer1.zero_grad()
        output = body_model(betas=betas_init, body_pose=body_pose, global_orient=global_orient,
                            jaw_pose=jaw_pose, left_hand_pose=left_hand, right_hand_pose=right_hand)
        proj = _project_points(output.joints, cam, proj_indices)
        full_pose = torch.cat([global_orient, body_pose, jaw_pose, left_hand, right_hand], dim=1)
        loss = total_loss(proj, obs_kp, betas_init, full_pose, output.vertices,
                          height_cm / 100.0, kp_weights,
                          w_kp=w['keypoint'], w_shape=0.0, w_pose=w['reg_pose'], w_height=0.0)
        loss.backward()
        return loss

    for i in range(fit_cfg['stage1_iterations'] // 20):
        optimizer1.step(closure1)

    # Stage 2: Shape fitting
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    body_pose_s1 = body_pose.detach().clone()
    global_orient_s1 = global_orient.detach().clone()

    opt_params2 = [betas]
    optimizer2 = torch.optim.LBFGS(opt_params2, lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure2():
        optimizer2.zero_grad()
        output = body_model(betas=betas, body_pose=body_pose_s1, global_orient=global_orient_s1,
                            jaw_pose=jaw_pose, left_hand_pose=left_hand, right_hand_pose=right_hand)
        proj = _project_points(output.joints, cam, proj_indices)
        full_pose = torch.cat([global_orient_s1, body_pose_s1, jaw_pose, left_hand, right_hand], dim=1)
        loss = total_loss(proj, obs_kp, betas, full_pose, output.vertices,
                          height_cm / 100.0, kp_weights,
                          w_kp=w['keypoint'], w_shape=w['reg_shape'], w_pose=0.0, w_height=w['height'])
        loss.backward()
        return loss

    for i in range(fit_cfg['stage2_iterations'] // 20):
        optimizer2.step(closure2)

    # Stage 3: Joint refinement
    global_orient_f = global_orient_s1.detach().clone().requires_grad_(True)
    body_pose_f = body_pose_s1.detach().clone().requires_grad_(True)
    betas_f = betas.detach().clone().requires_grad_(True)

    opt_params3 = [global_orient_f, body_pose_f, betas_f]
    optimizer3 = torch.optim.LBFGS(opt_params3, lr=0.5, max_iter=20, line_search_fn='strong_wolfe')

    def closure3():
        optimizer3.zero_grad()
        output = body_model(betas=betas_f, body_pose=body_pose_f, global_orient=global_orient_f,
                            jaw_pose=jaw_pose, left_hand_pose=left_hand, right_hand_pose=right_hand)
        proj = _project_points(output.joints, cam, proj_indices)
        full_pose = torch.cat([global_orient_f, body_pose_f, jaw_pose, left_hand, right_hand], dim=1)
        loss = total_loss(proj, obs_kp, betas_f, full_pose, output.vertices,
                          height_cm / 100.0, kp_weights,
                          w_kp=w['keypoint'], w_shape=w['reg_shape'] * 0.5,
                          w_pose=w['reg_pose'] * 0.5, w_height=w['height'])
        loss.backward()
        return loss

    for i in range(fit_cfg['stage3_iterations'] // 20):
        optimizer3.step(closure3)

    final_model = body_model(betas=betas_f, body_pose=body_pose_f, global_orient=global_orient_f,
                             jaw_pose=jaw_pose, left_hand_pose=left_hand, right_hand_pose=right_hand)

    full_pose_tensor = torch.cat([global_orient_f, body_pose_f, jaw_pose, left_hand, right_hand, betas_f], dim=1)

    return {
        'betas': betas_f.detach().cpu(),
        'pose': full_pose_tensor.detach().cpu(),
        'vertices': final_model.vertices.detach().cpu(),
        'faces': torch.tensor(body_model.faces_tensor, dtype=torch.long),
        'joints': final_model.joints.detach().cpu(),
    }
