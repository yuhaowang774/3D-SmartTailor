"""SMPL-X 3D人体网格生成并导出为GLB文件

从MediaPipe关键点估算体型比例 → 调整SMPL-X模板 → 导出GLB

用法:
    mesh, glb_path = generate_body_glb(kp_front, kp_side, height_cm, config, out_dir)
"""

import os, sys
import numpy as np
import torch

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'body_models', 'smplx', 'models', 'smplx')
sys.path.insert(0, MODEL_PATH)

# 预加载SMPL-X模型（模块级缓存）
_smplx_model = None


def _get_smplx_model(gender='neutral', model_path=None):
    """加载SMPL-X模型（单例缓存）"""
    global _smplx_model
    if _smplx_model is not None:
        return _smplx_model

    if model_path is None:
        model_path = MODEL_PATH

    from smplx import SMPLX
    _smplx_model = SMPLX(
        model_path,
        gender=gender,
        num_betas=10,
        num_expression_coeffs=10,
    ).eval()
    return _smplx_model


def estimate_body_params_from_keypoints(kp_front, kp_height_cm):
    """
    从关键点估算简单的体型缩放参数。

    返回: dict with 'beta', 'scale'
      - beta:  (10,) 体型参数向量(近似)
      - scale: 全局缩放因子(物理尺寸)
      - pelvis_y: 髋部参考高度(用于测量)
    """
    valid = kp_front[kp_front[:, 3] > 0.5]

    if len(valid) < 8:
        # 关键点不足,使用中性体型
        return {
            'beta': np.zeros(10, dtype=np.float32),
            'scale': kp_height_cm / 170.0,
            'pelvis_y': 0.0,
        }

    # 提取有效关键点的2D坐标
    pts = valid[:, :2]

    # 肩膀宽度（左肩→右肩）
    shoulder_idx = [5, 6]  # left_shoulder, right_shoulder
    if all(i < len(kp_front) and kp_front[i, 3] > 0.5 for i in shoulder_idx):
        shoulder_px = np.linalg.norm(kp_front[5, :2] - kp_front[6, :2])
    else:
        shoulder_px = 0.0

    # 髋部宽度（左髋→右髋）
    hip_idx = [11, 12]
    if all(i < len(kp_front) and kp_front[i, 3] > 0.5 for i in hip_idx):
        hip_px = np.linalg.norm(kp_front[11, :2] - kp_front[12, :2])
    else:
        hip_px = 0.0

    # 躯干高度（肩中→髋中）
    if shoulder_px > 0 and hip_px > 0:
        shoulder_mid = (kp_front[5, :2] + kp_front[6, :2]) / 2
        hip_mid = (kp_front[11, :2] + kp_front[12, :2]) / 2
        torso_px = abs(shoulder_mid[1] - hip_mid[1])
    else:
        torso_px = 0.0

    # 总关键点高度（头顶→脚底）
    if len(pts) > 25:
        kp_height_px = abs(pts[:, 1].max() - pts[:, 1].min())
    else:
        kp_height_px = 0.0

    # ---- 从像素比例估算体型beta ----
    beta = np.zeros(10, dtype=np.float32)

    if torso_px > 0 and kp_height_px > 0:
        torso_ratio = torso_px / kp_height_px

        if shoulder_px > 0:
            shoulder_ratio = shoulder_px / kp_height_px
            # Beta 0: 整体胖瘦 (用肩宽比例估算)
            # 正常肩宽/身高 ≈ 0.22, 偏差映射到beta[0]
            beta[0] = (shoulder_ratio - 0.22) / 0.22 * 2.0
            beta[0] = np.clip(beta[0], -2.0, 2.0)

        if hip_px > 0:
            hip_ratio = hip_px / kp_height_px
            # Beta 2: 臀部宽度
            beta[2] = (hip_ratio - 0.18) / 0.18 * 2.0
            beta[2] = np.clip(beta[2], -2.0, 2.0)

        # Beta 1: 身高/躯干比例
        beta[1] = (torso_ratio - 0.30) / 0.30 * 1.5
        beta[1] = np.clip(beta[1], -2.0, 2.0)

    # 全局缩放
    scale = kp_height_cm / 100.0  # cm → 米, SMPL-X默认身高~1.7m

    # 骨盆参考高度(用于测量模块)
    pelvis_y = 0.0
    if len(pts) > 12:
        pelvis_y = float(np.median(pts[11:13, 1])) if len(pts) >= 12 else 0.0

    return {
        'beta': beta,
        'scale': scale,
        'pelvis_y': pelvis_y,
    }


def generate_smplx_mesh(kp_front, kp_side, height_cm, config):
    """
    从关键点生成SMPL-X网格。

    返回: vertices (10475,3) numpy, faces (20894,3) numpy, 单位: 米
    """
    model = _get_smplx_model()
    params = estimate_body_params_from_keypoints(kp_front, height_cm)

    device = config.get('device', 'cpu')
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    else:
        device = 'cpu'
        model = model.cpu()

    beta = torch.tensor(params['beta'][None, :], dtype=torch.float32).to(device)
    scale = params['scale']

    with torch.no_grad():
        output = model(
            betas=beta,
            expression=torch.zeros(1, 10).to(device),
            body_pose=torch.zeros(1, 63).to(device),  # Neutral A-pose (21 joints × 3)
            return_verts=True,
        )
    vertices = output.vertices.cpu().numpy()[0]
    joints = output.joints.cpu().numpy()[0]

    # 应用全局缩放（使最终身高≈指定身高）
    current_height = vertices[:, 1].max() - vertices[:, 1].min()
    target_height = height_cm / 100.0  # cm → m
    height_scale = target_height / max(current_height, 0.01)
    vertices *= height_scale
    joints *= height_scale

    faces = model.faces_tensor.cpu().numpy()

    return vertices, faces, joints


def export_glb(vertices, faces, out_path):
    """导出网格为GLB文件"""
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    # 添加皮肤色材质
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh,
        vertex_colors=np.array([[240, 200, 180]] * len(vertices), dtype=np.uint8)
    )

    stage_dir = os.path.dirname(out_path)
    if stage_dir:
        os.makedirs(stage_dir, exist_ok=True)
    mesh.export(out_path, file_type='glb')
    return out_path


def generate_body_glb(kp_front, kp_side, height_cm, config, out_dir):
    """
    生成3D人体模型并导出GLB。

    返回: (glb_path, vertices, faces, joints)
    """
    vertices, faces, joints = generate_smplx_mesh(kp_front, kp_side, height_cm, config)
    glb_path = os.path.join(out_dir, 'body_model.glb')
    export_glb(vertices, faces, glb_path)
    return glb_path, vertices, faces, joints
