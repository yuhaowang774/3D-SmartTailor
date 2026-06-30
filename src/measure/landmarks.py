"""MediaPipe 33 关键点索引常量

参考: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
33 个关键点覆盖面部、上肢、躯干、下肢, 用于椭球人体建模.
"""

# MediaPipe Pose 33 关键点索引
MEDIAPIPE_JOINTS = {
    # 面部 (0-10)
    'nose':              0,
    'left_eye_inner':    1,
    'left_eye':          2,
    'left_eye_outer':    3,
    'right_eye_inner':   4,
    'right_eye':         5,
    'right_eye_outer':   6,
    'left_ear':          7,
    'right_ear':         8,
    'mouth_left':        9,
    'mouth_right':       10,
    # 上肢 (11-22)
    'left_shoulder':     11,
    'right_shoulder':    12,
    'left_elbow':        13,
    'right_elbow':       14,
    'left_wrist':        15,
    'right_wrist':       16,
    'left_pinky':        17,
    'right_pinky':       18,
    'left_index':        19,
    'right_index':       20,
    'left_thumb':        21,
    'right_thumb':       22,
    # 躯干 (23-24)
    'left_hip':          23,
    'right_hip':         24,
    # 下肢 (25-32)
    'left_knee':         25,
    'right_knee':        26,
    'left_ankle':        27,
    'right_ankle':       28,
    'left_heel':         29,
    'right_heel':        30,
    'left_foot_index':   31,
    'right_foot_index':  32,
}

# 别名 (兼容旧代码)
JOINTS = MEDIAPIPE_JOINTS
SMPLX_JOINTS = MEDIAPIPE_JOINTS  # 旧名称保留, 索引含义已变


def get_joint_positions(kp_3d) -> dict:
    """
    从 3D 关键点 (MediaPipe 33) 提取关节位置.

    Args:
        kp_3d: (33, 4) 或 (1, 33, 4) numpy/torch, [X_m, Y_m, Z_m, vis]

    Returns:
        dict: {joint_name: np.ndarray (3,)} 位置, 单位: 米
    """
    import numpy as np

    # 统一为 numpy (33, 4)
    if hasattr(kp_3d, 'detach'):
        arr = kp_3d.detach().cpu().numpy()
    else:
        arr = np.asarray(kp_3d)
    if arr.ndim == 3:
        arr = arr[0]

    positions = {}
    for name, idx in MEDIAPIPE_JOINTS.items():
        if idx < arr.shape[0]:
            positions[name] = arr[idx, :3].astype(np.float64)
        else:
            positions[name] = np.zeros(3, dtype=np.float64)
    return positions
