"""从 3D mesh 提取 6 项人体尺寸

SAM 3D Body 输出真实三角网格 (非椭球), 采用 mesh 截面方案:
  - 围度: 用 trimesh.section 在水平面 y=h 切割, 取凸包周长
  - 长度: 用 3D 关键点 (如有) 计算, 否则用 mesh 包围盒估算

测量项:
  - chest_cm: 胸围 (肩下 10cm 处截面)
  - waist_cm: 腰围 (髋到胸之间扫描最小值)
  - hip_cm:   臀围 (髋部最宽处)
  - shoulder_width_cm: 肩宽
  - sleeve_length_cm:  袖长
  - pants_length_cm:   裤长
"""

import numpy as np
import trimesh


# ============================================================
# 主入口
# ============================================================

def extract_measurements(
    vertices,
    faces=None,
    joints=None,
    joint_names=None,
    height_cm: float = None,
    scale_factor: float = 1.0,
) -> dict:
    """
    从 3D mesh + (可选) 关节 提取 6 项尺寸.

    Args:
        vertices: (V, 3) 顶点 (米)
        faces:    (F, 3) 面片索引 (可选, 无则从 vertices 推断)
        joints:   (J, 3) 关节坐标 (可选)
        joint_names: 关节名称列表
        height_cm: 用户身高 (用于归一化, 可选)
        scale_factor: 额外缩放因子 (云端结果可能需要)

    Returns:
        dict: 6 项尺寸 (cm)
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim == 3:
        verts = verts[0]
    if scale_factor != 1.0:
        verts = verts * scale_factor
        if joints is not None:
            joints = np.asarray(joints, dtype=np.float64) * scale_factor

    # 构建 trimesh 对象
    if faces is not None:
        f = np.asarray(faces, dtype=np.int64)
        if f.ndim == 3:
            f = f[0]
        mesh = trimesh.Trimesh(vertices=verts, faces=f, process=False)
    else:
        # 无 faces, 用 alpha shape 或凸包
        mesh = trimesh.Trimesh(vertices=verts, process=True)
        if len(mesh.faces) == 0:
            mesh = trimesh.convex.convex_hull(verts)

    # 用身高归一化 (云端输出尺度可能不正确)
    if height_cm is not None:
        height_m = height_cm / 100.0
        # 当前 mesh 的 Y 跨度
        y_range = float(verts[:, 1].max() - verts[:, 1].min())
        if y_range > 0.1:
            target_scale = height_m / y_range
            # 只在差距较大时归一化 (避免误校正)
            if 0.5 < target_scale < 2.0:
                verts = verts * target_scale
                mesh = trimesh.Trimesh(vertices=verts, faces=f if faces is not None else None, process=False)
                if joints is not None:
                    joints = np.asarray(joints, dtype=np.float64) * target_scale

    # Y 归零: 让脚底 (Y_min) = 0, 后续 shoulder_y/hip_y 用比例计算才正确
    y_shift = float(verts[:, 1].min())
    verts = verts.copy()
    verts[:, 1] -= y_shift
    mesh = trimesh.Trimesh(vertices=verts, faces=f if faces is not None else None, process=False)
    if joints is not None:
        joints = np.asarray(joints, dtype=np.float64)
        joints = joints.copy()
        joints[:, 1] -= y_shift

    # 提取关节位置 (如有)
    jpos = _parse_joints(joints, joint_names)

    # ----- 关键 Y 高度 -----
    # 默认: 用 mesh 的 Y 范围推算
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    y_range = y_max - y_min

    if jpos and 'left_shoulder' in jpos and 'left_hip' in jpos:
        shoulder_y = (jpos['left_shoulder'][1] + jpos['right_shoulder'][1]) / 2.0
        hip_y = (jpos['left_hip'][1] + jpos['right_hip'][1]) / 2.0
    else:
        # 无关节, 用比例估算 (解剖学: 肩≈0.82h, 髋≈0.53h)
        shoulder_y = y_min + y_range * 0.82
        hip_y = y_min + y_range * 0.53

    chest_y = shoulder_y - 0.10
    waist_y_min = hip_y + 0.02
    waist_y_max = shoulder_y - 0.08
    hip_y_min = hip_y - 0.05
    hip_y_max = hip_y + 0.05

    # ----- 围度 (mesh 截面 + 凸包周长) -----
    chest_g = _mesh_section_girth(mesh, chest_y) * 100
    waist_g = _find_min_girth_in_range(mesh, waist_y_min, waist_y_max, n_samples=12) * 100
    hip_g = _find_max_girth_in_range(mesh, hip_y_min, hip_y_max, n_samples=10) * 100

    # ----- 肩宽 -----
    if jpos and 'left_shoulder' in jpos and 'right_shoulder' in jpos:
        shoulder_w = float(np.linalg.norm(jpos['right_shoulder'] - jpos['left_shoulder'])) * 100
    else:
        # 用肩峰高度 (约 0.85h) 的截面 X 跨度估算肩宽
        # shoulder_y=0.82h 偏低, 切到肩膀下方含上臂区域, 导致肩宽偏大
        # 肩峰处手臂已离开躯干, X 跨度即为真实肩宽
        shoulder_peak_y = shoulder_y + 0.03 * y_range
        shoulder_pts = _mesh_section_points_2d(mesh, shoulder_peak_y)
        # 肩峰高度无截面时 (合成 mesh 或躯干较短), 回退到 shoulder_y
        if len(shoulder_pts) == 0:
            shoulder_pts = _mesh_section_points_2d(mesh, shoulder_y)
        if len(shoulder_pts) > 0:
            x_span = float(shoulder_pts[:, 0].max() - shoulder_pts[:, 0].min())
            # T-pose (手臂水平张开) 时 X 跨度含手臂长度, 需过滤
            # A-pose 或手臂微张时肩峰处手臂已离开, 不需过滤
            if x_span > 0.6:
                shoulder_pts = _filter_torso_points(shoulder_pts)
            shoulder_w = float(shoulder_pts[:, 0].max() - shoulder_pts[:, 0].min()) * 100
        else:
            shoulder_w = 0.0

    # ----- 袖长 -----
    if jpos and all(k in jpos for k in ['left_shoulder', 'left_elbow', 'left_wrist']):
        sleeve = _bone_chain_length(jpos, ['left_shoulder', 'left_elbow', 'left_wrist']) * 100
    else:
        sleeve = y_range * 0.30 * 100  # 经验比例

    # ----- 裤长 -----
    if jpos and all(k in jpos for k in ['left_hip', 'left_knee', 'left_ankle']):
        pants = _bone_chain_length(jpos, ['left_hip', 'left_knee', 'left_ankle']) * 100
    else:
        pants = y_range * 0.45 * 100  # 经验比例

    return {
        'chest_cm': round(float(chest_g), 1),
        'waist_cm': round(float(waist_g), 1),
        'hip_cm': round(float(hip_g), 1),
        'shoulder_width_cm': round(float(shoulder_w), 1),
        'sleeve_length_cm': round(float(sleeve), 1),
        'pants_length_cm': round(float(pants), 1),
    }


# ============================================================
# 辅助函数
# ============================================================

def _parse_joints(joints, joint_names) -> dict:
    """把关节数组解析为 {name: position} 字典"""
    if joints is None:
        return {}
    j = np.asarray(joints, dtype=np.float64)
    if j.ndim == 3:
        j = j[0]
    if joint_names is None:
        return {}
    return {name: j[i] for i, name in enumerate(joint_names) if i < len(j)}


def _mesh_section_points_2d(mesh: trimesh.Trimesh, y_height: float) -> np.ndarray:
    """mesh 在 y=y_height 处的截面点 (XZ 平面, 2D).

    手动实现 mesh-plane 交线: 遍历每个三角面, 若三顶点的 Y 跨越 y_height,
    则计算交线段端点. 不依赖 shapely.

    Returns:
        (N, 2) XZ 平面点数组; 空数组表示无截面.
    """
    try:
        v = mesh.vertices
        f = mesh.faces
        if len(f) == 0 or len(v) == 0:
            return np.zeros((0, 2))

        # 每个面三顶点的 Y 值
        y0 = v[f[:, 0], 1]
        y1 = v[f[:, 1], 1]
        y2 = v[f[:, 2], 1]

        # 面是否跨越 y_height 平面
        y_min_face = np.minimum(np.minimum(y0, y1), y2)
        y_max_face = np.maximum(np.maximum(y0, y1), y2)
        crossing = (y_min_face <= y_height) & (y_max_face >= y_height)

        if not crossing.any():
            return np.zeros((0, 2))

        crossing_faces = f[crossing]
        y0c = v[crossing_faces[:, 0], 1]
        y1c = v[crossing_faces[:, 1], 1]
        y2c = v[crossing_faces[:, 2], 1]

        pts_list = []
        for i in range(len(crossing_faces)):
            face = crossing_faces[i]
            ys = np.array([y0c[i], y1c[i], y2c[i]])
            verts_face = v[face]

            edge_pts = []
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                ya, yb = ys[a], ys[b]
                if (ya <= y_height <= yb) or (yb <= y_height <= ya):
                    if ya == yb:
                        continue
                    t = (y_height - ya) / (yb - ya)
                    p = verts_face[a] + t * (verts_face[b] - verts_face[a])
                    edge_pts.append(p)
                if len(edge_pts) >= 2:
                    break
            if len(edge_pts) == 2:
                pts_list.append(edge_pts[0])
                pts_list.append(edge_pts[1])

        if len(pts_list) < 3:
            return np.zeros((0, 2))

        return np.array(pts_list)[:, [0, 2]]  # 投影到 XZ 平面
    except Exception:
        return np.zeros((0, 2))


def _mesh_section_girth(mesh: trimesh.Trimesh, y_height: float, exclude_limbs: bool = True) -> float:
    """mesh 在 y=y_height 处的截面凸包周长 (米).

    Args:
        exclude_limbs: True 时排除手臂/腿的截面, 只保留躯干主体.
                      解决 A-pose 下手臂贴体或 T-pose 手臂张开导致围度膨胀的问题.
    """
    pts_2d = _mesh_section_points_2d(mesh, y_height)
    if len(pts_2d) < 3:
        return 0.0

    # 排除手臂: 用 X 坐标滑动窗口密度法分离躯干和手臂
    if exclude_limbs and len(pts_2d) > 10:
        pts_2d = _filter_torso_points(pts_2d)

    if len(pts_2d) < 3:
        return 0.0

    # 凸包周长
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
        return float(np.sum(np.linalg.norm(
            np.diff(hull_pts, axis=0, append=hull_pts[:1]), axis=1
        )))
    except Exception:
        return 0.0


def _filter_torso_points(pts_2d: np.ndarray) -> np.ndarray:
    """从截面点中分离躯干主体, 排除手臂.

    策略 (滑动窗口密度法, 稳健):
      1. 对 X 坐标排序
      2. 用固定宽度窗口 (0.28m, 成年躯干 X 跨度上限) 扫描
      3. 找包含最多点的窗口作为躯干主体
      4. 保留该窗口内的点

    原理: A-pose 下手臂贴体, 躯干顶点密度远高于手臂.
         手臂顶点稀疏且分布在躯干两侧, 0.28m 窗口能完整覆盖
         躯干主体 (胸/腰/臀) 同时排除两侧手臂.

    Args:
        pts_2d: (N, 2) XZ 平面点

    Returns:
        (M, 2) 躯干点集
    """
    if len(pts_2d) < 10:
        return pts_2d

    x_coords = pts_2d[:, 0]
    x_sorted = np.sort(x_coords)
    n = len(x_sorted)

    # 躯干 X 跨度上限: 成年人肩宽 ~0.42m, 胸部 ~0.32m, 腰部 ~0.28m
    # 用 0.28m 作为窗口宽度, 能覆盖胸/腰/臀躯干主体, 排除手臂
    window_width = 0.28

    best_count = 0
    best_left = float(x_sorted[0])
    best_right = float(x_sorted[0]) + window_width

    # 滑动窗口找密度最高区域 (双指针, O(n))
    left = 0
    for right_idx in range(n):
        while x_sorted[right_idx] - x_sorted[left] > window_width:
            left += 1
        count = right_idx - left + 1
        if count > best_count:
            best_count = count
            best_left = float(x_sorted[left])
            best_right = float(x_sorted[right_idx])

    # 加裕度 (避免切掉躯干边缘, 影响凸包周长)
    margin = 0.03
    torso_x_min = best_left - margin
    torso_x_max = best_right + margin

    keep_mask = (x_coords >= torso_x_min) & (x_coords <= torso_x_max)
    filtered = pts_2d[keep_mask]

    return filtered if len(filtered) >= 3 else pts_2d


def _find_min_girth_in_range(mesh, y_min, y_max, n_samples=12) -> float:
    """在 [y_min, y_max] 范围扫描最小围度"""
    best = float('inf')
    for y in np.linspace(y_min, y_max, n_samples):
        g = _mesh_section_girth(mesh, y)
        if 0 < g < best:
            best = g
    return best if best < float('inf') else 0.0


def _find_max_girth_in_range(mesh, y_min, y_max, n_samples=10) -> float:
    """在 [y_min, y_max] 范围扫描最大围度"""
    best = 0.0
    for y in np.linspace(y_min, y_max, n_samples):
        g = _mesh_section_girth(mesh, y)
        if g > best:
            best = g
    return best


def _bone_chain_length(jpos: dict, names: list) -> float:
    """关节链总长度 (米)"""
    total = 0.0
    for i in range(len(names) - 1):
        a = jpos.get(names[i])
        b = jpos.get(names[i + 1])
        if a is None or b is None:
            continue
        total += float(np.linalg.norm(b - a))
    return total
