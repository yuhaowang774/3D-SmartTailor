"""几何计算: 椭球截面 + 解析椭圆周长 + 凸包

椭球被水平平面 y=h 切割:
  - 截面是椭圆 (可能在椭球局部坐标系中)
  - 解析公式: 在椭球局部系中, 若椭球轴 (a, b, c) 沿 (X, Y, Z),
    则 y=h 截面为 XZ 平面椭圆, 半轴 a*sqrt(1-(h/b)²) 和 c*sqrt(1-(h/b)²)
  - 周长用 Ramanujan 近似: P ≈ π [ 3(a+b) - √((3a+b)(a+3b)) ], 误差 < 0.1%
"""

import numpy as np
from scipy.spatial import ConvexHull


def convex_hull_perimeter(points_2d: np.ndarray) -> float:
    """2D 点集的凸包周长"""
    if len(points_2d) < 3:
        return 0.0
    hull = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]
    perimeter = np.sum(np.linalg.norm(np.diff(hull_pts, axis=0, append=hull_pts[:1]), axis=1))
    return float(perimeter)


def ellipse_perimeter(a: float, b: float) -> float:
    """
    Ramanujan 第一近似公式计算椭圆周长.

    Args:
        a, b: 椭圆两半轴长度

    Returns:
        周长 (与 a, b 同单位)

    误差 < 0.1% 当 a/b < 3
    """
    if a <= 0 or b <= 0:
        return 0.0
    return float(np.pi * (3.0 * (a + b) -
                          np.sqrt((3.0 * a + b) * (a + 3.0 * b))))


def ellipsoid_cross_section_perimeter(
    center: np.ndarray,
    scales: np.ndarray,
    rotation_quat: np.ndarray,
    y_height: float,
) -> float:
    """
    单个椭球被水平平面 y=y_height 切割后的截面椭圆周长 (米).

    步骤:
      1. 把 y_height 转换到椭球局部坐标系 (考虑旋转)
      2. 在局部系中, 椭球方程: x²/a² + y²/b² + z²/c² = 1, 平面 y = y_local
         → 截面是 x²/(a²(1-t²)) + z²/(c²(1-t²)) = 1, t = y_local / b
      3. 截面半轴: a' = a*sqrt(1-t²), c' = c*sqrt(1-t²)
      4. 截面回到世界系的 XZ 平面 (但旋转可能让截面不水平, 简化: 用世界系 y 直接切)

    简化处理: 我们假设椭球的局部 Y 轴接近世界 Y 方向 (人体骨架主轴竖直),
    旋转主要在水平面内. 这种情况下上述近似成立.

    Args:
        center: (3,) 椭球中心 (世界系)
        scales: (3,) 半轴 (a, b, c) — b 是沿局部 Y 的半轴
        rotation_quat: (4,) (w, x, y, z)
        y_height: 水平平面 y 值 (世界系)

    Returns:
        截面周长 (米), 若不相交返回 0
    """
    a, b, c = float(scales[0]), float(scales[1]), float(scales[2])
    cy = float(center[1])

    # 局部 Y 在世界 Y 方向的分量 (用于计算真实的 y 偏移)
    # 旋转矩阵的第二列 = 局部 Y 在世界系的基向量
    q = np.asarray(rotation_quat, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    # R 的第二列 (对应局部 Y 在世界系的方向)
    R_col1 = np.array([
        2 * (x * y - z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z + x * w),
    ])
    # 局部 Y 在世界 Y 方向的投影
    cos_tilt = abs(R_col1[1])

    # 椭球中心到截面的有向距离 (世界 Y 系)
    dy = y_height - cy
    # 投影到椭球局部 Y 轴
    if cos_tilt < 1e-6:
        # 局部 Y 完全水平, 不与水平面相交 (退化)
        # 用 b 作为厚度近似
        return 0.0
    t = dy / cos_tilt  # 局部 Y 坐标

    # 椭球内部判断: |t| < b
    if abs(t) >= b:
        return 0.0

    # 截面椭圆半轴 (在椭球局部 XZ 平面)
    factor = np.sqrt(max(0.0, 1.0 - (t / b) ** 2))
    a_sec = a * factor
    c_sec = c * factor

    # 截面椭圆在世界系的周长 (考虑旋转可能让它倾斜)
    # 简化: 假设截面在水平面内, 直接用 Ramanujan 计算
    # (椭球局部 XZ 旋转到世界 XZ 后, 椭圆形状不变, 周长不变)
    return ellipse_perimeter(a_sec, c_sec)


def ellipsoid_cross_section_points(
    center: np.ndarray,
    scales: np.ndarray,
    rotation_quat: np.ndarray,
    y_height: float,
    n_points: int = 32,
) -> np.ndarray:
    """
    采样椭球截面椭圆上的 2D 点 (XZ 平面), 用于凸包合并多椭球截面.

    返回: (n_points, 2) [X, Z] 或空数组 (不相交)
    """
    a, b, c = float(scales[0]), float(scales[1]), float(scales[2])
    cy = float(center[1])

    q = np.asarray(rotation_quat, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    R_col1 = np.array([
        2 * (x * y - z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z + x * w),
    ])
    cos_tilt = abs(R_col1[1])
    dy = y_height - cy
    if cos_tilt < 1e-6:
        return np.zeros((0, 2), dtype=np.float64)
    t = dy / cos_tilt
    if abs(t) >= b:
        return np.zeros((0, 2), dtype=np.float64)

    factor = np.sqrt(max(0.0, 1.0 - (t / b) ** 2))
    a_sec = a * factor
    c_sec = c * factor

    # 在局部 XZ 平面采样椭圆
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    local_xz = np.stack([a_sec * np.cos(theta), np.zeros_like(theta), c_sec * np.sin(theta)], axis=-1)  # (n, 3)

    # 旋转到世界系
    R = _quat_to_matrix(q)
    world_xz = local_xz @ R.T  # (n, 3)
    # 平移到截面位置
    world_xz = world_xz + center
    # 投影到 XZ
    return world_xz[:, [0, 2]]


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 3×3 旋转矩阵"""
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


# ============================================================
# 旧接口兼容 (基于 mesh 的截面, 仍保留用于测试和回退)
# ============================================================

def cross_section_perimeter(vertices, faces, y_height, tolerance=0.005):
    """旧 mesh 截面周长 (保留供测试使用)"""
    import trimesh
    import torch
    if torch.is_tensor(vertices):
        verts = vertices.squeeze(0) if vertices.dim() == 3 else vertices
        verts_np = verts.detach().cpu().numpy()
    else:
        verts_np = np.asarray(vertices)
    if torch.is_tensor(faces):
        faces_np = faces.detach().cpu().numpy()
    else:
        faces_np = np.asarray(faces)
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    plane_origin = np.array([0, y_height, 0])
    plane_normal = np.array([0, 1, 0])
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return 0.0
    if hasattr(section, 'vertices') and len(section.vertices) > 0:
        points_2d = section.vertices[:, [0, 2]]
        return convex_hull_perimeter(points_2d)
    return 0.0


def geodesic_length(vertices, faces, start_idx, end_idx):
    """测地线长度 (mesh 上 Dijkstra), 保留用于兼容"""
    import trimesh
    import torch
    if torch.is_tensor(vertices):
        verts = vertices.squeeze(0) if vertices.dim() == 3 else vertices
        verts_np = verts.detach().cpu().numpy()
    else:
        verts_np = np.asarray(vertices)
    if torch.is_tensor(faces):
        faces_np = faces.detach().cpu().numpy()
    else:
        faces_np = np.asarray(faces)
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    try:
        distance = mesh.geodesic_distance(start_idx, end_idx)
        return float(distance) if distance is not None else float(np.linalg.norm(verts_np[start_idx] - verts_np[end_idx]))
    except Exception:
        return float(np.linalg.norm(verts_np[start_idx] - verts_np[end_idx]))
