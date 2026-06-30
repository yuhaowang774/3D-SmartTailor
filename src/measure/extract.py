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
    if joints is not None:
        joints = np.asarray(joints, dtype=np.float64)
        joints = joints.copy()
        joints[:, 1] -= y_shift

    # X 居中: PIFuHD 输出的 mesh 常偏离原点 (身体中心可能位于 X≈-0.7 等).
    # 用上躯干 (胸/肩区域, 重建最干净) 的中位 X 估计身体中心, 然后平移 X 使身体中心=0.
    # 这样后续过滤器可用 |X| < 阈值 来移除远离身体的漂浮伪影 (PIFuHD 在 hip/thigh 交界处常见).
    y_min_tmp = float(verts[:, 1].min())
    y_max_tmp = float(verts[:, 1].max())
    y_range_tmp = y_max_tmp - y_min_tmp
    if y_range_tmp > 0.1:
        x_shift = _estimate_body_center_x(verts, faces, y_min_tmp, y_range_tmp)
    else:
        x_shift = 0.0
    if abs(x_shift) > 1e-4:
        verts[:, 0] -= x_shift
        if joints is not None:
            joints = joints.copy()
            joints[:, 0] -= x_shift
    mesh = trimesh.Trimesh(vertices=verts, faces=f if faces is not None else None, process=False)

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
    # 构建完整围度曲线: PIFuHD 在 hip/thigh 交界处常有重建缺口, 单点测量会失效.
    # 改为细粒度扫描 + 无效截面插值, 再从曲线提取胸/腰/臀.
    girth_curve = _build_girth_curve(mesh, y_min, y_max)

    chest_g = _girth_at(girth_curve, chest_y) * 100
    waist_g = _min_girth_in_range(girth_curve, waist_y_min, waist_y_max) * 100

    # 臀围: 检查 hip 范围是否有有效截面. 若全部破损 (PIFuHD 缺口), 线性插值会
    # 低估臀围 (臀围是局部最大值, 落在缺口内), 改用解剖学比例从胸围估计.
    hip_valid = _count_valid_sections(mesh, hip_y_min, hip_y_max)
    if hip_valid > 0:
        hip_g = _max_girth_in_range(girth_curve, hip_y_min, hip_y_max) * 100
    else:
        # 臀围 ≈ 胸围 × 1.02 (成年男性解剖学比例)
        hip_g = chest_g * 1.02

    # ----- 肩宽 -----
    if jpos and 'left_shoulder' in jpos and 'right_shoulder' in jpos:
        shoulder_w = float(np.linalg.norm(jpos['right_shoulder'] - jpos['left_shoulder'])) * 100
    else:
        # 扫描肩部区域多个高度, 取过滤后最大 X 跨度作为肩宽
        # 肩峰位置因模型而异 (SAM3D ~0.85h, PIFuHD ~0.82h), 扫描范围覆盖两者
        # 用 0.42m 宽窗口过滤 (含肩膀, 排除手臂)
        scan_min = shoulder_y - 0.02 * y_range
        scan_max = shoulder_y + 0.04 * y_range
        best_x_span = 0.0
        for y_test in np.linspace(scan_min, scan_max, 8):
            pts = _mesh_section_points_2d(mesh, y_test)
            if len(pts) < 3:
                continue
            # 用宽窗口过滤 (0.42m 含肩膀, 排除手臂)
            if len(pts) > 10:
                pts = _filter_torso_points(pts, window_width=0.42)
            if len(pts) < 3:
                continue
            x_span = float(pts[:, 0].max() - pts[:, 0].min())
            # 排除颈/头区域 (X 跨度 < 0.15m 说明切到脖子)
            if x_span > 0.15 and x_span > best_x_span:
                best_x_span = x_span
        shoulder_w = best_x_span * 100

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


def _estimate_body_center_x(verts: np.ndarray, faces, y_min: float, y_range: float) -> float:
    """估计身体在 X 方向的中心位置 (用于 PIFuHD mesh 偏离原点的情况).

    用上躯干 (胸/肩区域, 重建最干净) 多个高度的截面, 取每高度 X 的 25-75 分位中点
    (对漂浮伪影/手臂稳健), 再取中位数作为身体中心.

    Returns:
        body_center_x (mesh 坐标系下的 X 值)
    """
    if faces is None or len(faces) == 0 or y_range <= 0.1:
        return 0.0
    mesh_tmp = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    candidates = []
    for y_frac in [0.78, 0.80, 0.82, 0.84, 0.86]:
        y = y_min + y_range * y_frac
        pts = _mesh_section_points_2d(mesh_tmp, y)
        if len(pts) < 20:
            continue
        x_vals = pts[:, 0]
        q25, q75 = np.percentile(x_vals, [25, 75])
        candidates.append(float((q25 + q75) / 2.0))
    if not candidates:
        return 0.0
    return float(np.median(candidates))


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
                      用 X 坐标滑动窗口密度法找身体簇 (局部中心, 对身体偏移稳健).

    注意: 返回 0.0 表示该高度无截面或截面破损 (PIFuHD 重建缺口).
          调用方应用围度曲线插值处理破损截面, 而非直接使用.
    """
    pts_2d = _mesh_section_points_2d(mesh, y_height)
    if len(pts_2d) < 3:
        return 0.0

    # 排除手臂/伪影: 局部密度窗口过滤 (找身体簇, 对身体偏离原点稳健)
    if exclude_limbs and len(pts_2d) > 10:
        filtered = _filter_torso_points(pts_2d)
        if len(filtered) >= 3:
            girth = _convex_hull_perimeter(filtered)
            # 截面完整时 girth > 0.6m; 破损截面 (PIFuHD 缺口) 会偏小, 返回 0 让曲线插值处理
            if girth >= 0.6:
                return girth
            return 0.0
    return _convex_hull_perimeter(pts_2d)


def _convex_hull_perimeter(pts_2d: np.ndarray) -> float:
    """计算 2D 点集的凸包周长 (米)"""
    if len(pts_2d) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts_2d)
        hull_pts = pts_2d[hull.vertices]
        return float(np.sum(np.linalg.norm(
            np.diff(hull_pts, axis=0, append=hull_pts[:1]), axis=1
        )))
    except Exception:
        return 0.0


def _filter_torso_points(pts_2d: np.ndarray, window_width: float = 0.28) -> np.ndarray:
    """从截面点中分离躯干主体, 排除手臂.

    策略 (滑动窗口密度法):
      1. 对 X 坐标排序
      2. 用指定宽度窗口扫描, 找密度最高的窗口
      3. 保留该窗口内的点 + 裕度

    Args:
        pts_2d: (N, 2) XZ 平面点
        window_width: 窗口宽度 (米). 躯干用 0.28m, 肩部用 0.42m (含肩)

    Returns:
        (M, 2) 躯干点集
    """
    if len(pts_2d) < 10:
        return pts_2d

    x_coords = pts_2d[:, 0]
    x_sorted = np.sort(x_coords)
    n = len(x_sorted)

    best_count = 0
    best_left = float(x_sorted[0])
    best_right = float(x_sorted[0]) + window_width

    left = 0
    for right_idx in range(n):
        while x_sorted[right_idx] - x_sorted[left] > window_width:
            left += 1
        count = right_idx - left + 1
        if count > best_count:
            best_count = count
            best_left = float(x_sorted[left])
            best_right = float(x_sorted[right_idx])

    margin = 0.03
    torso_x_min = best_left - margin
    torso_x_max = best_right + margin

    keep_mask = (x_coords >= torso_x_min) & (x_coords <= torso_x_max)
    filtered = pts_2d[keep_mask]

    return filtered if len(filtered) >= 3 else pts_2d


def _build_girth_curve(mesh, y_min: float, y_max: float,
                       step: float = 0.02) -> list:
    """构建身高围度曲线 (细粒度扫描 + 破损截面插值).

    PIFuHD 在 hip/thigh 交界处常有重建缺口, 单个高度的截面可能破损 (girth=0).
    本函数:
      1. 以 step 步长扫描 y_min~y_max, 计算每个高度的围度
      2. 标记无效截面 (girth < 0.5m)
      3. 用线性插值从最近的有效邻居填充无效截面

    Returns:
        [(y, girth), ...] 已插值的围度曲线 (按 y 升序)
    """
    # 跳过脚底 (y<0.15) 和头顶 (y>0.97*range), 这些区域无躯干截面
    y_lo = max(y_min + 0.15, 0.15)
    y_hi = y_min + (y_max - y_min) * 0.97
    ys = np.arange(y_lo, y_hi, step)
    raw = []
    for y in ys:
        g = _mesh_section_girth(mesh, float(y))
        raw.append((float(y), g))

    # 标记有效/无效
    valid_idx = [i for i, (_, g) in enumerate(raw) if 0.6 <= g <= 2.0]
    if not valid_idx:
        return raw  # 全部无效, 原样返回 (调用方兜底)

    # 线性插值填充无效区间
    valid_ys = np.array([raw[i][0] for i in valid_idx])
    valid_gs = np.array([raw[i][1] for i in valid_idx])
    result = []
    for y, g in raw:
        if 0.6 <= g <= 2.0:
            result.append((y, g))
        else:
            # 插值 (valid_ys 已排序)
            interp = float(np.interp(y, valid_ys, valid_gs))
            result.append((y, interp))
    return result


def _girth_at(curve: list, y: float) -> float:
    """从围度曲线取指定高度的围度 (线性插值)."""
    if not curve:
        return 0.0
    ys = np.array([p[0] for p in curve])
    gs = np.array([p[1] for p in curve])
    return float(np.interp(y, ys, gs))


def _min_girth_in_range(curve: list, y_min: float, y_max: float) -> float:
    """围度曲线在 [y_min, y_max] 范围内的最小值."""
    gs = [g for y, g in curve if y_min <= y <= y_max]
    return min(gs) if gs else 0.0


def _max_girth_in_range(curve: list, y_min: float, y_max: float) -> float:
    """围度曲线在 [y_min, y_max] 范围内的最大值."""
    gs = [g for y, g in curve if y_min <= y <= y_max]
    return max(gs) if gs else 0.0


def _count_valid_sections(mesh, y_min: float, y_max: float,
                          step: float = 0.02, min_valid: float = 0.6) -> int:
    """统计 [y_min, y_max] 范围内有效 (非破损) 截面的数量.

    用于判断某区域是否被 PIFuHD 重建缺口覆盖.
    """
    count = 0
    for y in np.arange(y_min, y_max, step):
        g = _mesh_section_girth(mesh, float(y))
        if min_valid <= g <= 2.0:
            count += 1
    return count


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
