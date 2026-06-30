import torch
import numpy as np
from .geometry import cross_section_perimeter, geodesic_length
from .landmarks import get_joint_positions

def extract_measurements(vertices, faces, joints):
    if vertices.dim() == 3:
        vertices = vertices.squeeze(0)  # (1,V,3) → (V,3)
    jpos = get_joint_positions(joints)
    verts_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # 胸围: 使用spine2关节(胸椎位置)作为胸围测量平面
    # 而非(shoulder+elbow)/2, 后者在手臂下垂或抬起时偏差巨大
    chest_y = jpos['spine2'][1]
    
    pelvis_y = jpos['pelvis'][1]
    # 腰围搜索范围: pelvis到spine2之间
    waist_y = _find_waist_level(verts_np, faces_np, pelvis_y, chest_y)

    knee_y = jpos['left_knee'][1]
    hip_y = _find_hip_level(verts_np, faces_np, pelvis_y, knee_y)

    chest_girth = cross_section_perimeter(vertices, faces, chest_y) * 100
    waist_girth = cross_section_perimeter(vertices, faces, waist_y) * 100
    hip_girth = cross_section_perimeter(vertices, faces, hip_y) * 100

    left_shoulder = torch.tensor(jpos['left_shoulder'])
    right_shoulder = torch.tensor(jpos['right_shoulder'])
    shoulder_width = float(torch.norm(right_shoulder - left_shoulder)) * 100

    sleeve_len = _compute_arm_length(vertices, faces, jpos, side='left') * 100
    pants_len = _compute_leg_length(vertices, faces, jpos, side='left') * 100

    return {
        'chest_cm': round(chest_girth, 1),
        'waist_cm': round(waist_girth, 1),
        'hip_cm': round(hip_girth, 1),
        'shoulder_width_cm': round(shoulder_width, 1),
        'sleeve_length_cm': round(sleeve_len, 1),
        'pants_length_cm': round(pants_len, 1),
    }

def _find_waist_level(verts_np, faces_np, pelvis_y, chest_max_y, n_samples=8):
    import trimesh
    from .geometry import convex_hull_perimeter
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    y_range = np.linspace(pelvis_y, chest_max_y, n_samples)
    min_perimeter = float('inf')
    best_y = (pelvis_y + chest_max_y) / 2.0
    for y_val in y_range:
        plane_origin = np.array([0, y_val, 0])
        plane_normal = np.array([0, 1, 0])
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None or len(section.vertices) < 10:
            continue
        pts_2d = section.vertices[:, [0, 2]]
        perimeter = convex_hull_perimeter(pts_2d)
        if 0 < perimeter < min_perimeter:
            min_perimeter = perimeter
            best_y = y_val
    return best_y

def _find_hip_level(verts_np, faces_np, pelvis_y, knee_y, n_samples=8):
    import trimesh
    from .geometry import convex_hull_perimeter
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    y_range = np.linspace(pelvis_y, knee_y, n_samples)
    max_perimeter = 0.0
    best_y = pelvis_y
    for y_val in y_range:
        plane_origin = np.array([0, y_val, 0])
        plane_normal = np.array([0, 1, 0])
        section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        if section is None or len(section.vertices) < 10:
            continue
        pts_2d = section.vertices[:, [0, 2]]
        perimeter = convex_hull_perimeter(pts_2d)
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            best_y = y_val
    return best_y

def _compute_arm_length(vertices, faces, jpos, side='left'):
    prefix = f'{side}_'
    shoulder_idx = _closest_vertex(vertices, jpos[f'{prefix}shoulder'])
    elbow_idx = _closest_vertex(vertices, jpos[f'{prefix}elbow'])
    wrist_idx = _closest_vertex(vertices, jpos[f'{prefix}wrist'])
    upper = geodesic_length(vertices, faces, shoulder_idx, elbow_idx)
    lower = geodesic_length(vertices, faces, elbow_idx, wrist_idx)
    return upper + lower

def _compute_leg_length(vertices, faces, jpos, side='left'):
    prefix = f'{side}_'
    hip_idx = _closest_vertex(vertices, jpos[f'{prefix}hip'])
    knee_idx = _closest_vertex(vertices, jpos[f'{prefix}knee'])
    ankle_idx = _closest_vertex(vertices, jpos[f'{prefix}ankle'])
    upper = geodesic_length(vertices, faces, hip_idx, knee_idx)
    lower = geodesic_length(vertices, faces, knee_idx, ankle_idx)
    return upper + lower

def _closest_vertex(vertices, point):
    point_t = torch.tensor(point, dtype=vertices.dtype, device=vertices.device)
    distances = torch.norm(vertices - point_t, dim=1)
    return int(torch.argmin(distances))
