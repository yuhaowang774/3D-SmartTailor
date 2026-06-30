import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull

def convex_hull_perimeter(points_2d: np.ndarray) -> float:
    if len(points_2d) < 3:
        return 0.0
    hull = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]
    perimeter = np.sum(np.linalg.norm(np.diff(hull_pts, axis=0, append=hull_pts[:1]), axis=1))
    return float(perimeter)

def _squeeze_verts(vertices):
    """确保vertices为(V, 3)形状，去除可能的batch维度"""
    if vertices.dim() == 3:
        return vertices.squeeze(0)
    return vertices


def cross_section_perimeter(vertices, faces, y_height, tolerance=0.005):
    verts = _squeeze_verts(vertices)
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
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
    verts = _squeeze_verts(vertices)
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    try:
        distance = mesh.geodesic_distance(start_idx, end_idx)
        return float(distance) if distance is not None else float(np.linalg.norm(verts_np[start_idx] - verts_np[end_idx]))
    except:
        return float(np.linalg.norm(verts_np[start_idx] - verts_np[end_idx]))
