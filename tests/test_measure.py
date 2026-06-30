import torch
import numpy as np


def test_convex_hull_square():
    """2D正方形的凸包周长应为4"""
    from src.measure.geometry import convex_hull_perimeter
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
    perimeter = convex_hull_perimeter(pts)
    assert abs(perimeter - 4.0) < 0.01, f"Expected 4.0, got {perimeter}"


def test_cross_section_unit_cube():
    """1×1×1立方体的截面周长应为4"""
    from src.measure.geometry import cross_section_perimeter
    vertices = torch.tensor([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ])
    faces = torch.tensor([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ])
    perimeter = cross_section_perimeter(vertices, faces, y_height=0.5)
    assert 3.8 < perimeter < 4.2, f"Expected ~4, got {perimeter}"


def test_landmark_locations():
    """验证SMPL-X关键关节索引在合理范围内"""
    from src.measure.landmarks import SMPLX_JOINTS
    assert SMPLX_JOINTS['left_shoulder'] == 16
    assert SMPLX_JOINTS['right_shoulder'] == 17
    assert SMPLX_JOINTS['left_elbow'] == 18
    assert SMPLX_JOINTS['right_elbow'] == 19
    assert SMPLX_JOINTS['left_wrist'] == 20
    assert SMPLX_JOINTS['right_wrist'] == 21
    assert SMPLX_JOINTS['left_hip'] == 1
    assert SMPLX_JOINTS['right_hip'] == 2
    assert SMPLX_JOINTS['left_knee'] == 4
    assert SMPLX_JOINTS['right_knee'] == 5
    assert SMPLX_JOINTS['left_ankle'] == 7
    assert SMPLX_JOINTS['right_ankle'] == 8


def test_extract_measurements_on_ellipsoid():
    """椭球体上的测量提取应返回合法值"""
    import trimesh as tm
    from src.measure.extract import extract_measurements
    
    mesh = tm.creation.icosphere(subdivisions=3, radius=1.0)
    verts = mesh.vertices.copy()
    verts[:, 0] *= 0.15
    verts[:, 1] *= 0.20
    verts[:, 2] *= 0.12
    vertices = torch.tensor(verts, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    joints = torch.zeros(1, 55, 3)
    joints[0, 16] = torch.tensor([-0.15, 0.20, 0.0])
    joints[0, 17] = torch.tensor([0.15, 0.20, 0.0])
    joints[0, 18] = torch.tensor([0.15, 0.05, 0.0])
    joints[0, 20] = torch.tensor([0.15, -0.05, 0.0])
    joints[0, 1] = vertices[vertices[:, 1].argmin()]
    joints[0, 2] = vertices[vertices[:, 1].argmin()]
    joints[0, 4] = torch.tensor([0.0, -0.05, 0.12])
    joints[0, 7] = torch.tensor([0.0, -0.15, 0.12])
    
    result = extract_measurements(vertices, faces, joints)
    
    for key in ['chest_cm', 'waist_cm', 'hip_cm', 'shoulder_width_cm']:
        assert key in result, f"Missing key: {key}"
        assert 5 < result[key] < 200, f"{key} = {result[key]} out of range"
