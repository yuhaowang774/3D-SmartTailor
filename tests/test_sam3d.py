"""测试 SAM 3D Body 集成模块

不依赖实际 SAM 3D Body 模型 (避免 GPU/模型下载前置条件),
用合成 mesh + 真实 GLB 文件 (如有) 验证:
  - reconstruction 抽象层导入正常
  - GlbFileBackend 解析 GLB 正常
  - extract_measurements 从 mesh 提取尺寸正常
  - render_mesh_to_image 渲染正常
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 真实 GLB 文件路径 (来自 sam3d.org, 用于端到端测试)
REAL_GLB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'body_models', 'person_0.glb')


def _make_synthetic_human_mesh(height_m: float = 1.70) -> tuple:
    """生成简化的人体 mesh (圆柱组合, 沿 Y 轴) 用于测试

    Returns: (vertices (V,3), faces (F,3))
    """
    import trimesh
    meshes = []

    # 躯干: 圆柱沿 Y 轴, 从髋(0.90)到肩(1.40), 半径 0.13
    torso = trimesh.creation.cylinder(
        radius=0.13, segment=[[0, 0.90, 0], [0, 1.40, 0]], sections=24,
    )
    meshes.append(torso)

    # 头: 球, 中心在 1.55, 半径 0.10
    head = trimesh.creation.icosphere(radius=0.10, subdivisions=2)
    head.apply_translation([0, 1.55, 0])
    meshes.append(head)

    # 左腿: 圆柱沿 Y, 从踝(0.05)到髋(0.90), 半径 0.06
    leg_l = trimesh.creation.cylinder(
        radius=0.06, segment=[[0.07, 0.05, 0], [0.07, 0.90, 0]], sections=16,
    )
    meshes.append(leg_l)

    # 右腿
    leg_r = trimesh.creation.cylinder(
        radius=0.06, segment=[[-0.07, 0.05, 0], [-0.07, 0.90, 0]], sections=16,
    )
    meshes.append(leg_r)

    # 左臂: 圆柱沿 Y, 从腕(1.00)到肩(1.40), 半径 0.045
    arm_l = trimesh.creation.cylinder(
        radius=0.045, segment=[[0.20, 1.00, 0], [0.20, 1.40, 0]], sections=16,
    )
    meshes.append(arm_l)

    # 右臂
    arm_r = trimesh.creation.cylinder(
        radius=0.045, segment=[[-0.20, 1.00, 0], [-0.20, 1.40, 0]], sections=16,
    )
    meshes.append(arm_r)

    # 合并
    combined = trimesh.util.concatenate(meshes)

    # 缩放到目标身高 (原始 Y 范围 0.05~1.65 = 1.60m)
    y_range = combined.vertices[:, 1].max() - combined.vertices[:, 1].min()
    scale = height_m / y_range
    combined.apply_scale(scale)

    return combined.vertices, combined.faces


# ============================================================
# 测试 1: reconstruction 抽象层
# ============================================================

def test_reconstruction_imports():
    """验证 reconstruction 模块导入正常"""
    from reconstruction import BodyReconstructor, ReconstructionResult, create_reconstructor
    assert BodyReconstructor is not None
    assert ReconstructionResult is not None
    assert create_reconstructor is not None


def test_create_reconstructor_unknown_backend():
    """未知后端应抛 ValueError"""
    from reconstruction import create_reconstructor
    with pytest.raises(ValueError, match="Unknown"):
        create_reconstructor({'reconstruction': {'backend': 'nonexistent'}})


def test_create_reconstructor_default_is_glb_file():
    """默认后端应为 glb_file"""
    from reconstruction import create_reconstructor
    rec = create_reconstructor({})
    assert rec.backend_name == 'glb_file'


def test_sam3d_local_backend_unavailable_without_model():
    """本地后端在模型未下载时应 is_available()=False"""
    from reconstruction import create_reconstructor
    cfg = {'reconstruction': {'backend': 'sam3d_local'}}
    rec = create_reconstructor(cfg)
    assert rec.backend_name == 'sam3d_local'
    assert rec.is_available() is False


def test_glb_file_backend_always_available():
    """GLB 后端始终可用"""
    from reconstruction import create_reconstructor
    cfg = {'reconstruction': {'backend': 'glb_file'}}
    rec = create_reconstructor(cfg)
    assert rec.backend_name == 'glb_file'
    assert rec.is_available() is True


# ============================================================
# 测试 2: GlbFileBackend 解析真实 GLB
# ============================================================

@pytest.mark.skipif(not os.path.exists(REAL_GLB_PATH),
                    reason=f"真实 GLB 文件不存在: {REAL_GLB_PATH}")
def test_glb_file_backend_parse_real_glb():
    """解析真实 person_0.glb 文件"""
    from reconstruction.glb_file import GlbFileBackend

    backend = GlbFileBackend({})
    result = backend.reconstruct_from_glb_path(REAL_GLB_PATH)

    # 验证 mesh 数据
    assert result.vertices.shape[1] == 3
    assert result.faces.shape[1] == 3
    assert len(result.vertices) > 1000, "GLB 顶点数太少"
    assert len(result.faces) > 1000, "GLB 面片数太少"

    # 验证 metadata
    assert result.metadata['backend'] == 'glb_file'
    assert result.metadata['n_vertices'] == len(result.vertices)
    assert result.metadata['n_faces'] == len(result.faces)

    # 验证 GLB 二进制
    assert result.glb_bytes is not None
    assert len(result.glb_bytes) > 1000

    # Y 跨度应接近人体高度 (1.5~2.0m)
    y_span = float(result.vertices[:, 1].max() - result.vertices[:, 1].min())
    assert 1.0 < y_span < 2.5, f"Y span {y_span}m 不合理"


@pytest.mark.skipif(not os.path.exists(REAL_GLB_PATH),
                    reason=f"真实 GLB 文件不存在: {REAL_GLB_PATH}")
def test_glb_file_backend_rejects_image_input():
    """GLB 后端不支持从图像重建"""
    from reconstruction.glb_file import GlbFileBackend
    backend = GlbFileBackend({})
    with pytest.raises(NotImplementedError):
        backend.reconstruct(np.zeros((100, 100, 3), dtype=np.uint8))


# ============================================================
# 测试 3: 测量提取 (从 mesh)
# ============================================================

def test_extract_measurements_from_synthetic_mesh():
    """从合成人体 mesh 提取 6 项尺寸, 验证范围合理"""
    from measure.extract import extract_measurements

    verts, faces = _make_synthetic_human_mesh(height_m=1.70)
    result = extract_measurements(
        vertices=verts,
        faces=faces,
        height_cm=170.0,
    )

    expected_keys = {'chest_cm', 'waist_cm', 'hip_cm',
                     'shoulder_width_cm', 'sleeve_length_cm', 'pants_length_cm'}
    assert set(result.keys()) == expected_keys, f"Missing keys: {expected_keys - set(result.keys())}"

    # 数值范围合理性 (成年男性 170cm, 合成 mesh 比真人略宽)
    assert 60 < result['chest_cm'] < 140, f"Chest {result['chest_cm']} out of range"
    assert 50 < result['waist_cm'] < 130, f"Waist {result['waist_cm']} out of range"
    assert 60 < result['hip_cm'] < 140, f"Hip {result['hip_cm']} out of range"
    assert 30 < result['shoulder_width_cm'] < 60, f"Shoulder {result['shoulder_width_cm']} out of range"
    assert 30 < result['sleeve_length_cm'] < 80, f"Sleeve {result['sleeve_length_cm']} out of range"
    assert 60 < result['pants_length_cm'] < 120, f"Pants {result['pants_length_cm']} out of range"

    # 腰围应 < 胸围 (解剖学常识)
    assert result['waist_cm'] < result['chest_cm'], \
        f"Waist {result['waist_cm']} should be < chest {result['chest_cm']}"


@pytest.mark.skipif(not os.path.exists(REAL_GLB_PATH),
                    reason=f"真实 GLB 文件不存在: {REAL_GLB_PATH}")
def test_extract_measurements_from_real_glb():
    """从真实 person_0.glb 提取尺寸, 验证落入人体合理区间"""
    from reconstruction.glb_file import GlbFileBackend
    from measure.extract import extract_measurements

    backend = GlbFileBackend({})
    result = backend.reconstruct_from_glb_path(REAL_GLB_PATH)
    measurements = extract_measurements(
        vertices=result.vertices,
        faces=result.faces,
        height_cm=170.0,
    )

    # 真实 SAM 3D Body mesh 应落入人体合理区间
    assert 60 < measurements['chest_cm'] < 150, f"Chest {measurements['chest_cm']} out of range"
    assert 50 < measurements['waist_cm'] < 150, f"Waist {measurements['waist_cm']} out of range"
    assert 60 < measurements['hip_cm'] < 150, f"Hip {measurements['hip_cm']} out of range"
    assert 25 < measurements['shoulder_width_cm'] < 60, f"Shoulder {measurements['shoulder_width_cm']} out of range"
    assert 30 < measurements['sleeve_length_cm'] < 80, f"Sleeve {measurements['sleeve_length_cm']} out of range"
    assert 50 < measurements['pants_length_cm'] < 120, f"Pants {measurements['pants_length_cm']} out of range"


def test_extract_measurements_height_normalization():
    """身高归一化: 不同身高输入应得到不同尺寸"""
    from measure.extract import extract_measurements

    verts_170, faces_170 = _make_synthetic_human_mesh(height_m=1.70)
    result_170 = extract_measurements(vertices=verts_170, faces=faces_170, height_cm=170.0)

    verts_180, faces_180 = _make_synthetic_human_mesh(height_m=1.80)
    result_180 = extract_measurements(vertices=verts_180, faces=faces_180, height_cm=180.0)

    # 高个子胸围应更大
    assert result_180['chest_cm'] > result_170['chest_cm'], \
        f"Taller person should have larger chest: {result_180['chest_cm']} vs {result_170['chest_cm']}"


# ============================================================
# 测试 4: 渲染
# ============================================================

def test_render_mesh_to_image(tmp_path):
    """渲染 mesh 到 PNG"""
    from output.render import render_mesh_to_image

    verts, faces = _make_synthetic_human_mesh(height_m=1.70)
    out_path = str(tmp_path / "test_render.png")

    result_path = render_mesh_to_image(
        vertices=verts, faces=faces,
        out_path=out_path, title="Test",
    )

    assert result_path == out_path
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000


# ============================================================
# 测试 5: ReconstructionResult 数据类
# ============================================================

def test_reconstruction_result_dataclass():
    """ReconstructionResult 数据类字段"""
    from reconstruction import ReconstructionResult

    verts = np.zeros((10, 3), dtype=np.float32)
    faces = np.zeros((5, 3), dtype=np.int32)
    result = ReconstructionResult(
        vertices=verts,
        faces=faces,
        glb_bytes=b'fake_glb',
        metadata={'backend': 'test'},
    )

    assert result.vertices.shape == (10, 3)
    assert result.glb_bytes == b'fake_glb'
    assert result.metadata['backend'] == 'test'
    assert result.joints is None
