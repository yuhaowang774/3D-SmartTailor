"""3D 模型可视化渲染

提供:
  - render_mesh_to_image: matplotlib 3D 绘制三角网格, 输出 PNG
  - render_ellipsoids_to_image: (旧, 保留兼容) 椭球绘制
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def render_mesh_to_image(
    vertices,
    faces=None,
    out_path: str = None,
    title: str = "3D Mesh",
    elev: float = 10,
    azim: float = -90,
    color: str = 'skin',
) -> str:
    """
    用 matplotlib 3D 绘制三角网格.

    Args:
        vertices: (V, 3) 顶点
        faces: (F, 3) 面片 (可选, 用于绘制三角面)
        out_path: 输出 PNG 路径
        title: 标题
        elev, azim: 视角

    Returns:
        out_path
    """
    verts = vertices.detach().cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
    if verts.ndim == 3:
        verts = verts[0]

    # 交换 Y 和 Z 坐标用于显示, 使身高 (原 Y 轴) 显示在 Z 轴 (垂直方向)
    # 原始: X=左右, Y=身高, Z=前后
    # 显示: X=左右, Y=前后, Z=身高
    verts_disp = verts.copy()
    verts_disp[:, [1, 2]] = verts_disp[:, [2, 1]]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if faces is not None:
        fcs = faces.detach().cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
        if fcs.ndim == 3:
            fcs = fcs[0]
        # 绘制三角面 (Poly3DCollection)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangles = verts_disp[fcs]
        coll = Poly3DCollection(triangles, alpha=0.8, facecolor=[0.9, 0.75, 0.65],
                                edgecolor='none')
        ax.add_collection3d(coll)
    else:
        ax.scatter(verts_disp[:, 0], verts_disp[:, 1], verts_disp[:, 2], s=0.5, c=[0.9, 0.75, 0.65])

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X (Left-Right, m)')
    ax.set_ylabel('Y (Front-Back, m)')
    ax.set_zlabel('Z (Height, m)')
    ax.set_title(title)

    # 等比例坐标轴 (基于交换后的坐标)
    max_range = max(np.ptp(verts_disp[:, 0]), np.ptp(verts_disp[:, 1]), np.ptp(verts_disp[:, 2])) / 2.0 + 0.1
    mid_x = verts_disp[:, 0].mean()
    mid_y = verts_disp[:, 1].mean()
    mid_z = verts_disp[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    return out_path


def render_front_and_side_mesh(vertices, faces, out_dir):
    """生成正面 + 侧面 2 张预览图"""
    os.makedirs(out_dir, exist_ok=True)
    render_mesh_to_image(
        vertices, faces,
        os.path.join(out_dir, 'mesh_front.png'),
        title="Front View", elev=0, azim=0,
    )
    render_mesh_to_image(
        vertices, faces,
        os.path.join(out_dir, 'mesh_side.png'),
        title="Side View", elev=0, azim=-90,
    )


# ============================================================
# 旧接口兼容 (椭球渲染, 已弃用但保留导入兼容)
# ============================================================

def render_ellipsoids_to_image(*args, **kwargs):
    """已弃用: 椭球渲染, 改用 render_mesh_to_image"""
    raise DeprecationWarning(
        "render_ellipsoids_to_image 已弃用 (椭球方案已移除), "
        "请改用 render_mesh_to_image"
    )


def render_front_and_side(*args, **kwargs):
    """已弃用"""
    raise DeprecationWarning("请改用 render_front_and_side_mesh")
