"""3D 人体重建抽象接口

定义统一接口 BodyReconstructor, 所有后端 (本地/云端) 都实现这个接口.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ReconstructionResult:
    """重建结果容器

    Attributes:
        vertices: (V, 3) 3D 网格顶点 (米)
        faces:    (F, 3) 三角面片索引
        joints:   (J, 3) 关键关节 3D 坐标 (米), 可选
        joint_names: 关节名称列表 (与 joints 对齐), 可选
        glb_bytes: GLB 文件二进制数据 (如有, 直接用于前端下载)
        img_with_overlay: (H, W, 3) uint8 可视化图 (2D 关键点叠加), 可选
        metadata: dict 额外信息 (推理时间, 模型版本等)
    """
    vertices: np.ndarray
    faces: np.ndarray
    joints: Optional[np.ndarray] = None
    joint_names: Optional[list] = None
    glb_bytes: Optional[bytes] = None
    img_with_overlay: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


class BodyReconstructor:
    """3D 人体重建统一接口"""

    backend_name: str = "base"

    def reconstruct(self, img_rgb: np.ndarray) -> ReconstructionResult:
        """
        从单张 RGB 图像重建 3D 人体.

        Args:
            img_rgb: (H, W, 3) uint8 RGB 图像

        Returns:
            ReconstructionResult

        Raises:
            RuntimeError: 重建失败 (模型加载失败, 网络错误, 未检测到人体等)
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """检查后端是否可用 (模型已加载/网络可达)"""
        return False


def create_reconstructor(config: dict) -> BodyReconstructor:
    """根据配置创建重建器实例"""
    backend = config.get('reconstruction', {}).get('backend', 'glb_file')

    if backend == 'pifuhd_local':
        from .pifuhd_local import PifuhdLocalBackend
        return PifuhdLocalBackend(config)
    elif backend == 'glb_file':
        from .glb_file import GlbFileBackend
        return GlbFileBackend(config)
    else:
        raise ValueError(f"Unknown reconstruction backend: {backend}")


def list_available_backends(config: dict) -> list:
    """列出所有后端及其可用性状态

    Returns:
        [{"name": "glb_file", "available": True, "label": "GLB 上传"}, ...]
    """
    backends = []
    # GLB 上传 (始终可用)
    backends.append({
        'name': 'glb_file',
        'available': True,
        'label': 'GLB 文件上传',
        'description': '上传 sam3d.org 生成的 GLB 文件',
    })
    # PIFuHD 本地
    try:
        from .pifuhd_local import PifuhdLocalBackend
        p = PifuhdLocalBackend(config)
        backends.append({
            'name': 'pifuhd_local',
            'available': p.is_available(),
            'label': 'PIFuHD 图片重建',
            'description': '单张照片本地生成 3D 模型 (需 GPU + 模型)',
        })
    except Exception:
        backends.append({
            'name': 'pifuhd_local',
            'available': False,
            'label': 'PIFuHD 图片重建',
            'description': '单张照片本地生成 3D 模型 (需 GPU + 模型)',
        })
    return backends
