"""3D 人体重建模块

提供统一接口 BodyReconstructor, 支持多种后端:
  - GlbFileBackend:    GLB 文件上传 (默认, 即开即用)
  - PifuhdLocalBackend: PIFuHD 本地推理 (单张照片 → 3D mesh, 需 GPU + 模型)
  - Sam3dLocalBackend:  SAM 3D Body 本地推理 (需 HuggingFace 权限 + 模型)

通过 config['reconstruction']['backend'] 切换.
"""

from .base import BodyReconstructor, ReconstructionResult, create_reconstructor, list_available_backends

__all__ = [
    'BodyReconstructor',
    'ReconstructionResult',
    'create_reconstructor',
    'list_available_backends',
]
