"""3D 人体重建模块

提供统一接口 BodyReconstructor, 支持两种后端:
  - Sam3dLocalBackend:  本地部署 SAM 3D Body 模型 (路线B, 推荐)
  - Sam3dCloudBackend:  通过 Playwright 自动化调用 sam3d.org (路线A, 临时)

通过 config['reconstruction']['backend'] 切换.
"""

from .base import BodyReconstructor, ReconstructionResult, create_reconstructor

__all__ = ['BodyReconstructor', 'ReconstructionResult', 'create_reconstructor']
