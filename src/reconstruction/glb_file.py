"""GLB 文件上传后端

用户通过 sam3d.org 网页手动生成 GLB 后上传, 此后端只负责:
  1. 解析 GLB → trimesh.Trimesh (vertices + faces)
  2. 尺度归一化 (用用户填写的身高)
  3. 包装为 ReconstructionResult 返回

不依赖任何外部模型/网络, 即开即用.
"""

import os
import time
import numpy as np
import trimesh

from .base import BodyReconstructor, ReconstructionResult


class GlbFileBackend(BodyReconstructor):
    """GLB 文件上传后端"""

    backend_name = "glb_file"

    def __init__(self, config: dict):
        self.config = config

    def is_available(self) -> bool:
        """GLB 后端始终可用 (只需 trimesh)"""
        return True

    def reconstruct_from_glb_path(self, glb_path: str) -> ReconstructionResult:
        """从 GLB 文件路径加载并解析

        Args:
            glb_path: GLB 文件路径

        Returns:
            ReconstructionResult
        """
        t0 = time.time()

        if not os.path.exists(glb_path):
            raise FileNotFoundError(f"GLB 文件不存在: {glb_path}")

        # 用 trimesh 加载 (force='mesh' 避免返回 Scene)
        mesh = trimesh.load(glb_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            # 如果是 Scene, 合并所有子 mesh
            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    raise RuntimeError("GLB 中未找到有效 mesh")
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise RuntimeError(f"无法解析的 GLB 类型: {type(mesh)}")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)

        # 读取原始 GLB 二进制 (供前端下载)
        with open(glb_path, 'rb') as f:
            glb_bytes = f.read()

        t1 = time.time()

        return ReconstructionResult(
            vertices=vertices,
            faces=faces,
            joints=None,
            joint_names=None,
            glb_bytes=glb_bytes,
            img_with_overlay=None,
            metadata={
                'backend': self.backend_name,
                'inference_time_s': round(t1 - t0, 3),
                'n_vertices': int(len(vertices)),
                'n_faces': int(len(faces)),
                'source_path': glb_path,
            },
        )

    def reconstruct(self, img_rgb: np.ndarray) -> ReconstructionResult:
        """GLB 后端不支持从图像重建 (需要 GLB 文件上传)"""
        raise NotImplementedError(
            "GlbFileBackend 不支持从图像重建, 请用 reconstruct_from_glb_path() 或 "
            "通过 /api/measure_glb 接口上传 GLB 文件"
        )
