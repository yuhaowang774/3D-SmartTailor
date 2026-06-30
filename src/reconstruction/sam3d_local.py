"""SAM 3D Body 本地部署后端 (路线B)

依赖:
  - sam-3d-body 仓库 (external/sam-3d-body)
  - HuggingFace 模型权限 + 下载的 checkpoints/sam-3d-body-dinov3/

调用方式 (参考官方 demo.py):
  estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
  outputs = estimator.process_one_image(img_rgb)
  # outputs 含: vertices, faces, joints, ...

首次使用前:
  1. 申请 HuggingFace 权限: https://huggingface.co/facebook/sam-3d-body-dinov3
  2. 下载模型: hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
  3. 克隆仓库: git clone https://github.com/facebookresearch/sam-3d-body external/sam-3d-body
  4. 安装依赖: pip install -r external/sam-3d-body/requirements.txt
"""

import os
import sys
import time
import numpy as np

from .base import BodyReconstructor, ReconstructionResult


_SAM3D_REPO_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'sam-3d-body')
_SAM3D_REPO_PATH = os.path.abspath(_SAM3D_REPO_PATH)


class Sam3dLocalBackend(BodyReconstructor):
    """SAM 3D Body 本地推理后端"""

    backend_name = "sam3d_local"

    def __init__(self, config: dict):
        self.config = config
        self._estimator = None
        self._faces = None

        rec_cfg = config.get('reconstruction', {})
        self.checkpoint_path = rec_cfg.get(
            'checkpoint_path',
            os.path.join('checkpoints', 'sam-3d-body-dinov3', 'model.ckpt')
        )
        self.mhr_path = rec_cfg.get(
            'mhr_path',
            os.path.join('checkpoints', 'sam-3d-body-dinov3', 'assets', 'mhr_model.pt')
        )
        self.device = rec_cfg.get('device', 'cuda')

    def _ensure_repo_in_path(self):
        """把 sam-3d-body 仓库加入 sys.path"""
        if not os.path.isdir(_SAM3D_REPO_PATH):
            raise RuntimeError(
                f"SAM 3D Body 仓库未找到: {_SAM3D_REPO_PATH}\n"
                f"请执行: git clone https://github.com/facebookresearch/sam-3d-body {_SAM3D_REPO_PATH}"
            )
        if _SAM3D_REPO_PATH not in sys.path:
            sys.path.insert(0, _SAM3D_REPO_PATH)

    def _load_estimator(self):
        """加载 SAM 3D Body 模型 (懒加载, 首次调用时)"""
        if self._estimator is not None:
            return

        self._ensure_repo_in_path()

        if not os.path.exists(self.checkpoint_path):
            raise RuntimeError(
                f"SAM 3D Body 模型未找到: {self.checkpoint_path}\n"
                f"请先申请 HuggingFace 权限并下载模型:\n"
                f"  1. 访问 https://huggingface.co/facebook/sam-3d-body-dinov3 申请访问\n"
                f"  2. hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3"
            )

        try:
            from notebook.utils import setup_sam_3d_body
        except ImportError as e:
            raise RuntimeError(
                f"无法导入 SAM 3D Body, 请先安装依赖:\n"
                f"  pip install -r {_SAM3D_REPO_PATH}/requirements.txt\n"
                f"原始错误: {e}"
            )

        print(f"[Sam3dLocal] 加载模型: {self.checkpoint_path}")
        self._estimator = setup_sam_3d_body(
            checkpoint_path=self.checkpoint_path,
            mhr_path=self.mhr_path,
            device=self.device,
        )
        # 缓存 faces (拓扑)
        self._faces = self._estimator.faces
        print(f"[Sam3dLocal] 模型加载完成")

    def is_available(self) -> bool:
        """检查本地模型是否可用"""
        return (
            os.path.isdir(_SAM3D_REPO_PATH)
            and os.path.exists(self.checkpoint_path)
        )

    def reconstruct(self, img_rgb: np.ndarray) -> ReconstructionResult:
        """
        从单张 RGB 图像重建 3D 人体 mesh.

        Args:
            img_rgb: (H, W, 3) uint8 RGB

        Returns:
            ReconstructionResult
        """
        t0 = time.time()
        self._load_estimator()

        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        # SAM 3D Body 期望 RGB 输入
        outputs = self._estimator.process_one_image(img_rgb)
        t1 = time.time()

        # outputs 结构 (参考官方 demo):
        #   - vertices: (V, 3) 顶点
        #   - joints:   (J, 3) 关节 (MHR rig)
        #   - img_with_overlay: 可视化图
        # 实际字段需根据 sam_3d_body/notebook/utils.py 确认
        vertices = np.asarray(outputs.get('vertices', outputs.get('mesh_vertices', [])))
        if len(vertices) == 0:
            raise RuntimeError("SAM 3D Body 未检测到人体或返回空 mesh")

        faces = self._faces
        if faces is None:
            faces = np.asarray(outputs.get('faces', []))

        joints = None
        joint_names = None
        if 'joints' in outputs:
            joints = np.asarray(outputs['joints'])
            joint_names = outputs.get('joint_names', None)

        img_overlay = outputs.get('img_with_overlay', None)
        if img_overlay is not None:
            img_overlay = np.asarray(img_overlay, dtype=np.uint8)

        # 生成 GLB 二进制 (用 trimesh)
        glb_bytes = self._mesh_to_glb(vertices, faces)

        return ReconstructionResult(
            vertices=vertices.astype(np.float32),
            faces=faces.astype(np.int32),
            joints=joints.astype(np.float32) if joints is not None else None,
            joint_names=joint_names,
            glb_bytes=glb_bytes,
            img_with_overlay=img_overlay,
            metadata={
                'backend': self.backend_name,
                'inference_time_s': round(t1 - t0, 2),
                'n_vertices': int(len(vertices)),
                'n_faces': int(len(faces)),
            },
        )

    def _mesh_to_glb(self, vertices: np.ndarray, faces: np.ndarray) -> bytes:
        """网格 → GLB 二进制"""
        import trimesh
        import io

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # 顶点颜色 (皮肤色)
        rgba = np.zeros((len(vertices), 4), dtype=np.uint8)
        rgba[:, :3] = [220, 180, 160]
        rgba[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)

        buf = io.BytesIO()
        mesh.export(buf, file_type='glb')
        return buf.getvalue()
