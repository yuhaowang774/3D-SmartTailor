"""PIFuHD 本地部署后端

PIFuHD (CVPR 2020) 从单张 RGB 图像重建高分辨率 3D 人体 mesh.
仓库: https://github.com/facebookresearch/pifuhd (Public archive, 可用)

依赖:
  - pifuhd 仓库 (external/pifuhd)
  - 预训练模型 (checkpoints/pifuhd/net_G)
  - PyTorch + GPU (推荐 8GB+ 显存)

首次使用前:
  1. 克隆仓库: git clone https://github.com/facebookresearch/pifuhd external/pifuhd
  2. 下载模型: cd external/pifuhd && sh scripts/download_trained_model.sh
     (模型保存到 external/pifuhd/checkpoints/pifuhd/net_G)
  3. 安装依赖: pip install -r external/pifuhd/requirements.txt

调用流程 (子进程方式, 避免直接导入 PIFuHD 的复杂依赖):
  1. 保存输入图片到临时目录
  2. 生成 {name}_rect.txt 裁剪框 (用 OpenCV 人体轮廓检测, 无需 OpenPose)
  3. 调用 python -m apps.simple_test --use_rect
  4. 读取输出 obj 文件, 用 trimesh 加载
"""

import os
import sys
import time
import shutil
import subprocess
import tempfile
import numpy as np

from .base import BodyReconstructor, ReconstructionResult


_PIFUHD_REPO_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'external', 'pifuhd')
_PIFUHD_REPO_PATH = os.path.abspath(_PIFUHD_REPO_PATH)

# 默认模型路径 (scripts/download_trained_model.sh 下载到 checkpoints/pifuhd.pt)
_DEFAULT_CKPT = os.path.join(_PIFUHD_REPO_PATH, 'checkpoints', 'pifuhd.pt')


class PifuhdLocalBackend(BodyReconstructor):
    """PIFuHD 本地推理后端

    通过子进程调用 PIFuHD 的 simple_test.py, 避免直接导入其依赖
    (PIFuHD 依赖旧版 PyTorch API, 直接导入可能与主项目冲突)
    """

    backend_name = "pifuhd_local"

    def __init__(self, config: dict):
        self.config = config
        rec_cfg = config.get('reconstruction', {})
        pifuhd_cfg = rec_cfg.get('pifuhd_local', {})

        self.ckpt_path = pifuhd_cfg.get('checkpoint_path', _DEFAULT_CKPT)
        # 转为绝对路径 (子进程工作目录是 external/pifuhd, 相对路径会解析错误)
        if self.ckpt_path and not os.path.isabs(self.ckpt_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            self.ckpt_path = os.path.join(project_root, self.ckpt_path)
        self.device = pifuhd_cfg.get('device', config.get('device', 'cuda'))
        # PIFuHD 输出分辨率 (默认 512, 越高越精细但越慢)
        self.resolution = pifuhd_cfg.get('resolution', 512)

    def is_available(self) -> bool:
        """检查 PIFuHD 后端是否可用"""
        return (
            os.path.isdir(_PIFUHD_REPO_PATH)
            and os.path.exists(self.ckpt_path)
            and os.path.exists(os.path.join(_PIFUHD_REPO_PATH, 'apps', 'simple_test.py'))
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

        if img_rgb.dtype != np.uint8:
            img_rgb = (img_rgb * 255).astype(np.uint8)

        # 检查环境
        if not os.path.isdir(_PIFUHD_REPO_PATH):
            raise RuntimeError(
                f"PIFuHD 仓库未找到: {_PIFUHD_REPO_PATH}\n"
                f"请执行: git clone https://github.com/facebookresearch/pifuhd {_PIFUHD_REPO_PATH}"
            )
        if not os.path.exists(self.ckpt_path):
            raise RuntimeError(
                f"PIFuHD 模型未找到: {self.ckpt_path}\n"
                f"请执行: cd {_PIFUHD_REPO_PATH} && sh scripts/download_trained_model.sh"
            )

        # 创建临时工作目录
        work_dir = tempfile.mkdtemp(prefix='pifuhd_')
        try:
            # 1. 保存输入图片 (PIFuHD 用文件名作为输出 obj 名)
            img_path = os.path.join(work_dir, 'input.png')
            import cv2
            # PIFuHD 期望 RGB, cv2 用 BGR, 需转换
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img_bgr)

            # 2. 生成裁剪框 rect.txt (无需 OpenPose, 用人体轮廓检测)
            rect_path = os.path.join(work_dir, 'input_rect.txt')
            self._write_crop_rect(img_rgb, rect_path)

            # 3. 调用 PIFuHD simple_test (子进程)
            obj_path = self._run_pifuhd(work_dir, img_path)

            # 4. 加载输出的 obj
            import trimesh
            mesh = trimesh.load(obj_path, force='mesh', process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                raise RuntimeError(f"PIFuHD 输出无法解析: {obj_path}")

            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.int32)

            t1 = time.time()

            # 保存 mesh 到持久路径 (供调试/前端下载)
            debug_dir = os.path.join(_PIFUHD_REPO_PATH, '..', '..', 'data', 'body_models')
            os.makedirs(debug_dir, exist_ok=True)
            debug_obj = os.path.join(debug_dir, 'pifuhd_output.obj')
            try:
                mesh.export(debug_obj)
                print(f"[PifuhdLocal] mesh 已保存到 {debug_obj}")
            except Exception as e:
                print(f"[PifuhdLocal] 保存 mesh 失败: {e}")

            # PIFuHD 输出的 mesh 坐标系:
            #   X=左右, Y=上下(身高), Z=前后
            #   尺度约为米, 但可能需要根据身高归一化
            # 测量模块会用 height_cm 归一化, 这里保持原始输出

            # 生成 GLB (供前端下载)
            glb_bytes = self._mesh_to_glb(vertices, faces)

            return ReconstructionResult(
                vertices=vertices,
                faces=faces,
                joints=None,  # PIFuHD 不输出关节
                joint_names=None,
                glb_bytes=glb_bytes,
                img_with_overlay=None,
                metadata={
                    'backend': self.backend_name,
                    'inference_time_s': round(t1 - t0, 2),
                    'n_vertices': int(len(vertices)),
                    'n_faces': int(len(faces)),
                    'resolution': self.resolution,
                },
            )
        finally:
            # 清理临时目录 (保留用于调试, 可注释掉)
            shutil.rmtree(work_dir, ignore_errors=True)

    def _write_crop_rect(self, img_rgb: np.ndarray, rect_path: str):
        """生成人体裁剪框 (无需 OpenPose)

        策略: 用 OpenCV 边缘检测 + 最大轮廓估计人体边界框.
        若检测失败, 回退到图片中央 90% 区域.

        PIFuHD rect.txt 格式: "x1 y1 x2 y2" (像素坐标)
        """
        import cv2

        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = 0, 0, w, h

        try:
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            # 高斯模糊 + Canny 边缘
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            # 找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # 取最大轮廓的包围盒
                largest = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(largest)
                if cw * ch > (w * h * 0.1):  # 轮廓面积 > 10% 图片才算有效
                    x1, y1, x2, y2 = x, y, x + cw, y + ch
        except Exception:
            pass  # 回退到整张图

        # 确保坐标在图片范围内
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        with open(rect_path, 'w') as f:
            f.write(f"{x1} {y1} {x2} {y2}\n")

    def _run_pifuhd(self, work_dir: str, img_path: str) -> str:
        """调用 PIFuHD simple_test.py (子进程)

        Returns:
            输出 obj 文件路径
        """
        out_dir = os.path.join(work_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            sys.executable, '-m', 'apps.simple_test',
            '--input_path', work_dir,
            '--out_path', out_dir,
            '--ckpt_path', self.ckpt_path,
            '--use_rect',  # 用 rect.txt 而非 keypoints.json
            '--resolution', str(self.resolution),
        ]

        # GPU ID (从配置读取)
        gpu_id = self.config.get('gpu_id', 0)
        if self.device == 'cuda':
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        else:
            # CPU 模式 (非常慢)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ''

        print(f"[PifuhdLocal] 运行: {' '.join(cmd)}")
        print(f"[PifuhdLocal] 工作目录: {_PIFUHD_REPO_PATH}")

        result = subprocess.run(
            cmd,
            cwd=_PIFUHD_REPO_PATH,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )

        if result.returncode != 0:
            stderr_tail = result.stderr[-2000:] if result.stderr else ''
            stdout_tail = result.stdout[-2000:] if result.stdout else ''
            raise RuntimeError(
                f"PIFuHD 推理失败 (exit {result.returncode}):\n"
                f"stdout: {stdout_tail}\n"
                f"stderr: {stderr_tail}"
            )

        # PIFuHD 输出: {out_path}/result/input.obj
        obj_path = os.path.join(out_dir, 'result', 'input.obj')
        if not os.path.exists(obj_path):
            # 尝试查找任意 obj 文件
            obj_files = []
            for root, dirs, files in os.walk(out_dir):
                for fn in files:
                    if fn.endswith('.obj'):
                        obj_files.append(os.path.join(root, fn))
            if obj_files:
                obj_path = obj_files[0]
            else:
                raise RuntimeError(
                    f"PIFuHD 未生成 obj 文件, 输出目录: {out_dir}\n"
                    f"stdout: {result.stdout[-1000:]}"
                )

        print(f"[PifuhdLocal] 输出 obj: {obj_path}")
        return obj_path

    def _mesh_to_glb(self, vertices: np.ndarray, faces: np.ndarray) -> bytes:
        """网格 → GLB 二进制"""
        import trimesh
        import io

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rgba = np.zeros((len(vertices), 4), dtype=np.uint8)
        rgba[:, :3] = [220, 180, 160]  # 皮肤色
        rgba[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)

        buf = io.BytesIO()
        mesh.export(buf, file_type='glb')
        return buf.getvalue()
