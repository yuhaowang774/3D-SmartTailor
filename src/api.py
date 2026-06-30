"""3D-SmartTailor API — 3D 人体量体 (GLB 上传 / PIFuHD 图片重建)

启动: uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
"""

import sys, os, base64, time, uuid, io
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from reconstruction import create_reconstructor, list_available_backends
from reconstruction.glb_file import GlbFileBackend
from measure.extract import extract_measurements
from output.render import render_mesh_to_image

with open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'), 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="3D-SmartTailor", description="3D Human Body Measurement (GLB / PIFuHD / SAM 3D Body)")

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# 懒加载重建器缓存 (按 backend_name 缓存)
_reconstructors = {}


def get_reconstructor(backend_name: str = None):
    """获取指定后端的重建器 (懒加载 + 缓存)

    Args:
        backend_name: 后端名 (glb_file / pifuhd_local / sam3d_local).
                     None 则用 config 默认.
    """
    if backend_name is None:
        backend_name = config.get('reconstruction', {}).get('backend', 'glb_file')

    if backend_name not in _reconstructors:
        # 临时覆盖 config 的 backend 字段
        cfg = dict(config)
        cfg['reconstruction'] = dict(config.get('reconstruction', {}))
        cfg['reconstruction']['backend'] = backend_name
        _reconstructors[backend_name] = create_reconstructor(cfg)
    return _reconstructors[backend_name]


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding='utf-8')
    return HTMLResponse("<h1>3D-SmartTailor API</h1>")


@app.get("/api/health")
async def health():
    rec = get_reconstructor()
    return {
        "status": "ok",
        "backend": rec.backend_name,
        "backend_available": rec.is_available(),
    }


@app.get("/api/backends")
async def list_backends():
    """列出所有可用的重建后端 (供前端选择)"""
    return {"backends": list_available_backends(config)}


@app.post("/api/measure_glb")
async def measure_glb(
    glb: UploadFile = File(...),
    height_cm: float = Form(None),
):
    """
    上传 GLB 文件 (来自 sam3d.org) → 提取 6 项人体尺寸

    返回 JSON:
      - measurements: 6 项估算尺寸 (cm)
      - glb_base64: 原始 GLB (base64, 供前端 3D 预览)
      - preview: mesh 预览图 (base64 PNG)
      - n_vertices / n_faces: mesh 统计
      - processing_time_s: 处理耗时
    """
    filename = glb.filename or ""
    if not (filename.lower().endswith('.glb') or filename.lower().endswith('.gltf')):
        raise HTTPException(400, f"需要 .glb 或 .gltf 文件, 收到: {filename}")

    t0 = time.time()
    req_id = uuid.uuid4().hex[:8]
    out_dir = OUTPUTS_DIR / req_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        glb_bytes = await glb.read()
        if len(glb_bytes) < 500:
            raise HTTPException(400, "GLB file too small")

        glb_path = str(out_dir / "body_model.glb")
        with open(glb_path, 'wb') as f:
            f.write(glb_bytes)

        # GLB 上传接口始终用 GlbFileBackend
        reconstructor = GlbFileBackend(config)
        result = reconstructor.reconstruct_from_glb_path(glb_path)

        measurements = extract_measurements(
            vertices=result.vertices,
            faces=result.faces,
            joints=result.joints,
            joint_names=result.joint_names,
            height_cm=height_cm,
        )

        # 预览图
        preview_b64 = _render_preview(result, out_dir)

        elapsed = time.time() - t0

        return {
            "measurements": measurements,
            "height_cm": height_cm,
            "glb_base64": base64.b64encode(glb_bytes).decode(),
            "glb_filename": f"body_model_{req_id}.glb",
            "preview": preview_b64,
            "n_vertices": result.metadata.get('n_vertices', 0),
            "n_faces": result.metadata.get('n_faces', 0),
            "backend": result.metadata.get('backend', ''),
            "processing_time_s": round(elapsed, 2),
            "req_id": req_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.post("/api/measure_image")
async def measure_image(
    image: UploadFile = File(...),
    height_cm: float = Form(None),
    backend: str = Form("pifuhd_local"),
):
    """
    上传图片 → PIFuHD/SAM3D 本地重建 → 提取 6 项人体尺寸

    Args:
        image: 上传的图片文件 (jpg/png)
        height_cm: 用户身高 (可选, 用于尺度归一化)
        backend: 重建后端 (pifuhd_local / sam3d_local)

    返回 JSON: 同 /api/measure_glb
    """
    # 校验文件类型
    filename = image.filename or ""
    if not (filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))):
        raise HTTPException(400, f"需要图片文件 (jpg/png/bmp/webp), 收到: {filename}")

    t0 = time.time()
    req_id = uuid.uuid4().hex[:8]
    out_dir = OUTPUTS_DIR / req_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_bytes = await image.read()
        if len(img_bytes) < 100:
            raise HTTPException(400, "Image file too small")

        # 解码图片为 RGB numpy
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(400, "无法解码图片")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 保存原图 (供调试)
        cv2.imwrite(str(out_dir / "input.png"), img_bgr)

        # 获取重建后端
        reconstructor = get_reconstructor(backend)
        if not reconstructor.is_available():
            raise HTTPException(
                503,
                f"后端 '{backend}' 不可用. "
                f"PIFuHD 需要先克隆仓库并下载模型, 详见 README."
            )

        # 重建
        result = reconstructor.reconstruct(img_rgb)

        # 测量
        measurements = extract_measurements(
            vertices=result.vertices,
            faces=result.faces,
            joints=result.joints,
            joint_names=result.joint_names,
            height_cm=height_cm,
        )

        # 预览图
        preview_b64 = _render_preview(result, out_dir)

        elapsed = time.time() - t0

        # 保存生成的 GLB
        glb_b64 = ""
        if result.glb_bytes:
            glb_path = out_dir / "body_model.glb"
            with open(glb_path, 'wb') as f:
                f.write(result.glb_bytes)
            glb_b64 = base64.b64encode(result.glb_bytes).decode()

        return {
            "measurements": measurements,
            "height_cm": height_cm,
            "glb_base64": glb_b64,
            "glb_filename": f"body_model_{req_id}.glb",
            "preview": preview_b64,
            "input_image": base64.b64encode(img_bytes).decode(),
            "n_vertices": result.metadata.get('n_vertices', 0),
            "n_faces": result.metadata.get('n_faces', 0),
            "backend": result.metadata.get('backend', backend),
            "inference_time_s": result.metadata.get('inference_time_s', 0),
            "processing_time_s": round(elapsed, 2),
            "req_id": req_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")


def _render_preview(result, out_dir) -> str:
    """渲染 mesh 预览图, 返回 base64"""
    try:
        preview_path = str(out_dir / "preview.png")
        render_mesh_to_image(
            result.vertices, result.faces,
            preview_path,
            title=f"{result.metadata.get('backend', '?')} ({result.metadata.get('n_vertices', 0)} verts)",
        )
        with open(preview_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"[WARN] preview render failed: {e}")
        return ""


@app.get("/api/download/{req_id}")
async def download_glb(req_id: str):
    """下载已处理的 GLB 模型文件"""
    glb_path = OUTPUTS_DIR / req_id / "body_model.glb"
    if not glb_path.exists():
        raise HTTPException(404, "Model not found")
    return FileResponse(
        str(glb_path),
        media_type="model/gltf-binary",
        filename="body_model.glb"
    )
