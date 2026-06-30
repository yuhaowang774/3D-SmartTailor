"""3D-SmartTailor API — SAM 3D Body GLB 上传量体

启动: uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
"""

import sys, os, base64, time, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import yaml
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from reconstruction import create_reconstructor
from reconstruction.glb_file import GlbFileBackend
from measure.extract import extract_measurements
from output.render import render_mesh_to_image

with open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'), 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="3D-SmartTailor", description="3D Human Body Measurement (SAM 3D Body / GLB Upload)")

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# 懒加载重建器 (首次请求时初始化)
_reconstructor = None


def get_reconstructor():
    global _reconstructor
    if _reconstructor is None:
        _reconstructor = create_reconstructor(config)
    return _reconstructor


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


@app.post("/api/measure_glb")
async def measure_glb(
    glb: UploadFile = File(...),
    height_cm: float = Form(None),
):
    """
    上传 GLB 文件 (来自 sam3d.org) → 提取 6 项人体尺寸

    流程:
      1. 接收 GLB 文件
      2. 用 GlbFileBackend 解析为 mesh
      3. 用身高 (可选) 归一化尺度
      4. 从 mesh 截面提取围度 + 长度

    返回 JSON:
      - measurements: 6 项估算尺寸 (cm)
      - glb_base64: 原始 GLB (base64, 供前端 3D 预览)
      - preview: mesh 预览图 (base64 PNG)
      - n_vertices / n_faces: mesh 统计
      - processing_time_s: 处理耗时
    """
    # 校验文件类型
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

        # 保存 GLB
        glb_path = str(out_dir / "body_model.glb")
        with open(glb_path, 'wb') as f:
            f.write(glb_bytes)

        # 用 GlbFileBackend 解析
        reconstructor = get_reconstructor()
        if not isinstance(reconstructor, GlbFileBackend):
            # 配置可能指向 sam3d_local, 但 GLB 上传接口始终用 GlbFileBackend
            reconstructor = GlbFileBackend(config)

        if not reconstructor.is_available():
            raise HTTPException(503, f"Backend not available: {reconstructor.backend_name}")

        result = reconstructor.reconstruct_from_glb_path(glb_path)

        # 尺寸提取
        measurements = extract_measurements(
            vertices=result.vertices,
            faces=result.faces,
            joints=result.joints,
            joint_names=result.joint_names,
            height_cm=height_cm,
        )

        # 预览图
        preview_path = str(out_dir / "preview.png")
        try:
            render_mesh_to_image(
                result.vertices, result.faces,
                preview_path,
                title=f"GLB Analysis ({result.metadata.get('n_vertices', 0)} verts)",
            )
            with open(preview_path, 'rb') as f:
                preview_b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            print(f"[WARN] preview render failed: {e}")
            preview_b64 = ""

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
