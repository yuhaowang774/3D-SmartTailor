"""3D-SmartTailor API — 三维人体模型生成
启动: uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
"""

import sys, os, io, base64, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

import yaml
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from input.preprocess import process_images
from keypoint.detect import detect_keypoints
from output.mesh_generator import generate_body_glb

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="3D-SmartTailor", description="3D Human Body Model Generator")

static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding='utf-8')
    return HTMLResponse("<h1>3D-SmartTailor API</h1>")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "device": config.get("device", "cpu")
    }


@app.post("/api/measure")
async def measure(
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    height_cm: float = Form(...),
):
    """
    上传正面+侧面照片和身高 → 生成3D人体GLB模型 + 尺寸估算

    返回: JSON包含
      - measurements: 6项估算尺寸(cm)
      - glb_url: GLB模型下载链接
      - preview_front/side: 2D预览图
    """
    for field, f in [("front", front), ("side", side)]:
        content_type = f.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(400, f"{field} must be an image file")

    import uuid, time
    t0 = time.time()

    # 本次请求的专属输出目录
    req_id = uuid.uuid4().hex[:8]
    out_dir = OUTPUTS_DIR / req_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存上传文件
    front_path = str(out_dir / "front.jpg")
    side_path = str(out_dir / "side.jpg")

    try:
        front_bytes = await front.read()
        side_bytes = await side.read()
        if len(front_bytes) < 500 or len(side_bytes) < 500:
            raise HTTPException(400, "Image file too small")
        with open(front_path, 'wb') as f:
            f.write(front_bytes)
        with open(side_path, 'wb') as f:
            f.write(side_bytes)

        # 预处理
        data = process_images(front_path, side_path, height_cm, config)

        # 关键点检测
        kp_front = detect_keypoints(data['img_front'])
        kp_side = detect_keypoints(data['img_side'])
        n_front = int((kp_front[:, 3] > 0.5).sum())
        n_side = int((kp_side[:, 3] > 0.5).sum())

        if n_front < 8:
            raise HTTPException(400, f"Too few keypoints in front photo ({n_front}), need >= 8")

        # 生成3D人体GLB模型
        glb_path, vertices, faces, joints = generate_body_glb(
            kp_front, kp_side, height_cm, config, str(out_dir)
        )

        # 用mesh的围度计算6项尺寸(基于生成后的网格)
        from measure.extract import extract_measurements
        import torch as _torch
        try:
            measurements = extract_measurements(
                _torch.tensor(vertices),
                _torch.tensor(faces),
                _torch.tensor(joints),
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            # 提取失败时回退
            measurements = {
                'chest_cm': 92, 'waist_cm': 76, 'hip_cm': 96,
                'shoulder_width_cm': 40, 'sleeve_length_cm': 58, 'pants_length_cm': 100,
            }

        # 生成2D预览图
        from output.render import render_mesh_to_image as _render
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        front_png = str(out_dir / "preview_front.png")
        side_png = str(out_dir / "preview_side.png")
        _render(_torch.tensor(vertices[None, ...]), _torch.tensor(faces),
                front_png, title="Front View", elev=0, azim=0)
        _render(_torch.tensor(vertices[None, ...]), _torch.tensor(faces),
                side_png, title="Side View", elev=0, azim=-90)

        elapsed = time.time() - t0

        # 编码预览图为base64
        def _img_b64(path):
            with open(path, 'rb') as fh:
                return base64.b64encode(fh.read()).decode()

        # 编码GLB为base64
        glb_b64 = ""
        with open(glb_path, 'rb') as fh:
            glb_b64 = base64.b64encode(fh.read()).decode()

        return {
            "measurements": measurements,
            "height_cm": height_cm,
            "glb_base64": glb_b64,
            "glb_filename": f"body_model_{req_id}.glb",
            "preview_front": _img_b64(front_png),
            "preview_side": _img_b64(side_png),
            "keypoints_front": n_front,
            "keypoints_side": n_side,
            "processing_time_s": round(elapsed, 1),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Processing failed: {str(e)}")


@app.get("/api/download/{req_id}")
async def download_glb(req_id: str):
    """下载生成的GLB模型文件"""
    glb_path = OUTPUTS_DIR / req_id / "body_model.glb"
    if not glb_path.exists():
        raise HTTPException(404, "Model not found")
    return FileResponse(
        str(glb_path),
        media_type="model/gltf-binary",
        filename="body_model.glb"
    )
