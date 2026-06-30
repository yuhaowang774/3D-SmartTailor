"""3D-SmartTailor REST API 服务层
启动: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import sys, os, io, base64, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import yaml
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from input.preprocess import process_images
from keypoint.detect import detect_keypoints
from fitting.smplify import fit_smplx
from measure.extract import extract_measurements
from output.render import render_mesh_to_image
from apply_calibration import calibrate_measurements

# 加载配置（服务启动时一次性加载）
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="3D-SmartTailor", description="AI Body Measurement API")

# 静态文件(前端页面)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    """返回前端页面"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding='utf-8')
    return HTMLResponse("<h1>3D-SmartTailor API</h1><p>Frontend not found. Try POST /api/measure</p>")


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
    接收正面+侧面照片和身高，返回 6 项人体尺寸。

    请求: multipart/form-data
      - front: 正面全身照 (jpg/png)
      - side:  侧面全身照 (jpg/png)
      - height_cm: 身高(cm), 浮点数

    返回: JSON
      {
        "measurements": {chest_cm, waist_cm, hip_cm, shoulder_width_cm, sleeve_length_cm, pants_length_cm},
        "height_cm": 170.0,
        "mesh_image_front": "base64...",
        "mesh_image_side": "base64...",
        "processing_time_s": 12.3
      }
    """
    # 校验文件类型
    for field, f in [("front", front), ("side", side)]:
        content_type = f.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(400, f"{field} must be an image file")

    # 保存上传文件到临时目录
    tmp_dir = tempfile.mkdtemp(prefix="tailor_")
    front_path = os.path.join(tmp_dir, "front.jpg")
    side_path = os.path.join(tmp_dir, "side.jpg")

    try:
        front_bytes = await front.read()
        side_bytes = await side.read()

        if len(front_bytes) < 500 or len(side_bytes) < 500:
            raise HTTPException(400, "Image file too small")

        with open(front_path, 'wb') as f:
            f.write(front_bytes)
        with open(side_path, 'wb') as f:
            f.write(side_bytes)

        import time
        t0 = time.time()

        # 模块1: 预处理
        data = process_images(front_path, side_path, height_cm, config)

        # 模块2: 关键点
        kp_front = detect_keypoints(data['img_front'])
        kp_side = detect_keypoints(data['img_side'])

        n_front = int((kp_front[:, 3] > 0.5).sum())
        n_side = int((kp_side[:, 3] > 0.5).sum())

        if n_front < 8:
            raise HTTPException(400, f"Too few keypoints in front photo (detected {n_front}, need >= 8)")
        if n_side < 5:
            kp_side = kp_front.copy()

        # 模块3: SMPL拟合
        smpl_data = fit_smplx(kp_front, kp_side, height_cm, config)

        # 模块4: 测量提取
        raw_measurements = extract_measurements(
            smpl_data['vertices'].to(torch.device('cpu')),
            smpl_data['faces'],
            smpl_data['joints']
        )

        # 校准
        try:
            measurements = calibrate_measurements(raw_measurements, smpl_data['betas'])
        except Exception:
            measurements = raw_measurements

        # 模块5: 可视化渲染
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        front_buf = io.BytesIO()
        front_png = os.path.join(tmp_dir, "front_mesh.png")
        render_mesh_to_image(
            smpl_data['vertices'], smpl_data['faces'],
            front_png, title="3D Body Model - Front", elev=0, azim=0
        )
        if os.path.exists(front_png):
            with open(front_png, 'rb') as f:
                front_buf.write(f.read())

        side_buf = io.BytesIO()
        side_png = os.path.join(tmp_dir, "side_mesh.png")
        render_mesh_to_image(
            smpl_data['vertices'], smpl_data['faces'],
            side_png, title="3D Body Model - Side", elev=0, azim=-90
        )
        if os.path.exists(side_png):
            with open(side_png, 'rb') as f:
                side_buf.write(f.read())

        elapsed = time.time() - t0

        return {
            "measurements": measurements,
            "height_cm": height_cm,
            "mesh_image_front": base64.b64encode(front_buf.getvalue()).decode() if front_buf.getvalue() else "",
            "mesh_image_side": base64.b64encode(side_buf.getvalue()).decode() if side_buf.getvalue() else "",
            "processing_time_s": round(elapsed, 1),
            "keypoints_detected": f"front={n_front}, side={n_side}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
