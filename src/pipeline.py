"""
3D-SmartTailor — SAM 3D Body 单照片量体管线

用法:
    python src/pipeline.py --image <path> --height 170 --out-dir outputs/subject_001

流程:
    1. 加载单张照片
    2. 调用 SAM 3D Body (本地或云端) 重建 3D mesh
    3. 从 mesh 提取 6 项人体尺寸
    4. 导出 GLB + 预览图

配置:
    config.yaml 中 reconstruction.backend:
      - 'sam3d_local': 本地部署 (推荐, 需 HuggingFace 权限 + 模型下载)
      - 'sam3d_cloud': Playwright 调用官网 (临时验证, 不可商用)
"""

import argparse
import os
import sys
import time
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from reconstruction import create_reconstructor
from measure.extract import extract_measurements
from output.render import render_mesh_to_image


def main():
    parser = argparse.ArgumentParser(description='3D-SmartTailor — SAM 3D Body 量体管线')
    parser.add_argument('--image', required=True, help='单人照片路径')
    parser.add_argument('--height', type=float, default=None, help='身高(cm), 可选 (用于尺度归一化)')
    parser.add_argument('--out-dir', default='outputs/default', help='输出目录')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("3D-SmartTailor — SAM 3D Body 单照片量体管线")
    print("=" * 60)

    # ====== [1/4] 加载图像 ======
    t0 = time.time()
    print(f"[1/4] 加载图像: {args.image}")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"  ✗ 无法读取图像: {args.image}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"  ✓ 图像尺寸: {w}x{h}")
    t1 = time.time()

    # ====== [2/4] SAM 3D Body 重建 ======
    print(f"[2/4] SAM 3D Body 3D 重建")
    backend = config.get('reconstruction', {}).get('backend', 'sam3d_local')
    print(f"  后端: {backend}")
    reconstructor = create_reconstructor(config)

    if not reconstructor.is_available():
        print(f"  ✗ 后端不可用: {backend}")
        if backend == 'sam3d_local':
            print("  请按以下步骤配置:")
            print("  1. 申请 HuggingFace 权限: https://huggingface.co/facebook/sam-3d-body-dinov3")
            print("  2. 克隆仓库: git clone https://github.com/facebookresearch/sam-3d-body external/sam-3d-body")
            print("  3. 下载模型: hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3")
            print("  4. 安装依赖: pip install -r external/sam-3d-body/requirements.txt")
        elif backend == 'sam3d_cloud':
            print("  请安装 Playwright: pip install playwright && playwright install chromium")
        sys.exit(1)

    try:
        result = reconstructor.reconstruct(img_rgb)
        t2 = time.time()
        print(f"  ✓ 完成 ({t2-t1:.1f}s)")
        print(f"  顶点数: {result.metadata.get('n_vertices', '-')}")
        print(f"  面片数: {result.metadata.get('n_faces', '-')}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ 重建失败: {e}")
        sys.exit(1)

    # ====== [3/4] 尺寸提取 ======
    print("[3/4] 提取人体尺寸")
    try:
        measurements = extract_measurements(
            vertices=result.vertices,
            faces=result.faces,
            joints=result.joints,
            joint_names=result.joint_names,
            height_cm=args.height,
        )
        t3 = time.time()
        print(f"  ✓ 完成 ({t3-t2:.1f}s)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ✗ 测量失败: {e}")
        sys.exit(1)

    # ====== [4/4] 输出 ======
    print("[4/4] 保存结果")
    # GLB 文件
    glb_path = os.path.join(args.out_dir, 'body_model.glb')
    if result.glb_bytes:
        with open(glb_path, 'wb') as f:
            f.write(result.glb_bytes)
    else:
        # 从 vertices/faces 生成 GLB
        import trimesh
        mesh = trimesh.Trimesh(vertices=result.vertices, faces=result.faces, process=False)
        mesh.export(glb_path, file_type='glb')

    # 预览图
    try:
        render_mesh_to_image(
            result.vertices, result.faces,
            os.path.join(args.out_dir, 'mesh_preview.png'),
            title="SAM 3D Body Reconstruction",
        )
    except Exception as e:
        print(f"  预览图生成失败 (非致命): {e}")

    # 保存测量结果
    from output.export import save_measurements_json
    save_measurements_json(measurements, args.height, os.path.join(args.out_dir, 'measurements.json'))

    t4 = time.time()
    print(f"  ✓ 完成 ({t4-t3:.1f}s)")

    # ====== 汇总 ======
    print("\n" + "=" * 60)
    print("测量结果:")
    for key, val in measurements.items():
        print(f"  {key:.<25} {val} cm")
    print(f"\n总耗时: {t4-t0:.1f}s")
    print(f"输出目录: {args.out_dir}")
    print(f"  - measurements.json")
    print(f"  - body_model.glb  ({os.path.getsize(glb_path)/1024:.0f} KB)")
    print(f"  - mesh_preview.png")


if __name__ == '__main__':
    main()
