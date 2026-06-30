"""
3D-SmartTailor Phase 1 Pipeline

用法:
    python src/pipeline.py --front <path> --side <path> --height 170 --out-dir outputs/subject_001
"""

import argparse
import os
import sys
import time
import yaml
import torch

from input.preprocess import process_images
from keypoint.detect import detect_keypoints
from fitting.smplify import fit_smplx
from measure.extract import extract_measurements
from output.export import save_measurements_json
from output.render import render_front_and_side


def main():
    parser = argparse.ArgumentParser(description='3D-SmartTailor Phase 1 Pipeline')
    parser.add_argument('--front', required=True, help='Path to front photo')
    parser.add_argument('--side', required=True, help='Path to side photo')
    parser.add_argument('--height', type=float, required=True, help='Height in cm')
    parser.add_argument('--out-dir', default='outputs/default', help='Output directory')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 50)
    print("3D-SmartTailor Phase 1: Body Measurement Pipeline")
    print("=" * 50)

    t0 = time.time()
    print("[1/5] Loading images...")
    data = process_images(args.front, args.side, args.height, config)
    t1 = time.time()
    print(f"  Done ({t1 - t0:.1f}s)")

    print("[2/5] MediaPipe keypoint detection...")
    kp_front = detect_keypoints(data['img_front'])
    kp_side = detect_keypoints(data['img_side'])
    n_front = (kp_front[:, 3] > 0.5).sum()
    n_side = (kp_side[:, 3] > 0.5).sum()
    print(f"  Front: {n_front}/33 valid, Side: {n_side}/33 valid")
    t2 = time.time()
    print(f"  Done ({t2 - t1:.1f}s)")

    print("[3/5] SMPLify-X body fitting (GPU)...")
    try:
        smpl_data = fit_smplx(kp_front, kp_side, args.height, config)
        t3 = time.time()
        print(f"  Done ({t3 - t2:.1f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Check: SMPL-X model file exists at path in config.yaml")
        sys.exit(1)

    print("[4/5] Extracting body measurements...")
    measurements = extract_measurements(
        smpl_data['vertices'].to(torch.device('cpu')),
        smpl_data['faces'],
        smpl_data['joints'],
    )
    t4 = time.time()
    print(f"  Done ({t4 - t3:.1f}s)")

    print("[5/5] Saving results...")
    save_measurements_json(measurements, args.height,
                           os.path.join(args.out_dir, 'measurements.json'))
    if config['output']['render_mesh']:
        render_front_and_side(smpl_data['vertices'], smpl_data['faces'], args.out_dir)
    t5 = time.time()
    print(f"  Done ({t5 - t4:.1f}s)")

    print("\n" + "=" * 50)
    print("Measurement Results:")
    for key, val in measurements.items():
        print(f"  {key}: {val} cm")
    print(f"\nTotal time: {t5 - t0:.1f}s")
    print(f"Output: {args.out_dir}")


if __name__ == '__main__':
    main()
