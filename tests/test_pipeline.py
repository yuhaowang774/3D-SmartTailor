"""
End-to-end test: requires GPU + SMPL-X model + test photos.

If environment not ready, test auto-skips.
"""

import pytest
import os
import json
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_photos')


def _has_test_data():
    if not os.path.isdir(DATA_DIR):
        return False
    for entry in os.listdir(DATA_DIR):
        subj_dir = os.path.join(DATA_DIR, entry)
        if os.path.isdir(subj_dir):
            front = os.path.join(subj_dir, 'front.jpg')
            side = os.path.join(subj_dir, 'side.jpg')
            gt = os.path.join(subj_dir, 'ground_truth.json')
            if os.path.isfile(front) and os.path.isfile(side) and os.path.isfile(gt):
                return True
    return False


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _has_smplx_model():
    import yaml
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    if not os.path.exists(config_path):
        return False
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return os.path.exists(cfg['model']['smplx_path'])


@pytest.mark.skipif(not _has_test_data(), reason="No test photos found")
@pytest.mark.skipif(not _has_gpu(), reason="No GPU available")
@pytest.mark.skipif(not _has_smplx_model(), reason="SMPL-X model not downloaded")
def test_pipeline_end_to_end():
    """Run pipeline on each test subject and validate output."""
    n_tested = 0
    for entry in os.listdir(DATA_DIR):
        subj_dir = os.path.join(DATA_DIR, entry)
        if not os.path.isdir(subj_dir):
            continue

        front = os.path.join(subj_dir, 'front.jpg')
        side = os.path.join(subj_dir, 'side.jpg')
        gt_file = os.path.join(subj_dir, 'ground_truth.json')

        if not (os.path.isfile(front) and os.path.isfile(side) and os.path.isfile(gt_file)):
            continue

        with open(gt_file) as f:
            gt = json.load(f)

        out_dir = os.path.join(PROJECT_ROOT, 'outputs', f'test_{entry}')
        cmd = [
            sys.executable, os.path.join(PROJECT_ROOT, 'src', 'pipeline.py'),
            '--front', front, '--side', side,
            '--height', str(gt['height']),
            '--out-dir', out_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        assert result.returncode == 0, f"Pipeline failed for {entry}: {result.stderr}"

        mj = os.path.join(out_dir, 'measurements.json')
        assert os.path.isfile(mj), f"measurements.json not generated for {entry}"

        with open(mj) as f:
            output = json.load(f)

        assert 'measurements' in output
        m = output['measurements']
        for key in ['chest_cm', 'waist_cm', 'hip_cm', 'shoulder_width_cm',
                     'sleeve_length_cm', 'pants_length_cm']:
            assert key in m, f"Missing measurement: {key}"
            assert 10 < m[key] < 300, f"{key} = {m[key]} out of range"

        assert os.path.isfile(os.path.join(out_dir, 'mesh_front.png'))
        assert os.path.isfile(os.path.join(out_dir, 'mesh_side.png'))

        # Accuracy check (if ground truth available)
        if 'measurements' in gt:
            for key in ['chest', 'waist', 'hip']:
                pred_key = f'{key}_cm'
                if pred_key in m and key in gt['measurements']:
                    error = abs(m[pred_key] - gt['measurements'][key])
                    print(f"  {entry}/{key}: pred={m[pred_key]:.1f}, gt={gt['measurements'][key]:.1f}, err={error:.1f}cm")

        n_tested += 1

    assert n_tested > 0, "No valid test subjects found"
    print(f"\nTested {n_tested} subject(s)")
