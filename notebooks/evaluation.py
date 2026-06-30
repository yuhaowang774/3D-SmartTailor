# Cell 1: Environment setup
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'test_photos')

MEASUREMENT_KEYS = ['chest_cm', 'waist_cm', 'hip_cm', 'shoulder_width_cm', 'sleeve_length_cm', 'pants_length_cm']

# Cell 2: Collect all results
results = []
for subj in os.listdir(DATA_DIR):
    subj_dir = os.path.join(DATA_DIR, subj)
    out_dir = os.path.join(OUTPUTS_DIR, f'test_{subj}')
    gt_path = os.path.join(subj_dir, 'ground_truth.json')
    pred_path = os.path.join(out_dir, 'measurements.json')

    if os.path.isfile(gt_path) and os.path.isfile(pred_path):
        with open(gt_path) as f:
            gt = json.load(f)
        with open(pred_path) as f:
            pred = json.load(f)
        results.append({'subject': subj, 'gt': gt, 'pred': pred})

print(f"Found {len(results)} results")

# Cell 3: Compute MAE per measurement
errors = defaultdict(list)

for r in results:
    gt_m = r['gt'].get('measurements', {})
    pred_m = r['pred'].get('measurements', {})
    for key, cm_key in [('chest_cm', 'chest'), ('waist_cm', 'waist'), ('hip_cm', 'hip'),
                          ('shoulder_width_cm', 'shoulder_width'),
                          ('sleeve_length_cm', 'sleeve_length'),
                          ('pants_length_cm', 'pants_length')]:
        if key in pred_m and cm_key in gt_m:
            errors[key].append(abs(pred_m[key] - gt_m[cm_key]))

print("\nPrecision Report:")
print(f"{'Measurement':<22} {'MAE(cm)':<10} {'Samples':<10} {'Status':<10}")
print("-" * 52)
all_pass = True
for key in MEASUREMENT_KEYS:
    if errors[key]:
        mae = np.mean(errors[key])
        n = len(errors[key])
        status = 'PASS' if mae <= 2.0 else 'FAIL'
        if mae > 2.0:
            all_pass = False
        print(f"{key:<22} {mae:<10.2f} {n:<10} {status:<10}")
    else:
        print(f"{key:<22} {'no data':<10}")

if all_pass and len(results) >= 3:
    print(f"\nAll 6 measurements meet Phase 1 target (MAE <= 2.0cm)")
else:
    print(f"\nSome measurements did not meet target")

# Cell 4: Error visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, key in enumerate(MEASUREMENT_KEYS):
    ax = axes[i]
    if errors[key]:
        ax.bar(range(len(errors[key])), errors[key])
        ax.axhline(y=2.0, color='r', linestyle='--', label='2cm target')
        ax.set_title(key)
        ax.set_ylabel('Absolute Error (cm)')
        ax.set_xlabel('Subject')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
plt.tight_layout()
os.makedirs(OUTPUTS_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUTS_DIR, 'error_analysis.png'), dpi=150)
plt.show()
