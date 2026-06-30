"""模块4验证: 多体型A-pose测量单调性 + 合理性"""
import sys; sys.path.insert(0, 'src')
import torch, numpy as np
import smplx
from measure.extract import extract_measurements

device = 'cpu'
model = smplx.create('data/body_models/smplx/models', model_type='smplx',
                      gender='neutral', num_betas=10, batch_size=1)

# A-pose: 手臂自然下垂（-30度绕Z轴）
a_pose = torch.zeros(1, 63)
a_pose[0, 0:3] = torch.tensor([0.0, 0.0, 0.0])       # left_arm rotation
a_pose[0, 3:6] = torch.tensor([0.0, 0.0, -0.5])       # right_arm
# 使用更接近人体比例的beta系数
betas_test = {
    'slim(-2)':   [-2.0,  0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'normal(0)':  [0.0,   0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'plump(+2)':  [2.0,  -0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

print("=" * 70)
print(f"{'体型':<15} {'胸围':<8} {'腰围':<8} {'臀围':<8} {'肩宽':<8} {'袖长':<8} {'裤长':<8}")
print("-" * 70)

results = {}
for name, betas_arr in betas_test.items():
    betas = torch.tensor([betas_arr], dtype=torch.float32)
    output = model(betas=betas, body_pose=a_pose,
                   global_orient=torch.zeros(1,3),
                   jaw_pose=torch.zeros(1,3),
                   left_hand_pose=torch.zeros(1,6),
                   right_hand_pose=torch.zeros(1,6))

    m = extract_measurements(output.vertices, model.faces_tensor, output.joints)
    results[name] = m

    print(f"{name:<15} {m['chest_cm']:<8.1f} {m['waist_cm']:<8.1f} {m['hip_cm']:<8.1f} "
          f"{m['shoulder_width_cm']:<8.1f} {m['sleeve_length_cm']:<8.1f} {m['pants_length_cm']:<8.1f}")

print("-" * 70)

# 验证单调性
sl = results['slim(-2)']
nm = results['normal(0)']
pl = results['plump(+2)']

checks = [
    ('胸围单调性', 'chest_cm', sl['chest_cm'] < nm['chest_cm'] < pl['chest_cm']),
    ('腰围单调性', 'waist_cm', sl['waist_cm'] < nm['waist_cm'] < pl['waist_cm']),
    ('臀围单调性', 'hip_cm',   sl['hip_cm']   < nm['hip_cm']   < pl['hip_cm']),
    ('肩宽合理性', 'shoulder_width_cm', 25 < nm['shoulder_width_cm'] < 50),
    ('袖长合理性', 'sleeve_length_cm',  40 < nm['sleeve_length_cm'] < 80),
    ('裤长合理性', 'pants_length_cm',   80 < nm['pants_length_cm'] < 120),
]

all_ok = True
for name, key, ok in checks:
    status = '[OK]' if ok else '[FAIL]'
    if not ok: all_ok = False
    val = results['normal(0)'][key] if key in ['shoulder_width_cm','sleeve_length_cm','pants_length_cm'] else f"{sl[key]:.0f}<{nm[key]:.0f}<{pl[key]:.0f}"
    print(f"  {status} {name}: {val}")

print()
if all_ok:
    print("[PASS] All 6 checks passed - Module 4 validated")
else:
    print("[FAIL] Some checks failed")
