"""方案A v2: 用SMPL-X关节回归器从EHF真值网格精确提取尺寸"""
import sys; sys.path.insert(0, 'src')
import torch, numpy as np, os, trimesh
import smplx
from measure.extract import extract_measurements

device = 'cpu'
# 加载SMPL-X模型以获取关节回归器
model = smplx.create('data/body_models/smplx/models', model_type='smplx',
                      gender='neutral', num_betas=10, batch_size=1)
J_regressor = model.J_regressor  # (55, 10475) sparse tensor

ehf_dir = 'data/ehf/EHF'
all_m = []
h_range = []

for frame in sorted(os.listdir(ehf_dir)):
    if not frame.endswith('_align.ply'):
        continue
    ply_path = os.path.join(ehf_dir, frame)
    
    gt = trimesh.load(ply_path)
    verts = torch.tensor(gt.vertices, dtype=torch.float32).unsqueeze(0)  # (1, V, 3)
    
    # 用SMPL-X关节回归器计算精确关节位置
    joints = torch.einsum('ij,bjk->bik', [J_regressor, verts])  # (1, 55, 3)
    
    h = (verts[0,:,1].max() - verts[0,:,1].min()).item() * 100
    
    try:
        m = extract_measurements(verts, 
                                 torch.tensor(model.faces_tensor, dtype=torch.long),
                                 joints)
        all_m.append(m)
        h_range.append(h)
    except Exception as e:
        continue

n = len(all_m)
print(f"EHF {n}/100 frames processed")
print()

keys = ['chest_cm', 'waist_cm', 'hip_cm', 'shoulder_width_cm', 'sleeve_length_cm', 'pants_length_cm']
ranges = {'chest_cm': (75,130), 'waist_cm': (55,110), 'hip_cm': (75,130),
          'shoulder_width_cm': (30,55), 'sleeve_length_cm': (45,75), 'pants_length_cm': (80,120)}

print(f"{'指标':<22} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10} {'检测':<8}")
print("-" * 72)

all_ok = True
for key in keys:
    vals = sorted([m[key] for m in all_m])
    # 去除首尾5%的异常值
    trim = int(n * 0.05)
    vals_trim = vals[trim:-trim] if trim > 0 else vals
    
    mn, mx = min(vals_trim), max(vals_trim)
    avg, std = np.mean(vals_trim), np.std(vals_trim)
    
    lo, hi = ranges[key]
    ok = lo <= avg <= hi
    if not ok: all_ok = False
    status = "[OK]" if ok else "[?]"
    print(f"  {status} {key:<18} {mn:<10.1f} {mx:<10.1f} {avg:<10.1f} {std:<10.1f} {lo}-{hi}")

print()
if h_range:
    print(f"身高: {min(h_range):.0f}-{max(h_range):.0f}cm (avg={np.mean(h_range):.0f}cm)")
    
if all_ok:
    print("\n[PASS] Plan A: EHF reference measurements established")
else:
    print("\n[MIXED] Some measurements outside typical range - check landmark definitions")
