"""快速统计模型: 2D关键点+身高 → 6项尺寸
完全不依赖SMPL拟合, 秒级出结果, 用于Web Demo
"""
import sys; sys.path.insert(0, 'src')
import torch, numpy as np, os, pickle, json
import smplx
from measure.extract import extract_measurements
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("Training statistical measurement predictor from 2D keypoints...")

device = 'cpu'
model = smplx.create('data/body_models/smplx/models', model_type='smplx',
                      gender='neutral', num_betas=10, batch_size=1)
faces = torch.tensor(model.faces_tensor, dtype=torch.long)

# A-pose
body_pose = torch.zeros(1, 63)
body_pose[0, 0:3] = torch.tensor([0.0, 0.0, -0.4])
body_pose[0, 3:6] = torch.tensor([0.0, 0.0, 0.4])

# 生成300样本: 随机beta → 2D关键点特征 → 尺寸
np.random.seed(42)
n = 300
betas_synth = np.clip(np.random.randn(n, 10) * 1.5, -4, 4)

features = []  # 每行: [height_cm, shoulder_ratio, torso_ratio, leg_ratio]
targets = {k: [] for k in ['chest_cm','waist_cm','hip_cm','shoulder_width_cm','sleeve_length_cm','pants_length_cm']}

for i in range(n):
    betas = torch.tensor([betas_synth[i]], dtype=torch.float32)
    output = model(betas=betas, body_pose=body_pose, global_orient=torch.zeros(1,3),
                   jaw_pose=torch.zeros(1,3), left_hand_pose=torch.zeros(1,6), right_hand_pose=torch.zeros(1,6))
    
    joints = output.joints[0].detach().numpy()  # (127, 3)
    verts = output.vertices[0].detach().numpy()  # (10475, 3)
    
    # 提取2D关键点投影特征(模拟正面视角, 正交投影)
    # SMPL-X body joints: 16=shoulder_L, 17=shoulder_R, 1=hip_L, 2=hip_R
    # 4=knee_L, 5=knee_R, 7=ankle_L, 8=ankle_R, 20=wrist_L, 21=wrist_R
    # 0=pelvis, 15=head
    
    height = verts[:,1].max() - verts[:,1].min()
    shoulder_w = np.linalg.norm(joints[16] - joints[17])  # 左右肩宽(3D)
    hip_w = np.linalg.norm(joints[1] - joints[2])         # 左右髋宽(3D)
    torso_h = joints[16,1] - joints[1,1]                   # 肩到髋高度
    leg_h = joints[1,1] - joints[7,1]                      # 髋到踝高度
    
    # 特征: 身高 + 比例
    feat = [
        height * 100,           # 身高cm
        shoulder_w / height,    # 肩宽比例
        hip_w / height,         # 髋宽比例
        torso_h / height,       # 躯干比例
        leg_h / height,         # 腿长比例
    ]
    features.append(feat)
    
    m = extract_measurements(output.vertices, faces, output.joints)
    for k in targets:
        targets[k].append(m[k])

X = np.array(features)
for k in targets:
    targets[k] = np.array(targets[k])

# 训练Ridge回归
scalers = {}
regressors = {}
scores = {}

for k in ['chest_cm','waist_cm','hip_cm','shoulder_width_cm','sleeve_length_cm','pants_length_cm']:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reg = Ridge(alpha=1.0)
    reg.fit(X_scaled, targets[k])
    
    from sklearn.model_selection import cross_val_score
    r2 = cross_val_score(reg, X_scaled, targets[k], cv=5, scoring='r2').mean()
    mae = np.mean(np.abs(reg.predict(X_scaled) - targets[k]))
    
    scalers[k] = scaler
    regressors[k] = reg
    scores[k] = {'r2': r2, 'mae': mae}
    
    print(f"  {k}: R2={r2:.4f}  MAE={mae:.1f}cm")

# 保存模型
os.makedirs('models', exist_ok=True)
with open('models/kp_to_measurement.pkl', 'wb') as f:
    pickle.dump({'scalers': scalers, 'regressors': regressors, 'scores': scores}, f)

print(f"\n[OK] Saved to models/kp_to_measurement.pkl")
