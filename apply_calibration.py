"""应用校准: 管线输出 → 校准后尺寸"""
import sys; sys.path.insert(0, 'src')
import pickle, numpy as np

calib = pickle.load(open('models/beta_to_measurement.pkl', 'rb'))
models = calib['models']

def calibrate_measurements(raw_measurements, betas):
    """用beta参数校正原始尺寸"""
    if hasattr(betas, 'dim') and betas.dim() == 2:
        b = betas[0].numpy().reshape(1, -1)
    else:
        b = np.array(betas).reshape(1, -1)
    
    corrected = {}
    for k in ['chest_cm','waist_cm','hip_cm','shoulder_width_cm','sleeve_length_cm','pants_length_cm']:
        if k in raw_measurements and k in models:
            pred = models[k].predict(b)[0]
            corrected[k] = round(pred, 1)
    
    return corrected

# 演示: 对合成中性体型应用校准
print("Calibration Model Ready")
print("=" * 50)
print(f"Input: beta(10d) + raw measurements -> calibrated measurements")
print()

# 测试几个体型
for label, beta_val in [("Slim", -2.0), ("Normal", 0.0), ("Plump", 2.0)]:
    betas = np.array([[beta_val, 0,0,0,0,0,0,0,0,0]])
    # 模拟管线输出的原始尺寸(未经校准)
    raw = {k: models[k].predict(betas)[0] for k in models}
    corrected = calibrate_measurements(raw, betas)
    
    print(f"  {label} (beta[0]={beta_val}):")
    print(f"    chest: {raw['chest_cm']:.1f} -> {corrected['chest_cm']:.1f}")
    print(f"    waist: {raw['waist_cm']:.1f} -> {corrected['waist_cm']:.1f}")
    print(f"    hip:   {raw['hip_cm']:.1f} -> {corrected['hip_cm']:.1f}")

print()
print("[OK] Ready to integrate into pipeline.py as post-processing step")
