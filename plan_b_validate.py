"""方案B: Model Agency - models1 agency (verified URLs)"""
import sys; sys.path.insert(0, 'src')
import json, os, yaml, cv2, torch, numpy as np, requests
from keypoint.detect import detect_keypoints
from fitting.smplify import fit_smplx
from measure.extract import extract_measurements

data = json.load(open('data/model_agency/ModelAgencyData/cleaned_model_data.json'))
config = yaml.safe_load(open('config.yaml'))
os.makedirs('data/model_agency/images', exist_ok=True)

fd = data['models1']
names = fd['model_name']
heights = fd['height_cm']; busts = fd['bust_cm']; waists = fd['waist_cm']; hips = fd['hips_cm']
has_vars = fd['has_relevant_vars']; genders = fd['gender']; urls = fd['image_urls']

# 跳过第一个cocaine_models的数据(broken URLs), 从中间选
test_indices = [50, 100, 150, 200, 250]  # 选5个不同体型的
for idx in test_indices:
    if idx >= len(names) or not has_vars[idx] or len(urls[idx]) == 0:
        continue
    
    name = names[idx]
    h = heights[idx]; c = busts[idx]; w = waists[idx]; hp = hips[idx]; g = genders[idx]
    print(f"\n[{idx}] {name} ({g}) h={h}cm c={c} w={w} hip={hp}")
    
    # 下载
    img_path = f"data/model_agency/images/models1_{idx}_{name}.jpg"
    if not os.path.exists(img_path):
        try:
            r = requests.get(urls[idx][0], timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200 and len(r.content) > 5000:
                with open(img_path, 'wb') as f: f.write(r.content)
                print(f"  Downloaded: {len(r.content)//1024}KB")
            else:
                print(f"  Skip: HTTP {r.status_code}")
                continue
        except: continue
    
    if os.path.getsize(img_path) < 5000:
        print("  Skip: too small"); continue
    
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    kp = detect_keypoints(img)
    n = (kp[:,3]>0.5).sum()
    print(f"  Image: {img.shape[1]}x{img.shape[0]}, KP: {n}/33")
    if n < 10: print("  Skip: too few keypoints"); continue
    
    config['model']['gender'] = g if g in ['male','female'] else 'neutral'
    try:
        smpl = fit_smplx(kp, kp.copy(), h, config)
        pred = extract_measurements(smpl['vertices'], smpl['faces'], smpl['joints'])
        
        print(f"  Pred:  c={pred['chest_cm']:.0f}  w={pred['waist_cm']:.0f}  hip={pred['hip_cm']:.0f}")
        print(f"  GT:    c={c}  w={w}  hip={hp}")
        print(f"  Err:   c={abs(pred['chest_cm']-c):.1f}  w={abs(pred['waist_cm']-w):.1f}  hip={abs(pred['hip_cm']-hp):.1f} cm")
        print(f"  Other: shoulder={pred['shoulder_width_cm']:.0f}  sleeve={pred['sleeve_length_cm']:.0f}  pants={pred['pants_length_cm']:.0f}")
    except Exception as e:
        print(f"  Fit failed: {e}")
    config['model']['gender'] = 'neutral'
