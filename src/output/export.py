import json
import os


def save_measurements_json(measurements, height_cm, out_path, confidence="medium", warnings=None):
    result = {
        'height_cm': height_cm,
        'measurements': measurements,
        'confidence': confidence,
        'warnings': warnings or [],
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
