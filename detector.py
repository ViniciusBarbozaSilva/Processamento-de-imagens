import json
import os
from feature_extractor import FeatureExtractor

class DeepfakeDetector:

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config", "config.json")

        print("Carregando config de:", config_path)  # debug

        with open(config_path) as f:
            self.config = json.load(f)

    def analyze(self, image):
        features = FeatureExtractor.extract(image)

        freq = features["frequency_score"]
        rgb = features["rgb_score"]
        hsv = features["hsv_score"]

        freq_norm = freq / 100
        rgb_norm = rgb / 0.001
        hsv_norm = hsv / 0.001
        
        final_score = (freq_norm * 0.5) + (rgb_norm * 0.25) + (hsv_norm * 0.25)
        is_suspicious = final_score > self.config["final_threshold"]

        print("FREQ:", freq_norm)
        print("RGB:", rgb_norm)
        print("HSV:", hsv_norm)
        print("FINAL SCORE:", final_score)
        
        result = {
            "frequency_score": float(freq),
            "rgb_score": float(rgb),
            "hsv_score": float(hsv),
            "final_score": float(final_score),
            "is_suspicious": bool(is_suspicious)
        }

        return result