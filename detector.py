import json
import os
from feature_extractor import FeatureExtractor
import numpy as np
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

        # --- NORMALIZAÇÃO DINÂMICA (MinMax) ---
        # Frequência
        min_f, max_f = self.config["min_freq"], self.config["max_freq"]
        freq_norm = (freq - min_f) / (max_f - min_f + 1e-5)
        
        # RGB
        min_rgb, max_rgb = self.config["min_rgb"], self.config["max_rgb"]
        rgb_norm = (rgb - min_rgb) / (max_rgb - min_rgb + 1e-5)
        
        # HSV
        min_hsv, max_hsv = self.config["min_hsv"], self.config["max_hsv"]
        hsv_norm = (hsv - min_hsv) / (max_hsv - min_hsv + 1e-5)

        # Garante que os scores fiquem rigidamente entre 0 e 1 (regra matemática)
        freq_norm = float(np.clip(freq_norm, 0.0, 1.0))
        rgb_norm = float(np.clip(rgb_norm, 0.0, 1.0))
        hsv_norm = float(np.clip(hsv_norm, 0.0, 1.0))
        
        # --- CÁLCULO DO SCORE FINAL ---
        # Pesos equilibrados: 50% Frequência, 25% Histograma RGB, 25% Histograma HSV
        final_score = (freq_norm * 0.60) + (rgb_norm * 0.30) + (hsv_norm * 0.10)
        
        # Classificação baseada no final_threshold (0.5)
        is_suspicious = final_score > self.config["final_threshold"]

        print(f"FREQ Normalizado: {freq_norm:.4f} (Original: {freq:.4f})")
        print(f"RGB Normalizado:  {rgb_norm:.4f} (Original: {rgb:.6f})")
        print(f"HSV Normalizado:  {hsv_norm:.4f} (Original: {hsv:.6f})")
        print(f"FINAL SCORE:      {final_score:.4f}")
        
        return {
            "frequency_score": float(freq),
            "rgb_score": float(rgb),
            "hsv_score": float(hsv),
            "final_score": float(final_score),
            "is_suspicious": bool(is_suspicious)
        }