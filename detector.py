import json
import os
from feature_extractor import FeatureExtractor
import numpy as np

class DeepfakeDetector:

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config", "config.json")
        with open(config_path) as f:
            self.config = json.load(f)

    def analyze(self, image):
        features = FeatureExtractor.extract(image)

        freq = features["frequency_score"]
        rgb = features["rgb_score"]
        hsv = features["hsv_score"]

        # 1. Normalização da Frequência baseada no mapeamento real (2.15 a 2.65)
        min_f, max_f = self.config["min_freq"], self.config["max_freq"]
        freq_norm = (freq - min_f) / (max_f - min_f + 1e-5)
        freq_norm = float(np.clip(freq_norm, 0.0, 1.0))
        
        # 2. Ajuste do teto do RGB: subimos o teto de 0.002 para 0.015 
        # para que o valor das suas fotos reais (ex: 0.006) não estoure em 1.0000
        rgb_norm = (rgb - 0.0001) / (0.015 - 0.0001 + 1e-5)
        rgb_norm = float(np.clip(rgb_norm, 0.0, 1.0))

        # Score Final Combinado (Pega o maior nível de suspeita)
        final_score = max(freq_norm, rgb_norm)
        
        is_suspicious = final_score > self.config["final_threshold"]

        print(f"FREQ Normalizado: {freq_norm:.4f} (Original: {freq:.4f})")
        print(f"RGB Normalizado:  {rgb_norm:.4f} (Original: {rgb:.6f})")
        print(f"SCORE FINAL DE SUSPEITA: {final_score:.4f}")
        
        # Corrigido: Usamos float() em TODOS os valores numéricos para o JSON não quebrar
        return {
            "is_suspicious": bool(is_suspicious),
            "final_score": float(final_score),
            "frequency_score": float(freq),
            "rgb_score": float(rgb)
        }