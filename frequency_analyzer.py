import numpy as np
import cv2

class FrequencyAnalyzer:
    
    @staticmethod
    def compute_fft(image_gray):
        # Garante tamanho fixo para manter a escala da FFT idêntica
        resized = cv2.resize(image_gray, (512, 512), interpolation=cv2.INTER_AREA)
        f = np.fft.fft2(resized)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        # Aplica log para comprimir a escala dinâmica e facilitar a análise
        return np.log(magnitude + 1e-5)
    
    @staticmethod
    def analyze_frequencies(magnitude):
        h, w = magnitude.shape
        
        # Extrai apenas a coroa mais externa (as altíssimas frequências onde a IA falha)
        # Ignoramos completamente o centro (baixas frequências/conteúdo da imagem)
        high_frequencies = np.copy(magnitude)
        high_frequencies[h//3:2*h//3, w//3:2*w//3] = 0
        
        # Calculamos a variação (desvio padrão) desse ruído de alta frequência.
        # Imagens naturais têm ruído uniforme (desvio baixo). IA tem picos matemáticos (desvio alto).
        return float(np.std(high_frequencies))

    @staticmethod
    def analyze(image_gray):
        magnitude = FrequencyAnalyzer.compute_fft(image_gray)
        return FrequencyAnalyzer.analyze_frequencies(magnitude)