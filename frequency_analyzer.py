import numpy as np

class FrequencyAnalyzer:
    
    @staticmethod
    def compute_fft(image_gray):
        f = np.fft.fft2(image_gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        return magnitude
    
    @staticmethod
    def analyze_frequencies(magnitude):
        h, w = magnitude.shape
        
        low = magnitude[h//4:3*h//4, w//4:3*w//4]
        high = np.copy(magnitude)
        high[h//4:3*h//4, w//4:3*w//4] = 0
        
        low_mean = np.mean(low)
        high_mean = np.mean(high)
        
        return low_mean / (high_mean + 1e-5)

    @staticmethod
    def analyze(image_gray):
        magnitude = FrequencyAnalyzer.compute_fft(image_gray)
        return FrequencyAnalyzer.analyze_frequencies(magnitude)