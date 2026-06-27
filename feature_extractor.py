import cv2
import numpy as np
from image_processor import ImageProcessor
from frequency_analyzer import FrequencyAnalyzer
from color_analyzer import ColorAnalyzer

class FeatureExtractor:

    @staticmethod
    def extract(image):
        # 1. CORREÇÃO: Fazer um CROP central para não destruir os pixels da IA com interpolação
        h, w = image.shape[:2]
        start_y = max(0, h//2 - 256)
        start_x = max(0, w//2 - 256)
        image_cropped = image[start_y:start_y+512, start_x:start_x+512]
        
        # Garantia de que a imagem terá 512x512 caso a original seja muito pequena
        image_resized = ImageProcessor.prepare_image(image_cropped, target_size=(512, 512))
        
        # 2. CORREÇÃO: A FFT deve analisar a imagem CRUA (sem blur)
        gray_raw = ImageProcessor.to_grayscale(image_resized)
        freq_score = FrequencyAnalyzer.analyze(gray_raw)
        
        # 3. Suaviza ruídos APÓS a FFT para preparar a análise de cores/bordas
        image_blurred = ImageProcessor.remove_sensor_noise(image_resized)
        
        # 4. CORREÇÃO: Laplaciano aplicado na imagem COM BLUR. 
        # Isso ignora a textura do tecido preto no fundo das imagens e foca nos artefatos da IA
        laplacian = cv2.Laplacian(image_blurred, cv2.CV_64F)
        laplacian_8bit = cv2.convertScaleAbs(laplacian)
        
        rgb_hist = ColorAnalyzer.rgb_histogram(laplacian_8bit)
        hsv_hist = ColorAnalyzer.hsv_histogram(laplacian_8bit)

        rgb_score = ColorAnalyzer.histogram_score(rgb_hist)
        hsv_score = ColorAnalyzer.histogram_score(hsv_hist)

        return {
            "frequency_score": freq_score,
            "rgb_score": rgb_score,
            "hsv_score": hsv_score
        }