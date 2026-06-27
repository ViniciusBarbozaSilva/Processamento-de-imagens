import cv2
from image_processor import ImageProcessor
from frequency_analyzer import FrequencyAnalyzer
from color_analyzer import ColorAnalyzer

class FeatureExtractor:

    @staticmethod
    def extract(image):
        # 1. Padroniza o tamanho (Zera a diferença de Megapixels entre celulares)
        image_resized = ImageProcessor.prepare_image(image, target_size=(512, 512))
        
        # 2. Suaviza ruídos e nitidez artificial nativa dos sensores dos aparelhos
        image_blurred = ImageProcessor.remove_sensor_noise(image_resized)
        
        # 3. Transforma em escala de cinza e analisa frequências (FFT)
        gray = ImageProcessor.to_grayscale(image_blurred)
        freq_score = FrequencyAnalyzer.analyze(gray)
        
        # 4. Extração de histogramas baseados em bordas (Laplaciano)
        # Convertemos para escala de cinza o Laplaciano para caçar ruído micro-estrutural
        laplacian = cv2.Laplacian(image_resized, cv2.CV_64F)
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