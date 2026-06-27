import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def prepare_image(image, target_size=(512, 512)):
        """Padroniza o tamanho da imagem para neutralizar resoluções diferentes."""
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def remove_sensor_noise(image):
        """Suaviza o sharpening artificial gerado pelas câmeras dos celulares."""
        return cv2.GaussianBlur(image, (3, 3), 0)
    
    @staticmethod
    def enhance_contrast(image):
        return cv2.equalizeHist(image)
    
    @staticmethod
    def sharpen(image):
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(image, -1, kernel)