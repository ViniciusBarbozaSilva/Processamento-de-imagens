import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def load_image(path):
        return cv2.imread(path)
    
    @staticmethod
    def to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def enhance_contrast(image):
        return cv2.equalizeHist(image)
    
    @staticmethod
    def gaussian_blur(image):
        return cv2.GaussianBlur(image, (5, 5), 1.5)
    
    @staticmethod
    def sharpen(image):
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        return cv2.filter2D(image, -1, kernel)