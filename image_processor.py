import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)

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
    

    def normalize_size(self, width=300, height=300, keep_aspect_ratio=True):
        if self.image is None:
            print(f"Arquivo não encontrado: {self.path}")
            return None

        h, w = self.image.shape[:2]

        if keep_aspect_ratio:
            # calcula proporção
            r = min(width / w, height / h)
            new_w, new_h = int(w * r), int(h * r)
            return cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            # força exatamente width x height
            return cv2.resize(self.image, (width, height), interpolation=cv2.INTER_AREA)


