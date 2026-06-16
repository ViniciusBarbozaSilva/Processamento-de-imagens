import cv2
import numpy as np

class ColorAnalyzer:

    @staticmethod
    def rgb_histogram(image):
        hist = []
        for i in range(3):
            h = cv2.calcHist([image], [i], None, [256], [0,256])
            hist.append(h.flatten())
        return hist

    @staticmethod
    def hsv_histogram(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = []
        for i in range(3):
            h = cv2.calcHist([hsv], [i], None, [256], [0,256])
            hist.append(h.flatten())
        return hist

    @staticmethod
    def detect_color_anomaly(histograms):
        variances = []
        for hist in histograms:
            hist = hist / (np.sum(hist) + 1e-5)
            mean = np.mean(hist)
            var = np.mean((hist - mean) ** 2)
            variances.append(var)

        return np.mean(variances)

    @staticmethod
    def histogram_score(histograms):
        return ColorAnalyzer.detect_color_anomaly(histograms)