from image_processor import ImageProcessor
from frequency_analyzer import FrequencyAnalyzer
from color_analyzer import ColorAnalyzer

class FeatureExtractor:

    @staticmethod
    def extract(image):
        gray = ImageProcessor.to_grayscale(image)

        freq_score = FrequencyAnalyzer.analyze(gray)

        rgb_hist = ColorAnalyzer.rgb_histogram(image)
        hsv_hist = ColorAnalyzer.hsv_histogram(image)

        rgb_score = ColorAnalyzer.histogram_score(rgb_hist)
        hsv_score = ColorAnalyzer.histogram_score(hsv_hist)

        return {
            "frequency_score": freq_score,
            "rgb_score": rgb_score,
            "hsv_score": hsv_score
        }