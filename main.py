import cv2
import json
import os
from detector import DeepfakeDetector
from image_processor import ImageProcessor

def main():
    image_path = "images/modificada.png"

    if not os.path.exists(image_path):
        print("Imagem não encontrada!")
        return


    # image = cv2.imread(image_path)
    processor = ImageProcessor(image_path)
    image = processor.normalize_size(width=300, height=300, keep_aspect_ratio=True)

    if image is None:
        print("Erro ao carregar imagem!")
        return

    detector = DeepfakeDetector()
    result = detector.analyze(image)

    print("\nResultado da análise:")
    for k, v in result.items():
        print(f"{k}: {v}")

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/results.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nResultado salvo em outputs/results.json")

if __name__ == "__main__":
    main()