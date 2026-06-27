import cv2
import json
import os
import sys
from detector import DeepfakeDetector

def main():
    image_path = "images/7.jpg"  # Caminho padrão para teste
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Erro: A imagem '{image_path}' não foi encontrada!")
        return

    image = cv2.imread(image_path)

    if image is None:
        print("Erro ao carregar imagem!")
        return

    print(f"Analisando o arquivo: {image_path}...")
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