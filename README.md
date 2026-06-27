# Processamento-de-imagens
Repositório para o projeto da disciplina Processamento de imagens do professor Lucas Cambuim, UFRPE
# Image Manipulation Detection using Frequency and Color Analysis

## Overview

This project implements a modular image analysis pipeline designed to detect potential manipulations in digital images, including deepfakes or artificially altered content.

The system uses classical image processing techniques, combining spatial and frequency-domain analysis to extract features and compute a final suspicion score.

---

## Architecture

The project follows a modular design, where each component is responsible for a specific stage of the pipeline:

* **Image Processing Layer**

  * Responsible for loading and resizing images
  * Maintains aspect ratio and standardizes input

* **Feature Extraction Layer**

  * Extracts relevant features from different domains:

    * Frequency (FFT)
    * Color (RGB)
    * Color space (HSV)

* **Analysis Layer**

  * Each analyzer computes a partial score
  * Results are aggregated into a final score

* **Detection Layer**

  * Combines all features and determines whether the image is suspicious

---

## Project Structure

```
.
├── main.py                     # Application entry point
├── detector.py                # Orchestrates detection logic
├── image_processor.py         # Image loading and preprocessing
│
├── feature_extractor.py       # Feature extraction interface
├── frequency_analyzer.py      # Frequency domain analysis (FFT)
├── color_analyzer.py          # RGB/HSV analysis
├── benford_analyzer.py        # Statistical anomaly detection
│
├── config/
│   └── config.json            # Thresholds and configuration
│
├── images/                    # Input images
├── outputs/                   # Analysis results
```

---

## How It Works

### 1. Image Input

The system loads an image from the `images/` directory or from a path provided via command line.

### 2. Preprocessing

The image is resized using:

* Fixed dimensions (300x300)
* Aspect ratio preservation

Handled by:

```
ImageProcessor.normalize_size()
```

---

### 3. Feature Extraction

The system extracts multiple features:

* **Frequency Analysis**

  * Uses Fourier Transform to detect unnatural patterns

* **Color Analysis**

  * RGB distribution
  * HSV distribution

* **Statistical Analysis**

  * Benford’s Law for anomaly detection

---

### 4. Scoring

Each analyzer returns a score. These scores are combined into a final metric inside:

```
DeepfakeDetector.analyze()
```

---

### 5. Classification

The result is compared with a threshold defined in:

```
config/config.json
```

---

## Installation

### Requirements

* Python 3.10+
* Libraries:

  * opencv-python
  * numpy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Usage

### Run the project

```bash
python main.py
```

### Run with custom image

```bash
python main.py caminho/para/imagem.png
```

---

## Output

The system returns:

* Individual scores (frequency, color, statistical)
* Final score
* Classification (suspicious or not)

Results are saved in:

```
outputs/results.json
```

---

## Example Output

```json
{
  "frequency_score": 12.48,
  "color_score": 0.0021,
  "benford_score": 0.15,
  "final_score": 0.31,
  "is_suspicious": false
}
```

---

## Limitations

* Heuristic-based approach (no machine learning)
* May not detect advanced deepfake techniques
* Sensitive to image quality and compression

---

## Future Improvements

* Integration with deep learning models
* Adaptive threshold tuning
* Noise pattern analysis
* Dataset-based validation
* Implementation of Support Vector Machines (SVM) for improved decision boundaries
* Implementation of Random Forest for feature-based ensemble learning

---

## Author

Vinicius Gustavo Barboza Silva \
João Manoel de melo barbosa \
Antony Albuquerque Silva

---

## License

This project is intended for academic and educational purposes.
