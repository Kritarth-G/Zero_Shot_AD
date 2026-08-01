# Project README
# Zero-Shot Anomaly Detection via WinCLIP

## 1. Environment Setup & Dependencies

This code requires Python 3.8+ and pip. To install the required dependencies, run:

pip install -r requirements.txt

## 2. Key Dependencies

| Package | Source |
|---|---|
| PyTorch & Torchvision | `pip install torch torchvision` |
| OpenAI CLIP | `pip install git+https://github.com/openai/CLIP.git` |
| OpenCV | `pip install opencv-python` |
| Scikit-Learn | `pip install scikit-learn` |

---


## 3. Training Commands

WinCLIP is a **zero-shot anomaly detection** framework. Therefore, no explicit training scripts or commands are required. The model leverages pre-trained vision-language features directly.

---

## 4. Evaluation Commands

To run the evaluation on labeled datasets (MVTec AD, Casting Data, Magnetic Tile):

1. Ensure your data is located in `../04_data/`
2. Navigate to the `03_code` directory and run:

```bash
python -m scripts.run_eval
```

3. Follow the interactive prompt and select **options 1 through 4**.

Output metrics will be automatically saved to `../05_results/`.

---

## 6. Demo / Inference Commands

To run unlabelled inference (e.g., on a custom test set or demo data):

1. Place your images in `../04_data/sample_inputs/`
2. Run:

```bash
python -m scripts.run_eval
```

3. Select **option 5** when prompted.

Anomaly Scores will be saved to `../05_results/main_results.csv`.
