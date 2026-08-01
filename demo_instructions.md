# Demo Instructions

This document outlines the steps to run a live demonstration of our zero-shot anomaly detection pipeline using the WinCLIP framework.

---

## 1. Prerequisites

Before running the demo, ensure you have set up the Python environment as detailed in `03_code/README.md`.

- All dependencies from `requirements.txt` must be installed.
- You must have an active internet connection the first time you run the script so the OpenAI CLIP model weights can be downloaded automatically.

---

## 2. Preparing the Demo Inputs

1. For the live demo, we evaluate unlabelled images to classify them as either **Normal** or **Anomaly**. Copy or move the images you wish to test into the designated inference folder: `04_data/sample_input/`.

> **Note:** The script recursively scans this directory, so you can place images directly inside or within subfolders.

---

## 3. Running the Demo

Once your environment is set up and your images are in place, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the `03_code` directory of the project:

```bash
cd path/to/Group16_Zero_Shot_AD/03_code
```

3. Launch the interactive evaluation script:

```bash
python -m scripts.run_eval
```

---

## 4. Expected Output
The script will process each image in the `sample_input` directory and calculate an anomaly score. 

* **Console Output:** You will see a progress bar indicating the inference status.
* **Results File:** Upon completion, the raw scores will be saved to a CSV file located at:
  `05_results/main_results.csv`

**Understanding the CSV Results:**
* **Filename:** The relative path of the processed image.
* **Anomaly_Score:** The raw confidence score calculated by the model (continuous float value). 

*Note: Because zero-shot models experience baseline shifts depending on the complexity of the object being analyzed, we output the raw continuous scores rather than applying a hardcoded `0.5` threshold. Higher scores indicate a higher likelihood of an anomaly. To convert these to binary labels, plot or analyze the distribution of these scores to find the natural valley/cutoff point for your specific data.*