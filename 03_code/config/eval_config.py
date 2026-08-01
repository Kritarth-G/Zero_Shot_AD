import os
import torch

# Base Paths (Relative to the 03_code directory)
BASE_DATASET_DIR = "../04_data"
RESULTS_DIR = "../05_results"

# Model Configurations
MODEL_NAME = "ViT-B/16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 100.0
WINDOW_SIZES = [2, 3]

# Data Normalization parameters (OpenAI CLIP standards)
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = (224, 224)