import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from configs.eval_config import NORM_MEAN, NORM_STD, IMAGE_SIZE, DEVICE
from src.datasets import MVTecDataset, CastingDataset, MagneticTileDataset, TestDataset

def compute_metrics(labels, scores):
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_per_threshold = (
        (2 * precision[:-1] * recall[:-1])
        / (precision[:-1] + recall[:-1] + 1e-8)
    )
    best_idx = f1_per_threshold.argmax()
    f1_max = float(f1_per_threshold[best_idx])
    best_threshold = float(thresholds[best_idx])

    return auroc, auprc, f1_max, best_threshold

def evaluate_winclip(winclip, dataset_path, dataset_type="mvtec", category=None):
    class_name = category if category else dataset_type.replace('_', ' ')
    winclip.encode_text_cpe(class_name)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    if dataset_type in ["mvtec_ad", "mvtec_ad_synthetic"]:
        dataset = MVTecDataset(dataset_path, category, transform, mask_transform)
    elif dataset_type == "casting_data":
        dataset = CastingDataset(dataset_path, transform, mask_transform)
    elif dataset_type == "magnetic_tile":
        dataset = MagneticTileDataset(dataset_path, transform, mask_transform)
    else:
        raise ValueError("Unknown dataset type")

    if len(dataset) == 0:
        return None, None, None

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    img_scores, img_labels = [], []

    for image, mask, label, weights in tqdm(dataloader, desc=f"Evaluating {class_name}"):
        image = image.to(DEVICE)
        weights = weights.to(DEVICE).squeeze(0) 

        image_score = winclip.extract_window_features(image, weights)

        img_scores.append(image_score)
        img_labels.append(label.item())

    img_auroc, img_auprc, img_f1_max, img_best_thr = compute_metrics(img_labels, img_scores)

    return img_auroc, img_auprc, img_f1_max

def infer_winclip(winclip, dataset_path, class_name="object"):
    winclip.encode_text_cpe(class_name)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    dataset = TestDataset(dataset_path, transform)
    if len(dataset) == 0:
        print(f"No images found in {dataset_path}.")
        return []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []

    for image, filename, weights in tqdm(dataloader, desc="Inferring sample_input"):
        image = image.to(DEVICE)
        weights = weights.to(DEVICE).squeeze(0) 

        image_score = winclip.extract_window_features(image, weights)
        
        # Output only the filename and the raw continuous score
        results.append((filename[0], float(image_score)))
        
    return results