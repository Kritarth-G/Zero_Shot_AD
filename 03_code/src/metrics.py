import cv2
import numpy as np
import torch

def compute_blur_alpha_ratio(img_gray):
    """ Hyperparameter-free blur metric using Self-Referential Laplacian Ratio. """
    v_orig = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    img_blurred = cv2.GaussianBlur(img_gray, (21, 21), 0)
    v_blur = cv2.Laplacian(img_blurred, cv2.CV_64F).var()
    alpha_blur = v_blur / (v_orig + 1e-7)
    return float(np.clip(alpha_blur, 0.0, 1.0))

def compute_exposure_alphas(img_gray):
    """ Measures absolute deviation from ideal sensor exposure (127.5). """
    mu = np.mean(img_gray)
    if mu < 127.5:
        alpha_dark = (127.5 - mu) / 127.5
        alpha_bright = 0.0
    else:
        alpha_dark = 0.0
        alpha_bright = (mu - 127.5) / 127.5
    return float(alpha_dark), float(alpha_bright)

def compute_grain_alpha(img_gray):
    """ Bounded grain metric using Shannon Entropy. """
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).ravel()
    hist_prob = hist / (hist.sum() + 1e-7)
    H = -np.sum(hist_prob * np.log2(hist_prob + 1e-7))
    alpha_grain = (H / 8.0) ** 2
    return float(np.clip(alpha_grain, 0.0, 1.0))

def get_dynamic_weights(image_pil):
    """ Computes the weights array for a given image, keeping Clean prior dominant. """
    img_gray = np.array(image_pil.convert('L'))

    alpha_blur = compute_blur_alpha_ratio(img_gray)
    alpha_dark, alpha_bright = compute_exposure_alphas(img_gray)
    alpha_grain = compute_grain_alpha(img_gray)

    # [Clean Prior, Blur, Dark, Bright, Grain]
    alphas = np.array([1.0, alpha_blur, alpha_dark, alpha_bright, alpha_grain], dtype=np.float32)
    return torch.tensor(alphas)