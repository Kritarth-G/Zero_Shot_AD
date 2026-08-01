# generate_synthetic_mvtec.py
import os
import cv2
import numpy as np
import shutil

# Dynamically find the absolute paths based on where this script is located
# __file__ is 03_code/scripts/generate_synthetic_mvtec.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Navigate up two levels from 'scripts/' to reach the main project root, then into '04_data/'
SOURCE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../04_data/mvtec_ad"))
TARGET_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../04_data/mvtec_ad_synthetic"))

def apply_random_combined_degradation(image):
    """
    Applies Blur, Darkness, and Grain with randomized severity per image 
    to simulate unpredictable real-world factory conditions.
    """
    # 1. Random Blur (Simulate variable out-of-focus / motion)
    # Sigma controls the blur intensity. 
    blur_sigma = np.random.uniform(1.0, 3.0)
    img_blur = cv2.GaussianBlur(image, (7, 7), blur_sigma)
    
    # 2. Random Darkness (Simulate fluctuating lighting)
    # Multiplier between 0.3 (very dark) and 0.8 (slightly dim)
    dark_factor = np.random.uniform(0.3, 0.8)
    img_dark = np.clip(img_blur * dark_factor, 0, 255).astype(np.uint8)
    
    # 3. Random Grain/Noise (Simulate variable sensor ISO noise)
    mean = 0
    noise_std = np.random.uniform(15, 40) 
    noise = np.random.normal(mean, noise_std, img_dark.shape).astype(np.float32)
    img_noisy = np.clip(img_dark + noise, 0, 255).astype(np.uint8)
    
    return img_noisy

def generate_full_synthetic_dataset():
    print(f"Generating full randomized synthetic dataset from:\n  {SOURCE_DIR}")
    print(f"Target directory:\n  {TARGET_DIR}\n")
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} not found. Ensure you have downloaded and placed MVTec AD in 04_data/mvtec_ad.")
        return

    image_count = 0

    # os.walk will automatically dig into every category, train/test, and defect folder
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Calculate where this exact folder should live in the target directory
        relative_path = os.path.relpath(root, SOURCE_DIR)
        target_path = os.path.join(TARGET_DIR, relative_path)
        
        # Create the mirrored folder if it doesn't exist
        os.makedirs(target_path, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_path, file)
                
                # STRICT CHECK: Is this a ground truth mask?
                # We check the directory path to see if it belongs to the ground truth
                if 'ground_truth' in root:
                    # Copy the mask exactly as it is without any degradation
                    shutil.copy2(src_file, dst_file)
                else:
                    # Read, degrade with random variables, and save the regular image
                    img = cv2.imread(src_file)
                    if img is not None:
                        degraded_img = apply_random_combined_degradation(img)
                        cv2.imwrite(dst_file, degraded_img)
                        image_count += 1
                        
                        if image_count % 500 == 0:
                            print(f"Processed {image_count} images...")

    print(f"\nSuccess! Full synthetic dataset ({image_count} degraded images + untouched masks) saved to:\n{TARGET_DIR}")

if __name__ == "__main__":
    generate_full_synthetic_dataset()