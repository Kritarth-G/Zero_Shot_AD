import os
import sys
import csv
import numpy as np

# Add the 03_code directory to the system path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.eval_config import BASE_DATASET_DIR, RESULTS_DIR, MODEL_NAME, DEVICE
from src.winclip import WinCLIP
from src.evaluate import evaluate_winclip, infer_winclip

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    datasets_menu = {
        "1": "mvtec_ad",
        "2": "mvtec_ad_synthetic",
        "3": "casting_data",
        "4": "magnetic_tile",
        "5": "sample_inputs"
    }

    print("=" * 40)
    print(" WinCLIP Interactive Evaluator")
    print("=" * 40)
    for key, name in datasets_menu.items():
        print(f" [{key}] {name}")
    print("=" * 40)

    choice = input("Enter the number of the dataset to evaluate (1-5): ").strip()
    if choice not in datasets_menu:
        sys.exit()

    selected_dataset = datasets_menu[choice]
    dataset_path = os.path.join(BASE_DATASET_DIR, selected_dataset)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        sys.exit()

    print("\nLoading CLIP Model (this happens only once)...")
    shared_winclip = WinCLIP(model_name=MODEL_NAME, device=DEVICE)

    if choice == "5":
        # Ask the user what the object is
        print("\n--- Inference Settings ---")
        obj_name = input("Enter the object name (e.g., 'metal casting', 'pump impeller') [default: object]: ").strip()
        if not obj_name:
            obj_name = "object"
            
        print(f"\nRunning inference for '{obj_name}'...")
        
        # Pass the arguments to infer_winclip (no threshold needed)
        results = infer_winclip(shared_winclip, dataset_path, class_name=obj_name)
        
        csv_filename = os.path.join(RESULTS_DIR, "main_results.csv")
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Update headers to strictly output the continuous score
            writer.writerow(["Filename", "Anomaly_Score"])
            for res in results:
                writer.writerow(res)
                
        print("\n" + "=" * 60)
        print(f" Inference Complete! Raw scores successfully saved to: {csv_filename}")
        print("=" * 60)

    else:
        img_aurocs, img_auprcs, img_f1s = [], [], []
        csv_filename = os.path.join(RESULTS_DIR, f"{selected_dataset}_evaluation_results.csv")
        
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Category", "Image_AUROC", "Image_AUPRC", "Image_F1_Max"])

            if selected_dataset in ["mvtec_ad", "mvtec_ad_synthetic"]:
                categories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
                for category in categories:
                    result = evaluate_winclip(shared_winclip, dataset_path, dataset_type=selected_dataset, category=category)
                    if result[0] is not None:
                        img_auc, img_auprc, img_f1 = result
                        img_aurocs.append(img_auc)
                        img_auprcs.append(img_auprc)
                        img_f1s.append(img_f1)
                        writer.writerow([category, img_auc, img_auprc, img_f1])
            else:
                result = evaluate_winclip(shared_winclip, dataset_path, dataset_type=selected_dataset)
                if result[0] is not None:
                    img_auc, img_auprc, img_f1 = result
                    img_aurocs.append(img_auc)
                    img_auprcs.append(img_auprc)
                    img_f1s.append(img_f1)
                    writer.writerow([selected_dataset, img_auc, img_auprc, img_f1])
            
            if img_aurocs:
                writer.writerow(["MEAN_AVERAGE", np.mean(img_aurocs), np.mean(img_auprcs), np.mean(img_f1s)])

        if img_aurocs:
            print("\n" + "=" * 60)
            print(f" FINAL MEAN RESULTS FOR: {selected_dataset.upper()}")
            print("=" * 60)
            print("  [Image-level]")
            print(f"    AUROC :  {np.mean(img_aurocs)  * 100:.1f}%")
            print(f"    AUPRC :  {np.mean(img_auprcs)  * 100:.1f}%")
            print(f"    F1-max:  {np.mean(img_f1s)     * 100:.1f}%")
            print("\n" + "=" * 60)
            print(f" Full metrics successfully saved to: {csv_filename}")
            print("=" * 60)