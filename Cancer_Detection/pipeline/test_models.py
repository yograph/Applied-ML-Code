#!/usr/bin/env python3
"""
Script to load the three trained models (model_0, model_1, model_2) and evaluate them
on the exact same test samples. For each model, this will:

  - Run inference on the test set
  - Print the classification report (precision, recall, F1) to the terminal
  - Print the confusion matrix (raw and normalized) to the terminal
  - Print key metrics: accuracy, balanced accuracy, Matthews’s CC, Cohen’s Kappa, ROC-AUC
  - Save a CSV of [file path, true label, predicted label, prob_no_cancer, prob_cancer]

Usage:
    python models/main_model/test_models.py
"""

import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score
)
from torchvision import transforms
from PIL import Image

# Import necessary classes/functions from main_convnext_model.py
from main_convnext_model import (
    Creating_Convnet,
    CsvBackedImageDataset,
    ExtendedEvaluation,
    default_device
)


# Directory containing your test images (flat structure; labels from CSV)
TEST_IMG_DIR   = "data/test_images"

# Path to the same CSV metadata used in training
METADATA_CSV   = "data/csv_file/train.csv"

# Directory where your saved weights live:
#   models/main_model/model_weights/model_0_weights.pth, etc.
MODEL_SAVE_DIR = "models/main_model/model_weights"

# Directory to which we will write CSVs of predictions
PREDICTIONS_OUTDIR = "models/main_model/predictions"
os.makedirs(PREDICTIONS_OUTDIR, exist_ok=True)

def main():
    # 1) Device
    device = default_device()
    print(f"Evaluating on device: {device}\n")

    # 2) Load the same metadata.csv
    metadata_df = pd.read_csv(METADATA_CSV)

    # 3) Define exactly the same transforms used in training
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 4) Build the test dataset
    print("→ Building TEST dataset…")
    test_ds = CsvBackedImageDataset(
        root_dir=TEST_IMG_DIR,
        metadata_df=metadata_df,
        transform=data_transforms
    )

    # 5) DataLoader for test set
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 6) Class names (binary)
    class_names = ["no_cancer", "cancer"]

    # 7) Loop over the three saved weight files
    for model_idx in range(3):
        weight_path = os.path.join(MODEL_SAVE_DIR, f"model_{model_idx}_weights.pth")
        if not os.path.exists(weight_path):
            print(f"⚠️  Weight file not found: {weight_path}")
            continue

        # 7a) Instantiate a fresh ConvNet via Creating_Convnet
        base = Creating_Convnet()
        model = base.creating_single_model().to(device)

        # 7b) Load the saved weights
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # 7c) Run ExtendedEvaluation on the test_loader
        evaluator = ExtendedEvaluation(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names
        )
        metrics = evaluator.compute_all()

        # 7d) Get raw predictions for printing
        y_true, y_pred, y_prob = evaluator.get_predictions()

        # 7e) Print results
        print(f"\n========== Model {model_idx} Evaluation on TEST Set ==========\n")

        # Classification report
        print(">>> Classification Report:\n")
        print(classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0
        ))

        # Confusion matrix (raw)
        cm = confusion_matrix(y_true, y_pred)
        print(">>> Confusion Matrix (raw counts):")
        print(cm)

        # Confusion matrix (normalized by true‐row sums)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        print("\n>>> Confusion Matrix (normalized):")
        print(np.round(cm_norm, 4))

        # Key metrics from metrics dict
        print("\n>>> Key Metrics:")
        print(f"  • Accuracy             : {metrics['accuracy']:.4f}")
        print(f"  • Balanced Accuracy    : {metrics['balanced_accuracy']:.4f}")
        print(f"  • Matthews Corrcoef    : {metrics['matthews_corrcoef']:.4f}")
        print(f"  • Cohen's Kappa        : {metrics['cohen_kappa']:.4f}")
        if metrics['roc_auc_ovo'] is not None:
            print(f"  • ROC-AUC (OVO)        : {metrics['roc_auc_ovo']:.4f}")
        else:
            print(f"  • ROC-AUC (OVO)        : n/a")

        # 7f) Save predictions to CSV
        # The test_ds.samples list holds (image_path, label) in the same order as DataLoader
        file_paths = [fp for (fp, _) in test_ds.samples]
        probs_arr = np.stack(y_prob, axis=0)  # shape [N, 2]
        df = pd.DataFrame({
            "file_path":       file_paths,
            "true_label":      y_true,
            "pred_label":      y_pred,
            "prob_no_cancer":  probs_arr[:, 0],
            "prob_cancer":     probs_arr[:, 1]
        })
        out_csv = os.path.join(PREDICTIONS_OUTDIR, f"predictions_model_{model_idx}.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n→ Saved predictions for model_{model_idx} to:\n   {out_csv}")

        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
