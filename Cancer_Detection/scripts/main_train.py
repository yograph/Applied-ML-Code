#!/usr/bin/env python3
"""
Main script to train three ConvNet models on the split data.

Usage:
    python scripts/main_train.py
"""

import os
import torch
from torchvision import datasets, transforms
from models.main_model.training_utils import FullTrainingOfModel

# ←── EDIT THIS PATH ──→ where to find your split data
TRAIN_DATA_DIR = "data/train_images"
VALID_DATA_DIR = "data/validation_images"
TEST_DATA_DIR  = "data/test_images"

# ←── EDIT THIS PATH ──→ where to save your model weights
MODEL_SAVE_DIR = "models/main_model/model_weights"

# 1) Define your transforms (must match what convnet_one expects)
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# 2) Build ImageFolder datasets
train_ds = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=data_transforms)
val_ds   = datasets.ImageFolder(root=VALID_DATA_DIR, transform=data_transforms)
test_ds  = datasets.ImageFolder(root=TEST_DATA_DIR, transform=data_transforms)

# 3) Get class names (ImageFolder uses subfolder names as class labels in alphabetical order)
class_names = train_ds.classes  # e.g. ["0", "1"] if your subfolders are named '0' and '1'

# 4) Instantiate FullTrainingOfModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = FullTrainingOfModel(
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    batch_size=32,
    lr=1e-4,
    num_epochs=10,
    device=device,
    class_names=class_names,
    gamma=2.0
)

# 5) Initialize (e.g. 3 separate ConvNets)
trainer.initialize_models(num_models=3)

# 6) Train & save
trainer.train_and_save(MODEL_SAVE_DIR)

# 7) Evaluate on val/test and produce metrics + plots
eval_results = trainer.evaluate_models()
print("\n=== Evaluation Results ===")
for name, res in eval_results.items():
    print(f"\n{name} → VAL ACCURACY: {res['val']['accuracy']:.4f}, TEST ACCURACY: {res['test']['accuracy']:.4f}")
