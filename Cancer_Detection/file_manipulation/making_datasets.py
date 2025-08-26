#!/usr/bin/env python3
"""
Script to balance a cancer image dataset, split into train/validation/test sets,
and move image files into corresponding directories organized by patient_id.

Assumptions:
- The CSV file is at: data/train.csv
- Original images are in: data/images/
- Image files are named as <patient_id>_<image_id>.<ext>, where <ext> is one of:
  .jpg, .jpeg, .png, .bmp, .gif, .tiff
- The script will create (if not existing):
    data/train_images/
    data/validation_images/
    data/test_images/
  Each of these will contain subfolders named by patient_id, into which images are moved.
- Any image not selected for the balanced splits remains in data/images/ untouched.

Usage:
    python balance_and_split_dataset.py
"""

import os
import shutil
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===

# Path to the CSV file
CSV_PATH = "data/csv_file/train.csv"

# Directory containing the original images
ORIGINAL_IMG_DIR = "data/dataset_images/train_images_processed_512"

# Base directories for the new splits
TRAIN_DIR = "data/train_images"
VAL_DIR = "data/validation_images"
TEST_DIR = "data/test_images"

# Ratio of negatives to positives
NEGATIVE_MULTIPLIER = 3

# Desired splits
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

# Random seed for reproducibility
RANDOM_STATE = 42

# Common image extensions to try when locating files
IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]


def find_image_file(patient_id: str, image_id: str, img_dir: str):
    """
    Given a patient_id and image_id (both strings) and the directory to search,
    try to locate a file named <patient_id>_<image_id> with any of the common extensions.
    Return the full path if found, or None otherwise.
    """
    base_name = f"{patient_id}_{image_id}"
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(img_dir, f"{base_name}{ext}")
        if os.path.isfile(candidate):
            return candidate
    # As a fallback, glob for any extension (if casing differs or extension is unexpected)
    pattern = os.path.join(img_dir, f"{base_name}.*")
    matches = glob.glob(pattern)
    for m in matches:
        lower_ext = os.path.splitext(m)[1].lower()
        if lower_ext in IMG_EXTENSIONS:
            return m
    return None


def make_patient_subfolder(base_dir: str, patient_id: str):
    """
    Ensure that a subfolder named patient_id exists inside base_dir.
    Return the path to that subfolder.
    """
    subfolder = os.path.join(base_dir, str(patient_id))
    os.makedirs(subfolder, exist_ok=True)
    return subfolder


def move_images_for_split(df_split: pd.DataFrame, destination_base: str):
    """
    For each row in df_split (which contains 'patient_id' and 'image_id'),
    find the corresponding image file in ORIGINAL_IMG_DIR (which should be named
    <patient_id>_<image_id>.<ext>) and move it into destination_base/<patient_id>/.
    If the image file is not found, print a warning.
    """
    for _, row in df_split.iterrows():
        patient_id = str(row["patient_id"])
        image_id = str(row["image_id"])

        # Attempt to locate the source file
        src_path = find_image_file(patient_id, image_id, ORIGINAL_IMG_DIR)
        if src_path is None:
            print(
                f"Warning: Image file for patient_id={patient_id}, image_id={image_id} "
                f"not found in {ORIGINAL_IMG_DIR}. Skipping."
            )
            continue

        # Ensure patient subfolder exists
        dest_subfolder = make_patient_subfolder(destination_base, patient_id)

        # Construct destination path
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(dest_subfolder, file_name)

        # If a file with the same name already exists, append a counter to avoid collision
        if os.path.exists(dst_path):
            stem, ext = os.path.splitext(file_name)
            counter = 1
            while True:
                candidate_name = f"{stem}_{counter}{ext}"
                candidate_path = os.path.join(dest_subfolder, candidate_name)
                if not os.path.exists(candidate_path):
                    dst_path = candidate_path
                    break
                counter += 1

        # Move the file
        shutil.move(src_path, dst_path)


def main():
    # 1. Load CSV
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Verify required columns exist
    required_cols = {"patient_id", "image_id", "cancer"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"CSV must contain columns: {required_cols}")

    # 2. Filter positive cases (cancer == 1)
    positives = df[df["cancer"] == 1].copy()
    num_positives = len(positives)
    print(f"Total positive (cancer=1) samples: {num_positives}")

    # 3. Filter negative cases (cancer == 0) and sample NEGATIVE_MULTIPLIER * num_positives
    negatives = df[df["cancer"] == 0].copy()
    num_to_sample = NEGATIVE_MULTIPLIER * num_positives
    if num_to_sample > len(negatives):
        raise ValueError(
            f"Not enough negative samples to sample {num_to_sample} (only found {len(negatives)})"
        )
    neg_sampled = negatives.sample(n=num_to_sample, random_state=RANDOM_STATE)
    print(f"Sampled {len(neg_sampled)} negative (cancer=0) samples (ratio {NEGATIVE_MULTIPLIER}:1).")

    # 4. Combine and shuffle
    balanced_df = pd.concat([positives, neg_sampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    total_balanced = len(balanced_df)
    print(f"Total balanced dataset size: {total_balanced}")

    # 5. Split into train (60%), validation (20%), test (20%), stratified by 'cancer'
    # First split: train vs temp (val+test)
    train_df, temp_df = train_test_split(
        balanced_df,
        test_size=(1 - TRAIN_RATIO),
        stratify=balanced_df["cancer"],
        random_state=RANDOM_STATE,
    )
    # Second split: validation vs test from temp
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),  # to split temp into val/test equally
        stratify=temp_df["cancer"],
        random_state=RANDOM_STATE,
    )

    # Print split sizes
    print(f"Train set: {len(train_df)} samples ({len(train_df) / total_balanced:.2%} of total)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df) / total_balanced:.2%} of total)")
    print(f"Test set: {len(test_df)} samples ({len(test_df) / total_balanced:.2%} of total)")

    # 6. Create destination base directories if they don't exist
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    # 7. Move images for each split
    print("Moving training images...")
    move_images_for_split(train_df, TRAIN_DIR)
    print("Moving validation images...")
    move_images_for_split(val_df, VAL_DIR)
    print("Moving test images...")
    move_images_for_split(test_df, TEST_DIR)

    print("Done. Images have been moved. Unused images remain in the original folder.")


if __name__ == "__main__":
    main()
