#!/usr/bin/env python3
"""
Script to select all BI-RADS 4 & 5 images plus a 50% random sample of BI-RADS 1 images,
based on two possible annotation files, and copy them into a train_images directory
organized by patient_id.

It looks for:
  data/csv_file/breast-level_annotations.csv
  data/csv_file/finding_annotations.csv

If either (or both) exist, it loads them, renames their birads column to 'birads',
extracts the numeric part, and concatenates the results.  Any image present in one but
not the other is still included.

Usage:
    python select_images_trainig.py
"""

import os
import shutil
import glob
import pandas as pd
from pathlib import Path

# === CONFIGURATION ===

CSV_DIR              = "csv_file"
BREAST_CSV           = os.path.join(CSV_DIR, "breast-level_annotations.csv")
FINDING_CSV          = os.path.join(CSV_DIR, "finding_annotations.csv")

ORIGINAL_IMG_DIR     = "images_png"
TRAIN_DIR            = "train_images"

PATIENT_COL          = "patient_id"
IMAGE_COL            = "image_id"
BI_RADS_COL          = "birads"

BI_RADS_POSITIVE     = {4, 5}
BI_RADS_NEG_SAMPLE   = 1
NEG_SAMPLE_FRAC      = 0.5
RANDOM_STATE         = 42

IMG_EXTENSIONS       = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}



def find_image_file(patient_id: str, image_id: str, img_dir: str) -> str:
    """
    Given a patient_id and image_id (both strings) and the directory to search,
    try to locate a file named <patient_id>_<image_id> with any of the common extensions.
    Returns the full path to the file if found, or None if not found.
    
    Args:
        patient_id (str): The ID of the patient.
        image_id (str): The ID of the image.
        img_dir (str): The directory to search for the image file.
    Returns:
        str: Full path to the image file if found, otherwise None."""
    base = f"{patient_id}_{image_id}"
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(img_dir, base + ext)
        if os.path.isfile(candidate):
            return candidate
    for m in glob.glob(os.path.join(img_dir, base + ".*")):
        if Path(m).suffix.lower() in IMG_EXTENSIONS:
            return m
    return None

def make_patient_subfolder(base_dir: str, patient_id: str) -> str:
    """
    Create a subfolder for the given patient_id under base_dir.
    If the folder already exists, it will not raise an error.
    Returns the path to the created or existing folder.

    Args:
        base_dir (str): The base directory where patient subfolders will be created.
        patient_id (str): The ID of the patient for whom the subfolder is created.
    Returns:
        str: The path to the created or existing patient subfolder.
    """
    folder = os.path.join(base_dir, str(patient_id))
    os.makedirs(folder, exist_ok=True)
    return folder

def load_and_unify_annotations() -> pd.DataFrame:
    """
    Load and unify annotations from breast-level and finding-level CSV files.
    If both files exist, they are concatenated. If only one exists, it is used.
    The resulting DataFrame contains columns for patient_id, image_id, and birads.
    Returns:
        pd.DataFrame: A DataFrame with columns 'patient_id', 'image_id', and 'birads'.
    Raises:
        FileNotFoundError: If neither CSV file exists.
        KeyError: If the required columns are not present in the CSV files.
    """
    dfs = []
    # breast‐level
    if os.path.isfile(BREAST_CSV):
        df1 = pd.read_csv(BREAST_CSV)
        if not {"study_id", "image_id", "breast_birads"}.issubset(df1.columns):
            raise KeyError("breast-level_annotations.csv must contain study_id,image_id,breast_birads")
        df1 = df1[["study_id", "image_id", "breast_birads"]].rename(
            columns={
                "study_id": PATIENT_COL,
                "image_id": IMAGE_COL,
                "breast_birads": BI_RADS_COL
            }
        )
        dfs.append(df1)
    # finding‐level
    if os.path.isfile(FINDING_CSV):
        df2 = pd.read_csv(FINDING_CSV)
        if not {"study_id", "image_id", "finding_birads"}.issubset(df2.columns):
            raise KeyError("finding_annotations.csv must contain study_id,image_id,finding_birads")
        df2 = df2[["study_id", "image_id", "finding_birads"]].rename(
            columns={
                "study_id": PATIENT_COL,
                "image_id": IMAGE_COL,
                "finding_birads": BI_RADS_COL
            }
        )
        dfs.append(df2)
    if not dfs:
        raise FileNotFoundError("No annotation CSV found in data/csv_file/")
    # concatenate & drop duplicates (keep first)
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=[PATIENT_COL, IMAGE_COL], keep="first")
    # --- FIX: extract numeric part of BI-RADS ---
    # e.g. "BI-RADS 2" → "2" → int 2
    df[BI_RADS_COL] = (
        df[BI_RADS_COL]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    return df

def main():
    df = load_and_unify_annotations()
    print(f"🔍 Loaded {len(df)} annotated images")

    # filter and sample
    df_pos      = df[df[BI_RADS_COL].isin(BI_RADS_POSITIVE)]
    df_b1       = df[df[BI_RADS_COL] == BI_RADS_NEG_SAMPLE]
    df_b1_samp  = df_b1.sample(frac=NEG_SAMPLE_FRAC, random_state=RANDOM_STATE)
    df_sel      = pd.concat([df_pos, df_b1_samp], ignore_index=True)
    print(f"Selected {len(df_pos)} BI-RADS 4/5 + {len(df_b1_samp)} BI-RADS 1 = {len(df_sel)} images")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    missing = 0
    for _, row in df_sel.iterrows():
        pid, iid = str(row[PATIENT_COL]), str(row[IMAGE_COL])
        src = find_image_file(pid, iid, ORIGINAL_IMG_DIR)
        if src is None:
            print(f"⚠️  Missing image file for {pid}_{iid}")
            missing += 1
            continue
        dest_folder = make_patient_subfolder(TRAIN_DIR, pid)
        dst_path = os.path.join(dest_folder, os.path.basename(src))
        # avoid overwrite
        if os.path.exists(dst_path):
            stem, ext = os.path.splitext(os.path.basename(src))
            cnt = 1
            while True:
                cand = os.path.join(dest_folder, f"{stem}_{cnt}{ext}")
                if not os.path.exists(cand):
                    dst_path = cand
                    break
                cnt += 1
        shutil.copy(src, dst_path)

    print(f"✅ Done! Copied {len(df_sel)-missing} images, skipped {missing} missing → {TRAIN_DIR}")

if __name__ == "__main__":
    main()
