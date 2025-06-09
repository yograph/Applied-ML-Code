#!/usr/bin/env python3
import os
import subprocess
import pandas as pd
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import imageio
from tqdm import tqdm
from joblib import Parallel, delayed
import dicomsdl
import shutil

# -------------------------
# STEP 1: Filter BI-RADS 4 & 5 Study and Image IDs
# -------------------------

def filter_birads(csv_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Load annotations
    df1 = pd.read_csv(os.path.join(csv_dir, 'breast-level_annotations.csv'))
    df2 = pd.read_csv(os.path.join(csv_dir, 'finding_annotations.csv'))

    # Filter BI-RADS 4 & 5 (preserve duplicates)
    df1_filt = df1[df1['breast_birads'].isin(['BI-RADS 4', 'BI-RADS 5'])]
    df2_filt = df2[df2['finding_birads'].isin(['BI-RADS 4', 'BI-RADS 5'])]

    # Write study_id and image_id lists (including duplicates)
    path1 = os.path.join(out_dir, 'study_image_br4_5.txt')
    df1_filt[['study_id', 'image_id']].to_csv(path1, index=False, header=False)

    path2 = os.path.join(out_dir, 'extra_study_image_br4_5.txt')
    df2_filt[['study_id', 'image_id']].to_csv(path2, index=False, header=False)

    return [path1, path2]

# -------------------------
# STEP 2: Download Selected DICOMs (ensure all 4 views exist)
# -------------------------

def download_dicoms(list_files: list, download_dir: str, user: str, password: str):
    os.makedirs(download_dir, exist_ok=True)
    BASE_URL = 'https://physionet.org/files/vindr-mammo/1.0.0/images'
    study_image_pairs = []
    # read both files, each line: study_id,image_id
    for lf in list_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    study_image_pairs.append((parts[0], parts[1]))

    # group image_ids by study
    study_map = {}
    for sid, img in study_image_pairs:
        study_map.setdefault(sid, set()).add(img)

    for sid, images in study_map.items():
        study_dir = Path(download_dir) / sid
        existing = {p.stem for p in study_dir.glob('*.dicom')} if study_dir.exists() else set()
        missing = images - existing
        if not missing:
            print(f"All requested DICOMs present for study {sid}, skipping.")
            continue
        # download entire study folder (wget will skip existing ones)
        url = f"{BASE_URL}/{sid}/"
        cmd = [
            'wget', '-r', '-l', '1', '-np', '-c', '-nH',
            '--cut-dirs=3', '-A', '*.dicom',
            '-P', download_dir,
            '--user', user,
            '--password', password,
            url
        ]
        print(f"Downloading DICOMs for study {sid}, missing {len(missing)} files → {download_dir}")
        subprocess.run(cmd, check=True)

# -------------------------
# STEP 3: Convert DICOMs to PNG (skip existing files)
# -------------------------

class DicomToPNGConverter:
    def __init__(self, parent_dir, save_dir, resize_to=512,
                 n_jobs=16, backend='loky', prefer='threads'):
        self.parent_dir = parent_dir
        self.save_dir = save_dir
        self.resize_to = resize_to
        os.makedirs(self.save_dir, exist_ok=True)
        self.all_files = list(Path(self.parent_dir).rglob('*.dicom')) + \
                         list(Path(self.parent_dir).rglob('*.dcm'))
        self.fail_counter = Counter()
        self.n_jobs = n_jobs
        self.backend = backend
        self.prefer = prefer

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_LINEAR):
        h, w = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / h
            dim = (int(w * r), height)
        else:
            r = width / w
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=inter)

    def apply_window(self, img, center, width):
        mn = center - width/2
        mx = center + width/2
        img = np.clip(img, mn, mx)
        return (img - mn) / (mx - mn)

    def dicom_to_array(self, path: Path):
        dcm = dicomsdl.open(str(path))
        data = dcm.pixelData().astype(np.float32)
        photo = dcm.getPixelDataInfo()['PhotometricInterpretation']
        if photo == 'MONOCHROME1':
            data = data.max() - data
        try:
            cen = dcm.getMeta('0028|1050')
            wid = dcm.getMeta('0028|1051')
            cen = float(cen[0]) if isinstance(cen, (list,tuple)) else float(cen)
            wid = float(wid[0]) if isinstance(wid, (list,tuple)) else float(wid)
            data = self.apply_window(data, cen, wid)
        except Exception:
            data = (data - data.min()) / (data.max() - data.min())
        h, w = data.shape
        if w > h:
            data = self.image_resize(data, width=self.resize_to)
        else:
            data = self.image_resize(data, height=self.resize_to)
        return (data * 255).astype(np.uint8)

    def process_file(self, path: Path):
        try:
            rel = path.relative_to(self.parent_dir)
            out_path = Path(self.save_dir) / rel.with_suffix('.png')
            if out_path.exists():
                return
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img = self.dicom_to_array(path)
            imageio.imwrite(str(out_path), img)
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            self.fail_counter['fail'] += 1

    def run(self):
        Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)(
            delayed(self.process_file)(p) for p in tqdm(self.all_files)
        )
        print(f"✅ Processed {len(self.all_files)} files, failed {self.fail_counter['fail']}")

# -------------------------
# MAIN EXECUTION
# -------------------------
if __name__ == '__main__':
    CSV_DIR = 'csv_file'
    TXT_DIR = 'text_files'
    DICOM_DIR = os.path.join(os.getcwd(), '..', 'DICOMFILES')
    PNG_DIR = os.path.join(os.getcwd(), '..', 'PNGFILES')
    TRAIN_DIR = os.path.join(os.getcwd(), '..', 'train_images')
    USER = 'yfarahat'
    PASSWORD = 'Dubai$123'

    # Step 1: Filter and save study and image IDs (preserve all duplicates)
    list_paths = filter_birads(CSV_DIR, TXT_DIR)

    # Step 2: Download selected DICOMs (ensure requested images exist)
    download_dicoms(list_paths, DICOM_DIR, USER, PASSWORD)

    # Step 3: Convert to PNG (skip existing files)
    converter = DicomToPNGConverter(parent_dir=DICOM_DIR, save_dir=PNG_DIR)
    converter.run()

    # Step 4: Move PNGFILES to train_images
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    shutil.move(PNG_DIR, TRAIN_DIR)
    print(f"All PNGs organized under: {TRAIN_DIR}")
