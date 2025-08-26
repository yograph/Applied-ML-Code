"""
This script processes DICOM files into PNG images.
It reads DICOM files, applies photometric corrections, windowing, and resizing,
and saves the processed images in JPEG2000 format.
"""

import os
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import imageio
from tqdm import tqdm
from joblib import Parallel, delayed
import dicomsdl


class DicomProcessor:
    def __init__(self, parent_dir: str, save_dir: str, resize_to: int = 512,
                 n_jobs: int = 16, backend: str = 'loky', prefer: str = 'threads'):
        """
        Initialize the DICOM processor.

        Args:
            parent_dir (str): Base directory containing 'train_images' subfolder.
            save_dir (str): Directory where processed images will be saved.
            resize_to (int): Target size for the longer side of the image.
            n_jobs (int): Number of parallel jobs.
            backend (str): Joblib backend to use.
            prefer (str): Joblib prefer setting.
        """
        self.parent_dir = parent_dir
        self.resize_to = resize_to
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Gather all DICOM files under train_images
        self.image_dir = Path(self.parent_dir) / "train_images"
        self.all_dcm_files = list(self.image_dir.rglob("*.dcm"))
        self.fail_counter = Counter()

        # Parallelization settings
        self.n_jobs = n_jobs
        self.backend = backend
        self.prefer = prefer

    def image_resize(self, image: np.ndarray, width: int = None, height: int = None,
                     inter=cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize an image while maintaining aspect ratio.
        """
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def apply_window(self, image: np.ndarray, window_center: float,
                     window_width: float) -> np.ndarray:
        """
        Apply DICOM windowing to normalize pixel values.
        """
        img = image.copy().astype(np.float32)
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        img = np.clip(img, min_val, max_val)
        img = (img - min_val) / (max_val - min_val)
        return img

    def dicom_file_to_array(self, path: Path) -> np.ndarray:
        """
        Read a DICOM file, apply photometric corrections, windowing, and resize.
        Returns a uint8 image array in range [0, 255].
        """
        dicom = dicomsdl.open(str(path))
        data = dicom.pixelData().astype(np.float32)
        photometric = dicom.getPixelDataInfo()['PhotometricInterpretation']

        # Invert MONOCHROME1
        if photometric == "MONOCHROME1":
            data = data.max() - data

        # Windowing
        try:
            center = dicom.getMeta("0028|1050")
            width = dicom.getMeta("0028|1051")
            if isinstance(center, list):
                center = float(center[0])
            else:
                center = float(center)
            if isinstance(width, list):
                width = float(width[0])
            else:
                width = float(width)
            data = self.apply_window(data, center, width)
        except Exception:
            # Fallback to min-max normalization
            data = (data - data.min()) / (data.max() - data.min())

        # Resize to target
        h, w = data.shape
        if w > h:
            data = self.image_resize(data, width=self.resize_to)
        else:
            data = self.image_resize(data, height=self.resize_to)

        # Scale to 0-255
        return (data * 255).astype(np.uint8)

    def process_file(self, path: Path) -> None:
        """
        Process a single DICOM file and save as JPEG2000.
        """
        try:
            parent_folder = path.parent.name
            save_subdir = os.path.join(self.save_dir, parent_folder)
            os.makedirs(save_subdir, exist_ok=True)

            processed_img = self.dicom_file_to_array(path)
            save_path = os.path.join(save_subdir, f"{path.stem}.jp2")
            imageio.imwrite(save_path, processed_img, format='JP2')
        except Exception as e:
            print(f"[ERROR] Failed: {path} — {e}")
            self.fail_counter["fail"] += 1

    def run(self) -> None:
        """
        Run the processing pipeline over all DICOM files in parallel.
        """
        Parallel(n_jobs=self.n_jobs, backend=self.backend, prefer=self.prefer)(
            delayed(self.process_file)(path) for path in tqdm(self.all_dcm_files, total=len(self.all_dcm_files))
        )

        print(f"✅ Done! Processed {len(self.all_dcm_files)} images.")
        print(f"❌ Failed: {self.fail_counter['fail']}")


if __name__ == "__main__":
    RESIZE_TO = 512
    PARENT_DIR = "/kaggle/input/rsna-breast-cancer-detection"
    SAVE_DIR = f"/kaggle/working/train_image_processed_jp2000_{RESIZE_TO}"

    processor = DicomProcessor(
        parent_dir=PARENT_DIR,
        save_dir=SAVE_DIR,
        resize_to=RESIZE_TO,
        n_jobs=16,
    )
    processor.run()
