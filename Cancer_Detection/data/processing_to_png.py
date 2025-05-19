"""
This chode changed dicom to PNG


"""
import os
from pathlib import Path
import shutil
import pydicom
import numpy as np
import cv2
from multiprocessing import Pool
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut

class DicomToPngConverter:
    """
    Converts a directory of DICOM images (organized in subdirectories) to PNG files,
    resizing them to a specified resolution and saving to an output directory.
    """

    def __init__(self, input_dir, output_dir, resize_to=1024, num_workers=2):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.resize_to = resize_to
        self.num_workers = num_workers

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_LINEAR):
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

    def dicom_file_to_array(self, path):
        dicom = pydicom.dcmread(str(path))
        data = apply_voi_lut(dicom.pixel_array, dicom)
        data = (data - data.min()) / (data.max() - data.min())
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = 1 - data
        height, width = data.shape
        if width > height:
            data = self.image_resize(data, width=self.resize_to)
        else:
            data = self.image_resize(data, height=self.resize_to)
        return (data * 255).astype(np.uint8)

    def process_directory(self, directory_path):
        """
        Processes all DICOM files in a single directory and saves them as PNGs.
        """
        parent = Path(directory_path).name
        out_dir = self.output_dir / parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for image_path in Path(directory_path).iterdir():
            try:
                arr = self.dicom_file_to_array(image_path)
                out_path = out_dir / f"{image_path.stem}.png"
                cv2.imwrite(str(out_path), arr)
            except Exception as e:
                print(f"Failed to process {image_path}: {e}")

    def run(self):
        """
        Removes any existing output directory, recreates it, and processes all DICOM subdirectories.
        """
        # Equivalent to `rm -rf` on the output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        # Recreate the (now empty) output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        directories = [d for d in self.input_dir.iterdir() if d.is_dir()]
        if self.num_workers and self.num_workers > 1:
            with Pool(self.num_workers) as pool:
                list(tqdm(pool.imap(self.process_directory, directories),
                          total=len(directories),
                          desc="Processing directories"))
        else:
            for d in tqdm(directories, desc="Processing directories"):
                self.process_directory(d)


