%%time

import pydicom
import numpy as np
import cv2
import os
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from pathlib import Path
from pydicom.pixel_data_handlers.util import apply_voi_lut

RESIZE_TO = 1024

!rm -rf train_images_processed_cv2_vl_asp_{1024}
!mkdir train_images_processed_cv2_vl_asp_{1024}


def image_resize(image, width = None, height = None, inter = cv2.INTER_LINEAR):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized


# https://www.kaggle.com/code/tanlikesmath/brain-tumor-radiogenomic-classification-eda/notebook
def dicom_file_to_array(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = apply_voi_lut(dicom.pixel_array, dicom)
    # This is normalization for the data
    data = (data - data.min()) / (data.max() - data.min())
    
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1 - data
    
    height, width = data.shape
    if width > height:
         data = image_resize(data, width = 1024)
    else:
         data = image_resize(data, height = 1024)
    
    data = (data * 255).astype(np.uint8)
    return data

directories = list(Path('../input/rsna-breast-cancer-detection/train_images').iterdir())

def process_directory(directory_path):
    parent_directory = str(directory_path).split('/')[-1]
    # This is to make a new file in the directory. Change this to where you have to store your data
    !mkdir -p train_images_processed_cv2_vl_asp_{1024}/{parent_directory}
    for image_path in directory_path.iterdir():
        processed_ary = dicom_file_to_array(image_path)
        
        cv2.imwrite(
            f'train_images_processed_cv2_vl_asp_{1024}/{parent_directory}/{image_path.stem}.png',
            processed_ary
        )
        
import multiprocessing as mp

with mp.Pool(64) as p:
    # using 64 cores to code the values
    p.map(process_directory, directories)