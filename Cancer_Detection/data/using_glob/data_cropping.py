import cv2
import numpy as np
from PIL import Image

def crop_coords(img):
    """Get coordinates for cropping breast region from image"""
    try:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(breast_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return (0, 0, img.shape[1], img.shape[0])
        cnt = max(cnts, key=cv2.contourArea)
        return cv2.boundingRect(cnt)
    except:
        return (0, 0, img.shape[1], img.shape[0])

def truncation_normalization(img):
    """Normalize image with truncation and handle edge cases"""
    try:
        non_zero = img[img != 0]
        if len(non_zero) == 0:
            return img.astype(np.float32)
        Pmin, Pmax = np.percentile(non_zero, 5), np.percentile(non_zero, 99)
        if Pmax - Pmin < 1e-6:
            norm = np.zeros_like(img, dtype=np.float32)
            norm[img != 0] = img[img != 0]
        else:
            clipped = np.clip(img, Pmin, Pmax)
            norm = (clipped - Pmin) / (Pmax - Pmin)
            norm = np.nan_to_num(norm)
        norm[img == 0] = 0
        return norm
    except:
        return img.astype(np.float32)

def clahe(img, clip):
    try:
        if img.size == 0:
            return img.astype(np.uint8)
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        _clahe = cv2.createCLAHE(clipLimit=clip)
        return _clahe.apply(img_uint8)
    except:
        return (img * 255).astype(np.uint8)

def process_image(img_path, img_size=512):
    """Full image processing pipeline that falls back to original image on errors"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        original_img = img.copy()
        try:
            x, y, w, h = crop_coords(img)
            crop = img[y:y+h, x:x+w]
        except:
            crop = img
        try:
            norm = truncation_normalization(crop)
        except:
            norm = crop.astype(np.float32) / 255.0
        try:
            cl1 = clahe(norm, 1.0)
            cl2 = clahe(norm, 2.0)
        except:
            cl1 = cl2 = (norm * 255).astype(np.uint8)
        try:
            merged = cv2.merge([
                np.clip(norm * 255, 0, 255).astype(np.uint8),
                cl1, cl2
            ])
        except:
            merged = cv2.merge([
                np.clip(norm * 255, 0, 255).astype(np.uint8),
                np.clip(norm * 255, 0, 255).astype(np.uint8),
                np.clip(norm * 255, 0, 255).astype(np.uint8)
            ])
        try:
            resized = cv2.resize(merged, (img_size, img_size))
            return Image.fromarray(resized)
        except:
            resized = cv2.resize(original_img, (img_size, img_size))
            return Image.fromarray(cv2.merge([resized, resized, resized]))
    except:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            blank = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            return Image.fromarray(blank)
        resized = cv2.resize(img, (img_size, img_size))
        return Image.fromarray(cv2.merge([resized, resized, resized]))
