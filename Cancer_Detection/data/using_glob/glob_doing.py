import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

KAGGLE = True

class GLOB_USAGE:
    OUTPUT_RECT_DIR  = "output/rectangles"
    OUTPUT_FINAL_DIR = "output/final"

    def __init__(self):
        if KAGGLE:
            pattern = "../input/rsna-mammography-images-as-pngs/images_as_pngs_1024/train_images_processed_1024/*/*"
        else:
            pattern = "../Cancer_Detection/data/dataset_images/rsna-mammography-images-as-pngs/images_as_pngs_1024/train_images_processed_1024/*/*"
        self.train_images = glob.glob(pattern)
        self.crops = []

        os.makedirs(self.OUTPUT_RECT_DIR,  exist_ok=True)
        os.makedirs(self.OUTPUT_FINAL_DIR, exist_ok=True)

    def crop_coords(self, img):
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(breast_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)
        return cv2.boundingRect(cnt)

    def truncation_normalization(self, img):
        non_zero = img[img != 0]
        # This is to ensure that images with very shiny 
        Pmin, Pmax = np.percentile(non_zero, 5), np.percentile(non_zero, 99)
        clipped = np.clip(img, Pmin, Pmax)
        norm = (clipped - Pmin) / (Pmax - Pmin)
        norm[img == 0] = 0
        return norm

    def clahe(self, img, clip):
        
        clahe = cv2.createCLAHE(clipLimit=clip)
        return clahe.apply((img * 255).astype(np.uint8))

    def save_with_rectangles(self):
        for idx, img_path in enumerate(self.train_images):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            x_coords, y_coords, width, height = self.crop_coords(img)

            canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(canvas, (x_coords, y_coords), (x_coords + width, y_coords + height), (0, 0, 255), 2)

            out_path = os.path.join(self.OUTPUT_RECT_DIR, f"rect_{idx:03d}.png")
            cv2.imwrite(out_path, canvas)

            crop = img[y_coords : y_coords + height, x_coords : x_coords + width]
            self.crops.append(crop)

    def save_final_images(self, img_size=512):
        for idx, crop in enumerate(self.crops):
            norm = self.truncation_normalization(crop)
            cl1  = self.clahe(norm, 1.0)
            cl2  = self.clahe(norm, 2.0)

            merged = cv2.merge((
                (norm * 255).astype(np.uint8),
                cl1,
                cl2
            ))
            resized = cv2.resize(merged, (img_size, img_size))

            out_path = os.path.join(self.OUTPUT_FINAL_DIR, f"final_{idx:03d}.png")
            cv2.imwrite(out_path, resized)

    def show_image(self, idx, kind="rect"):
        """
        Display one of the saved images in grayscale.
        idx  : integer index (0-based) matching the filename suffix
        kind : "rect" to show rect_* or "final" to show final_* 
        """
        if kind == "rect":
            path = os.path.join(self.OUTPUT_RECT_DIR,  f"rect_{idx:03d}.png")
        else:
            path = os.path.join(self.OUTPUT_FINAL_DIR, f"final_{idx:03d}.png")

        # load as grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="bone")   # or 'gray' if you prefer
        plt.axis("off")
        plt.show()

