#!/usr/bin/env python3
"""
Main runner: apply ROI extraction and classify via ConvNeXt-small ensemble.
Usage:
    python models/main_model/run_roi_filters.py --img path/to/image.png --device mps
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms, models
from PIL import Image

# Local imports (ensure this file is in the same folder as roi_extract.py)
from roi_extract import (
    extract_roi_otsu,
    extract_roi_adaptive,
    extract_roi_canny
)

# ─── Adjust these paths as needed ─────────────────────────────────────────
# Folder where your saved model weights live:
#   models/main_model/model_weights/model_0_weights.pth, etc.
WEIGHTS_DIR = Path(__file__).resolve().parent / "model_weights"

# Number of classes must be 2 (no_cancer vs. cancer)
NUM_CLASSES = 2
# ─────────────────────────────────────────────────────────────────────────────

def load_class_model(weights_path, num_classes, device):
    """
    Instantiate a ConvNeXt-small, adjust its head to `num_classes`, then load matching weights.
    """
    # 1) Load official convnext_small architecture (no pre-trained weights)
    model = models.convnext_small(pretrained=False)

    # 2) Adjust the classifier head for `num_classes`
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, num_classes)

    # 3) Load saved state_dict, but only keep matching keys
    sd = torch.load(weights_path, map_location=device)
    model_state = model.state_dict()
    matched = {k: v for k, v in sd.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)

    model.to(device).eval()
    return model


def classify_crop(crop, models, transform, device):
    """
    Given a NumPy‐array crop (H×W×C, BGR), run each model in `models` on it,
    then return a list of per‐model probability vectors (length=num_classes).
    """
    # Convert BGR→RGB, then PIL→Tensor→Normalized
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = transform(pil).unsqueeze(0).to(device)  # shape [1, 3, 224, 224]

    all_probs = []
    with torch.no_grad():
        for m in models:
            logits = m(t)  # shape [1, num_classes]
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    return all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help="Path to a single image (PNG/JPG) to classify via ROI")
    parser.add_argument('--device', default='', help="Device to use: 'cuda', 'mps', or 'cpu'")
    args = parser.parse_args()

    # 1) Pick device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}\n")

    # 2) Build the three ConvNeXt‐small models
    w0 = WEIGHTS_DIR / "model_0_weights.pth"
    w1 = WEIGHTS_DIR / "model_1_weights.pth"
    w2 = WEIGHTS_DIR / "model_2_weights.pth"
    for w in (w0, w1, w2):
        if not w.exists():
            print(f"❌ Weight file not found: {w}")
            return

    print("Loading models…")
    cls_models = [
        load_class_model(str(w0), NUM_CLASSES, device),
        load_class_model(str(w1), NUM_CLASSES, device),
        load_class_model(str(w2), NUM_CLASSES, device)
    ]

    # 3) Image transforms (must match training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 4) Load the full image (BGR)
    img_path = Path(args.img)
    if not img_path.exists():
        print(f"❌ Failed to find image: {img_path}")
        return
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"❌ Failed to load image: {img_path}")
        return

    # 5) Convert to grayscale and run ROI extractors
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    methods = [extract_roi_otsu, extract_roi_adaptive, extract_roi_canny]

    roi_box = None
    for fn in methods:
        box, area_pct = fn(gray)
        if box is not None and area_pct > 0.01:
            print(f"→ Using ROI from {fn.__name__}, area = {area_pct:.2%}")
            roi_box = box
            break

    if roi_box:
        x0, y0, x1, y1 = roi_box
        crop = img_bgr[y0:y1, x0:x1]
    else:
        print("→ No valid ROI found; using full image")
        crop = img_bgr

    # 6) Classify the crop with all three models
    all_probs = classify_crop(crop, cls_models, transform, device)
    # all_probs is a list of three arrays, each shape [2], e.g. [ [p0, p1], [p0, p1], [p0, p1] ]

    # 7) Average the probabilities across the 3 models
    avg_prob = np.mean(np.stack(all_probs, axis=0), axis=0)  # shape [2]

    # 8) Print per‐model results and final average
    for i, probs in enumerate(all_probs):
        top_class = int(np.argmax(probs))
        print(f"Model {i} → class {top_class} (p0={probs[0]:.4f}, p1={probs[1]:.4f})")

    print("\n→ Final (averaged) probabilities:")
    print(f"    no_cancer (class 0) = {avg_prob[0]:.4f}")
    print(f"       cancer (class 1) = {avg_prob[1]:.4f}")

    final_class = int(np.argmax(avg_prob))
    print(f"\n❯ Final predicted class = {final_class}  (p={avg_prob[final_class]:.4f})")


if __name__ == '__main__':
    main()
