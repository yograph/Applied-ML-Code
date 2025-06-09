#!/usr/bin/env python3
"""
Main runner: apply ROI extraction and classify via ConvNeXt-small.
Save this file at `models/main_model/run_roi_filters.py`.
Usage:
    python run_roi_filters.py --img path/to/image.png --device cpu
"""
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.models import convnext_small

# import ROI methods (local import)
from roi_extract import (
    extract_roi_otsu,
    extract_roi_adaptive,
    extract_roi_canny
)

def load_class_model(weights_path, num_classes, device):
    # instantiate ConvNeXt-small
    model = convnext_small(pretrained=False)
    # adjust classifier head for num_classes
    if hasattr(model, 'classifier'):
        in_f = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Linear(in_f, num_classes)
    else:
        in_f = model.head.in_features
        model.head = torch.nn.Linear(in_f, num_classes)

    # load state dict and filter mismatched keys
    sd = torch.load(weights_path, map_location=device)
    model_state = model.state_dict()
    # keep only matching parameters
    matched = {k: v for k, v in sd.items()
               if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    model.to(device).eval()
    return model


def classify_crop(crop, models, transform, device):
    t = transform(crop).unsqueeze(0).to(device)
    results = []
    with torch.no_grad():
        for m in models:
            logits = m(t)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
            results.append(probs)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--num_classes', type=int, default=1)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))

    base = Path.cwd() / 'models' / 'main_model' / 'model_weights'
    w0 = base / 'model_0_best.pth'
    w1 = base / 'model_1_best.pth'
    w2 = base / 'model_2_best.pth'

    cls_models = [
        load_class_model(str(w0), args.num_classes, device),
        load_class_model(str(w1), args.num_classes, device),
        load_class_model(str(w2), args.num_classes, device),
    ]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    img = cv2.imread(args.img)
    if img is None:
        print(f"❌ Failed to load {args.img}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    methods = [extract_roi_otsu, extract_roi_adaptive, extract_roi_canny]

    roi = None
    for fn in methods:
        xyxy, pct = fn(gray)
        if xyxy and pct > 0.01:
            print(f"Using {fn.__name__}, area {pct:.2%}")
            roi = xyxy
            break
    if roi:
        x0,y0,x1,y1 = roi
        crop = img[y0:y1, x0:x1]
    else:
        print("No valid ROI; using full image")
        crop = img

    probs = classify_crop(crop, cls_models, transform, device)
    for i, pr in enumerate(probs):
        top = int(pr.argmax())
        print(f"Classifier {i}: class {top}, p={pr[top]:.4f}")

if __name__ == '__main__':
    main()