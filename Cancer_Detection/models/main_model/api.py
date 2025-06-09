# ===== File: models/main_model/api.py =====
#!/usr/bin/env python3
"""
FastAPI service for ROI extraction + ConvNeXt-small classification.
Save at `models/main_model/api.py`.
Run with: uvicorn models.main_model.api:app --reload --port 8000
"""
import io
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.models import convnext_small

# import ROI extraction
from models.main_model.roi_extract import (
    extract_roi_otsu,
    extract_roi_adaptive,
    extract_roi_canny
)

app = FastAPI(title="ROI Classifier API")

# load models once at startup
def load_models(num_classes, device):
    base = Path(__file__).parent / 'model_weights'
    weights = [base / f"model_{i}_best.pth" for i in range(3)]
    cls = []
    for w in weights:
        model = convnext_small(pretrained=False)
        if hasattr(model, 'classifier'):
            in_f = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Linear(in_f, num_classes)
        else:
            in_f = model.head.in_features
            model.head = torch.nn.Linear(in_f, num_classes)
        sd = torch.load(w, map_location=device)
        st = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in st and v.shape == st[k].shape}
        st.update(matched)
        model.load_state_dict(st)
        model.to(device).eval()
        cls.append(model)
    return cls

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLS_MODELS = load_models(num_classes=1, device=DEVICE)
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read image
    data = await file.read()
    img_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # try ROI methods in order
    methods = [extract_roi_otsu, extract_roi_adaptive, extract_roi_canny]
    roi_bbox = None
    used = None
    for fn in methods:
        bbox, pct = fn(gray)
        if bbox and pct > 0.01:
            roi_bbox = bbox
            used = fn.__name__
            break
    if roi_bbox:
        x0, y0, x1, y1 = roi_bbox
        crop = img[y0:y1, x0:x1]
    else:
        used = 'full_image'
        crop = img

    # classification
    t = TRANSFORM(crop).unsqueeze(0).to(DEVICE)
    preds = []
    with torch.no_grad():
        for idx, m in enumerate(CLS_MODELS):
            logits = m(t)
            probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy().tolist()
            preds.append({"classifier": idx, "probs": probs})

    return JSONResponse({
        "roi_method": used,
        "bbox": roi_bbox,
        "predictions": preds
    })
