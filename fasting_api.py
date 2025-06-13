from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
import torch
from torchvision.models import convnext_small
from torchvision import transforms
from PIL import Image
import io
import math

app = FastAPI(title="Cancer Detection API")

# Resolve weight path once, relative to this script
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "Cancer_Detection"/ "main_model" / "model_weights" / "convnext_final.pth"

def load_model():
    if not WEIGHTS_PATH.exists():
        raise RuntimeError(f"Model file not found at {WEIGHTS_PATH}")
    ckpt = torch.load(str(WEIGHTS_PATH), map_location="cpu")
    sd   = ckpt.get("model_state_dict", ckpt)

    head_keys = [k for k,v in sd.items()
                 if v.ndim==2 and v.shape[1]==768 and "classifier" in k]
    num_out   = sd[head_keys[0]].shape[0] if head_keys else 2

    model = convnext_small(pretrained=False, num_classes=num_out)
    filtered = {k:v for k,v in sd.items()
                if k in model.state_dict() and v.shape==model.state_dict()[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model

model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")

    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        # Get malignant probability
        if out.shape[1] == 2:
            p = torch.softmax(out, dim=1)[0,1].item()
        else:
            p = torch.sigmoid(out.squeeze(1)).item()

    # Compute predictive entropy as an uncertainty measure
    # H(p) = -[p log p + (1-p) log (1-p)]
    eps = 1e-12
    p_clamped = max(min(p, 1-eps), eps)
    entropy = - (p_clamped * math.log(p_clamped) +
                 (1-p_clamped) * math.log(1-p_clamped))

    return {
        "probability_malignant": p,
        "uncertainty_entropy":    entropy
    }
