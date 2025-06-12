#!/usr/bin/env python3
"""
Training pipeline for breast cancer detection using ConvNeXt Small,
with focal-plus-class-weighted loss, 1:1 class balance via WeightedRandomSampler,
advanced augmentations, Apple Silicon MPS, progress bars, and CSV of test predictions,
plus checkpointing to resume training.
"""

import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Small_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score
)
from tqdm.auto import tqdm

# --- Loss: Focal + Class Weight ---
class FocalLossWithClassWeight(nn.Module):
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 class_weight: torch.Tensor = None,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weight = class_weight  # tensor([w0, w1])
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        mod = (1 - p_t) ** self.gamma
        loss = alpha_t * mod * ce
        if self.class_weight is not None:
            w_t = self.class_weight[1] * targets + self.class_weight[0] * (1 - targets)
            loss = loss * w_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

# --- Image Processing Functions ---
def crop_coords(img):
    """Get coordinates for cropping breast region from image"""
    try:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(breast_mask.astype(np.uint8),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:  # if no contours found, return full image coordinates
            return (0, 0, img.shape[1], img.shape[0])
        cnt = max(cnts, key=cv2.contourArea)
        return cv2.boundingRect(cnt)
    except:
        return (0, 0, img.shape[1], img.shape[0])  # return full image on error

def truncation_normalization(img):
    """Normalize image with truncation and handle edge cases"""
    try:
        non_zero = img[img != 0]
        if len(non_zero) == 0:  # if image is all zeros, return as-is
            return img.astype(np.float32)
        
        Pmin, Pmax = np.percentile(non_zero, 5), np.percentile(non_zero, 99)
        if Pmax - Pmin < 1e-6:  # avoid division by zero
            norm = np.zeros_like(img, dtype=np.float32)
            norm[img != 0] = img[img != 0]  # keep original values
        else:
            clipped = np.clip(img, Pmin, Pmax)
            norm = (clipped - Pmin) / (Pmax - Pmin)
            norm = np.nan_to_num(norm)  # convert any NaN to 0
        
        norm[img == 0] = 0
        return norm
    except:
        return img.astype(np.float32)  # return original on error

def clahe(img, clip):
    """Apply CLAHE with safety checks"""
    try:
        if img.size == 0:
            return img.astype(np.uint8)  # return original if empty
        
        # Ensure input is in correct range and type
        img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip)
        return clahe.apply(img_uint8)
    except:
        return (img * 255).astype(np.uint8)  # return scaled original on error

def process_image(img_path, img_size=512):
    """Full image processing pipeline that falls back to original image on errors"""
    try:
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        
        original_img = img.copy()
        
        # Try cropping
        try:
            x, y, w, h = crop_coords(img)
            crop = img[y:y+h, x:x+w]
        except:
            crop = img  # fall back to full image
        
        # Try normalization
        try:
            norm = truncation_normalization(crop)
        except:
            norm = crop.astype(np.float32) / 255.0  # simple scaling
        
        # Try CLAHE
        try:
            cl1 = clahe(norm, 1.0)
            cl2 = clahe(norm, 2.0)
        except:
            cl1 = cl2 = (norm * 255).astype(np.uint8)
        
        # Merge channels
        try:
            merged = cv2.merge([
                np.clip(norm * 255, 0, 255).astype(np.uint8),
                cl1,
                cl2
            ])
        except:
            # If merging fails, just use the normalized image for all channels
            merged = cv2.merge([
                np.clip(norm * 255, 0, 255).astype(np.uint8),
                np.clip(norm * 255, 0, 255).astype(np.uint8),
                np.clip(norm * 255, 0, 255).astype(np.uint8)
            ])
        
        # Resize
        try:
            resized = cv2.resize(merged, (img_size, img_size))
            return Image.fromarray(resized)
        except:
            # If resize fails, return original image resized
            resized = cv2.resize(original_img, (img_size, img_size))
            return Image.fromarray(cv2.merge([resized, resized, resized]))
            
    except Exception as e:
        # If all else fails, return original image resized to target dimensions
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # As last resort, return blank image
            blank = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            return Image.fromarray(blank)
        resized = cv2.resize(img, (img_size, img_size))
        return Image.fromarray(cv2.merge([resized, resized, resized]))
# --- Data and Labels ---
def get_data_paths_and_labels():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, 'data')
    csv_dir = os.path.join(data_dir, 'csv_file')
    ds1 = os.path.join(data_dir, 'images_png')
    ds2 = os.path.join(data_dir, 'train_images_processed_512')

    df1 = pd.read_csv(os.path.join(csv_dir, 'breast-level_annotations.csv'))
    df2 = pd.read_csv(os.path.join(csv_dir, 'train.csv'))

    pths, lbls = [], []
    for _, r in df1.iterrows():
        try:
            lvl = int(str(r['breast_birads']).split()[-1])
        except:
            continue
        lab = 1 if lvl >= 4 else 0
        fn = f"{r['study_id']}_{r['image_id']}.png"
        fp = os.path.join(ds1, fn)
        if os.path.exists(fp):
            pths.append(fp)
            lbls.append(lab)

    for _, r in df2.iterrows():
        fn = f"{int(r['patient_id'])}_{int(r['image_id'])}.png"
        fp = os.path.join(ds2, fn)
        if os.path.exists(fp):
            pths.append(fp)
            lbls.append(int(r['cancer']))

    if not pths:
        raise RuntimeError("No images found.")
    return pths, np.array(lbls, dtype=np.int64)

# --- Dataset ---
class BreastCancerDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = process_image(self.paths[i])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.float32)

# --- Main ---
def main():
    # checkpoint file
    ckpt_path = 'checkpoint.pth'

    # --- Load data ---
    all_paths, all_labels = get_data_paths_and_labels()
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        all_paths, all_labels,
        test_size=0.3, stratify=all_labels, random_state=42
    )
    vl_p, tst_p, vl_l, tst_l = train_test_split(
        tmp_p, tmp_l,
        test_size=0.5, stratify=tmp_l, random_state=42
    )

    # compute class weights for loss
    cnt_pos = tr_l.sum()
    cnt_neg = len(tr_l) - cnt_pos
    w0 = cnt_pos / len(tr_l)
    w1 = cnt_neg / len(tr_l)
    cw = torch.tensor([w0, w1], dtype=torch.float32)

    # sampler for 1:1 balance
    cls_counts = np.bincount(tr_l)
    weights = 1.0 / cls_counts[tr_l]
    samp = WeightedRandomSampler(weights,
                                 num_samples=2 * int(cls_counts.max()),
                                 replacement=True)

    # augmentations
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # dataloaders
    dev = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    ds_tr = BreastCancerDataset(tr_p, tr_l, transform=train_tf)
    ds_vl = BreastCancerDataset(vl_p, vl_l, transform=val_tf)
    ds_ts = BreastCancerDataset(tst_p, tst_l, transform=val_tf)
    ld_tr = DataLoader(ds_tr, batch_size=32, sampler=samp, num_workers=4)
    ld_vl = DataLoader(ds_vl, batch_size=32, shuffle=False, num_workers=4)
    ld_ts = DataLoader(ds_ts, batch_size=32, shuffle=False, num_workers=4)

    # model, loss, optimizer
    model = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, 1)
    model.to(dev)

    crit = FocalLossWithClassWeight(alpha=0.25,
                                    gamma=2.0,
                                    class_weight=cw.to(dev))
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-4)

    start_epoch = 1
    best_loss = float('inf')

    # --- Resume from checkpoint if available ---
    if os.path.isfile(ckpt_path):
        print(f"⏳ Loading checkpoint '{ckpt_path}'")
        checkpoint = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss   = checkpoint.get('best_loss', best_loss)
        print(f"✅ Resumed: epoch {start_epoch}, best_loss={best_loss:.4f}")

    # --- Training loop ---
    num_epochs = 6 # 5 overfit at 5
    for ep in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(ld_tr, desc=f"Train {ep}", unit="batch"):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x).view(-1)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(ld_tr.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in tqdm(ld_vl, desc=f" Val {ep}", unit="batch"):
                x, y = x.to(dev), y.to(dev)
                logits = model(x).view(-1)
                loss = crit(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y.long()).sum().item()
                total   += y.size(0)
        val_loss /= len(ld_vl.dataset)
        val_acc = correct / total
        print(f"E{ep}: Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}")

        # update best
        if val_loss < best_loss:
            best_loss = val_loss

        # save checkpoint
        ckpt = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_loss': best_loss
        }
        torch.save(ckpt, ckpt_path)
        print(f"💾 Saved checkpoint (epoch {ep})")

    # --- Testing & metrics ---
    model.eval()
    preds, probs, trues, paths = [], [], [], []
    for x, y in tqdm(ld_ts, desc="Test", unit="batch"):
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad():
            logits = model(x).view(-1)
        p = torch.sigmoid(logits)
        preds.extend((p >= 0.5).long().cpu().numpy())
        probs.extend(p.cpu().numpy())
        trues.extend(y.long().cpu().numpy())
        paths.extend(ds_ts.paths[:len(y)])

    # compute metrics
    cm   = confusion_matrix(trues, preds)
    roc  = roc_auc_score(trues, probs)
    prec = precision_score(trues, preds)
    rec  = recall_score(trues, preds)
    f1   = f1_score(trues, preds)
    mcc  = matthews_corrcoef(trues, preds)
    ap   = average_precision_score(trues, probs)
    # uncertainty as entropy
    eps = 1e-6
    unc = -np.array(probs)*np.log(np.array(probs)+eps) \
          - (1-np.array(probs))*np.log((1-np.array(probs))+eps)

    # save predictions to CSV
    df = pd.DataFrame({
        'path': paths,
        'true': trues,
        'pred': preds,
        'prob': probs,
        'uncertainty': unc
    })
    df.to_csv('test_predictions.csv', index=False)

    # print summary
    print('Confusion Matrix:\n', cm)
    print(f'ROC AUC: {roc:.4f}  PR AUC: {ap:.4f}')
    print(f'Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}  MCC: {mcc:.4f}')

    # save final model
    torch.save(model.state_dict(), 'convnext_final.pth')

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()