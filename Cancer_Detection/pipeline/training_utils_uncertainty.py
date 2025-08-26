#!/usr/bin/env python3
"""
ConvNeXt-Small breast-cancer detection with:
- PCA Lighting aug (utils.PCALighting)
- Optional PCA projection head (utils.build_pca_linear + utils.gather_convnext_features)
- Hyperparam tuning for focal alpha/gamma
- Validation-threshold optimisation via F-beta or MCC
- Uncertainty via MC-Dropout and/or TTA (entropy, expected entropy, BALD MI, variance)
- Grad-CAM overlays + saliency maps
- CSV with predictions & uncertainty
- Apple Silicon MPS support
- Base directory for data outside this script: --base-dir /path/to/project_root  (expects {base}/data/...)

You can still use your own:
  from Cancer_Detection.data.using_glob import process_image
  from Cancer_Detection.features import FocalLossWithClassWeight
If those aren’t available, local fallbacks are used.
"""

import os
import argparse
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Small_Weights
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, average_precision_score
)
from tqdm.auto import tqdm

# ---- Prefer your project utilities if available ----
try:
    from Cancer_Detection.data.using_glob import process_image as _custom_process_image
except Exception:
    _custom_process_image = None

try:
    from Cancer_Detection.features import FocalLossWithClassWeight as _custom_focal
except Exception:
    _custom_focal = None

from features import (
    PCALighting, gather_convnext_features, build_pca_linear,
    predictive_stats, tta_batch, GradCAM, upsample_like, overlay_heatmap, saliency_map
)

# -------------------- Reproducibility --------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -------------------- Local fallbacks --------------------
def _fallback_process_image(img_path, img_size=512):
    """Simple robust grayscale->3ch pipeline (fallback)."""
    import cv2, numpy as np
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Image.fromarray(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    img = cv2.resize(img, (img_size, img_size))
    return Image.fromarray(cv2.merge([img, img, img]))

class _FallbackFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.class_weight = class_weight
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        ce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        mod = (1 - p_t) ** self.gamma
        loss = alpha_t * mod * ce
        if self.class_weight is not None:
            w_t = self.class_weight[1] * targets + self.class_weight[0] * (1 - targets)
            loss = loss * w_t
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

FocalLossWithClassWeight = _custom_focal if _custom_focal is not None else _FallbackFocalLoss

def process_image(img_path):
    return _custom_process_image(img_path) if _custom_process_image is not None else _fallback_process_image(img_path)

# -------------------- Data -------------------------------
def get_data_paths_and_labels(base_dir) -> tuple:
    """
    Expects:
      {base_dir}/data/csv_file/breast-level_annotations.csv
      {base_dir}/data/csv_file/train.csv
      {base_dir}/data/images_png/{study_id}_{image_id}.png
      {base_dir}/data/train_images_processed_512/{patient_id}_{image_id}.png
    """
    data_dir = os.path.join(base_dir, 'data')
    csv_dir = os.path.join(data_dir, 'csv_file')
    ds1 = os.path.join(data_dir, 'images_png')
    ds2 = os.path.join(data_dir, 'train_images_processed_512')

    df1 = pd.read_csv(os.path.join(csv_dir, 'breast-level_annotations.csv'))
    df2 = pd.read_csv(os.path.join(csv_dir, 'train.csv'))

    pths, lbls = [], []
    # dataset 1 (BIRADS)
    for _, r in df1.iterrows():
        try:
            lvl = int(str(r['breast_birads']).split()[-1])
        except:
            continue
        lab = 1 if lvl >= 4 else 0
        fn = f"{r['study_id']}_{r['image_id']}.png"
        fp = os.path.join(ds1, fn)
        if os.path.exists(fp):
            pths.append(fp); lbls.append(lab)

    # dataset 2 (cancer)
    for _, r in df2.iterrows():
        fn = f"{int(r['patient_id'])}_{int(r['image_id'])}.png"
        fp = os.path.join(ds2, fn)
        if os.path.exists(fp):
            pths.append(fp); lbls.append(int(r['cancer']))

    if not pths:
        raise RuntimeError("No images found under base_dir='%s'." % base_dir)
    return pths, np.array(lbls, dtype=np.int64)

class BreastCancerDataset(Dataset):
    def __init__(self, paths, labels, transform=None) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = process_image(self.paths[i])
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.labels[i], dtype=torch.float32), self.paths[i]

# -------------------- Model ------------------------------
def build_model(dropout: float = 0.0, pca_layer: nn.Linear = None, pca_dim: int = 128):
    model = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
    if pca_layer is not None:
        new_classifier = nn.Sequential(
            model.classifier[0],      # LayerNorm2d
            nn.Flatten(1),
            pca_layer,                # frozen Linear(768 -> pca_dim)
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(pca_dim, 1),
        )
        model.classifier = new_classifier
    else:
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(in_f, 1)
        )
    return model

# -------------------- Training / Eval helpers ------------
def make_loaders(tr_paths, tr_labels, vl_paths, vl_labels, ts_paths, ts_labels,
                 batch_size, mean, std, pca_lighting=False, num_workers=4):
    train_tf_list = [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
    ]
    if pca_lighting:
        train_tf_list.append(PCALighting(alphastd=0.1))
    train_tf_list.extend([
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.5),
    ])
    train_tf = transforms.Compose(train_tf_list)

    val_tf = transforms.Compose([
        transforms.Resize(236),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds_tr = BreastCancerDataset(tr_paths, tr_labels, transform=train_tf)
    ds_vl = BreastCancerDataset(vl_paths, vl_labels, transform=val_tf)
    ds_ts = BreastCancerDataset(ts_paths, ts_labels, transform=val_tf)

    cls_counts = np.bincount(tr_labels, minlength=2)
    weights_cls = np.array([1.0 / max(1, c) for c in cls_counts])
    sample_weights = torch.from_numpy(weights_cls[tr_labels]).double()
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_labels), replacement=True)

    ld_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=False)
    ld_vl = DataLoader(ds_vl, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    ld_ts = DataLoader(ds_ts, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return ld_tr, ld_vl, ld_ts, ds_ts

def run_epoch(model, loader, device, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_sum, n = 0.0, 0
    probs, trues = [], []
    for x, y, _ in tqdm(loader, desc="Train" if is_train else "Eval", leave=False):
        x, y = x.to(device), y.to(device)
        if is_train: optimizer.zero_grad()
        logits = model(x).view(-1)
        loss = criterion(logits, y)
        if is_train:
            loss.backward(); optimizer.step()
        with torch.no_grad():
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.extend(p.tolist()); trues.extend(y.long().cpu().numpy().tolist())
        loss_sum += float(loss.detach()) * x.size(0)
        n += x.size(0)
    return loss_sum / max(1, n), np.array(probs), np.array(trues)

def compute_metrics(y_true, y_prob, thr=0.5, beta=1.0):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    ap  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    b2 = beta*beta
    fbeta = (1+b2)*prec*rec/(b2*prec+rec+1e-12) if (prec+rec)>0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if (y_pred.sum() not in [0, len(y_pred)]) else 0.0
    return dict(cm=cm, roc=roc, ap=ap, prec=prec, rec=rec, f1=f1, fbeta=fbeta, mcc=mcc)

def select_best_threshold(y_true, y_prob, metric="fbeta", beta=1.0):
    grid = np.linspace(0.05, 0.95, 19)
    best_thr, best_val = 0.5, -1
    for t in grid:
        M = compute_metrics(y_true, y_prob, thr=t, beta=beta)
        val = M['fbeta'] if metric == "fbeta" else M['mcc']
        if val > best_val:
            best_val, best_thr = val, t
    return best_thr, best_val

# -------------------- Pipeline ---------------------------
class BCPipeline:
    def __init__(self, args):
        self.args = args
        self.dev = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

        # data
        all_paths, all_labels = get_data_paths_and_labels(args.base_dir)
        tr_p, tmp_p, tr_l, tmp_l = train_test_split(
            all_paths, all_labels, test_size=0.3, stratify=all_labels, random_state=args.seed
        )
        vl_p, ts_p, vl_l, ts_l = train_test_split(
            tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=args.seed
        )

        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.ld_tr, self.ld_vl, self.ld_ts, self.ds_ts = make_loaders(
            tr_p, tr_l, vl_p, vl_l, ts_p, ts_l,
            batch_size=args.batch_size, mean=self.mean, std=self.std,
            pca_lighting=args.pca_lighting, num_workers=4
        )

        # class weights for loss
        cnt_pos = int(tr_l.sum()); cnt_neg = len(tr_l) - cnt_pos
        w0 = cnt_pos / max(1, (cnt_pos + cnt_neg))
        w1 = cnt_neg / max(1, (cnt_pos + cnt_neg))
        self.class_weight = torch.tensor([w0, w1], dtype=torch.float32, device=self.dev)

        # optional PCA head
        self.pca_layer = None
        if args.pca_head:
            tmp = build_model(dropout=args.dropout, pca_layer=None)
            tmp.to(self.dev).eval()
            feats = gather_convnext_features(tmp, self.ld_tr, self.dev, max_samples=args.pca_fit_samples)
            self.pca_layer = build_pca_linear(feats, out_dim=args.pca_dim)

    def _make_model(self, alpha, gamma):
        model = build_model(dropout=self.args.dropout, pca_layer=self.pca_layer, pca_dim=self.args.pca_dim)
        model.to(self.dev)
        crit = FocalLossWithClassWeight(alpha=alpha, gamma=gamma, class_weight=self.class_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return model, crit, opt

    def tune(self):
        if not self.args.tune:
            return self.args.alpha, self.args.gamma
        print(f"🔎 Tuning alpha/gamma for {self.args.tune_epochs} epochs per combo...")
        best_a, best_g, best_score = self.args.alpha, self.args.gamma, -1
        for a in self.args.alpha_grid:
            for g in self.args.gamma_grid:
                model, crit, opt = self._make_model(a, g)
                for ep in range(1, self.args.tune_epochs+1):
                    tr_loss, _, _    = run_epoch(model, self.ld_tr, self.dev, crit, opt)
                    vl_loss, p, yval = run_epoch(model, self.ld_vl, self.dev, crit, None)
                    thr, score = select_best_threshold(yval, p, metric=self.args.threshold_metric, beta=self.args.fbeta)
                    print(f"[a={a} g={g}] ep{ep} tr={tr_loss:.4f} vl={vl_loss:.4f} {self.args.threshold_metric}={score:.4f} thr={thr:.2f}")
                if score > best_score:
                    best_score, best_a, best_g = score, a, g
        print(f"✅ Best from tuning: alpha={best_a}, gamma={best_g} (val {self.args.threshold_metric}={best_score:.4f})")
        return best_a, best_g

    def train_and_validate(self, alpha, gamma):
        model, crit, opt = self._make_model(alpha, gamma)
        start_epoch, best_loss = 1, float('inf')

        if os.path.isfile(self.args.ckpt):
            print(f"⏳ Loading checkpoint '{self.args.ckpt}'")
            ckpt = torch.load(self.args.ckpt, map_location=self.dev)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_loss   = ckpt.get('best_loss', best_loss)
            print(f"✅ Resumed: epoch {start_epoch}, best_loss={best_loss:.4f}")

        for ep in range(start_epoch, self.args.epochs+1):
            tr_loss, _, _    = run_epoch(model, self.ld_tr, self.dev, crit, opt)
            vl_loss, p, yval = run_epoch(model, self.ld_vl, self.dev, crit, None)
            thr, score = select_best_threshold(yval, p, metric=self.args.threshold_metric, beta=self.args.fbeta)
            print(f"E{ep}: Train={tr_loss:.4f}  Val={vl_loss:.4f}  best_{self.args.threshold_metric.upper()}@thr={thr:.2f}={score:.4f}")
            if vl_loss < best_loss:
                best_loss = vl_loss
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'best_loss': best_loss,
                    'alpha': alpha, 'gamma': gamma
                }, self.args.ckpt)
                print(f"💾 Saved checkpoint (epoch {ep})")
        return model, crit

    @torch.no_grad()
    def test_with_uncertainty(self, model, crit):
        # reload best
        if os.path.isfile(self.args.ckpt):
            ckpt = torch.load(self.args.ckpt, map_location=self.dev)
            model.load_state_dict(ckpt['model_state_dict'])
            alpha = ckpt.get('alpha', self.args.alpha); gamma = ckpt.get('gamma', self.args.gamma)
            print(f"Using best checkpoint (alpha={alpha}, gamma={gamma})")

        # Best threshold from validation
        _, val_prob, val_true = run_epoch(model, self.ld_vl, self.dev, crit, None)
        best_thr, _ = select_best_threshold(val_true, val_prob, metric=self.args.threshold_metric, beta=self.args.fbeta)
        print(f"🔧 Selected threshold={best_thr:.2f} by {self.args.threshold_metric.upper()} (beta={self.args.fbeta})")

        model.eval()
        all_paths, y_true, p_mean, p_raw, var, ent, exp_ent, mi = [], [], [], [], [], [], [], []

        for x, y, paths in tqdm(self.ld_ts, desc="Test/Uncertainty"):
            x, y = x.to(self.dev), y.to(self.dev)

            # collect samples: base + TTA + MC-dropout passes
            samples = []

            # base forward
            logits = model(x).view(-1)
            samples.append(torch.sigmoid(logits).cpu().numpy())

            # TTA
            for xt in tta_batch(x, t=max(1, self.args.tta)):
                logits_t = model(xt).view(-1)
                samples.append(torch.sigmoid(logits_t).cpu().numpy())

            # MC Dropout (enable dropout by train() but don’t step)
            if self.args.mc_dropout > 0:
                model.train()
                for _ in range(self.args.mc_dropout):
                    logits_d = model(x).view(-1)
                    samples.append(torch.sigmoid(logits_d).cpu().numpy())
                model.eval()

            S = np.vstack(samples)  # [T, B]
            stats = predictive_stats(S)

            pm = stats['mean_prob']; pv = stats['var']; pe = stats['entropy']
            peh = stats['expected_entropy']; pmi = stats['mutual_info']

            p_mean.extend(pm.tolist())
            p_raw.extend(S[0].tolist())  # first (non-aug) as raw prob
            var.extend(pv.tolist()); ent.extend(pe.tolist())
            exp_ent.extend(peh.tolist()); mi.extend(pmi.tolist())
            y_true.extend(y.long().cpu().numpy().tolist())
            all_paths.extend(list(paths))

        y_true = np.array(y_true)
        p_mean = np.array(p_mean)
        preds = (p_mean >= best_thr).astype(int)

        # metrics with mean prob
        cm   = confusion_matrix(y_true, preds)
        roc  = roc_auc_score(y_true, p_mean) if len(np.unique(y_true)) == 2 else np.nan
        ap   = average_precision_score(y_true, p_mean) if len(np.unique(y_true)) == 2 else np.nan
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)
        mcc  = matthews_corrcoef(y_true, preds) if (preds.sum() not in [0, len(preds)]) else 0.0

        # save CSV
        df = pd.DataFrame({
            'path': all_paths,
            'true': y_true,
            'pred': preds,
            'prob_raw': np.array(p_raw),
            'prob_mean': p_mean,
            'var': np.array(var),
            'entropy': np.array(ent),
            'expected_entropy': np.array(exp_ent),
            'mutual_info': np.array(mi),
            'threshold_used': best_thr
        })
        df.to_csv('test_predictions.csv', index=False)

        print('Confusion Matrix:\n', cm)
        print(f'ROC AUC: {roc:.4f}  PR AUC: {ap:.4f}')
        print(f'Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}  MCC: {mcc:.4f}')
        torch.save(model.state_dict(), 'convnext_final.pth')

    @torch.no_grad()
    def explain(self, model, out_dir="explanations", topk=20, by="mutual_info"):
        """
        Generate Grad-CAM overlays and saliency heatmaps for top-K test items
        by uncertainty (mutual_info, entropy, var) or for all misclassifications
        if by='errors'.
        """
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        # First, get probabilities and uncertainties (single pass)
        paths_list, probs_list, labels_list = [], [], []
        for x, y, paths in tqdm(self.ld_ts, desc="Collect for explain"):
            x = x.to(self.dev)
            p = torch.sigmoid(model(x).view(-1)).cpu().numpy()
            probs_list.extend(p.tolist())
            labels_list.extend(y.numpy().tolist())
            paths_list.extend(list(paths))
        probs = np.array(probs_list); labels = np.array(labels_list)
        preds = (probs >= 0.5).astype(int)

        # Choose indices
        if by == "errors":
            idx = np.where(preds != labels)[0]
        else:
            # quick uncertainty proxy = |p-0.5| small -> uncertain; or use entropy
            if by == "entropy":
                u = -probs*np.log(probs+1e-6) - (1-probs)*np.log(1-probs+1e-6)
            elif by == "var":
                # var proxy from single pass (not ideal); fallback to |p-0.5|
                u = 0.25 - (probs-0.5)**2
            else:
                # mutual_info approximated with entropy here; full MI is in test_with_uncertainty
                u = -probs*np.log(probs+1e-6) - (1-probs)*np.log(1-probs+1e-6)
            idx = np.argsort(u)[::-1]

        idx = idx[:min(topk, len(idx))]
        print(f"Explaining {len(idx)} samples -> {out_dir}")

        # Grad-CAM setup
        cam = GradCAM(model)
        for i in tqdm(idx, desc="Grad-CAM"):
            # Recreate the exact preprocessed tensor to overlay correctly
            img_rgb = cv2.imread(paths_list[i], cv2.IMREAD_GRAYSCALE)
            if img_rgb is None: continue
            img_rgb = cv2.cvtColor(cv2.resize(img_rgb, (224, 224)), cv2.COLOR_GRAY2RGB)

            # Forward once on a single image (requires same val transform!):
            pil = Image.fromarray(img_rgb)
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
            x = tf(pil).unsqueeze(0).to(self.dev)

            logits = model(x).view(-1)
            heat_small = cam(x, logits)               # [1, h, w]
            heat = upsample_like(heat_small, x)[0].cpu().numpy()

            # overlay & save
            overlay = overlay_heatmap(img_rgb, heat, alpha=0.35)
            base = os.path.splitext(os.path.basename(paths_list[i]))[0]
            cv2.imwrite(os.path.join(out_dir, f"{base}_gradcam.jpg"), overlay[:, :, ::-1])  # BGR for imwrite

            # saliency
            s = saliency_map(model, x)[0].cpu().numpy()
            s_overlay = overlay_heatmap(img_rgb, s, alpha=0.35)
            cv2.imwrite(os.path.join(out_dir, f"{base}_saliency.jpg"), s_overlay[:, :, ::-1])

# -------------------- CLI -------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                   help="Project base dir that contains the 'data/' folder.")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--fbeta", type=float, default=1.0)
    p.add_argument("--threshold-metric", choices=["fbeta", "mcc"], default="fbeta")
    p.add_argument("--pca-lighting", action="store_true")
    p.add_argument("--pca-head", action="store_true")
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--pca-fit-samples", type=int, default=2000)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--tune-epochs", type=int, default=2)
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[0.25, 0.5])
    p.add_argument("--gamma-grid", type=float, nargs="+", default=[1.0, 2.0])
    p.add_argument("--dropout", type=float, default=0.0, help="dropout in classifier (enables MC-dropout)")
    p.add_argument("--mc-dropout", type=int, default=0, help="# MC-dropout passes per test batch (0=off)")
    p.add_argument("--tta", type=int, default=0, help="# TTA variants per batch (0,1,2,3,4)")
    p.add_argument("--ckpt", type=str, default="checkpoint.pth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["train", "test", "all", "explain"], default="all")
    p.add_argument("--explain-dir", type=str, default="explanations")
    p.add_argument("--explain-topk", type=int, default=20)
    p.add_argument("--explain-by", choices=["mutual_info", "entropy", "var", "errors"], default="mutual_info")
    args = p.parse_args()

    set_seed(args.seed)
    pipe = BCPipeline(args)
    if args.mode in ("train", "all"):
        a, g = pipe.tune()
        model, crit = pipe.train_and_validate(a, g)
    else:
        # if skipping training, still need model/crit for test/explain
        a, g = args.alpha, args.gamma
        model, crit, _ = pipe._make_model(a, g)

    if args.mode in ("test", "all"):
        pipe.test_with_uncertainty(model, crit)

    if args.mode == "explain":
        pipe.explain(model, out_dir=args.explain_dir, topk=args.explain_topk, by=args.explain_by)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
