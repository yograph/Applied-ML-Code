#!/usr/bin/env python3
"""
Contains:
- ExtendedEvaluation: runs through a DataLoader and computes many metrics & plots.
- FocalLoss: standard focal loss for multi‐class (with per‐class alpha).
- FullTrainingOfModel: inherits Creating_Convnet to build samplers, train multiple models,
  save/load weights, and evaluate on val/test.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score
)

# ←── IMPORT YOUR ConvNet BASE ──→
# Suppose your "convnet_one.py" lives in the same folder. Adjust the import path if needed.
from convnet_one import Creating_Convnet  

# ───── Utility: Choose default device ─────
def default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# ══════════════════════════════════════════
# 1) ExtendedEvaluation: compute metrics & plots
# ══════════════════════════════════════════
class ExtendedEvaluation:
    """
    Runs inference on a DataLoader, collects y_true, y_pred, y_prob,
    then computes classification_report, cm, balanced acc, MCC, Kappa, ROC‐AUC, etc.
    Also provides methods to plot confusion matrix and ROC curves.
    """
    def __init__(self, model, dataloader, device=None, class_names=None):
        self.device = device or default_device()
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.class_names = class_names or [str(i) for i in range(getattr(model, 'num_classes', 2))]

    def get_predictions(self):
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x, y in self.dataloader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
        return np.array(y_true), np.array(y_pred), np.array(y_prob)

    def compute_all(self):
        y_true, y_pred, y_prob = self.get_predictions()

        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovo')
        except:
            roc_auc = None

        precision, recall, f1, supp = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        accuracy = (y_true == y_pred).mean()

        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "balanced_accuracy": bal_acc,
            "matthews_corrcoef": mcc,
            "cohen_kappa": kappa,
            "roc_auc_ovo": roc_auc,
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": supp.tolist(),
            "accuracy": accuracy
        }

    def plot_confusion_matrix(self, normalize=False):
        cm = confusion_matrix(*self.get_predictions()[:2])
        if normalize:
            cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(self.class_names)),
            yticks=np.arange(len(self.class_names)),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title=("Normalized " if normalize else "") + "Confusion Matrix"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = f"{cm[i,j]:.2f}" if normalize else f"{int(cm[i,j])}"
                ax.text(j, i, val, ha='center', va='center')
        fig.tight_layout()
        return fig

    def plot_roc_curve(self):
        y_true, _, y_prob = self.get_predictions()
        n_classes = len(self.class_names)
        y_true_bin = np.eye(n_classes)[y_true]
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots()
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f"{self.class_names[i]} (AUC={roc_auc[i]:.2f})")
        ax.plot([0,1], [0,1], 'k--', label="Chance")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
        ax.legend(loc="lower right")
        return fig

# ════════════════════════════════════════
# 2) Utility: plot training/validation loss
# ════════════════════════════════════════
def plot_loss_curves(train_losses, val_losses):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves")
    ax.legend()
    return fig

# ══════════════════════════════════════════
# 3) FocalLoss: multi‐class focal loss
# ══════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    FL = - α_t * (1 - p_t)**γ * log(p_t)
    where p_t is the softmax prob of the true class.
    α can be a tensor of size [num_classes], for per‐class weighting.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        log_probs = F.log_softmax(logits, dim=1)  # [B, C]
        probs = torch.exp(log_probs)              # [B, C]
        targets = targets.view(-1, 1)             # [B, 1]
        log_p_t = log_probs.gather(1, targets).view(-1)   # [B]
        p_t     = probs.gather(1, targets).view(-1)       # [B]
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
        else:
            alpha_t = 1.0

        focal_term = (1.0 - p_t) ** self.gamma
        loss = -alpha_t * focal_term * log_p_t  # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# ════════════════════════════════════════════════════════════════════════════
# 4) FullTrainingOfModel: inherits Creating_Convnet to train multiple ConvNets,  
#    oversample minority class, save/load weights, and run extended evaluation.
# ════════════════════════════════════════════════════════════════════════════
class FullTrainingOfModel(Creating_Convnet):
    """
    Extends your Creating_Convnet to:
      1. Compute per‐class α for FocalLoss from an imbalanced train set.
      2. Build a WeightedRandomSampler for oversampling the minority class.
      3. Train N separate models (each with its own optimizer & criterion).
      4. Provide methods to save & load model weights.
      5. Evaluate on val & test via ExtendedEvaluation (metrics + plots).
    """

    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 batch_size=32,
                 lr=1e-4,
                 num_epochs=10,
                 device=None,
                 class_names=None,
                 gamma=2.0):
        # 1) Device selection (CUDA > MPS > CPU)
        self.device = device or default_device()
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.class_names = class_names
        self.gamma = gamma

        # 2) Compute per‐class counts from train_dataset
        all_labels = [int(label) for _, label in train_dataset]
        label_counts = Counter(all_labels)       # e.g. {0: 5200, 1: 100}
        num_classes = len(label_counts)
        total_count = float(len(all_labels))     # e.g. 5300

        # 2a) Build per‐class α for FocalLoss: α_i = total_count / (num_classes * count_i)
        alpha_list = []
        for cls_idx in range(num_classes):
            count_i = float(label_counts.get(cls_idx, 0))
            if count_i > 0:
                alpha_list.append(total_count / (num_classes * count_i))
            else:
                alpha_list.append(0.0)
        self.alpha = torch.tensor(alpha_list, dtype=torch.float32)

        # 2b) Build sample weights for WeightedRandomSampler: 1 / count(label)
        sw = [1.0 / float(label_counts[int(label)]) for _, label in train_dataset]
        sample_weights = torch.tensor(sw, dtype=torch.double)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 3) Build DataLoaders
        pin = True if (self.device.type in ["cuda", "mps"]) else False
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=pin
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=pin
        )

        # 4) Storage for models, optimizers, histories, eval results
        self.models = {}      # e.g. { "model_0": model_object, … }
        self.histories = {}   # e.g. { "model_0": { "optimizer": …, "criterion": …, "train_loss":[], "val_loss":[] } }
        self.eval_results = {}

    def initialize_models(self, num_models: int):
        """
        Instantiate `num_models` distinct ConvNet instances,
        each with its own optimizer (Adam) and FocalLoss(α, γ).
        """
        if num_models <= 0:
            raise ValueError("num_models must be at least 1.")

        for i in range(num_models):
            model = self.creating_single_model()
            model.to(self.device)

            alpha_device = self.alpha.to(self.device)
            criterion = FocalLoss(alpha=alpha_device, gamma=self.gamma, reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=self.lr)

            self.models[f"model_{i}"] = model
            self.histories[f"model_{i}"] = {
                "optimizer": optimizer,
                "criterion": criterion,
                "train_loss": [],
                "val_loss": []
            }

    def train(self):
        """
        Train each model in `self.models` for `self.num_epochs` epochs,
        using the WeightedRandomSampler in `self.train_loader`.
        """
        for name, model in self.models.items():
            history = self.histories[name]
            optimizer = history["optimizer"]
            criterion = history["criterion"]

            for epoch in range(1, self.num_epochs + 1):
                # ── Training Phase ──
                model.train()
                running = 0.0
                for images, labels in self.train_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running += loss.item() * images.size(0)

                train_loss = running / len(self.train_loader.dataset)
                history["train_loss"].append(train_loss)

                # ── Validation Phase ──
                model.eval()
                val_running = 0.0
                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_running += loss.item() * images.size(0)

                val_loss = val_running / len(self.val_loader.dataset)
                history["val_loss"].append(val_loss)

                print(f"{name} | Epoch {epoch}/{self.num_epochs}  "
                      f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    def save_model(self, name: str, filepath: str):
        """
        Save `self.models[name]` state_dict to the given filepath.
        """
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.models[name].state_dict(), filepath)
        print(f"💾 Saved {name} → {filepath}")

    def save_all_models(self, directory: str):
        """
        Save every model’s weights into `directory/<model_name>_weights.pth`.
        """
        os.makedirs(directory, exist_ok=True)
        for name in self.models:
            path = os.path.join(directory, f"{name}_weights.pth")
            self.save_model(name, path)

    def load_model(self, name: str, filepath: str):
        """
        Load state_dict into `self.models[name]`. If model does not exist yet,
        re‐initialize it via `creating_single_model()`.
        """
        if name not in self.models:
            model = self.creating_single_model()
            model.to(self.device)
            self.models[name] = model

        self.models[name].load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"🔃 Loaded weights for {name} from {filepath}")

    def train_and_save(self, save_dir: str):
        """
        Convenience: call `self.train()` and then `self.save_all_models(save_dir)`.
        """
        self.train()
        self.save_all_models(save_dir)

    def evaluate_models(self):
        """
        For each model in `self.models`, run ExtendedEvaluation on validation/test,
        store results in `self.eval_results[name] = {'val':…, 'test':…}`, and
        plot confusion matrices + ROC curves.
        """
        for name, model in self.models.items():
            results = {}
            for split, loader in [("val", self.val_loader), ("test", self.test_loader)]:
                evaluator = ExtendedEvaluation(
                    model,
                    loader,
                    device=self.device,
                    class_names=self.class_names
                )
                metrics = evaluator.compute_all()
                results[split] = metrics

                # Plot & show confusion matrix
                fig_cm = evaluator.plot_confusion_matrix(normalize=False)
                fig_cm.suptitle(f"{name} – {split} Confusion Matrix")
                fig_cm.show()

                # Plot & show normalized confusion matrix
                fig_cm_n = evaluator.plot_confusion_matrix(normalize=True)
                fig_cm_n.suptitle(f"{name} – {split} Confusion Matrix (Normalized)")
                fig_cm_n.show()

                # Plot & show ROC curves
                fig_roc = evaluator.plot_roc_curve()
                fig_roc.suptitle(f"{name} – {split} ROC Curves")
                fig_roc.show()

            self.eval_results[name] = results
        return self.eval_results

    def plot_all_loss_curves(self):
        """
        For each model in `self.histories`, draw its train/val loss curve.
        """
        for name, history in self.histories.items():
            fig = plot_loss_curves(history["train_loss"], history["val_loss"])
            fig.suptitle(f"{name} Loss Curves")
            fig.show()
