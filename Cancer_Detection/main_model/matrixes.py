#!/usr/bin/env python3
"""
Plot your model’s metrics using the actual test‐set CSV and hard‐coded
training & validation losses from your terminal output.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────
base_dir         = os.path.dirname(os.path.abspath(__file__))
predictions_dir  = os.path.join(base_dir, 'predictions')

TEST_CSV_PATH    = os.path.join(predictions_dir, 'test_predictions.csv')

# Your test CSV must have these columns:
TRUE_LABEL_COL   = 'true'   # ground‐truth
SCORE_COL        = 'prob'   # predicted probability

# Hard‐coded training history from your logs:
HISTORY = {
    'loss':     [0.0165, 0.0105, 0.0070, 0.0053, 0.0044, 0.0039],
    'val_loss': [0.0065, 0.0075, 0.0079, 0.0111, 0.0123, 0.0093]
}
# ─────────────────────────────────────────────────────────────────────────

def load_test_data():
    df = pd.read_csv(TEST_CSV_PATH)
    y_true   = df[TRUE_LABEL_COL].values
    y_scores = df[SCORE_COL].values
    return y_true, y_scores

def plot_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc      = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1],'--', label='Chance (AUC = 0.5000)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def plot_pr(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc               = auc(recall, precision)
    prevalence           = np.mean(y_true)

    plt.figure()
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.4f})')
    plt.hlines(prevalence, xmin=0, xmax=1, linestyles='--',
               label=f'Chance = prevalence ({prevalence:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_confusion(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (threshold = {threshold})')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['Neg','Pos'])
    plt.yticks(ticks, ['Neg','Pos'])
    plt.xlabel('Predicted')
    plt.ylabel('True')

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.show()

def plot_loss(history):
    epochs = np.arange(1, len(history['loss'])+1)
    plt.figure()
    plt.plot(epochs, history['loss'],     marker='o', label='Train Loss')
    plt.plot(epochs, history['val_loss'], marker='o', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss (hard‐coded)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # load
    y_true, y_scores = load_test_data()

    # ROC
    plot_roc(y_true, y_scores)

    # PR
    plot_pr(y_true, y_scores)

    # Confusion matrix
    plot_confusion(y_true, y_scores, threshold=0.5)

    # Loss curves (hard-coded)
    plot_loss(HISTORY)

    # Print numeric summaries
    cm      = confusion_matrix(y_true, (y_scores>=0.5).astype(int))
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    pr_auc  = auc(*precision_recall_curve(y_true, y_scores)[:2])

    print("Confusion Matrix:")
    print(cm)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR  AUC: {pr_auc:.4f}")

if __name__ == '__main__':
    main()
