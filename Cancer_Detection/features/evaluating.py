import torch
import torch.nn as nn
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

class ExtendedEvaluation:
    """
    Extended evaluation toolkit for imbalanced classification.
    Computes a wide range of metrics and provides plotting utilities.
    """
    def __init__(self, model, dataloader, device=None, class_names=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.class_names = class_names or [str(i) for i in range(getattr(model, 'num_classes', 2))]

    def get_predictions(self):
        """Run inference and collect true labels, preds, and probabilities."""
        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for x, y in self.dataloader:
                x = x.to(self.device)
                logits = self.model(x)
                probs = nn.functional.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
        return np.array(y_true), np.array(y_pred), np.array(y_prob)

    def compute_all(self):
        """Compute and return dictionary of all metrics."""
        y_true, y_pred, y_prob = self.get_predictions()
        # Basic classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        # Core metrics
        cm = confusion_matrix(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        try:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovo')
        except Exception:
            roc_auc = None
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        # Overall accuracy
        accuracy = (y_true == y_pred).mean()

        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'balanced_accuracy': balanced_acc,
            'matthews_corrcoef': mcc,
            'cohen_kappa': kappa,
            'roc_auc_ovo': roc_auc,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist(),
            'accuracy': accuracy
        }

    def plot_confusion_matrix(self, normalize=False):
        """
        Plot the confusion matrix as a heatmap.
        Args:
            normalize (bool): If True, normalize by row (true label) sums.
        Returns:
            fig: Matplotlib figure object.
        """
        cm = confusion_matrix(*self.get_predictions()[:2])
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(self.class_names)),
            yticks=np.arange(len(self.class_names)),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ylabel='True label',
            xlabel='Predicted label',
            title=('Normalized ' if normalize else '') + 'Confusion Matrix'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                        ha='center', va='center')
        fig.tight_layout()
        return fig

    def plot_roc_curve(self):
        """
        Plot ROC curves (one-vs-rest) for each class.
        Returns:
            fig: Matplotlib figure object.
        """
        y_true, _, y_prob = self.get_predictions()
        n_classes = len(self.class_names)
        # Binarize true labels
        y_true_bin = np.eye(n_classes)[y_true]
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots()
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f"{self.class_names[i]} (AUC = {roc_auc[i]:.2f})")
        ax.plot([0, 1], [0, 1], 'k--', label='Chance')
        ax.set(
            xlabel='False Positive Rate',
            ylabel='True Positive Rate',
            title='Receiver Operating Characteristic'
        )
        ax.legend(loc='lower right')
        return fig

# Utility function (outside class) to plot training/validation loss curves

def plot_loss_curves(train_losses, val_losses):
    """
    Plot training and validation loss over epochs.

    Args:
        train_losses (list of float): Training loss at each epoch.
        val_losses (list of float): Validation loss at each epoch.

    Returns:
        fig: Matplotlib figure object.
    """
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set(
        xlabel='Epoch',
        ylabel='Loss',
        title='Training and Validation Loss'
    )
    ax.legend()
    return fig
