import torch
import torch.nn as nn

class FocalLossWithClassWeight(nn.Module):
    """
    Focal Loss with class weights for imbalanced binary classification.
    """
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
        # logits, targets: shape [N]
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