# utils.py
import os
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------- PCA Lighting -----------------------
class PCALighting(object):
    """
    AlexNet-style PCA lighting noise (on tensors in [0,1]).
    Apply after ToTensor(), before Normalize().
    """
    DEFAULT_EIGVAL = torch.tensor([0.2175, 0.0188, 0.0045])
    DEFAULT_EIGVEC = torch.tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])

    def __init__(self, alphastd: float = 0.1, eigval=None, eigvec=None):
        self.alphastd = alphastd
        self.eigval = eigval if eigval is not None else self.DEFAULT_EIGVAL
        self.eigvec = eigvec if eigvec is not None else self.DEFAULT_EIGVEC

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.alphastd == 0:
            return img
        alpha = torch.normal(mean=0.0, std=self.alphastd, size=(3,), device=img.device)
        rgb = (self.eigvec.to(img.device) @ (self.eigval.to(img.device) * alpha))
        out = img.clone()
        for c in range(3):
            out[c, :, :] = out[c, :, :] + rgb[c]
        return torch.clamp(out, 0.0, 1.0)

# -------------------- PCA Head (pure PyTorch) ------------
@torch.no_grad()
def gather_convnext_features(model: nn.Module, dataloader, device, max_samples=2000):
    """
    Collect penultimate (pre-final-linear) features for PCA.
    Works with torchvision ConvNeXt (classifier[-1] is final Linear).
    """
    model.eval()
    feats = []
    captured = {}

    def hook(module, inp, out):
        captured['x'] = inp[0].detach()

    handle = model.classifier[-1].register_forward_pre_hook(hook)
    seen = 0
    for x, *_ in dataloader:
        x = x.to(device)
        _ = model(x)  # trigger hook
        f = captured['x']
        feats.append(f.cpu())
        seen += x.size(0)
        if seen >= max_samples:
            break
    handle.remove()
    return torch.cat(feats, dim=0)[:max_samples]

def build_pca_linear(features: torch.Tensor, out_dim: int) -> nn.Linear:
    """
    Fit PCA on [N, D] and return a frozen nn.Linear(D -> out_dim)
    implementing y = (x - mu) @ W. No external libs needed.
    """
    with torch.no_grad():
        X = features.float()
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)  # [D, D]
        eigvals, eigvecs = torch.linalg.eigh(cov)    # ascending
        idx = torch.argsort(eigvals, descending=True)
        W = eigvecs[:, idx[:out_dim]]                # [D, out_dim]
        layer = nn.Linear(W.shape[0], W.shape[1], bias=True)
        layer.weight.data.copy_(W.T)
        layer.bias.data.copy_((-mu @ W).squeeze(0))
        for p in layer.parameters():
            p.requires_grad = False
        return layer

# -------------------- Uncertainty Metrics ----------------
def predictive_stats(prob_samples: np.ndarray, eps: float = 1e-6):
    """
    prob_samples: shape [T, N] where T = #samples (MC-dropout and/or TTA)
    Returns dict of mean_prob, var, entropy, expected_entropy, mutual_info.
    """
    if prob_samples.ndim != 2:
        raise ValueError("prob_samples must be [T, N]")
    p_mean = prob_samples.mean(axis=0)                    # [N]
    p_var  = prob_samples.var(axis=0)                     # [N]
    # Predictive entropy H(p_mean)
    H = -p_mean*np.log(p_mean+eps) - (1-p_mean)*np.log(1-p_mean+eps)
    # Expected entropy E[H(p_t)]
    Ht = -prob_samples*np.log(prob_samples+eps) - (1-prob_samples)*np.log(1-prob_samples+eps)
    EH = Ht.mean(axis=0)
    # BALD mutual information = H - E[H]
    MI = H - EH
    return dict(mean_prob=p_mean, var=p_var, entropy=H, expected_entropy=EH, mutual_info=MI)

# -------------------- TTA helper -------------------------
def tta_batch(x: torch.Tensor, t: int = 4):
    """
    Simple deterministic TTA variants (no randomness): identity, hflip, vflip, hvflip.
    If t>4, repeats the set to match t.
    """
    xs = [x,
          torch.flip(x, dims=[-1]),
          torch.flip(x, dims=[-2]),
          torch.flip(x, dims=[-1, -2])]
    if t <= 4:
        return xs[:t]
    reps = (t + 3) // 4
    xs = (xs * reps)[:t]
    return xs

# -------------------- Grad-CAM & Saliency ----------------
class GradCAM:
    """
    Minimal Grad-CAM for ConvNeXt. Hooks activations/gradients from a chosen layer.
    Default target layer: the last stage's last block depthwise conv.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        if target_layer is None:
            # Try ConvNeXt last stage last block dwconv
            try:
                target_layer = model.features[-1][-1].dwconv
            except Exception:
                # Fallback: last Conv2d found
                convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
                target_layer = convs[-1]

        def fwd_hook(m, i, o):
            self.activations = o.detach()

        def bwd_hook(m, gi, go):
            self.gradients = go[0].detach()

        self.handle_f = target_layer.register_forward_hook(fwd_hook)
        self.handle_b = target_layer.register_backward_hook(bwd_hook)

    def __del__(self):
        try:
            self.handle_f.remove(); self.handle_b.remove()
        except Exception:
            pass

    def __call__(self, x: torch.Tensor, target_logit: torch.Tensor):
        """
        x: input tensor [B,C,H,W]
        target_logit: scalar logit to backprop (e.g., model(x)[:,0] for positive class)
        returns Grad-CAM heatmaps in [0,1], shape [B, H, W]
        """
        self.model.zero_grad(set_to_none=True)
        target_logit.sum().backward(retain_graph=True)

        A = self.activations           # [B, C, H', W']
        G = self.gradients             # [B, C, H', W']
        weights = G.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * A).sum(dim=1)             # [B, H', W']
        cam = F.relu(cam)
        # normalise per-sample
        B, *_ = cam.shape
        cams = []
        for i in range(B):
            c = cam[i]
            if c.max() > 0:
                c = (c - c.min()) / (c.max() - c.min())
            else:
                c = torch.zeros_like(c)
            cams.append(c)
        return torch.stack(cams, dim=0)  # [B, H', W']

def upsample_like(cam: torch.Tensor, x: torch.Tensor):
    """Upsample cam [B,h,w] to x spatial size [B,C,H,W]."""
    cam = cam.unsqueeze(1)  # [B,1,h,w]
    cam_up = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam_up.squeeze(1)  # [B,H,W]

def overlay_heatmap(rgb_uint8: np.ndarray, heat01: np.ndarray, alpha: float = 0.35):
    """
    rgb_uint8: HxWx3 uint8
    heat01: HxW float in [0,1]
    returns overlay uint8
    """
    heat = np.uint8(255 * heat01.clip(0, 1))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]  # to RGB
    overlay = (alpha * heat + (1 - alpha) * rgb_uint8.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return overlay

# -------------------- Saliency ---------------------------
def saliency_map(model: nn.Module, x: torch.Tensor, target_index: int = 0):
    """
    Vanilla saliency: gradient of target logit wrt input.
    Returns |grad| per-pixel in [0,1], shape [B, H, W].
    """
    x = x.clone().detach().requires_grad_(True)
    logits = model(x).view(-1)  # [B]
    target = logits  # default: class-1 logit
    model.zero_grad(set_to_none=True)
    target.sum().backward()
    g = x.grad.detach()  # [B,C,H,W]
    g = g.abs().max(dim=1)[0]  # [B,H,W]
    # normalise per-sample
    B = g.shape[0]
    outs = []
    for i in range(B):
        s = g[i]
        if s.max() > 0:
            s = (s - s.min()) / (s.max() - s.min())
        else:
            s = torch.zeros_like(s)
        outs.append(s)
    return torch.stack(outs, dim=0)
