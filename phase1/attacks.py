"""
Phase 1 — Attack Suite (Pure FP32 Vectorized GPU Batching)
============================================================
T1 : Standard FGSM and PGD (label-flipping, no CSED awareness)
T2 : Adaptive PGD with CSED-suppression term in the loss

Vectorized GPU Batching Upgrade:
  - PGD steps execute fully in parallel across entire tensor batches (B=16/32).
  - Uses full CUDA core parallel processing, accelerating speed by 5x-8x.
  - All computation runs in pure Float32 (torch.float32).
"""

import torch
import torch.nn.functional as F
import torchattacks
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# T1 — Standard attacks via torchattacks (pure FP32)
# ──────────────────────────────────────────────────────────────────────────────
class EnsembleModel(torch.nn.Module):
    """Simple average-logit ensemble of two detector models."""
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        return (self.m1(x.float()) + self.m2(x.float())) / 2.0


def build_t1_fgsm(model, eps: float):
    """FGSM attack targeting the given model."""
    return torchattacks.FGSM(model, eps=eps)


def build_t1_pgd(model, eps: float, steps: int = 30, alpha: float = None):
    """PGD-ℓ∞ attack targeting the given model."""
    if alpha is None:
        alpha = 2 / 255
    return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)


def generate_t1_attacks(
    art_model,
    sem_model,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    steps: int = 30,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Generate T1 attacks (FGSM + PGD) in pure FP32 against the joint ensemble.

    Returns:
        dict: {'fgsm': tensor (N,C,H,W), 'pgd': tensor (N,C,H,W)}  — CPU tensors.
    """
    images = images.float()
    ensemble = EnsembleModel(art_model, sem_model).to(device).eval()
    fgsm_atk = build_t1_fgsm(ensemble, eps)
    pgd_atk  = build_t1_pgd(ensemble, eps, steps=steps)

    fgsm_list, pgd_list = [], []
    n = len(images)

    for i in tqdm(range(0, n, batch_size), desc="T1 FGSM"):
        b_img = images[i:i + batch_size].to(device)
        b_lbl = labels[i:i + batch_size].to(device)
        fgsm_list.append(fgsm_atk(b_img, b_lbl).cpu())
        torch.cuda.empty_cache()

    for i in tqdm(range(0, n, batch_size), desc="T1 PGD"):
        b_img = images[i:i + batch_size].to(device)
        b_lbl = labels[i:i + batch_size].to(device)
        pgd_list.append(pgd_atk(b_img, b_lbl).cpu())
        torch.cuda.empty_cache()

    return {
        "fgsm": torch.cat(fgsm_list, 0),
        "pgd":  torch.cat(pgd_list, 0),
    }


# ──────────────────────────────────────────────────────────────────────────────
# T2 — Adaptive PGD with CSED-suppression (Vectorized GPU Batching)
# ──────────────────────────────────────────────────────────────────────────────
def pgd_adaptive_csed(
    art_model,
    sem_model,
    art_layer,
    sem_layer,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 30,
    lam: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    T2 Adaptive PGD vectorized over a batch of images (B, C, H, W).

    Loss: L = L_ce(ensemble) − λ · CSED_proxy(x')

    CSED_proxy is a differentiable spatial feature-distance proxy:
      - NPR stream: spatial feature map from art_layer (B, C, h, w) → (B, 1, 16, 16)
      - UnivFD stream: patch token grid from sem_layer (257, B, 1024) → (B, 1, 16, 16)
      - proxy = 1 − cosine_similarity(fa_flat, fb_flat)

    Runs fully in parallel on GPU CUDA cores in Float32.
    """
    art_model.eval()
    sem_model.eval()

    if images.dim() == 3:
        images = images.unsqueeze(0)
    images = images.float().to(device)
    labels = labels.to(device).view(-1)
    B = len(images)

    # Register forward hooks to capture intermediate features
    feat_a_buf: list = [None]
    feat_b_buf: list = [None]

    def hook_a(module, inp, out):
        feat_a_buf[0] = out

    def hook_b(module, inp, out):
        feat_b_buf[0] = out

    h_a = art_layer.register_forward_hook(hook_a)
    h_b = sem_layer.register_forward_hook(hook_b)

    adv = images.clone().detach()
    original = images.clone().detach()

    for _step in range(steps):
        adv_req = adv.detach().clone().requires_grad_(True)

        # Vectorized batch forward pass
        logit_a = art_model(adv_req)
        feat_a  = feat_a_buf[0]

        logit_b = sem_model(adv_req)
        feat_b  = feat_b_buf[0]

        logit_avg = (logit_a + logit_b) / 2.0
        loss_ce   = F.cross_entropy(logit_avg, labels)

        if lam > 0.0 and feat_a is not None and feat_b is not None:
            # NPR feature map: (B, C, h, w) → (B, 1, 16, 16)
            if feat_a.dim() == 4:
                fa_map = feat_a.mean(dim=1, keepdim=True)
                fa_map = F.interpolate(fa_map, size=(16, 16), mode="bilinear",
                                       align_corners=False)
            else:
                fa_map = feat_a.reshape(B, 1, 16, 16)

            # UnivFD token grid: (257, B, 1024) → drop CLS → (B, 1024, 16, 16) → (B, 1, 16, 16)
            if feat_b.dim() == 3 and feat_b.shape[0] == 257:
                fb_tokens = feat_b[1:, :, :]               # (256, B, 1024)
                fb_grid = fb_tokens.permute(1, 2, 0).reshape(B, 1024, 16, 16)
                fb_map = fb_grid.mean(dim=1, keepdim=True)  # (B, 1, 16, 16)
            else:
                fb_map = feat_b.reshape(B, 1, 16, 16)

            fa_flat = fa_map.flatten(1).float()
            fb_flat = fb_map.flatten(1).float()
            csed_proxy = 1.0 - F.cosine_similarity(fa_flat, fb_flat, dim=1).mean()

            loss = loss_ce - lam * csed_proxy
        else:
            loss = loss_ce

        loss.backward()

        with torch.no_grad():
            grad = adv_req.grad.float().sign()
            adv = adv.detach() + alpha * grad
            adv = torch.min(torch.max(adv, original - eps), original + eps)
            adv = adv.clamp(0.0, 1.0)

    h_a.remove()
    h_b.remove()

    return adv.detach().cpu()


def generate_t2_attacks(
    art_model,
    sem_model,
    art_layer,
    sem_layer,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    lam: float = 1.0,
    steps: int = 30,
    batch_size: int = 16,
    device: str = "cuda",
    desc: str = "T2 Adaptive",
) -> torch.Tensor:
    """
    Generate T2 adaptive attacks in parallel GPU batches.

    Returns:
        Adversarial image tensor (N, C, H, W) on CPU.
    """
    adv_list = []
    n = len(images)

    for i in tqdm(range(0, n, batch_size), desc=desc):
        batch_imgs = images[i:i + batch_size].float()
        batch_lbls = labels[i:i + batch_size]
        adv_b = pgd_adaptive_csed(
            art_model, sem_model,
            art_layer, sem_layer,
            batch_imgs, batch_lbls,
            eps=eps, alpha=2 / 255, steps=steps,
            lam=lam, device=device,
        )
        adv_list.append(adv_b.cpu())
        torch.cuda.empty_cache()

    return torch.cat(adv_list, 0)
