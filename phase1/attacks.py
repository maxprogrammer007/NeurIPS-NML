"""
Phase 1 — Attack Suite (Pure FP32 Throughout)
===============================================
T1 : Standard FGSM and PGD (label-flipping, no CSED awareness)
T2 : Adaptive PGD with CSED-suppression term in the loss

Key upgrade from Phase 0:
  - ALL computation runs in pure Float32 (torch.float32).
  - The FP16 autocast block in Phase 0's pgd_adaptive_csed has been removed.
  - This prevents gradient underflow during PGD optimization, matching the
    authoritative Phase 0 run that achieved 99.67% ASR on PGD-T1.
  - Batch processing is added via generate_t1_attacks() and generate_t2_attacks()
    with tqdm progress reporting for large-N Phase 1 runs.
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
    batch_size: int = 4,
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
# T2 — Adaptive PGD with CSED-suppression (pure FP32, no autocast)
# ──────────────────────────────────────────────────────────────────────────────
def pgd_adaptive_csed(
    art_model,
    sem_model,
    art_layer,
    sem_layer,
    image: torch.Tensor,
    label: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 30,
    lam: float = 1.0,
    device: str = "cuda",
) -> torch.Tensor:
    """
    T2 Adaptive PGD:  L = L_ce(ensemble) − λ · CSED_proxy(x')

    CSED_proxy is a differentiable spatial feature-distance proxy:
      - NPR stream: spatial feature map from art_layer (B, C, h, w) → (B, 1, 16, 16)
      - UnivFD stream: patch token grid from sem_layer (257, B, 1024) → (B, 1, 16, 16)
      - proxy = 1 − cosine_similarity(fa_flat, fb_flat)

    Pure FP32 throughout — the FP16 autocast block from Phase 0 is intentionally
    removed. This allows gradients to flow without numerical underflow, matching
    the authoritative Phase 0 FP32 evaluation (99.67% ASR on PGD-T1).

    Args:
        art_model/sem_model:  Detector models in FP32.
        art_layer/sem_layer:  Layers to hook for feature proxy.
        image:                Single image (C,H,W) or (1,C,H,W), FP32.
        label:                True label tensor.
        lam:                  CSED suppression weight (0.0 = plain PGD).

    Returns:
        Adversarial image tensor (C,H,W) on CPU.
    """
    art_model.eval()
    sem_model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.float().to(device)
    label = label.to(device).view(-1)

    # Register forward hooks to capture intermediate features
    feat_a_buf: list = [None]
    feat_b_buf: list = [None]

    def hook_a(module, inp, out):
        feat_a_buf[0] = out

    def hook_b(module, inp, out):
        feat_b_buf[0] = out

    h_a = art_layer.register_forward_hook(hook_a)
    h_b = sem_layer.register_forward_hook(hook_b)

    adv = image.clone().detach()
    original = image.clone().detach()

    for _step in range(steps):
        adv_req = adv.detach().clone().requires_grad_(True)

        # Pure FP32 forward pass (no autocast)
        logit_a = art_model(adv_req)
        feat_a  = feat_a_buf[0]

        logit_b = sem_model(adv_req)
        feat_b  = feat_b_buf[0]

        logit_avg = (logit_a + logit_b) / 2.0
        loss_ce   = F.cross_entropy(logit_avg, label)

        if lam > 0.0 and feat_a is not None and feat_b is not None:
            # NPR feature map: (B, 2048, 7, 7) → (B, 1, 16, 16)
            if feat_a.dim() == 4:
                fa_map = feat_a.mean(dim=1, keepdim=True)
                fa_map = F.interpolate(fa_map, size=(16, 16), mode="bilinear",
                                       align_corners=False)
            else:
                fa_map = feat_a.reshape(-1, 1, 16, 16)

            # UnivFD token grid: (257, B, 1024) → drop CLS → (B, 1, 16, 16)
            if feat_b.dim() == 3 and feat_b.shape[0] == 257:
                fb_tokens = feat_b[1:, :, :]         # (256, B, 1024)
                B_dim = fb_tokens.shape[1]
                fb_grid = fb_tokens.permute(1, 2, 0).reshape(B_dim, 1024, 16, 16)
                fb_map = fb_grid.mean(dim=1, keepdim=True)  # (B, 1, 16, 16)
            else:
                fb_map = feat_b.reshape(-1, 1, 16, 16)

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

    return adv.detach().squeeze(0).cpu()


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
    batch_size: int = 4,
    device: str = "cuda",
    desc: str = "T2 Adaptive",
) -> torch.Tensor:
    """
    Generate T2 adaptive attacks for a batch of images in pure FP32.

    Args:
        batch_size: Number of images processed together per call.
                    For T2, images are processed individually (batch_size is
                    advisory here — pgd_adaptive_csed supports B>1 but the
                    hook shapes must match).

    Returns:
        Adversarial image tensor (N, C, H, W) on CPU.
    """
    adv_list = []
    n = len(images)

    for i in tqdm(range(0, n, batch_size), desc=desc):
        batch_imgs = images[i:i + batch_size].float()
        batch_lbls = labels[i:i + batch_size]
        adv = pgd_adaptive_csed(
            art_model, sem_model,
            art_layer, sem_layer,
            batch_imgs, batch_lbls,
            eps=eps, alpha=2 / 255, steps=steps,
            lam=lam, device=device,
        )
        # pgd_adaptive_csed squeezes dim 0 — re-unsqueeze for batch > 1
        if adv.dim() == 3:
            adv = adv.unsqueeze(0)
        adv_list.append(adv.cpu())
        torch.cuda.empty_cache()

    return torch.cat(adv_list, 0)
