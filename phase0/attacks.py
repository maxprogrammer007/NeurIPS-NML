"""
Phase 0 — Attack Suite
=======================
T1 : Standard FGSM and PGD (label-flipping, no CSED awareness)
T2 : Adaptive PGD with CSED-suppression term in the loss
"""

import torch
import torch.nn.functional as F
import torchattacks
from csed import get_gradcam_map, compute_csed
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# T1 — Standard attacks via torchattacks
# ──────────────────────────────────────────────────────────────────────────────
def build_t1_fgsm(model, eps: float):
    """FGSM attack targeting the provided model."""
    return torchattacks.FGSM(model, eps=eps)


def build_t1_pgd(model, eps: float, steps: int = 20, alpha: float = None):
    """PGD-ℓ∞ attack targeting the provided model."""
    if alpha is None:
        alpha = eps / 4
    return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)


def generate_t1_attacks(art_model, sem_model, images: torch.Tensor,
                        labels: torch.Tensor,
                        epsilons=(4/255, 8/255, 16/255),
                        device: str = "cuda") -> dict:
    """
    Generate T1 attacks at each epsilon against BOTH streams independently
    and also against a simple ensemble (average logits).
    Returns dict: eps → {'fgsm': tensor, 'pgd': tensor}
    """
    images = images.to(device)
    labels = labels.to(device)

    # Ensemble wrapper for joint attack
    class EnsembleModel(torch.nn.Module):
        def __init__(self, m1, m2):
            super().__init__()
            self.m1 = m1
            self.m2 = m2
        def forward(self, x):
            return (self.m1(x) + self.m2(x)) / 2

    ensemble = EnsembleModel(art_model, sem_model).to(device).eval()

    attacked = {}
    batch_size = 4
    n_samples = len(images)

    for eps in epsilons:
        fgsm = build_t1_fgsm(ensemble, eps)
        pgd  = build_t1_pgd(ensemble, eps)

        fgsm_list = []
        pgd_list  = []

        for i in range(0, n_samples, batch_size):
            b_img = images[i:i+batch_size].to(device)
            b_lbl = labels[i:i+batch_size].to(device)

            adv_f = fgsm(b_img, b_lbl)
            adv_p = pgd(b_img, b_lbl)

            fgsm_list.append(adv_f.cpu())
            pgd_list.append(adv_p.cpu())

            torch.cuda.empty_cache()

        attacked[eps] = {
            'fgsm': torch.cat(fgsm_list, dim=0),
            'pgd':  torch.cat(pgd_list, dim=0)
        }
    return attacked


# ──────────────────────────────────────────────────────────────────────────────
# T2 — Adaptive PGD with CSED-suppression term
# ──────────────────────────────────────────────────────────────────────────────
def pgd_adaptive_csed(art_model, sem_model,
                      art_layer, sem_layer,
                      image: torch.Tensor,
                      label: torch.Tensor,
                      eps: float = 8/255,
                      alpha: float = 2/255,
                      steps: int = 20,
                      lam: float = 1.0,
                      device: str = "cuda",
                      use_feature_proxy: bool = True) -> torch.Tensor:
    """
    T2 Adaptive PGD:  L = L_ce(ensemble) − λ · CSED_proxy(x')

    where CSED_proxy = L2( features_A(x') , features_B(x') )
    (differentiable feature-distance proxy as described in Phase 0 plan §4.2;
     noted as a known simplification — real CSED is still computed for evaluation)

    use_feature_proxy=True : use intermediate feature L2 (differentiable, fast)
    use_feature_proxy=False: differentiate through Grad-CAM cosine (slow, often
                             unstable — kept for ablation only)
    """
    image = image.to(device)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    label = label.to(device).view(-1)

    # Cache clean features for the proxy
    art_model.eval(); sem_model.eval()

    # Hook to get intermediate features
    feat_a_buf = [None]
    feat_b_buf = [None]

    def hook_a(module, inp, out):
        feat_a_buf[0] = out

    def hook_b(module, inp, out):
        feat_b_buf[0] = out

    h_a = art_layer.register_forward_hook(hook_a)
    h_b = sem_layer.register_forward_hook(hook_b)

    adv = image.clone().detach().requires_grad_(True)
    original = image.clone().detach()

    for step in range(steps):
        adv_req = adv.detach().clone().requires_grad_(True)

        logit_a = art_model(adv_req)
        feat_a  = feat_a_buf[0]

        logit_b = sem_model(adv_req)
        feat_b  = feat_b_buf[0]

        logit_avg = (logit_a + logit_b) / 2
        loss_ce   = F.cross_entropy(logit_avg, label)

        if use_feature_proxy and feat_a is not None and feat_b is not None:
            # feat_a: (1, 2048, 7, 7) → mean across channels → interpolate to (1, 1, 16, 16)
            if feat_a.dim() == 4:
                fa_map = feat_a.mean(dim=1, keepdim=True)
                fa_map = F.interpolate(fa_map, size=(16, 16), mode='bilinear', align_corners=False)
            else:
                fa_map = feat_a.view(1, 1, 16, 16)

            # feat_b: (257, 1, 1024) → drop CLS, permute to (1, 1024, 16, 16) → mean across channels
            if feat_b.dim() == 3:
                fb_tokens = feat_b[1:, 0, :] # (256, 1024)
                fb_grid = fb_tokens.reshape(16, 16, 1024).permute(2, 0, 1).unsqueeze(0) # (1, 1024, 16, 16)
                fb_map = fb_grid.mean(dim=1, keepdim=True) # (1, 1, 16, 16)
            else:
                fb_map = feat_b.view(1, 1, 16, 16)

            # Differentiable spatial CSED proxy: 1 - cosine_similarity of spatial maps
            fa_flat = fa_map.flatten(1)
            fb_flat = fb_map.flatten(1)
            csed_proxy = 1.0 - F.cosine_similarity(fa_flat, fb_flat, dim=1).mean()

            # Attack wants to SUPPRESS divergence → subtract λ · csed_proxy
            loss = loss_ce - lam * csed_proxy
        else:
            loss = loss_ce

        loss.backward()

        with torch.no_grad():
            grad = adv_req.grad.sign()
            adv = adv.detach() + alpha * grad
            # Project back into ε-ball and [0,1]
            adv = torch.min(torch.max(adv, original - eps), original + eps)
            adv = adv.clamp(0.0, 1.0)

    h_a.remove()
    h_b.remove()

    return adv.detach().squeeze(0).cpu()


def generate_t2_attacks(art_model, sem_model,
                        art_layer, sem_layer,
                        images: torch.Tensor,
                        labels: torch.Tensor,
                        eps: float = 8/255,
                        lam: float = 1.0,
                        steps: int = 20,
                        device: str = "cuda") -> torch.Tensor:
    """
    Generate T2 adaptive attacks for a batch of images.
    Returns tensor of adversarial images (CPU).
    """
    adv_list = []
    for img, lbl in zip(images, labels):
        adv = pgd_adaptive_csed(
            art_model, sem_model, art_layer, sem_layer,
            img, lbl, eps=eps, steps=steps, lam=lam, device=device
        )
        adv_list.append(adv)
        torch.cuda.empty_cache()
    return torch.stack(adv_list)
