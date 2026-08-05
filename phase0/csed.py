"""
Phase 0 — CSED (Cross-Stream Explanation Divergence) Pipeline
==============================================================
Computes Grad-CAM maps for both detectors and measures divergence
using cosine distance and Jensen–Shannon divergence.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.spatial.distance import jensenshannon


# ──────────────────────────────────────────────────────────────────────────────
# reshape_transform for CLIP ViT (required by pytorch-grad-cam for transformers)
# ──────────────────────────────────────────────────────────────────────────────
def clip_reshape_transform(tensor, height=16, width=16):
    """
    CLIP ViT-L/14 processes 224×224 images as 16×16 patches.
    Activation tensor from transformer resblocks has shape (257, Batch, 1024).
    """
    if tensor.dim() == 3 and tensor.shape[0] == 257:
        tensor = tensor.permute(1, 0, 2)  # → (Batch, 257, 1024)
    result = tensor[:, 1:, :]        # drop CLS token → (B, 256, 1024)
    result = result.reshape(result.shape[0], height, width, result.shape[2])
    result = result.permute(0, 3, 1, 2)   # → (B, 1024, 16, 16)
    return result


def get_gradcam_map(model, target_layer, image_tensor: torch.Tensor,
                    target_class: int = 1,
                    is_vit: bool = False) -> np.ndarray:
    """
    Extract a Grad-CAM heat-map for one image.

    Returns a 2D numpy array (H, W) in [0, 1], resized to 224×224.
    """
    reshape_fn = clip_reshape_transform if is_vit else None

    with GradCAM(model=model,
                 target_layers=[target_layer],
                 reshape_transform=reshape_fn) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0),
                            targets=targets)
    heatmap = grayscale_cam[0]  # (H, W) already in [0,1]
    # Resize to 224×224 for consistent comparison
    hm_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    hm_resized = F.interpolate(hm_t, size=(224, 224), mode='bilinear',
                               align_corners=False)
    return hm_resized.squeeze().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────────────
def _to_prob_dist(arr: np.ndarray) -> np.ndarray:
    """Flatten, clip negatives, normalise to a probability distribution."""
    flat = np.clip(arr.flatten(), 0, None)
    s = flat.sum()
    if s < 1e-8:
        return np.ones_like(flat) / flat.size   # uniform if map is blank
    return flat / s


# ──────────────────────────────────────────────────────────────────────────────
# CSED computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_csed(map_a: np.ndarray, map_b: np.ndarray) -> dict:
    """
    Given two Grad-CAM maps of the same shape, compute:
      - cosine_distance   ∈ [0, 2]
      - js_divergence     ∈ [0, 1]   (square-root form, bounded)

    Returns a dict so callers can choose which metric to use.
    """
    flat_a = map_a.flatten().astype(np.float64)
    flat_b = map_b.flatten().astype(np.float64)

    # Cosine distance (1 − cosine_similarity)
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        cos_dist = 1.0
    else:
        cos_sim = np.dot(flat_a, flat_b) / (norm_a * norm_b)
        cos_dist = 1.0 - float(cos_sim)

    # Jensen–Shannon divergence (sqrt form — a proper metric in [0,1])
    p = _to_prob_dist(map_a)
    q = _to_prob_dist(map_b)
    js = float(jensenshannon(p, q, base=2))   # base-2 JS in [0, 1]

    return {"cosine": cos_dist, "js": js}


# ──────────────────────────────────────────────────────────────────────────────
# Batch extractor
# ──────────────────────────────────────────────────────────────────────────────
def extract_csed_batch(art_model, sem_model,
                       art_layer, sem_layer,
                       images: torch.Tensor,
                       target_class: int = 1,
                       device: str = "cuda") -> list[dict]:
    """
    Process a batch of images and return a list of CSED dicts.
    Each dict: {'cosine': float, 'js': float}
    """
    results = []
    art_model.eval()
    sem_model.eval()
    for img in images:
        img = img.to(device)
        # Paradigm 1
        map_a = get_gradcam_map(art_model, art_layer, img,
                                target_class=target_class, is_vit=False)
        # Paradigm 2
        map_b = get_gradcam_map(sem_model, sem_layer, img,
                                target_class=target_class, is_vit=True)
        results.append(compute_csed(map_a, map_b))
    return results
