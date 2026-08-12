"""
Phase 1 — CSED Pipeline (Cross-Stream Explanation Divergence)
==============================================================
Upgraded from Phase 0 with:
  1. Pure FP32 throughout (no FP16 autocast in any Grad-CAM pass)
  2. Standard Grad-CAM (primary metric)
  3. Grad-CAM++ (architectural upgrade — better localization, tested in Phase 1)
  4. Multi-layer depth ensemble CSED (average divergence across multiple layers)

Phase 1 CSED modes:
  csed_mode='gradcam'      — Phase 0 baseline (1 target layer per stream)
  csed_mode='gradcam++'    — Grad-CAM++ (better localization for attribution)
  csed_mode='ensemble'     — Average CSED across multiple layers per stream

The degeneracy filter (std < 1e-5) and all AUC statistics are computed exactly
as in Phase 0, but now in pure FP32 throughout.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.spatial.distance import jensenshannon
from typing import Literal, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# ViT reshape transform (CLIP ViT-L/14, 16×16 patch grid)
# ──────────────────────────────────────────────────────────────────────────────
def clip_reshape_transform(tensor, height=16, width=16):
    """
    CLIP ViT-L/14 processes 224×224 images as 16×16 patches.
    Activation tensor from transformer resblocks: (257, Batch, 1024).
    Returns: (Batch, 1024, 16, 16)
    """
    if tensor.dim() == 3 and tensor.shape[0] == 257:
        tensor = tensor.permute(1, 0, 2)   # → (Batch, 257, 1024)
    result = tensor[:, 1:, :]              # drop CLS → (B, 256, 1024)
    result = result.reshape(result.shape[0], height, width, result.shape[2])
    result = result.permute(0, 3, 1, 2)   # → (B, 1024, 16, 16)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core Grad-CAM / Grad-CAM++ extractor (pure FP32)
# ──────────────────────────────────────────────────────────────────────────────
def get_gradcam_map(
    model,
    target_layer,
    image_tensor: torch.Tensor,
    target_class: int = 1,
    is_vit: bool = False,
    mode: Literal["gradcam", "gradcam++"] = "gradcam",
) -> np.ndarray:
    """
    Extract a Grad-CAM or Grad-CAM++ heat-map for one image.

    Args:
        model:          Detector model (NPRDetector or UnivFDDetector).
        target_layer:   Layer to hook for explanation.
        image_tensor:   Single image tensor (C, H, W) or (1, C, H, W), FP32.
        target_class:   Class index to explain (1 = AI/fake).
        is_vit:         Use CLIP ViT reshape transform.
        mode:           'gradcam' (Phase 0 baseline) or 'gradcam++'.

    Returns:
        2D numpy array (224, 224) in [0, 1].
    """
    reshape_fn = clip_reshape_transform if is_vit else None
    cam_cls = GradCAMPlusPlus if mode == "gradcam++" else GradCAM

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Ensure pure FP32 — critical to prevent FP16 underflow in backward pass
    image_tensor = image_tensor.float()

    with cam_cls(
        model=model,
        target_layers=[target_layer],
        reshape_transform=reshape_fn,
    ) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)

    heatmap = grayscale_cam[0]   # (H, W) in [0, 1]
    hm_t = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    hm_resized = F.interpolate(hm_t, size=(224, 224), mode="bilinear",
                               align_corners=False)
    return hm_resized.squeeze().numpy()


def get_ensemble_map(
    model,
    target_layers: List,
    image_tensor: torch.Tensor,
    target_class: int = 1,
    is_vit: bool = False,
    mode: Literal["gradcam", "gradcam++"] = "gradcam",
) -> np.ndarray:
    """
    Compute a depth-ensemble explanation map by averaging Grad-CAM/Grad-CAM++
    maps across multiple target layers.

    Returns:
        2D numpy array (224, 224), averaged across all layers, in [0, 1].
    """
    maps = []
    for layer in target_layers:
        m = get_gradcam_map(model, layer, image_tensor,
                            target_class=target_class,
                            is_vit=is_vit, mode=mode)
        maps.append(m)
    # Average across layers, then re-normalize to [0, 1]
    avg_map = np.mean(maps, axis=0)
    mn, mx = avg_map.min(), avg_map.max()
    if mx - mn > 1e-8:
        avg_map = (avg_map - mn) / (mx - mn)
    return avg_map


# ──────────────────────────────────────────────────────────────────────────────
# Degeneracy check
# ──────────────────────────────────────────────────────────────────────────────
DEGENERACY_THRESHOLD = 1e-5  # spatial std threshold for dying-ReLU detection

def is_degenerate(heatmap: np.ndarray) -> bool:
    """Return True if the heatmap is spatially flat (dying-ReLU signature)."""
    return bool(np.std(heatmap) < DEGENERACY_THRESHOLD)


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
    Compute CSED between two explanation maps.

    Returns:
        dict with:
          - 'cosine':  cosine distance ∈ [0, 2]  (primary Phase 1 metric)
          - 'js':      JS divergence ∈ [0, 1]    (sqrt form, secondary metric)
    """
    flat_a = map_a.flatten().astype(np.float64)
    flat_b = map_b.flatten().astype(np.float64)

    # Cosine distance
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        cos_dist = 1.0   # flat map → treated as maximal local distance
    else:
        cos_sim = np.dot(flat_a, flat_b) / (norm_a * norm_b)
        cos_dist = 1.0 - float(cos_sim)

    # Jensen–Shannon divergence (sqrt form, base-2, ∈ [0, 1])
    p = _to_prob_dist(map_a)
    q = _to_prob_dist(map_b)
    js = float(jensenshannon(p, q, base=2))

    return {"cosine": cos_dist, "js": js}


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample CSED extractor (pure FP32, all modes)
# ──────────────────────────────────────────────────────────────────────────────
def extract_csed_sample(
    art_model,
    sem_model,
    art_layer,
    sem_layer,
    image_tensor: torch.Tensor,
    target_class: int = 1,
    csed_mode: Literal["gradcam", "gradcam++", "ensemble"] = "gradcam",
    art_multi_layers: Optional[List] = None,
    sem_multi_layers: Optional[List] = None,
) -> dict:
    """
    Extract CSED metrics for a single image.

    Args:
        art_model/sem_model:     NPRDetector / UnivFDDetector.
        art_layer/sem_layer:     Primary target layers (for gradcam/gradcam++).
        image_tensor:            Single image (C,H,W) or (1,C,H,W), FP32.
        target_class:            Class to explain (1 = AI/fake).
        csed_mode:               Explanation method.
        art_multi_layers:        Layer list for art ensemble (csed_mode='ensemble').
        sem_multi_layers:        Layer list for sem ensemble (csed_mode='ensemble').

    Returns:
        dict: {
            'map_a': np.ndarray,   art explanation map (224,224)
            'map_b': np.ndarray,   sem explanation map (224,224)
            'is_deg_a': bool,      art map degenerate?
            'is_deg_b': bool,      sem map degenerate?
            'is_deg':  bool,       either degenerate?
            'cosine':  float,      cosine distance
            'js':      float,      JS divergence
        }
    """
    if csed_mode == "ensemble":
        art_layers = art_multi_layers or [art_layer]
        sem_layers = sem_multi_layers or [sem_layer]
        map_a = get_ensemble_map(art_model, art_layers, image_tensor,
                                 target_class=target_class, is_vit=False,
                                 mode="gradcam")
        map_b = get_ensemble_map(sem_model, sem_layers, image_tensor,
                                 target_class=target_class, is_vit=True,
                                 mode="gradcam")
    else:
        grad_mode = csed_mode  # 'gradcam' or 'gradcam++'
        map_a = get_gradcam_map(art_model, art_layer, image_tensor,
                                target_class=target_class, is_vit=False,
                                mode=grad_mode)
        map_b = get_gradcam_map(sem_model, sem_layer, image_tensor,
                                target_class=target_class, is_vit=True,
                                mode=grad_mode)

    deg_a = is_degenerate(map_a)
    deg_b = is_degenerate(map_b)
    csed = compute_csed(map_a, map_b)

    return {
        "map_a":    map_a,
        "map_b":    map_b,
        "is_deg_a": deg_a,
        "is_deg_b": deg_b,
        "is_deg":   deg_a or deg_b,
        "cosine":   csed["cosine"],
        "js":       csed["js"],
    }
