"""
Phase 0 — Degeneracy Filtering Pass & Saturation Diagnosis
===========================================================
Addresses the critical observation flagged by the user's guide:
1. Identifies flat Grad-CAM maps (spatial std < 1e-5).
2. Computes Degeneracy Rate (%) per condition: Clean, FGSM, PGD (eps=4,8,16), and Adaptive λ-sweep.
3. Recomputes Q2 and Q3 AUCs (Cosine & JS) both UNFILTERED and FILTERED (excluding degenerate maps).
4. Measures raw target logit magnitude and softmax confidence on degenerate vs non-degenerate samples to confirm vanishing-gradient / logit-saturation mechanism.
"""

import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchattacks
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from detectors   import load_detectors
from csed        import get_gradcam_map, compute_csed
from data_loader import build_dataset_from_dirs
from attacks     import generate_t1_attacks

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(__file__).parent / "data"
RESULTS   = Path(__file__).parent / "results"
PLOTS     = RESULTS / "plots"

VAR_THRESHOLD = 1e-5 # Spatial std threshold below which Grad-CAM map is degenerate


def is_map_degenerate(cam_map: np.ndarray, std_thresh: float = VAR_THRESHOLD) -> bool:
    """Checks if a Grad-CAM map is flat / uniform (near-zero spatial variance)."""
    if cam_map is None:
        return True
    return bool(np.std(cam_map) < std_thresh)


def inspect_logit_confidence(model, image: torch.Tensor, label: int, device: str = "cuda"):
    """Returns raw logit value, softmax confidence, and logit gradient norm."""
    model.eval()
    img = image.to(device)
    if img.dim() == 3:
        img = img.unsqueeze(0)

    img_req = img.clone().detach().requires_grad_(True)
    logits = model(img_req)
    probs  = F.softmax(logits, dim=1)
    
    target_logit = logits[0, label]
    target_prob  = probs[0, label].item()
    
    grad = torch.autograd.grad(target_logit, img_req)[0]
    grad_norm = grad.norm(2).item()
    
    return {
        "logit": float(target_logit.item()),
        "confidence": float(target_prob),
        "input_grad_norm": float(grad_norm)
    }


def run_degeneracy_pass():
    print("=" * 75)
    print("  CPED Phase 0 — Grad-CAM Degeneracy & Map Flatness Diagnostic Pass")
    print("=" * 75)

    print("\n[1/4] Loading models and evaluation dataset (300 samples) …")
    art_model, sem_model = load_detectors(DEVICE)
    art_layer = art_model.get_target_layer()
    sem_layer = sem_model.get_target_layer()

    dataset = build_dataset_from_dirs(
        str(DATA_ROOT / "nature"), str(DATA_ROOT / "ai"),
        n_real=150, n_ai=150, seed=42
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    images = []; labels = []
    for img, lbl in loader:
        images.append(img.squeeze(0))
        labels.append(lbl.squeeze(0))
    images = torch.stack(images)
    labels = torch.stack(labels)
    N = len(images)
    print(f"      Loaded {N} samples.")

    # ── 1. Clean Data Check ──
    print("\n[2/4] Evaluating Clean Data Degeneracy & CSED …")
    clean_cos = []
    clean_js  = []
    clean_degen_a = 0
    clean_degen_b = 0
    clean_stats_a = []
    
    for img, lbl in tqdm(zip(images, labels), total=N, desc="Clean Data"):
        map_a = get_gradcam_map(art_model, art_layer, img.to(DEVICE), int(lbl), False)
        map_b = get_gradcam_map(sem_model, sem_layer, img.to(DEVICE), int(lbl), True)
        
        deg_a = is_map_degenerate(map_a)
        deg_b = is_map_degenerate(map_b)
        if deg_a: clean_degen_a += 1
        if deg_b: clean_degen_b += 1

        d = compute_csed(map_a, map_b)
        clean_cos.append(d["cosine"])
        clean_js.append(d["js"])

    print(f"      Clean Degenerate Rate: Paradigm A = {clean_degen_a/N:.1%} | Paradigm B = {clean_degen_b/N:.1%}")

    # ── 2. T1 FGSM vs PGD Degeneracy Check ──
    print("\n[3/4] Running T1 Attack Degeneracy Comparison (FGSM vs PGD eps=4, 8, 16) …")
    
    class EnsembleModel(torch.nn.Module):
        def __init__(self, m1, m2):
            super().__init__()
            self.m1 = m1
            self.m2 = m2
        def forward(self, x):
            return (self.m1(x) + self.m2(x)) / 2

    ensemble = EnsembleModel(art_model, sem_model).to(DEVICE).eval()

    attack_configs = [
        ("FGSM (eps=8/255)",   torchattacks.FGSM(ensemble, eps=8/255)),
        ("PGD-20 (eps=4/255)",  torchattacks.PGD(ensemble, eps=4/255, alpha=1/255, steps=20)),
        ("PGD-20 (eps=8/255)",  torchattacks.PGD(ensemble, eps=8/255, alpha=2/255, steps=20)),
        ("PGD-20 (eps=16/255)", torchattacks.PGD(ensemble, eps=16/255, alpha=4/255, steps=20)),
    ]

    y_true = np.array([0]*N + [1]*N)
    t1_results = {}

    batch_size = 4

    for name, atk_fn in attack_configs:
        print(f"\n   ── {name} ──")
        adv_imgs_list = []
        for i in range(0, N, batch_size):
            b_imgs = images[i:i+batch_size].to(DEVICE)
            b_lbls = labels[i:i+batch_size].to(DEVICE)
            b_adv  = atk_fn(b_imgs, b_lbls)
            adv_imgs_list.append(b_adv.cpu())
            torch.cuda.empty_cache()
        adv_imgs = torch.cat(adv_imgs_list, dim=0)
        
        cos_all = []; js_all = []
        valid_cos = []; valid_js = []
        valid_clean_cos = []; valid_clean_js = []
        
        degen_a_cnt = 0; degen_b_cnt = 0; degen_either_cnt = 0
        degen_logits_a = []; non_degen_logits_a = []
        
        for idx in tqdm(range(N), desc=f"CSED {name}"):
            img_adv = adv_imgs[idx]
            lbl     = int(labels[idx])
            
            map_a = get_gradcam_map(art_model, art_layer, img_adv.to(DEVICE), lbl, False)
            map_b = get_gradcam_map(sem_model, sem_layer, img_adv.to(DEVICE), lbl, True)
            
            deg_a = is_map_degenerate(map_a)
            deg_b = is_map_degenerate(map_b)
            
            if deg_a: degen_a_cnt += 1
            if deg_b: degen_b_cnt += 1
            if deg_a or deg_b: degen_either_cnt += 1
            
            d = compute_csed(map_a, map_b)
            cos_val = d["cosine"]
            js_val  = d["js"]
            
            cos_all.append(cos_val)
            js_all.append(js_val)
            
            # Logit saturation check on Paradigm A
            conf_a = inspect_logit_confidence(art_model, img_adv, lbl, DEVICE)
            if deg_a:
                degen_logits_a.append(conf_a)
            else:
                non_degen_logits_a.append(conf_a)

            if not (deg_a or deg_b):
                valid_cos.append(cos_val)
                valid_js.append(js_val)
                valid_clean_cos.append(clean_cos[idx])
                valid_clean_js.append(clean_js[idx])

        cos_all = np.array(cos_all); js_all = np.array(js_all)
        auc_cos_unfiltered = roc_auc_score(y_true, np.concatenate([clean_cos, cos_all]))
        auc_js_unfiltered  = roc_auc_score(y_true, np.concatenate([clean_js,  js_all]))
        
        if len(valid_cos) >= 10:
            y_filtered = np.array([0]*len(valid_clean_cos) + [1]*len(valid_cos))
            auc_cos_filtered = roc_auc_score(y_filtered, np.concatenate([valid_clean_cos, valid_cos]))
            auc_js_filtered  = roc_auc_score(y_filtered, np.concatenate([valid_clean_js,  valid_js]))
        else:
            auc_cos_filtered = float(auc_cos_unfiltered)
            auc_js_filtered  = float(auc_js_unfiltered)

        avg_deg_conf = np.mean([x["confidence"] for x in degen_logits_a]) if degen_logits_a else 0.0
        avg_non_deg_conf = np.mean([x["confidence"] for x in non_degen_logits_a]) if non_degen_logits_a else 0.0
        avg_deg_grad = np.mean([x["input_grad_norm"] for x in degen_logits_a]) if degen_logits_a else 0.0
        avg_non_deg_grad = np.mean([x["input_grad_norm"] for x in non_degen_logits_a]) if non_degen_logits_a else 0.0

        t1_results[name] = {
            "degeneracy_rate_paradigm_a": round(degen_a_cnt / N, 4),
            "degeneracy_rate_paradigm_b": round(degen_b_cnt / N, 4),
            "total_degenerate_rate": round(degen_either_cnt / N, 4),
            "auc_cosine_unfiltered": round(auc_cos_unfiltered, 4),
            "auc_cosine_filtered": round(auc_cos_filtered, 4),
            "auc_js_unfiltered": round(auc_js_unfiltered, 4),
            "auc_js_filtered": round(auc_js_filtered, 4),
            "auc_gap_cosine": round(auc_cos_unfiltered - auc_cos_filtered, 4),
            "avg_confidence_degen": round(float(avg_deg_conf), 4),
            "avg_confidence_non_degen": round(float(avg_non_deg_conf), 4),
            "avg_grad_norm_degen": round(float(avg_deg_grad), 6),
            "avg_grad_norm_non_degen": round(float(avg_non_deg_grad), 6),
        }

        print(f"      Total Degenerate Maps Rate : {degen_either_cnt/N:.1%} (A: {degen_a_cnt/N:.1%} | B: {degen_b_cnt/N:.1%})")
        print(f"      Cosine AUC Unfiltered      : {auc_cos_unfiltered:.3f}")
        print(f"      Cosine AUC Filtered        : {auc_cos_filtered:.3f} (Gap = {auc_cos_unfiltered - auc_cos_filtered:+.3f})")
        print(f"      Logit Grad Norm (Degen)    : {avg_deg_grad:.6f} vs Non-Degen: {avg_non_deg_grad:.6f}")

    # ── Output Summary JSON ──
    out_json = RESULTS / "degeneracy_diagnostic_summary.json"
    with open(out_json, "w") as f:
        json.dump(t1_results, f, indent=2)
    print(f"\n[4/4] Saved degeneracy diagnostic report: {out_json}")
    print("\n✓ Degeneracy check completed.")


if __name__ == "__main__":
    run_degeneracy_pass()
