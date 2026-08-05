"""
Phase 0 — Per-Epsilon Analysis
================================
Generates AUC vs epsilon curves for T1 attacks.
Run AFTER run_phase0.py has completed.
"""

import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from detectors   import load_detectors
from csed        import get_gradcam_map, compute_csed
from data_loader import build_dataset_from_dirs
from attacks     import generate_t1_attacks

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(__file__).parent / "data"
RESULTS   = Path(__file__).parent / "results"
PLOTS     = RESULTS / "plots"

EPSILONS = [2/255, 4/255, 8/255, 12/255, 16/255]


def run_epsilon_sweep():
    print("[Epsilon sweep] Loading models and data …")
    art_model, sem_model = load_detectors(DEVICE)
    art_layer = art_model.get_target_layer()
    sem_layer = sem_model.get_target_layer()

    dataset = build_dataset_from_dirs(
        str(DATA_ROOT/"nature"), str(DATA_ROOT/"ai"),
        n_real=200, n_ai=200, seed=42
    )
    all_images = []; all_labels = []
    from torch.utils.data import DataLoader
    for img, lbl in DataLoader(dataset, batch_size=1):
        all_images.append(img.squeeze(0))
        all_labels.append(lbl.squeeze(0))
    all_images = torch.stack(all_images)
    all_labels = torch.stack(all_labels)
    N = len(all_images)

    # Clean CSED
    clean_cos = []; clean_js = []
    for img, lbl in tqdm(zip(all_images, all_labels), total=N, desc="Clean"):
        map_a = get_gradcam_map(art_model, art_layer, img.to(DEVICE), int(lbl), False)
        map_b = get_gradcam_map(sem_model, sem_layer, img.to(DEVICE), int(lbl), True)
        d = compute_csed(map_a, map_b)
        clean_cos.append(d["cosine"])
        clean_js.append(d["js"])
    clean_cos = np.array(clean_cos)
    clean_js  = np.array(clean_js)
    y_q2 = np.array([0]*N + [1]*N)

    t1_attacked = generate_t1_attacks(art_model, sem_model, all_images,
                                      all_labels, epsilons=EPSILONS, device=DEVICE)

    pgd_aucs_cos = []; pgd_aucs_js = []
    fgsm_aucs_cos = []; fgsm_aucs_js = []

    for eps in EPSILONS:
        adv_pgd  = t1_attacked[eps]["pgd"]
        adv_fgsm = t1_attacked[eps]["fgsm"]

        pgd_cos=[]; pgd_js=[]; fgsm_cos=[]; fgsm_js=[]
        for img_p, img_f, lbl in tqdm(zip(adv_pgd, adv_fgsm, all_labels),
                                       total=N, desc=f"eps={eps:.0%}"):
            lbl = int(lbl)
            mp_a = get_gradcam_map(art_model, art_layer, img_p.to(DEVICE), lbl, False)
            mp_b = get_gradcam_map(sem_model, sem_layer, img_p.to(DEVICE), lbl, True)
            dp = compute_csed(mp_a, mp_b)
            pgd_cos.append(dp["cosine"]); pgd_js.append(dp["js"])

            mf_a = get_gradcam_map(art_model, art_layer, img_f.to(DEVICE), lbl, False)
            mf_b = get_gradcam_map(sem_model, sem_layer, img_f.to(DEVICE), lbl, True)
            df = compute_csed(mf_a, mf_b)
            fgsm_cos.append(df["cosine"]); fgsm_js.append(df["js"])

        pgd_cos  = np.array(pgd_cos)
        pgd_js   = np.array(pgd_js)
        fgsm_cos = np.array(fgsm_cos)
        fgsm_js  = np.array(fgsm_js)

        pgd_aucs_cos.append(roc_auc_score(y_q2, np.concatenate([clean_cos, pgd_cos])))
        pgd_aucs_js.append( roc_auc_score(y_q2, np.concatenate([clean_js,  pgd_js])))
        fgsm_aucs_cos.append(roc_auc_score(y_q2, np.concatenate([clean_cos, fgsm_cos])))
        fgsm_aucs_js.append( roc_auc_score(y_q2, np.concatenate([clean_js,  fgsm_js])))

    eps_pct = [e*255 for e in EPSILONS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CSED AUC vs Perturbation Budget (T1)", fontsize=12)
    for ax, metric, pgd_a, fgsm_a in zip(
            axes,
            ["Cosine Distance", "JS Divergence"],
            [pgd_aucs_cos, pgd_aucs_js],
            [fgsm_aucs_cos, fgsm_aucs_js]):
        ax.plot(eps_pct, pgd_a,  "o-", color="#E53935", label="PGD")
        ax.plot(eps_pct, fgsm_a, "s--", color="#FB8C00", label="FGSM")
        ax.axhline(0.60, color="grey", ls=":", lw=1, label="Q3 bar")
        ax.axhline(0.65, color="black", ls=":", lw=1, label="Q2 bar")
        ax.set_xlabel("Perturbation ε (pixel/255)", fontsize=10)
        ax.set_ylabel(f"AUC (CSED {metric})", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylim(0.45, 1.05)
    plt.tight_layout()
    out = PLOTS / "epsilon_sweep.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[Sweep] Saved: {out}")


if __name__ == "__main__":
    run_epsilon_sweep()
