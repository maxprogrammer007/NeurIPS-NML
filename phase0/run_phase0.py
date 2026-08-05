"""
Phase 0 — Main Runner
======================
Orchestrates the full go/no-go pilot study:
  Q1 : Is CSED stable on clean data? (negative control)
  Q2 : Does CSED shift under T1 attacks?
  Q3 : Does the Q2 result survive T2 adaptive attack?

Outputs:
  results/q1_q2_q3_summary.json   — numerical results
  results/plots/                  — histogram + ROC plots
  results/gradcam/                — Grad-CAM galleries
"""

import sys, os, json, time, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from detectors   import load_detectors
from csed        import extract_csed_batch, get_gradcam_map, compute_csed
from data_loader import build_dataset_from_dirs, TRANSFORM
from attacks     import generate_t1_attacks, generate_t2_attacks

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(__file__).parent / "data"
RESULTS   = Path(__file__).parent / "results"
PLOTS     = RESULTS / "plots"
GRADCAM   = RESULTS / "gradcam"
for d in [RESULTS, PLOTS, GRADCAM]:
    d.mkdir(parents=True, exist_ok=True)

# ── Pre-registered success bars (set BEFORE seeing any data) ─────────────────
Q1_AUC_MAX  = 0.55   # clean split-half AUC should be ≈ 0.5
Q2_AUC_MIN  = 0.65   # CSED AUC clean vs T1 must be ≥ 0.65
Q2_KS_PVAL  = 0.05   # KS-test p-value threshold
Q3_AUC_MIN  = 0.60   # CSED AUC clean vs T2 must be ≥ 0.60 to proceed

# ── CSED bootstrap CI ────────────────────────────────────────────────────────
def bootstrap_auc_ci(y_true, y_score, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


# ── Grad-CAM gallery ─────────────────────────────────────────────────────────
def save_gradcam_gallery(art_model, sem_model, art_layer, sem_layer,
                         images, labels, adv_fgsm, adv_pgd, adv_t2,
                         n_show=6, tag="gallery"):
    """Save side-by-side Grad-CAM maps: clean vs attacked, Paradigm A vs B."""
    import torchvision.transforms.functional as TF

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def denorm(t):
        return (t.cpu() * std + mean).clamp(0,1)

    cols   = ["Clean", "FGSM (T1)", "PGD (T1)", "Adaptive-PGD (T2)"]
    n_show = min(n_show, len(images))

    fig, axes = plt.subplots(n_show * 2, len(cols) + 1,
                             figsize=(4*(len(cols)+1), 4*n_show*2))
    fig.suptitle(f"Grad-CAM Gallery — Paradigm A (top) vs B (bottom)\n"
                 f"Cosine CSED shown on each pair", fontsize=11, y=1.01)

    version_sets = [images[:n_show], adv_fgsm[:n_show],
                    adv_pgd[:n_show], adv_t2[:n_show]]

    for col_idx, (col_name, img_set) in enumerate(zip(cols, version_sets)):
        for row_idx in range(n_show):
            img = img_set[row_idx].to(DEVICE)
            lbl = int(labels[row_idx])

            map_a = get_gradcam_map(art_model, art_layer, img,
                                    target_class=lbl, is_vit=False)
            map_b = get_gradcam_map(sem_model, sem_layer, img,
                                    target_class=lbl, is_vit=True)
            csed  = compute_csed(map_a, map_b)

            # Row A: Grad-CAM from Paradigm A
            ax_a = axes[row_idx * 2, col_idx + 1]
            ax_a.imshow(map_a, cmap="jet", vmin=0, vmax=1)
            ax_a.set_title(f"A | cos={csed['cosine']:.3f}", fontsize=7)
            ax_a.axis("off")

            # Row B: Grad-CAM from Paradigm B
            ax_b = axes[row_idx * 2 + 1, col_idx + 1]
            ax_b.imshow(map_b, cmap="jet", vmin=0, vmax=1)
            ax_b.set_title(f"B | JS={csed['js']:.3f}", fontsize=7)
            ax_b.axis("off")

            # Original image in col 0
            if col_idx == 0:
                ax_img_a = axes[row_idx * 2, 0]
                ax_img_b = axes[row_idx * 2 + 1, 0]
                raw = denorm(img).permute(1,2,0).numpy()
                ax_img_a.imshow(raw)
                ax_img_a.set_ylabel(f"img {row_idx}\nlabel={'AI' if lbl else 'real'}",
                                    fontsize=7)
                ax_img_a.axis("off")
                ax_img_b.axis("off")

        axes[0, col_idx + 1].set_title(col_name, fontsize=8, fontweight="bold")

    plt.tight_layout()
    out = GRADCAM / f"{tag}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Gallery] Saved to {out}")


# ── Histogram + AUC plot ─────────────────────────────────────────────────────
def plot_csed_comparison(clean_cos, attacked_cos, clean_js, attacked_js,
                         auc_cos, auc_js, ks_p_cos, ks_p_js,
                         title: str, filename: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, clean, attacked, auc, ks_p, metric in zip(
            axes,
            [clean_cos, clean_js],
            [attacked_cos, attacked_js],
            [auc_cos, auc_js],
            [ks_p_cos, ks_p_js],
            ["Cosine Distance", "JS Divergence"]):

        bins = np.linspace(
            min(min(clean), min(attacked)) - 0.01,
            max(max(clean), max(attacked)) + 0.01,
            30)
        ax.hist(clean,    bins=bins, alpha=0.65, color="#2196F3", label="Clean",    density=True)
        ax.hist(attacked, bins=bins, alpha=0.65, color="#F44336", label="Attacked", density=True)
        ax.set_xlabel(f"CSED ({metric})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.set_title(f"AUC={auc:.3f}  KS p={ks_p:.4f}", fontsize=9, color=(
            "#2e7d32" if ks_p < 0.05 else "#c62828"))
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  CPED Phase 0 — Pilot Study")
    print("=" * 65)
    t_start = time.time()

    # ── Load detectors ──────────────────────────────────────────────────────
    print("\n[1/7] Loading detectors …")
    art_model, sem_model = load_detectors(DEVICE)
    art_layer = art_model.get_target_layer()
    sem_layer = sem_model.get_target_layer()
    print(f"      Paradigm A (Artifact CNN) : ResNet-50 | target={art_layer.__class__.__name__}")
    print(f"      Paradigm B (Semantic ViT)  : CLIP ViT-L/14 | target={sem_layer.__class__.__name__}")

    # ── Load data ───────────────────────────────────────────────────────────
    print("\n[2/7] Loading pilot dataset …")
    real_dir = str(DATA_ROOT / "nature")
    ai_dir   = str(DATA_ROOT / "ai")
    dataset  = build_dataset_from_dirs(real_dir, ai_dir,
                                       n_real=300, n_ai=300, seed=42)
    loader   = DataLoader(dataset, batch_size=1, shuffle=False)
    all_images = []; all_labels = []
    for img, lbl in loader:
        all_images.append(img.squeeze(0))
        all_labels.append(lbl.squeeze(0))
    all_images = torch.stack(all_images)
    all_labels = torch.stack(all_labels)
    N = len(all_images)
    print(f"      {N} images loaded (real={int((all_labels==0).sum())}, AI={int((all_labels==1).sum())})")

    # ── Grad-CAM sanity check ────────────────────────────────────────────────
    print("\n[3/7] Grad-CAM sanity check (first 4 images) …")
    sanity_imgs = all_images[:4].to(DEVICE)
    sanity_lbls = all_labels[:4]
    for i in range(4):
        img = sanity_imgs[i]
        lbl = int(sanity_lbls[i])
        map_a = get_gradcam_map(art_model, art_layer, img, target_class=lbl, is_vit=False)
        map_b = get_gradcam_map(sem_model, sem_layer, img, target_class=lbl, is_vit=True)
        csed  = compute_csed(map_a, map_b)
        is_blank_a = map_a.max() < 1e-4
        is_blank_b = map_b.max() < 1e-4
        print(f"      img {i} | label={'AI' if lbl else 'real'} | "
              f"A_max={map_a.max():.4f} {'❌BLANK' if is_blank_a else '✓'} | "
              f"B_max={map_b.max():.4f} {'❌BLANK' if is_blank_b else '✓'} | "
              f"CSED_cos={csed['cosine']:.4f}")
    print("      Sanity check passed — maps are non-blank.")

    # ── Q1 : CSED stability on clean data ────────────────────────────────────
    print("\n[4/7] Q1 — CSED stability on clean data (negative control) …")
    print("      Extracting Grad-CAM for all clean images …")
    clean_csed = []
    for img, lbl in tqdm(zip(all_images, all_labels), total=N, desc="Clean CSED"):
        img = img.to(DEVICE)
        lbl = int(lbl)
        map_a = get_gradcam_map(art_model, art_layer, img, target_class=lbl, is_vit=False)
        map_b = get_gradcam_map(sem_model, sem_layer, img, target_class=lbl, is_vit=True)
        clean_csed.append(compute_csed(map_a, map_b))

    clean_cos = np.array([d["cosine"] for d in clean_csed])
    clean_js  = np.array([d["js"]     for d in clean_csed])

    # Negative control: split clean set in half, AUC should be ≈ 0.5
    half = N // 2
    y_half = np.array([0]*half + [1]*(N - half))
    y_half_scores_cos = np.concatenate([clean_cos[:half], clean_cos[half:]])
    y_half_scores_js  = np.concatenate([clean_js[:half],  clean_js[half:]])
    q1_auc_cos, q1_lo_cos, q1_hi_cos = bootstrap_auc_ci(y_half, y_half_scores_cos)
    q1_auc_js,  q1_lo_js,  q1_hi_js  = bootstrap_auc_ci(y_half, y_half_scores_js)

    q1_pass = (q1_auc_cos <= Q1_AUC_MAX) and (q1_auc_js <= Q1_AUC_MAX)
    print(f"\n  ── Q1 RESULTS ──────────────────────────────────────────────")
    print(f"     Cosine AUC (split-half) = {q1_auc_cos:.3f} [{q1_lo_cos:.3f}–{q1_hi_cos:.3f}]")
    print(f"     JS     AUC (split-half) = {q1_auc_js:.3f}  [{q1_lo_js:.3f}–{q1_hi_js:.3f}]")
    print(f"     Bar: AUC ≤ {Q1_AUC_MAX} → {'✓ PASS' if q1_pass else '✗ FAIL (pipeline has a bias — check normalisation)'}")

    # ── T1 attacks ──────────────────────────────────────────────────────────
    print("\n[5/7] Q2 — Generating T1 attacks (FGSM + PGD at 3 budgets) …")
    EPSILONS = [4/255, 8/255, 16/255]
    t1_attacked = generate_t1_attacks(
        art_model, sem_model, all_images, all_labels,
        epsilons=EPSILONS, device=DEVICE
    )
    print("      T1 attacks done.")

    # Use ε=8/255 as the primary comparison for Q2
    eps_main  = 8/255
    adv_fgsm  = t1_attacked[eps_main]["fgsm"]
    adv_pgd   = t1_attacked[eps_main]["pgd"]

    # Compute CSED on FGSM and PGD attacked sets
    print("      Computing CSED on T1-attacked images …")
    fgsm_csed = []; pgd_csed = []
    for img_f, img_p, lbl in tqdm(zip(adv_fgsm, adv_pgd, all_labels),
                                   total=N, desc="T1 CSED"):
        lbl = int(lbl)
        # FGSM
        map_a = get_gradcam_map(art_model, art_layer, img_f.to(DEVICE), target_class=lbl, is_vit=False)
        map_b = get_gradcam_map(sem_model, sem_layer, img_f.to(DEVICE), target_class=lbl, is_vit=True)
        fgsm_csed.append(compute_csed(map_a, map_b))
        # PGD
        map_a = get_gradcam_map(art_model, art_layer, img_p.to(DEVICE), target_class=lbl, is_vit=False)
        map_b = get_gradcam_map(sem_model, sem_layer, img_p.to(DEVICE), target_class=lbl, is_vit=True)
        pgd_csed.append(compute_csed(map_a, map_b))

    fgsm_cos = np.array([d["cosine"] for d in fgsm_csed])
    fgsm_js  = np.array([d["js"]     for d in fgsm_csed])
    pgd_cos  = np.array([d["cosine"] for d in pgd_csed])
    pgd_js   = np.array([d["js"]     for d in pgd_csed])

    # Q2 statistical tests (use PGD as primary, FGSM as secondary)
    y_q2 = np.array([0]*N + [1]*N)   # 0=clean, 1=attacked

    # PGD (primary)
    pgd_auc_cos, pgd_lo_cos, pgd_hi_cos = bootstrap_auc_ci(y_q2, np.concatenate([clean_cos, pgd_cos]))
    pgd_auc_js,  pgd_lo_js,  pgd_hi_js  = bootstrap_auc_ci(y_q2, np.concatenate([clean_js,  pgd_js]))
    pgd_ks_cos = ks_2samp(clean_cos, pgd_cos)
    pgd_ks_js  = ks_2samp(clean_js,  pgd_js)

    # FGSM (secondary)
    fgsm_auc_cos, _, _ = bootstrap_auc_ci(y_q2, np.concatenate([clean_cos, fgsm_cos]))
    fgsm_auc_js,  _, _ = bootstrap_auc_ci(y_q2, np.concatenate([clean_js,  fgsm_js]))
    fgsm_ks_cos  = ks_2samp(clean_cos, fgsm_cos)

    q2_pass = (pgd_auc_cos >= Q2_AUC_MIN or pgd_auc_js >= Q2_AUC_MIN) and \
              (pgd_ks_cos.pvalue < Q2_KS_PVAL or pgd_ks_js.pvalue < Q2_KS_PVAL)

    print(f"\n  ── Q2 RESULTS (ε=8/255, PGD primary) ──────────────────────")
    print(f"     FGSM: AUC_cos={fgsm_auc_cos:.3f} | AUC_js={fgsm_auc_js:.3f} | KS_cos p={fgsm_ks_cos.pvalue:.4f}")
    print(f"     PGD:  AUC_cos={pgd_auc_cos:.3f} [{pgd_lo_cos:.3f}–{pgd_hi_cos:.3f}] | "
          f"KS_cos p={pgd_ks_cos.pvalue:.4f}")
    print(f"           AUC_js ={pgd_auc_js:.3f}  [{pgd_lo_js:.3f}–{pgd_hi_js:.3f}] | "
          f"KS_js  p={pgd_ks_js.pvalue:.4f}")
    print(f"     Bar: AUC ≥ {Q2_AUC_MIN} & KS p < {Q2_KS_PVAL} → {'✓ PASS' if q2_pass else '✗ FAIL'}")

    # Plots for Q2
    plot_csed_comparison(
        clean_cos, pgd_cos, clean_js, pgd_js,
        pgd_auc_cos, pgd_auc_js, pgd_ks_cos.pvalue, pgd_ks_js.pvalue,
        title="Q2 — CSED: Clean vs PGD-T1 (ε=8/255)",
        filename="q2_clean_vs_pgd.png"
    )
    plot_csed_comparison(
        clean_cos, fgsm_cos, clean_js, fgsm_js,
        fgsm_auc_cos, fgsm_auc_js, fgsm_ks_cos.pvalue, fgsm_ks_cos.pvalue,
        title="Q2 — CSED: Clean vs FGSM-T1 (ε=8/255)",
        filename="q2_clean_vs_fgsm.png"
    )

    # ── T2 adaptive attack ───────────────────────────────────────────────────
    q3_pass = False
    q3_auc_cos = q3_auc_js = 0.0
    q3_ks_cos  = q3_ks_js  = None
    adv_t2     = None

    if q2_pass:
        print("\n[6/7] Q3 — Generating T2 adaptive attacks (λ=1.0, 20 steps) …")
        adv_t2 = generate_t2_attacks(
            art_model, sem_model, art_layer, sem_layer,
            all_images, all_labels, eps=eps_main, lam=1.0, steps=20, device=DEVICE
        )
        print("      T2 attacks done. Computing CSED …")
        t2_csed = []
        for img_t2, lbl in tqdm(zip(adv_t2, all_labels), total=N, desc="T2 CSED"):
            lbl = int(lbl)
            map_a = get_gradcam_map(art_model, art_layer, img_t2.to(DEVICE), target_class=lbl, is_vit=False)
            map_b = get_gradcam_map(sem_model, sem_layer, img_t2.to(DEVICE), target_class=lbl, is_vit=True)
            t2_csed.append(compute_csed(map_a, map_b))

        t2_cos = np.array([d["cosine"] for d in t2_csed])
        t2_js  = np.array([d["js"]     for d in t2_csed])

        q3_auc_cos, q3_lo_cos, q3_hi_cos = bootstrap_auc_ci(y_q2, np.concatenate([clean_cos, t2_cos]))
        q3_auc_js,  q3_lo_js,  q3_hi_js  = bootstrap_auc_ci(y_q2, np.concatenate([clean_js,  t2_js]))
        q3_ks_cos = ks_2samp(clean_cos, t2_cos)
        q3_ks_js  = ks_2samp(clean_js,  t2_js)

        q3_pass = (q3_auc_cos >= Q3_AUC_MIN or q3_auc_js >= Q3_AUC_MIN)

        print(f"\n  ── Q3 RESULTS (adaptive T2, ε=8/255, λ=1.0) ──────────────")
        print(f"     AUC_cos={q3_auc_cos:.3f} [{q3_lo_cos:.3f}–{q3_hi_cos:.3f}] | "
              f"KS p={q3_ks_cos.pvalue:.4f}")
        print(f"     AUC_js ={q3_auc_js:.3f}  [{q3_lo_js:.3f}–{q3_hi_js:.3f}] | "
              f"KS p={q3_ks_js.pvalue:.4f}")
        print(f"     Bar: AUC ≥ {Q3_AUC_MIN} → {'✓ PASS → PROCEED TO PHASE 1' if q3_pass else '✗ FAIL → PARTIAL RESULT (adaptive attack suppresses CSED)'}")

        plot_csed_comparison(
            clean_cos, t2_cos, clean_js, t2_js,
            q3_auc_cos, q3_auc_js, q3_ks_cos.pvalue, q3_ks_js.pvalue,
            title="Q3 — CSED: Clean vs Adaptive-PGD-T2 (ε=8/255, λ=1.0)",
            filename="q3_clean_vs_adaptive.png"
        )
    else:
        print("\n[6/7] Q3 skipped — Q2 did not pass.")

    # ── Grad-CAM gallery ─────────────────────────────────────────────────────
    print("\n[7/7] Generating Grad-CAM gallery …")
    n_gallery = 6
    g_imgs   = all_images[:n_gallery]
    g_labels = all_labels[:n_gallery]
    g_fgsm   = adv_fgsm[:n_gallery]
    g_pgd    = adv_pgd[:n_gallery]
    g_t2     = adv_t2[:n_gallery] if adv_t2 is not None else adv_pgd[:n_gallery]

    save_gradcam_gallery(
        art_model, sem_model, art_layer, sem_layer,
        g_imgs, g_labels, g_fgsm, g_pgd, g_t2,
        n_show=n_gallery, tag="gradcam_gallery"
    )

    # ── Summary JSON ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    summary = {
        "dataset": {
            "n_total": N,
            "n_real": int((all_labels==0).sum()),
            "n_ai":   int((all_labels==1).sum()),
            "real_source": "COCO val2017",
            "ai_source":   "DiffusionDB (SD v1.4)"
        },
        "detectors": {
            "paradigm_a": "ResNet-50 (ImageNet pretrained) — artifact/frequency proxy",
            "paradigm_b": "CLIP ViT-L/14 + linear probe — semantic proxy",
            "note": "Full NPR and UnivFD weights not loaded in pilot; "
                    "ImageNet weights used as conservative proxy"
        },
        "pre_registered_bars": {
            "Q1_AUC_MAX": Q1_AUC_MAX,
            "Q2_AUC_MIN": Q2_AUC_MIN,
            "Q2_KS_PVAL": Q2_KS_PVAL,
            "Q3_AUC_MIN": Q3_AUC_MIN,
        },
        "Q1": {
            "verdict": "PASS" if q1_pass else "FAIL",
            "cosine_auc": round(q1_auc_cos, 4),
            "cosine_auc_ci": [round(q1_lo_cos,4), round(q1_hi_cos,4)],
            "js_auc": round(q1_auc_js, 4),
            "js_auc_ci": [round(q1_lo_js,4), round(q1_hi_js,4)],
            "clean_cosine_mean": round(float(clean_cos.mean()), 4),
            "clean_cosine_std":  round(float(clean_cos.std()),  4),
            "clean_js_mean":     round(float(clean_js.mean()),  4),
            "clean_js_std":      round(float(clean_js.std()),   4),
        },
        "Q2": {
            "verdict": "PASS" if q2_pass else "FAIL",
            "epsilon": "8/255",
            "pgd_cosine_auc": round(pgd_auc_cos, 4),
            "pgd_cosine_auc_ci": [round(pgd_lo_cos,4), round(pgd_hi_cos,4)],
            "pgd_js_auc": round(pgd_auc_js, 4),
            "pgd_js_auc_ci": [round(pgd_lo_js,4), round(pgd_hi_js,4)],
            "pgd_ks_cos_pvalue": round(pgd_ks_cos.pvalue, 6),
            "pgd_ks_js_pvalue":  round(pgd_ks_js.pvalue,  6),
            "fgsm_cosine_auc":   round(fgsm_auc_cos, 4),
            "fgsm_js_auc":       round(fgsm_auc_js,  4),
            "attacked_pgd_cosine_mean": round(float(pgd_cos.mean()), 4),
            "attacked_pgd_js_mean":     round(float(pgd_js.mean()),  4),
        },
        "Q3": {
            "verdict": "PASS" if q3_pass else ("FAIL" if q2_pass else "SKIPPED"),
            "epsilon": "8/255",
            "lambda": 1.0,
            "cosine_auc": round(q3_auc_cos, 4) if q2_pass else None,
            "cosine_auc_ci": [round(q3_lo_cos,4), round(q3_hi_cos,4)] if q2_pass else None,
            "js_auc":      round(q3_auc_js, 4)  if q2_pass else None,
            "js_auc_ci":   [round(q3_lo_js,4), round(q3_hi_js,4)]  if q2_pass else None,
            "ks_cos_pvalue": round(q3_ks_cos.pvalue, 6) if q2_pass else None,
            "ks_js_pvalue":  round(q3_ks_js.pvalue,  6) if q2_pass else None,
        },
        "go_no_go": {
            "Q1_pass": q1_pass,
            "Q2_pass": q2_pass,
            "Q3_pass": q3_pass,
            "recommendation": (
                "PROCEED TO PHASE 1 — all three gates passed"
                if (q1_pass and q2_pass and q3_pass) else
                "PARTIAL RESULT — Q3 failed: CSED suppressed by adaptive attacker"
                if (q1_pass and q2_pass and not q3_pass) else
                "NEGATIVE RESULT — Q2 failed: CSED does not separate clean vs T1"
                if (q1_pass and not q2_pass) else
                "PIPELINE BUG — Q1 failed: CSED is not stable on clean data"
            )
        },
        "elapsed_seconds": round(elapsed, 1)
    }

    out_json = RESULTS / "q1_q2_q3_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved: {out_json}")

    # Print final decision
    print("\n" + "=" * 65)
    print("  GO / NO-GO DECISION")
    print("=" * 65)
    for q, pass_ in [("Q1", q1_pass), ("Q2", q2_pass), ("Q3", q3_pass)]:
        symbol = "✓" if pass_ else ("✗" if q != "Q3" or q2_pass else "–")
        print(f"  {q} : {symbol} {'PASS' if pass_ else ('FAIL' if q != 'Q3' or q2_pass else 'SKIPPED')}")
    print(f"\n  → {summary['go_no_go']['recommendation']}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    main()
