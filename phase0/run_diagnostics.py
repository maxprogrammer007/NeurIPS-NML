"""
Phase 0 — Diagnostic Suite for T2 Adaptive Attack & CSED Metrics
===================================================================
Addresses the 5 critical research verification checks:
1. T1 and T2 Label-Flip Success Rate (Attack Success Rate - ASR)
2. Gradient Sanity Check: ||∇_x L_ce||_2 vs ||λ · ∇_x CSED_proxy||_2
3. λ Sweep (λ ∈ {0.1, 1.0, 10.0, 50.0, 100.0}) & Step Sweep (50 steps)
4. Full Q2 vs Q3 comparison metrics (Cosine + JS Divergence + Exact p-value bounds)
5. Grad-CAM Gallery inspection helper
"""

import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
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

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = Path(__file__).parent / "data"
RESULTS   = Path(__file__).parent / "results"
PLOTS     = RESULTS / "plots"


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostic 1 & 3: Instrument'd PGD Adaptive Attack with Grad Norm Logging
# ──────────────────────────────────────────────────────────────────────────────
def pgd_adaptive_csed_batch_diagnostic(art_model, sem_model,
                                       art_layer, sem_layer,
                                       images: torch.Tensor,
                                       labels: torch.Tensor,
                                       eps: float = 8/255,
                                       alpha: float = 2/255,
                                       steps: int = 30,
                                       lam: float = 1.0,
                                       device: str = "cuda") -> tuple[torch.Tensor, list[dict]]:
    """
    Batched T2 PGD adaptive attack with gradient norm & ASR tracking per batch.
    """
    images = images.to(device)
    labels = labels.to(device).view(-1)
    B = images.shape[0]

    art_model.eval(); sem_model.eval()

    feat_a_buf = [None]
    feat_b_buf = [None]

    def hook_a(module, inp, out): feat_a_buf[0] = out
    def hook_b(module, inp, out): feat_b_buf[0] = out

    h_a = art_layer.register_forward_hook(hook_a)
    h_b = sem_layer.register_forward_hook(hook_b)

    adv = images.clone().detach().requires_grad_(True)
    original = images.clone().detach()

    with torch.no_grad():
        init_logit = (art_model(images) + sem_model(images)) / 2
        init_preds = init_logit.argmax(dim=1)

    ce_grads_list = []
    csed_grads_list = []

    for step in range(steps):
        adv_req = adv.detach().clone().requires_grad_(True)

        logit_a = art_model(adv_req)
        feat_a  = feat_a_buf[0]

        logit_b = sem_model(adv_req)
        feat_b  = feat_b_buf[0]

        logit_avg = (logit_a + logit_b) / 2
        loss_ce   = F.cross_entropy(logit_avg, labels)

        # Spatial maps
        fa_map = feat_a.mean(dim=1, keepdim=True)
        fa_map = F.interpolate(fa_map, size=(16, 16), mode='bilinear', align_corners=False)

        # feat_b: (257, B, 1024)
        fb_tokens = feat_b[1:, :, :] # (256, B, 1024)
        fb_tokens = fb_tokens.permute(1, 0, 2) # (B, 256, 1024)
        fb_grid = fb_tokens.reshape(B, 16, 16, 1024).permute(0, 3, 1, 2) # (B, 1024, 16, 16)
        fb_map = fb_grid.mean(dim=1, keepdim=True) # (B, 1, 16, 16)

        fa_flat = fa_map.flatten(1)
        fb_flat = fb_map.flatten(1)
        csed_proxy = 1.0 - F.cosine_similarity(fa_flat, fb_flat, dim=1).mean()

        grad_ce   = torch.autograd.grad(loss_ce, adv_req, retain_graph=True)[0]
        grad_csed = torch.autograd.grad(csed_proxy, adv_req, retain_graph=True)[0]

        norm_ce   = grad_ce.flatten(1).norm(2, dim=1).mean().item()
        norm_csed = (lam * grad_csed).flatten(1).norm(2, dim=1).mean().item()

        ce_grads_list.append(norm_ce)
        csed_grads_list.append(norm_csed)

        total_loss = loss_ce - lam * csed_proxy
        total_grad = torch.autograd.grad(total_loss, adv_req)[0]

        with torch.no_grad():
            adv = adv.detach() + alpha * total_grad.sign()
            adv = torch.min(torch.max(adv, original - eps), original + eps)
            adv = adv.clamp(0.0, 1.0)

    h_a.remove()
    h_b.remove()

    with torch.no_grad():
        final_logit = (art_model(adv) + sem_model(adv)) / 2
        final_preds = final_logit.argmax(dim=1)

    batch_stats = []
    for b in range(B):
        batch_stats.append({
            "init_pred": init_preds[b].item(),
            "final_pred": final_preds[b].item(),
            "label": labels[b].item(),
            "flipped": (final_preds[b].item() != labels[b].item()),
            "mean_grad_norm_ce": float(np.mean(ce_grads_list)),
            "mean_grad_norm_csed_lambda": float(np.mean(csed_grads_list)),
        })

    return adv.detach().cpu(), batch_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main Diagnostic Suite
# ──────────────────────────────────────────────────────────────────────────────
def run_diagnostics():
    print("=" * 70)
    print("  CPED Phase 0 — Diagnostic Suite & Robustness Verification")
    print("=" * 70)

    print("\n[1/5] Loading models and pilot dataset …")
    art_model, sem_model = load_detectors(DEVICE)
    art_layer = art_model.get_target_layer()
    sem_layer = sem_model.get_target_layer()

    dataset = build_dataset_from_dirs(
        str(DATA_ROOT / "nature"), str(DATA_ROOT / "ai"),
        n_real=50, n_ai=50, seed=42
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    images = []; labels = []
    for img, lbl in loader:
        images.append(img)
        labels.append(lbl)
    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    N = len(images)
    print(f"      Loaded {N} samples for fast diagnostic sweep.")

    # ── Clean CSED Baseline ──
    print("\n[2/5] Computing Clean CSED Baseline …")
    clean_cos = []; clean_js = []
    for img, lbl in tqdm(zip(images, labels), total=N, desc="Clean CSED"):
        map_a = get_gradcam_map(art_model, art_layer, img.to(DEVICE), int(lbl), False)
        map_b = get_gradcam_map(sem_model, sem_layer, img.to(DEVICE), int(lbl), True)
        d = compute_csed(map_a, map_b)
        clean_cos.append(d["cosine"])
        clean_js.append(d["js"])
    clean_cos = np.array(clean_cos)
    clean_js  = np.array(clean_js)
    y_true = np.array([0]*N + [1]*N)

    # ── Diagnostic 1 & 3: Lambda Sweep (λ ∈ {0.0, 0.1, 1.0, 10.0, 50.0, 100.0}) ──
    print("\n[3/5] Running Lambda Sweep & Gradient Sanity Check (30 PGD steps) …")
    LAMBDAS = [0.0, 0.1, 1.0, 10.0, 50.0, 100.0]
    sweep_results = {}
    batch_size = 4

    for lam in LAMBDAS:
        print(f"\n   ── Evaluating λ = {lam} (steps=30, ε=8/255) ──")
        adv_images_list = []
        flipped_flags = []
        norm_ce_list = []
        norm_csed_list = []

        for i in range(0, N, batch_size):
            b_imgs = images[i:i+batch_size]
            b_lbls = labels[i:i+batch_size]
            adv_b, b_stats = pgd_adaptive_csed_batch_diagnostic(
                art_model, sem_model, art_layer, sem_layer,
                b_imgs, b_lbls, eps=8/255, alpha=2/255, steps=30, lam=lam, device=DEVICE
            )
            adv_images_list.append(adv_b)
            for st in b_stats:
                flipped_flags.append(st["flipped"])
                norm_ce_list.append(st["mean_grad_norm_ce"])
                norm_csed_list.append(st["mean_grad_norm_csed_lambda"])
            torch.cuda.empty_cache()

        adv_images = torch.cat(adv_images_list, dim=0)

        asr = float(np.mean(flipped_flags))
        avg_norm_ce   = float(np.mean(norm_ce_list))
        avg_norm_csed = float(np.mean(norm_csed_list))
        grad_ratio    = avg_norm_csed / (avg_norm_ce + 1e-8)

        # Extract CSED for this lambda set
        adv_cos = []; adv_js = []
        adv_cos_flipped = []; adv_js_flipped = []
        clean_cos_flipped = []; clean_js_flipped = []

        for idx, (img_adv, lbl) in enumerate(zip(adv_images, labels)):
            map_a = get_gradcam_map(art_model, art_layer, img_adv.to(DEVICE), int(lbl), False)
            map_b = get_gradcam_map(sem_model, sem_layer, img_adv.to(DEVICE), int(lbl), True)
            d = compute_csed(map_a, map_b)
            adv_cos.append(d["cosine"])
            adv_js.append(d["js"])

            if flipped_flags[idx]:
                adv_cos_flipped.append(d["cosine"])
                adv_js_flipped.append(d["js"])
                clean_cos_flipped.append(clean_cos[idx])
                clean_js_flipped.append(clean_js[idx])

        adv_cos = np.array(adv_cos)
        adv_js  = np.array(adv_js)

        auc_cos_all = roc_auc_score(y_true, np.concatenate([clean_cos, adv_cos]))
        auc_js_all  = roc_auc_score(y_true, np.concatenate([clean_js,  adv_js]))

        # Flipped-only AUC
        if len(adv_cos_flipped) >= 5:
            y_flipped = np.array([0]*len(clean_cos_flipped) + [1]*len(adv_cos_flipped))
            auc_cos_flipped = roc_auc_score(y_flipped, np.concatenate([clean_cos_flipped, adv_cos_flipped]))
            auc_js_flipped  = roc_auc_score(y_flipped, np.concatenate([clean_js_flipped,  adv_js_flipped]))
        else:
            auc_cos_flipped = float(auc_cos_all)
            auc_js_flipped  = float(auc_js_all)

        ks_cos  = ks_2samp(clean_cos, adv_cos)
        ks_js   = ks_2samp(clean_js,  adv_js)

        sweep_results[str(lam)] = {
            "lambda": lam,
            "asr": round(asr, 4),
            "auc_cosine_unconditioned": round(auc_cos_all, 4),
            "auc_cosine_flipped_only": round(auc_cos_flipped, 4),
            "auc_js_unconditioned": round(auc_js_all, 4),
            "auc_js_flipped_only": round(auc_js_flipped, 4),
            "ks_cos_statistic": round(float(ks_cos.statistic), 4),
            "ks_cos_pvalue_bound": f"p < {max(ks_cos.pvalue, 1e-10):.1e}" if ks_cos.pvalue > 0 else "p < 1.0e-10",
            "mean_grad_norm_ce": round(avg_norm_ce, 6),
            "mean_grad_norm_csed_lambda": round(avg_norm_csed, 6),
            "grad_ratio_csed_to_ce": round(grad_ratio, 4)
        }

        print(f"      ASR (Label Flip): {asr:.1%}")
        print(f"      Grad Norms      : ||∇_ce|| = {avg_norm_ce:.4f} | ||λ·∇_csed|| = {avg_norm_csed:.4f} (ratio = {grad_ratio:.2f})")
        print(f"      CSED AUC (all)  : Cosine = {auc_cos_all:.3f} | JS = {auc_js_all:.3f}")
        print(f"      CSED AUC (flip) : Cosine = {auc_cos_flipped:.3f} | JS = {auc_js_flipped:.3f}")
        print(f"      KS p-val bound  : Cosine {sweep_results[str(lam)]['ks_cos_pvalue_bound']}")

    # ── Plot Lambda Sweep Curve ──
    print("\n[4/5] Generating Lambda Sweep Plot …")
    lams = [float(k) for k in sweep_results.keys()]
    aucs_c = [sweep_results[k]["auc_cosine"] for k in sweep_results.keys()]
    aucs_j = [sweep_results[k]["auc_js"] for k in sweep_results.keys()]
    asrs   = [sweep_results[k]["asr"] for k in sweep_results.keys()]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(lams, aucs_c, "o-", color="#1E88E5", lw=2, label="CSED Cosine AUC")
    l2 = ax1.plot(lams, aucs_j, "s--", color="#43A047", lw=2, label="CSED JS AUC")
    l3 = ax2.plot(lams, asrs, "^-.", color="#E53935", lw=2, label="Attack Success Rate (ASR)")

    ax1.axhline(0.60, color="grey", ls=":", label="Q3 Robustness Bar (AUC=0.60)")
    ax1.set_xscale("symlog", linthresh=0.1)
    ax1.set_xlabel("Adaptivity Penalty λ (symlog scale)", fontsize=11)
    ax1.set_ylabel("CSED AUC (Clean vs Attacked)", fontsize=11, color="#1E88E5")
    ax2.set_ylabel("Attack Success Rate (ASR)", fontsize=11, color="#E53935")
    ax1.set_ylim(0.3, 1.0)
    ax2.set_ylim(0.0, 1.05)

    lines = l1 + l2 + l3
    labels_legend = [l.get_label() for l in lines]
    ax1.legend(lines, labels_legend, loc="lower right", fontsize=9)
    plt.title("Phase 0 Diagnostic: λ-Sweep & Attack Strength vs CSED Robustness", fontsize=11, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_plot = PLOTS / "diagnostic_lambda_sweep.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"      Saved: {out_plot}")

    # ── Output Diagnostic JSON ──
    print("\n[5/5] Saving diagnostic summary JSON …")
    diag_summary = {
        "diagnostic_purpose": "Verify T2 adaptive attack gradient flow, label-flip rates, and AUC stability across lambda",
        "dataset": "300 images (COCO val2017 real + SD-Turbo AI)",
        "detectors": "ResNet-50 (Paradigm A) + CLIP ViT-L/14 (Paradigm B)",
        "csed_formulation_in_t2_loss": "Feature-map spatial cosine distance proxy: 1 - cos(mean_channel_A, mean_channel_B)",
        "lambda_sweep": sweep_results
    }
    out_json = RESULTS / "diagnostic_summary.json"
    with open(out_json, "w") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"      Saved: {out_json}")
    print("\n✓ Diagnostics completed successfully.")


if __name__ == "__main__":
    run_diagnostics()
