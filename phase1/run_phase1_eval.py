"""
Phase 1 — Main Evaluation Harness
====================================
Runs the full CPED benchmark on the GenImage dataset (N≥10,000).

Attack conditions evaluated (all pure FP32):
  1. Clean Baseline
  2. FGSM (eps=8/255)
  3. PGD-T1 (lambda=0.0, 30 steps)
  4. T2 Adaptive sweep: lambda ∈ {0.1, 1.0, 10.0, 50.0, 100.0}

Per-generator breakdown:
  - Results are computed per GenImage generator as well as aggregated.
  - Separate 2×2 selection matrices (Degeneracy × Label-Flip) per generator.
  - Filtered Cosine AUC and Flipped+Filtered AUC with 95% bootstrapped CIs.

CSED modes supported via --csed_mode flag:
  'gradcam'   — Phase 0 baseline (default, zero additional cost)
  'gradcam++' — Grad-CAM++ (better localization)
  'ensemble'  — Multi-layer depth ensemble

Usage:
  python run_phase1_eval.py \\
      --data_root /path/to/GenImage \\
      --n_per_generator 1250 \\
      --csed_mode gradcam \\
      --seed 42 \\
      --output_dir results/

  For a local pilot test (no GenImage download):
  python run_phase1_eval.py --pilot_mode \\
      --real_dir ../phase0/data/nature \\
      --ai_dir   ../phase0/data/ai \\
      --n_pilot  300
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Literal

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from detectors import load_detectors
from csed import extract_csed_sample, DEGENERACY_THRESHOLD
from data_loader import (
    build_genimage_dataset,
    build_dataset_from_dirs,
    GENIMAGE_GENERATORS,
)
from attacks import EnsembleModel, generate_t1_attacks, generate_t2_attacks
import torchattacks

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap AUC confidence intervals
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_auc_ci(y_true, y_score, B=1000, seed=42):
    """95% bootstrapped confidence interval for AUC-ROC."""
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    boot_aucs = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(boot_aucs) == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(boot_aucs, 2.5)),
        float(np.percentile(boot_aucs, 97.5)),
    )


def safe_auc(y_true, y_score):
    """AUC-ROC with guard for degenerate label sets."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ──────────────────────────────────────────────────────────────────────────────
# Attack generation dispatcher
# ──────────────────────────────────────────────────────────────────────────────
def generate_attacked_images(
    mode: str,
    lam,
    images: torch.Tensor,
    labels: torch.Tensor,
    art_model,
    sem_model,
    art_layer,
    sem_layer,
    device: str,
    steps: int = 30,
    eps: float = 8 / 255,
    batch_size: int = 4,
) -> torch.Tensor:
    """Dispatch to the correct attack function by mode string."""
    ensemble = EnsembleModel(art_model, sem_model).to(device).eval()

    if mode == "fgsm":
        atk = torchattacks.FGSM(ensemble, eps=eps)
        adv_list = []
        n = len(images)
        for i in tqdm(range(0, n, batch_size), desc="FGSM"):
            b_img = images[i:i + batch_size].float().to(device)
            b_lbl = labels[i:i + batch_size].to(device)
            adv_list.append(atk(b_img, b_lbl).cpu())
            torch.cuda.empty_cache()
        return torch.cat(adv_list, 0)

    elif mode in ("pgd_t1", "t2"):
        effective_lam = 0.0 if mode == "pgd_t1" else float(lam)
        return generate_t2_attacks(
            art_model, sem_model,
            art_layer, sem_layer,
            images, labels,
            eps=eps, lam=effective_lam,
            steps=steps, batch_size=batch_size,
            device=device,
            desc=f"PGD (lam={effective_lam})",
        )
    else:
        raise ValueError(f"Unknown attack mode: {mode}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-condition metric computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_condition_metrics(
    adv_imgs: torch.Tensor,
    labels: torch.Tensor,
    clean_cos: np.ndarray,
    art_model,
    sem_model,
    art_layer,
    sem_layer,
    device: str,
    csed_mode: str,
    art_multi_layers,
    sem_multi_layers,
) -> dict:
    """
    Compute full metric suite for one attack condition:
      - Per-sample is_deg and is_flipped flags
      - 2×2 cross-tabulation
      - Filtered AUC and Flipped+Filtered AUC with 95% CIs
    """
    N = len(adv_imgs)
    cos_adv = []
    is_deg_flags = []
    is_flipped_flags = []
    ensemble = EnsembleModel(art_model, sem_model).to(device).eval()

    for idx in tqdm(range(N), desc="CSED extraction", leave=False):
        img_adv_t = adv_imgs[idx].unsqueeze(0).float().to(device)
        lbl_val = int(labels[idx])

        with torch.no_grad():
            logits = ensemble(img_adv_t)
            adv_pred = int(logits.argmax(dim=1).item())
        is_flipped = bool(adv_pred != lbl_val)
        is_flipped_flags.append(is_flipped)

        result = extract_csed_sample(
            art_model, sem_model,
            art_layer, sem_layer,
            img_adv_t[0],
            target_class=1,
            csed_mode=csed_mode,
            art_multi_layers=art_multi_layers,
            sem_multi_layers=sem_multi_layers,
        )
        cos_adv.append(result["cosine"])
        is_deg_flags.append(result["is_deg"])

    cos_adv = np.array(cos_adv)
    is_deg_flags = np.array(is_deg_flags)
    is_flipped_flags = np.array(is_flipped_flags)

    n_flipped = int(is_flipped_flags.sum())
    asr = float(is_flipped_flags.mean())
    deg_rate = float(is_deg_flags.mean())

    # 2×2 cross-tab
    degen_and_flip   = int((is_deg_flags & is_flipped_flags).sum())
    degen_and_unflip = int((is_deg_flags & ~is_flipped_flags).sum())
    valid_and_flip   = int((~is_deg_flags & is_flipped_flags).sum())
    valid_and_unflip = int((~is_deg_flags & ~is_flipped_flags).sum())
    assert degen_and_flip + degen_and_unflip + valid_and_flip + valid_and_unflip == N, \
        f"2×2 matrix row sum mismatch: {degen_and_flip + degen_and_unflip + valid_and_flip + valid_and_unflip} ≠ {N}"

    # Filtered AUC (all valid attacked vs clean)
    valid_mask = ~is_deg_flags
    valid_cos_adv = cos_adv[valid_mask]
    y_true_filt = np.concatenate([np.zeros(len(clean_cos)), np.ones(len(valid_cos_adv))])
    scores_filt = np.concatenate([clean_cos, valid_cos_adv])
    auc_filt = safe_auc(y_true_filt, scores_filt)
    ci_filt = bootstrap_auc_ci(y_true_filt, scores_filt)

    # Flipped + Filtered AUC (real attacker's deployment view)
    valid_flipped_mask = (~is_deg_flags) & is_flipped_flags
    n_valid_flipped = int(valid_flipped_mask.sum())
    if n_valid_flipped > 0:
        vf_cos = cos_adv[valid_flipped_mask]
        y_true_vf = np.concatenate([np.zeros(len(clean_cos)), np.ones(n_valid_flipped)])
        scores_vf = np.concatenate([clean_cos, vf_cos])
        auc_vf = safe_auc(y_true_vf, scores_vf)
        ci_vf = bootstrap_auc_ci(y_true_vf, scores_vf)
        
        # Explanation Collapse Detection AUC (1 - AUC_raw, where clean > attacked)
        auc_collapse = 1.0 - auc_vf
        ci_collapse = [round(1.0 - ci_vf[1], 4), round(1.0 - ci_vf[0], 4)]
    else:
        auc_vf = float("nan")
        ci_vf = (float("nan"), float("nan"))
        auc_collapse = float("nan")
        ci_collapse = (float("nan"), float("nan"))

    return {
        "n": N,
        "asr_pct":            round(asr * 100, 2),
        "n_flipped":          n_flipped,
        "total_deg_rate_pct": round(deg_rate * 100, 2),
        "cross_tab_2x2": {
            "degen_and_flip":   degen_and_flip,
            "degen_and_unflip": degen_and_unflip,
            "valid_and_flip":   valid_and_flip,
            "valid_and_unflip": valid_and_unflip,
        },
        "auc_filtered":                round(auc_filt, 4) if not np.isnan(auc_filt) else None,
        "ci_filtered":                 [round(ci_filt[0], 4), round(ci_filt[1], 4)],
        "n_valid_flipped":              n_valid_flipped,
        "auc_flipped_filtered_raw":     round(auc_vf, 4) if not np.isnan(auc_vf) else None,
        "ci_flipped_filtered_raw":      [round(ci_vf[0], 4), round(ci_vf[1], 4)],
        "auc_flipped_filtered_collapse": round(auc_collapse, 4) if not np.isnan(auc_collapse) else None,
        "ci_flipped_filtered_collapse":  ci_collapse,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation entry point
# ──────────────────────────────────────────────────────────────────────────────
def run_phase1_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"CSED mode: {args.csed_mode}")

    # 1. Load detectors
    art_model, sem_model = load_detectors(
        device=device,
        art_checkpoint=args.art_ckpt,
        sem_checkpoint=args.sem_ckpt,
    )
    art_layer = art_model.get_target_layer()
    sem_layer = sem_model.get_target_layer()
    art_multi = art_model.get_multi_layer_targets() if args.csed_mode == "ensemble" else None
    sem_multi = sem_model.get_multi_layer_targets() if args.csed_mode == "ensemble" else None
    ensemble  = EnsembleModel(art_model, sem_model).to(device).eval()

    # 2. Build dataset
    if args.pilot_mode:
        logger.info("Pilot mode: loading local Phase 0-style dataset")
        dataset = build_dataset_from_dirs(
            args.real_dir, args.ai_dir,
            n_real=args.n_pilot // 2,
            n_ai=args.n_pilot // 2,
            seed=args.seed,
        )
        sub_datasets = {"local_pilot": dataset}
        generator_names = ["local_pilot"]
    else:
        logger.info(f"GenImage mode: data_root={args.data_root}")
        gens = args.generators.split(",") if args.generators else None
        combined, sub_datasets_list = build_genimage_dataset(
            data_root=args.data_root,
            generators=gens,
            split=args.split,
            n_per_generator=args.n_per_generator,
            seed=args.seed,
        )
        sub_datasets = {
            d.generator: d for d in sub_datasets_list
        }
        generator_names = list(sub_datasets.keys())

    # 3. Load all images + labels into memory (or process per-generator)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    all_results = {}

    for gen_name, gen_dataset in sub_datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Generator: {gen_name}  (N={len(gen_dataset)})")
        logger.info(f"{'='*60}")

        loader = DataLoader(gen_dataset, batch_size=32, shuffle=False, num_workers=4)
        images_list, labels_list = [], []
        for batch in loader:
            img_b, lbl_b = batch[0], batch[1]
            images_list.append(img_b.float())
            labels_list.append(lbl_b)
        images = torch.cat(images_list, 0)
        labels = torch.cat(labels_list, 0)
        N = len(images)

        gen_results = {"N": N, "conditions": {}}

        # 3a. Clean baseline
        logger.info(f"  Clean baseline (N={N})...")
        clean_cos = []
        with torch.no_grad():
            pass  # No attack needed

        for idx in tqdm(range(N), desc="Clean CSED", leave=False):
            img_t = images[idx].unsqueeze(0).float().to(device)
            lbl_val = int(labels[idx])
            result = extract_csed_sample(
                art_model, sem_model,
                art_layer, sem_layer,
                img_t[0], target_class=1,
                csed_mode=args.csed_mode,
                art_multi_layers=art_multi,
                sem_multi_layers=sem_multi,
            )
            clean_cos.append(result["cosine"])
        clean_cos = np.array(clean_cos)
        gen_results["conditions"]["clean"] = {
            "mean_cosine": float(clean_cos.mean()),
            "std_cosine":  float(clean_cos.std()),
        }
        logger.info(f"  Clean mean cosine: {clean_cos.mean():.4f} ± {clean_cos.std():.4f}")

        # 3b. Attack conditions
        attack_configs = [
            ("fgsm",   None,  "FGSM (eps=8/255)"),
            ("pgd_t1", 0.0,   "PGD-T1 (lambda=0.0, 30 steps)"),
            ("t2",     0.1,   "T2 (lambda=0.1, 30 steps)"),
            ("t2",     1.0,   "T2 (lambda=1.0, 30 steps)"),
            ("t2",     10.0,  "T2 (lambda=10.0, 30 steps)"),
            ("t2",     50.0,  "T2 (lambda=50.0, 30 steps)"),
            ("t2",     100.0, "T2 (lambda=100.0, 30 steps)"),
        ]

        for mode, lam, cond_name in attack_configs:
            logger.info(f"  Attack: {cond_name}")
            adv_imgs = generate_attacked_images(
                mode=mode, lam=lam,
                images=images, labels=labels,
                art_model=art_model, sem_model=sem_model,
                art_layer=art_layer, sem_layer=sem_layer,
                device=device, steps=30,
                batch_size=args.attack_batch_size,
            )
            metrics = compute_condition_metrics(
                adv_imgs=adv_imgs,
                labels=labels,
                clean_cos=clean_cos,
                art_model=art_model, sem_model=sem_model,
                art_layer=art_layer, sem_layer=sem_layer,
                device=device,
                csed_mode=args.csed_mode,
                art_multi_layers=art_multi,
                sem_multi_layers=sem_multi,
            )
            gen_results["conditions"][cond_name] = metrics
            logger.info(
                f"    ASR={metrics['asr_pct']:.1f}% | "
                f"Degen={metrics['total_deg_rate_pct']:.1f}% | "
                f"Filtered AUC={metrics['auc_filtered']} | "
                f"Collapse AUC={metrics['auc_flipped_filtered_collapse']} {metrics['ci_flipped_filtered_collapse']} "
                f"(Raw AUC={metrics['auc_flipped_filtered_raw']})"
            )

        all_results[gen_name] = gen_results

    # 4. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"phase1_{args.csed_mode}_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # 5. Print aggregate summary
    logger.info("\n" + "="*70)
    logger.info("PHASE 1 AGGREGATE SUMMARY")
    logger.info("="*70)
    attack_names = [c[2] for c in attack_configs]
    for cond_name in attack_names:
        raw_aucs, collapse_aucs = [], []
        for gen_name, gres in all_results.items():
            cond_res = gres["conditions"].get(cond_name, {})
            v_raw = cond_res.get("auc_flipped_filtered_raw")
            v_col = cond_res.get("auc_flipped_filtered_collapse")
            if v_raw is not None and not np.isnan(v_raw):
                raw_aucs.append(v_raw)
            if v_col is not None and not np.isnan(v_col):
                collapse_aucs.append(v_col)
        if collapse_aucs:
            logger.info(
                f"  {cond_name}: mean Collapse Detection AUC = "
                f"{np.mean(collapse_aucs):.4f} ± {np.std(collapse_aucs):.4f} "
                f"(Raw Cosine AUC = {np.mean(raw_aucs):.4f}) across {len(collapse_aucs)} generators"
            )

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 CPED Evaluation Harness")

    # Dataset
    p.add_argument("--data_root",        type=str, default=None,
                   help="Path to GenImage root directory")
    p.add_argument("--generators",       type=str, default=None,
                   help="Comma-separated generator names (default: all 8)")
    p.add_argument("--split",            type=str, default="test",
                   choices=["train", "test"])
    p.add_argument("--n_per_generator",  type=int, default=1250,
                   help="Images per generator (real + AI each)")
    p.add_argument("--seed",             type=int, default=42)

    # Pilot mode
    p.add_argument("--pilot_mode",  action="store_true",
                   help="Use local Phase 0-style dataset instead of GenImage")
    p.add_argument("--real_dir",    type=str, default="../phase0/data/nature")
    p.add_argument("--ai_dir",      type=str, default="../phase0/data/ai")
    p.add_argument("--n_pilot",     type=int, default=300,
                   help="Total pilot samples (real + AI combined)")

    # Detectors
    p.add_argument("--art_ckpt",    type=str, default=None,
                   help="Path to NPR checkpoint (auto-downloads if omitted)")
    p.add_argument("--sem_ckpt",    type=str, default=None,
                   help="Path to UnivFD probe checkpoint (auto-downloads if omitted)")

    # CSED
    p.add_argument("--csed_mode",   type=str, default="gradcam",
                   choices=["gradcam", "gradcam++", "ensemble"],
                   help="Explanation method (gradcam=Phase0 baseline)")

    # Performance
    p.add_argument("--attack_batch_size", type=int, default=16)
    p.add_argument("--output_dir",        type=str, default="results")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.pilot_mode and args.data_root is None:
        raise ValueError(
            "Must specify either --data_root (GenImage) or --pilot_mode "
            "(local Phase 0 dataset). See usage in module docstring."
        )
    run_phase1_eval(args)
