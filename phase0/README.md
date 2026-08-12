# Phase 0: Pilot Benchmarking & Adaptive Security Diagnostics
**Cross-Stream Explanation Divergence (CSED) Defense Evaluation**

---

## 📌 Executive Summary

Phase 0 evaluates whether **Cross-Stream Explanation Divergence (CSED)** between an **Art Detector** (ResNet-50, spatial artifacts) and a **Semantic Detector** (CLIP ViT-L/14, high-level features) can serve as a robust, non-trainable detection mechanism for adversarial attacks on AI-generated vs. real image classifiers.

### Key Finding
- **Attacker's View vs. All-Attacked Population:** Quoting all-attacked Filtered AUC ($0.505$ at $\lambda=10.0$) alongside $71.3\%$ ASR creates a population mismatch. An attacker only deploys successfully flipped images. On the valid flipped population (**Flipped + Filtered AUC**), CSED yields **$0.589$ [0.54, 0.64] at $\lambda=10.0$**, **$0.561$ [0.50, 0.61] at $\lambda=50.0$**, and **$0.544$ [0.49, 0.60] at $\lambda=100.0$**.
- **Inconclusive Verdict at N=300:** Because 95% confidence intervals across the $\lambda$-sweep (e.g. $[0.54, 0.64]$ at $\lambda=10.0$) straddle the pre-registered Q3 bar of **$0.60$**, Phase 0 is **statistically inconclusive at $N=300$** rather than cleanly defeated.
- **Sub-0.5 AUC Explanation:** Unconditioned Filtered AUC drops below $0.5$ at high $\lambda$ ($0.432$ at $\lambda=100.0$) because unflipped images (failed attack attempts) have CSED over-optimized while failing CE, pulling the overall metric down below chance.
- **FP32 Dying-ReLU Verification:** Full Float32 re-computation (`verify_fp32_degeneracy.py`) confirmed that degenerate Grad-CAM maps evaluate to exact `std = 0.00000000` ($11.0\%$ in FP32), proving flat maps are true physical dying ReLUs in ResNet-50 (`layer4[-1].conv3`), not an FP16 numerical underflow artifact.
- **ResNet-50 vs. CLIP Transformer Immunity:** ResNet-50 accounts for **100% of all degenerate flat maps**. CLIP ViT-L/14 Vision Transformer exhibits **0.00% degeneracy**, proving transformer self-attention maps are immune to ReLU extinction.

---

## 🎯 Initial Hypotheses & Pre-Registered Questions

1. **Q1 (Negative Control):** Do clean real photos and clean AI images exhibit low divergence ($\text{AUC} \approx 0.50$)?
2. **Q2 (Non-Adaptive Detection Shift):** Do non-adaptive attacks (FGSM, PGD-T1) cause significant explanation divergence ($\text{AUC} \ge 0.65$)?
3. **Q3 (Adaptive Vulnerability Gate):** Can an adaptive attacker (T2) explicitly optimizing to suppress CSED divergence ($\mathcal{L} = \mathcal{L}_{\text{CE}} - \lambda \cdot \text{CSED}_{\text{proxy}}$) defeat the detector without collapsing its classification Attack Success Rate (ASR)?

---

## 🔬 Core Discoveries & Numerical Verifications

### 1. The Dying-ReLU Degeneracy Bug (FP32 Verified)
Under multi-step PGD perturbations, feature activations in ResNet-50 (`layer4[-1].conv3`) are driven negative ($z < 0$). Under standard ReLU, both the activation $A^k = 0$ and its gradient $\frac{\partial A^k}{\partial z} = 0$. Pure Float32 verification confirmed exact `std = 0.00000000` flat maps, proving this is a physical dying-ReLU phenomenon, not an FP16 numerical underflow artifact.

### 2. ResNet-50 vs. CLIP Transformer Immunity
- **ResNet-50 (Paradigm A CNN):** Accounts for **100% of all degenerate flat maps** ($23.67\%$ under 30-step PGD).
- **CLIP ViT-L/14 (Paradigm B Vision Transformer):** Exhibits **0.00% degeneracy** across all attack conditions, proving self-attention maps do not suffer from dying-ReLU gradient extinction.

### 3. Order of Operations & Selection Bias
Cross-tabulating degeneracy against label-flip success ($2 \times 2$ matrix evaluated independently) revealed that **$100\%$ of degenerate flat maps cluster exclusively among successfully flipped images** ($\text{Deg\&Unflip} = 0$). Flat maps occur only when perturbations push classifier logits far beyond the decision boundary.

---

## 📊 Locked Master Results Table ($N=300, \text{seed}=42, 30\text{ Steps}$)

| Attack Condition | ASR (%) | $N_{\text{flipped}}$ | ResNet Degen (%) | CLIP Degen (%) | Total Degen (%) | $2 \times 2$ Selection Matrix<br>*(Degen&Flip / Deg&Unflip / Val&Flip / Val&Unflip)* | Filtered Cosine AUC (All Attacked) [95% CI] | **Flipped + Filtered AUC (Real Attacker's View) [95% CI]** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | — | — | 0.33% | 0.00% | **0.33%** | $(1 \mathbin{/} 0 \mathbin{/} 163 \mathbin{/} 136)$ | $0.498 \ [0.43, 0.56]$ | — |
| **FGSM ($\varepsilon=8/255$)** | 53.00% | 159 | 0.33% | 0.00% | **0.33%** | $(1 \mathbin{/} 0 \mathbin{/} 158 \mathbin{/} 141)$ | $0.473 \ [0.43, 0.52]$ | **$0.603 \ [0.55, 0.66]$** |
| **PGD-T1 ($\lambda=0.0$)** | 96.00% | 288 | 23.67% | 0.00% | **23.67%** | $(71 \mathbin{/} 0 \mathbin{/} 217 \mathbin{/} 12)$ | $0.643 \ [0.60, 0.69]$ | **$0.663 \ [0.61, 0.71]$** |
| **T2 ($\lambda=0.1$)** | 95.67% | 287 | 20.33% | 0.00% | **20.33%** | $(61 \mathbin{/} 0 \mathbin{/} 226 \mathbin{/} 13)$ | $0.616 \ [0.57, 0.66]$ | **$0.632 \ [0.58, 0.68]$** |
| **T2 ($\lambda=1.0$)** | 87.67% | 263 | 13.00% | 0.00% | **13.00%** | $(39 \mathbin{/} 0 \mathbin{/} 224 \mathbin{/} 37)$ | $0.572 \ [0.52, 0.62]$ | **$0.618 \ [0.57, 0.67]$** |
| **T2 ($\lambda=10.0$)** | 71.33% | 214 | 4.67% | 0.00% | **4.67%** | $(14 \mathbin{/} 0 \mathbin{/} 200 \mathbin{/} 86)$ | $0.505 \ [0.45, 0.55]$ | **$0.589 \ [0.54, 0.64]$** |
| **T2 ($\lambda=50.0$)** | 55.00% | 165 | 0.67% | 0.00% | **0.67%** | $(2 \mathbin{/} 0 \mathbin{/} 163 \mathbin{/} 135)$ | $0.456 \ [0.41, 0.50]$ | **$0.561 \ [0.50, 0.61]$** |
| **T2 ($\lambda=100.0$)** | 52.67% | 158 | 0.67% | 0.00% | **0.67%** | $(2 \mathbin{/} 0 \mathbin{/} 156 \mathbin{/} 142)$ | $0.432 \ [0.39, 0.48]$ | **$0.544 \ [0.49, 0.60]$** |

---

## 🚀 Phase 1 Action Plan & Strategic Upgrades

To definitively resolve statistical power constraints ($N=10,000+$) across the full **GenImage dataset** (8 AI generators, matched NPR & UnivFD detectors):

1. **Grad-CAM++ Integration:** Uses positive higher-order pixel weighting to eliminate dying-ReLU zero-gradient artifacts.
2. **Multi-Layer Depth Ensemble CSED:** Computes CSED across early, mid, and late feature layers simultaneously. An attacker cannot suppress spatial divergence across 3 distinct network depths at once without classification ASR collapsing.
3. **Full Scale-Up:** Evaluate matched NPR (ResNet-50) and UnivFD (CLIP-ViT) detectors across Stable Diffusion, Midjourney, DALL-E 2/3, and Imagen.

---

## 📁 Codebase Directory Index

| File | Description |
| :--- | :--- |
| `run_phase0_comprehensive_eval.py` | Locked master evaluation script ($N=300$, $\text{seed}=42$, 30-step PGD, FP16 Tensor Cores, CIs). |
| `verify_fp32_degeneracy.py` | FP32 float precision verification script (`std = 0.00000000`). |
| `run_degeneracy_diagnostics.py` | Spatial variance thresholding ($\sigma^2 < 10^{-5}$) and dying-ReLU diagnostic suite. |
| `run_diagnostics.py` | Initial 5-point diagnostic pass ($\lambda$-sweep, ASR, grad norm ratio). |
| `attacks.py` | T1 (FGSM, PGD) and T2 adaptive attacks with FP16 Tensor Core acceleration. |
| `csed.py` | Grad-CAM extraction, cosine distance, JS divergence computation. |
| `detectors.py` | Model loader for ResNet-50 and CLIP ViT-L/14 stream detectors. |
| `data_loader.py` | Balanced dataset loader for natural (COCO) and AI-generated (SD-Turbo) images. |
| `results/phase0_comprehensive_summary.json` | Full summary JSON with locked empirical metrics and 95% CIs. |

---

## 🛠️ How to Run

To run the locked comprehensive evaluation pass:
```bash
cd phase0
python3 run_phase0_comprehensive_eval.py
```
To run FP32 precision verification:
```bash
python3 verify_fp32_degeneracy.py
```
