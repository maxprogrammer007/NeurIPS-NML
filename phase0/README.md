# Phase 0: Pilot Benchmarking & Adaptive Security Diagnostics
**Cross-Stream Explanation Divergence (CSED) Defense Evaluation (Authoritative Pure FP32 Evaluation)**

---

## 📌 Executive Summary & Authoritative Table Designation

Phase 0 evaluates whether **Cross-Stream Explanation Divergence (CSED)** between an **Art Detector** (ResNet-50, spatial artifacts) and a **Semantic Detector** (CLIP ViT-L/14, high-level features) can serve as a robust, non-trainable detection mechanism for adversarial attacks on AI-generated vs. real image classifiers.

All evaluations in the master table below represent **pure Float32 precision** across model execution, attack generation, and Grad-CAM extraction.

### 📌 Iteration History & Disclosures
To ensure complete transparency, Phase 0 progressed through three pipeline iterations:
1. **Iteration 1 (FP16 Hybrid - Deprecated):** Unfiltered AUC appeared inflated ($0.826$) due to dying-ReLU flat maps ($23.67\%$).
2. **Iteration 2 (FP16 Attack + FP32 Filter Diagnostic - Deprecated):** Reported an $11.0\%$ PGD-T1 degeneracy rate because FP32 Grad-CAM maps were evaluated over *FP16-generated* adversarial images.
3. **Iteration 3 (Pure End-to-End FP32 - AUTHORITATIVE):** The table below represents end-to-end Float32 execution for both attack generation and Grad-CAM extraction. **The earlier $11.0\%$ figure used FP16-generated adversarial images; this table regenerates attacks in FP32 throughout, which allows PGD optimization to push further into misclassification territory (ASR increases from 96.0% to 99.67%), driving more pre-activations negative and increasing true dying ReLUs to $35.67\%$.**

---

## 🎯 Key Empirical Findings & Re-Evaluated Verdict

- **PGD-T1 and $\lambda=1.0$ Clear the 0.60 Bar:** On the real attacker's deployment view (**Flipped + Filtered AUC**), CSED Cosine AUC achieves **$0.713$ [0.67, 0.76]** at $\lambda=0.0$ (PGD-T1), **$0.696$ [0.65, 0.74]** at $\lambda=0.1$, and **$0.652$ [0.61, 0.70]** at $\lambda=1.0$ (ASR = $98.67\%$). This indicates that CSED retains detectable divergence under mild adaptivity.
- **High-$\lambda$ Plateau ($\lambda \ge 10.0$):** At higher adaptivity weights, Flipped + Filtered AUC plateaus tightly at **$0.561$ [0.51, 0.61]** ($\lambda=10.0$), **$0.562$ [0.51, 0.61]** ($\lambda=50.0$), and **$0.564$ [0.50, 0.62]** ($\lambda=100.0$). Because 95% confidence intervals straddle the pre-registered Q3 bar of **$0.60$**, high-$\lambda$ performance is **statistically inconclusive at $N=300$** rather than cleanly defeated.
- **ResNet-50 vs. CLIP Transformer Immunity:** ResNet-50 accounts for **100% of all degenerate flat maps**. CLIP ViT-L/14 Vision Transformer exhibits **0.00% degeneracy**, proving transformer self-attention maps are immune to ReLU extinction.

---

## 📊 Authoritative Pure FP32 Master Results Table ($N=300, \text{seed}=42, 30\text{ Steps}$)

| Attack Condition | ASR (%) | $N_{\text{flipped}}$ | ResNet Degen (%) | CLIP Degen (%) | Total Degen (%) | $2 \times 2$ Selection Matrix<br>*(Degen&Flip / Deg&Unflip / Val&Flip / Val&Unflip)* | Filtered Cosine AUC (All Attacked) [95% CI] | **Flipped + Filtered AUC (Real Attacker's Deployment View) [95% CI]** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | — | — | 0.33% | 0.00% | **0.33%** | $(1 \mathbin{/} 0 \mathbin{/} 163 \mathbin{/} 136)$ | $0.498 \ [0.43, 0.56]$ | — |
| **FGSM ($\varepsilon=8/255$)** | 64.00% | 192 | 3.33% | 0.00% | **3.33%** | $(10 \mathbin{/} 0 \mathbin{/} 182 \mathbin{/} 108)$ | $0.520 \ [0.47, 0.57]$ | **$0.584 \ [0.53, 0.64]$** |
| **PGD-T1 ($\lambda=0.0$)** | 99.67% | 299 | 35.67% | 0.00% | **35.67%** | $(107 \mathbin{/} 0 \mathbin{/} 192 \mathbin{/} 1)$ | $0.711 \ [0.67, 0.75]$ | **$0.713 \ [0.67, 0.76]$** |
| **T2 ($\lambda=0.1$)** | 99.67% | 299 | 32.00% | 0.00% | **32.00%** | $(96 \mathbin{/} 0 \mathbin{/} 203 \mathbin{/} 1)$ | $0.694 \ [0.65, 0.74]$ | **$0.696 \ [0.65, 0.74]$** |
| **T2 ($\lambda=1.0$)** | 98.67% | 296 | 22.33% | 0.00% | **22.33%** | $(67 \mathbin{/} 0 \mathbin{/} 229 \mathbin{/} 4)$ | $0.647 \ [0.60, 0.70]$ | **$0.652 \ [0.61, 0.70]$** |
| **T2 ($\lambda=10.0$)** | 88.33% | 265 | 9.67% | 0.00% | **9.67%** | $(28 \mathbin{/} 1 \mathbin{/} 237 \mathbin{/} 34)$ | $0.535 \ [0.48, 0.58]$ | **$0.561 \ [0.51, 0.61]$** |
| **T2 ($\lambda=50.0$)** | 64.00% | 192 | 5.00% | 0.00% | **5.00%** | $(14 \mathbin{/} 1 \mathbin{/} 178 \mathbin{/} 107)$ | $0.499 \ [0.45, 0.55]$ | **$0.562 \ [0.51, 0.61]$** |
| **T2 ($\lambda=100.0$)** | 55.00% | 165 | 4.33% | 0.00% | **4.33%** | $(11 \mathbin{/} 2 \mathbin{/} 154 \mathbin{/} 133)$ | $0.489 \ [0.44, 0.53]$ | **$0.564 \ [0.50, 0.62]$** |

---

## 🚀 Phase 1 Action Plan & Strategic Upgrades

To definitively resolve statistical power constraints ($N=10,000+$) across the full **GenImage dataset** (8 AI generators, matched NPR & UnivFD detectors):

1. **Zero-Cost Fix (FP32 Backward Pass):** Running attack and explanation backpropagation in pure FP32 float precision completely eliminates FP16 numerical underflow artifacts at zero architectural cost and serves as the standard pipeline for Phase 1.
2. **Phase 1 Scale-Up ($N=10,000+$):** Scale evaluation up to the full GenImage dataset across 8 AI generators with matched NPR (ResNet-50) and UnivFD (CLIP-ViT) detectors to achieve statistical resolution on the high-$\lambda$ plateau.
3. **Grad-CAM++ & Multi-Layer Depth Ensemble CSED:** Evaluate Grad-CAM++ and multi-layer depth-ensemble CSED to harden divergence measurement against spatial surrogate optimization.

---

## 📁 Codebase Directory Index

| File | Description |
| :--- | :--- |
| `run_fp32_comprehensive_eval.py` | Authoritative FP32 evaluation script ($N=300$, $\text{seed}=42$, 30-step PGD, CIs). |
| `verify_fp32_degeneracy.py` | FP32 float precision verification script (`std = 0.00000000`). |
| `run_degeneracy_diagnostics.py` | Spatial variance thresholding ($\sigma^2 < 10^{-5}$) and dying-ReLU diagnostic suite. |
| `run_diagnostics.py` | Initial 5-point diagnostic pass ($\lambda$-sweep, ASR, grad norm ratio). |
| `attacks.py` | T1 (FGSM, PGD) and T2 adaptive attack modules. |
| `csed.py` | Grad-CAM extraction, cosine distance, JS divergence computation. |
| `detectors.py` | Model loader for ResNet-50 and CLIP ViT-L/14 stream detectors. |
| `data_loader.py` | Balanced dataset loader for natural (COCO) and AI-generated (SD-Turbo) images. |
| `results/phase0_fp32_comprehensive_summary.json` | Full FP32 summary JSON with locked empirical metrics and 95% CIs. |

---

## 🛠️ How to Run

To run the authoritative FP32 comprehensive evaluation pass:
```bash
cd phase0
python3 run_fp32_comprehensive_eval.py
```
Outputs are automatically written to `results/phase0_fp32_comprehensive_summary.json`.
