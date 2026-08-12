# Phase 1: GenImage Scale-Up CPED Benchmark
**Cross-Stream Explanation Divergence (CSED) — Large-Scale Evaluation**

---

## 📌 Motivation & Context

Phase 0 (N=300, locked seed=42, pure FP32) established the empirical baseline for CPED:
- **PGD-T1 and λ≤1.0 clear the pre-registered Flipped+Filtered AUC bar of 0.60** cleanly.
- **At λ≥10.0**, performance plateaus around **0.56** with 95% CIs straddling 0.60 — statistically inconclusive at N=300.

Phase 1 directly addresses this statistical power gap by scaling to **N≥10,000** across **8 GenImage generators**, and promotes the detector architecture from ImageNet stand-ins to **production-grade pretrained models** (NPR + UnivFD).

---

## 🎯 Pre-Registered Phase 1 Hypotheses

| Hypothesis | Description |
|---|---|
| **H1 (Statistical Resolution)** | At N≥10,000, the high-λ plateau (0.56) resolves: 95% CI falls cleanly below 0.60 (CSED defeated) or above 0.60 (CSED survives). |
| **H2 (Generator Heterogeneity)** | CSED Flipped+Filtered AUC varies across generators — high-frequency artifact generators (BigGAN, ADM) show higher divergence than latent-diffusion generators (SD v1.4/1.5). |
| **H3 (CSED Mode Ablation)** | Grad-CAM++ and multi-layer ensemble CSED show equal or higher AUC compared to baseline Grad-CAM at zero ASR cost. |

---

## 🚀 Quick Start

### Pilot Mode (no GenImage download required)
Test the Phase 1 pipeline locally using the Phase 0 dataset:
```bash
cd phase1
python run_phase1_eval.py \
    --pilot_mode \
    --real_dir ../phase0/data/nature \
    --ai_dir   ../phase0/data/ai \
    --n_pilot  300 \
    --csed_mode gradcam \
    --output_dir results/
```

### Full GenImage Mode
```bash
cd phase1
python run_phase1_eval.py \
    --data_root /path/to/GenImage \
    --n_per_generator 1250 \
    --csed_mode gradcam \
    --seed 42 \
    --output_dir results/
```

### Grad-CAM++ Mode
```bash
python run_phase1_eval.py \
    --data_root /path/to/GenImage \
    --n_per_generator 1250 \
    --csed_mode gradcam++ \
    --output_dir results/
```

### Multi-Layer Ensemble Mode
```bash
python run_phase1_eval.py \
    --data_root /path/to/GenImage \
    --n_per_generator 1250 \
    --csed_mode ensemble \
    --output_dir results/
```

---

## 📁 GenImage Dataset Layout

Download from the [GenImage Google Drive](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS?usp=sharing).

Expected directory structure:
```
GenImage/
  midjourney/
    test/
      ai/        ← AI-generated (label=1)
      nature/    ← Real ImageNet (label=0)
    train/
      ai/
      nature/
  stable_diffusion_v_1_4/
    test/
      ai/
      nature/
  ...
```

**8 supported generators:**
`midjourney`, `stable_diffusion_v_1_4`, `stable_diffusion_v_1_5`, `wukong`, `VQDM`, `ADM`, `glide`, `biggan`

---

## 🔬 Detector Architecture

| Paradigm | Model | Architecture | Pretrained Weights |
|---|---|---|---|
| **A (NPR)** | `NPRDetector` | ResNet-50 + binary head | NPR forensics checkpoint (auto-download) |
| **B (UnivFD)** | `UnivFDDetector` | Frozen CLIP ViT-L/14 + linear probe | UnivFD probe checkpoint (auto-download) |

Both models run in **pure Float32** (no FP16 autocast) to prevent gradient underflow during PGD optimization.

### Checkpoint Auto-Download
The first run will attempt to download pretrained weights to `phase1/checkpoints/`.
You can also manually provide paths:
```bash
python run_phase1_eval.py \
    --art_ckpt /path/to/NPR.pth \
    --sem_ckpt /path/to/univfd_probe.pth \
    ...
```

---

## 📊 CSED Modes

| Mode | Flag | Description |
|---|---|---|
| **Grad-CAM** | `--csed_mode gradcam` | Phase 0 baseline. Single target layer per stream. |
| **Grad-CAM++** | `--csed_mode gradcam++` | Better gradient localization. Same single target layer. |
| **Ensemble** | `--csed_mode ensemble` | Average CSED across 3 layers per stream (multi-depth). |

---

## 🗃️ Output Format

Results are written to `results/phase1_<csed_mode>_results.json`.

Structure:
```json
{
  "<generator_name>": {
    "N": 2500,
    "conditions": {
      "clean": { "mean_cosine": 0.123, "std_cosine": 0.045 },
      "PGD-T1 (lambda=0.0, 30 steps)": {
        "n": 2500,
        "asr_pct": 99.8,
        "n_flipped": 2495,
        "total_deg_rate_pct": 34.5,
        "cross_tab_2x2": {
          "degen_and_flip": 860, "degen_and_unflip": 0,
          "valid_and_flip": 1635, "valid_and_unflip": 5
        },
        "auc_filtered": 0.701,
        "ci_filtered": [0.68, 0.72],
        "n_valid_flipped": 1635,
        "auc_flipped_filtered": 0.705,
        "ci_flipped_filtered": [0.685, 0.725]
      }
    }
  }
}
```

---

## 📦 Codebase Index

| File | Description |
|---|---|
| `detectors.py` | NPRDetector + UnivFDDetector with checkpoint auto-loading and multi-layer target support. |
| `data_loader.py` | GenImage 8-generator balanced loader + Phase 0-compatible local loader. |
| `csed.py` | Pure FP32 CSED pipeline: Grad-CAM, Grad-CAM++, multi-layer ensemble. |
| `attacks.py` | T1 (FGSM, PGD) + T2 adaptive attacks, all pure FP32 (FP16 autocast removed). |
| `run_phase1_eval.py` | Main evaluation harness: per-generator metrics, 2×2 cross-tabs, Flipped+Filtered AUC with 95% CIs. |
| `results/` | Output JSON files (gitignored except for `.png` plots). |

---

## 🔗 Related

- [Phase 0 README](../phase0/README.md) — Pilot benchmarking (N=300, locked FP32 authoritative baseline)
- [GenImage GitHub](https://github.com/GenImage-Dataset/GenImage)
- [NPR: Rethinking Up-Sampling](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
- [UnivFD: Universal Fake Detection](https://github.com/Yuheng-Li/UniversalFakeDetect)
