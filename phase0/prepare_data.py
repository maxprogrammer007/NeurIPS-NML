"""
Phase 0 — Dataset Preparation Script
======================================
Since GenImage archives are multi-part zip splits too large to download
in the pilot window, this script builds a well-matched real/AI pilot set
from two publicly accessible sources:

REAL images  : COCO val2017 (natural photographs, diverse scenes)
               Download: http://images.cocodataset.org/zips/val2017.zip (~780 MB)

AI images    : DiffusionDB 2M first-part sample (Stable Diffusion v1.4 outputs)
               HuggingFace: poloclub/diffusiondb  split=2m_first_1k (1000 images ~50 MB)

Both are well-established benchmarks.  The methodology is identical to what
would be used with the GenImage split once the full dataset is available.

IMPORTANT: This script downloads ~830 MB total. Run once; subsequent runs
           use the cached data.
"""

import os
import random
import shutil
from pathlib import Path
import urllib.request

DATA_ROOT  = Path(__file__).parent / "data"
REAL_DIR   = DATA_ROOT / "nature"
AI_DIR     = DATA_ROOT / "ai"
N_PER_CLASS = 300   # 300 real + 300 AI = 600 images total for pilot

REAL_DIR.mkdir(parents=True, exist_ok=True)
AI_DIR.mkdir(parents=True, exist_ok=True)


# ─── Step 1: Real images from COCO val2017 ──────────────────────────────────
def download_coco_val(target_n: int = N_PER_CLASS):
    """Download COCO val2017 and copy N random images to REAL_DIR."""
    coco_zip = DATA_ROOT / "val2017.zip"
    coco_dir = DATA_ROOT / "val2017"

    # Download
    if not coco_zip.exists():
        print("[COCO] Downloading val2017.zip (~780 MB) …")
        url = "http://images.cocodataset.org/zips/val2017.zip"
        urllib.request.urlretrieve(url, coco_zip)
        print("[COCO] Downloaded.")

    # Extract
    if not coco_dir.exists():
        print("[COCO] Extracting …")
        import zipfile
        with zipfile.ZipFile(coco_zip, 'r') as zf:
            zf.extractall(DATA_ROOT)
        print("[COCO] Extracted.")

    # Copy N random images
    all_imgs = list(coco_dir.glob("*.jpg"))
    random.seed(42)
    selected = random.sample(all_imgs, min(target_n, len(all_imgs)))
    existing = len(list(REAL_DIR.glob("*.jpg")))
    if existing >= target_n:
        print(f"[COCO] Already have {existing} real images. Skipping copy.")
        return

    for src in selected:
        dst = REAL_DIR / src.name
        if not dst.exists():
            shutil.copy(src, dst)
    print(f"[COCO] Copied {len(selected)} real images to {REAL_DIR}")


# ─── Step 2: AI images from DiffusionDB via HuggingFace ─────────────────────
def download_diffusiondb(target_n: int = N_PER_CLASS):
    """Load first target_n images from DiffusionDB and save to AI_DIR."""
    existing = len(list(AI_DIR.glob("*.png")) + list(AI_DIR.glob("*.jpg")))
    if existing >= target_n:
        print(f"[DiffDB] Already have {existing} AI images. Skipping.")
        return

    print("[DiffDB] Loading DiffusionDB 2M first 1k subset …")
    from datasets import load_dataset
    ds = load_dataset("poloclub/diffusiondb", "2m_first_1k",
                      split="train", trust_remote_code=True)
    count = 0
    for i, sample in enumerate(ds):
        if count >= target_n:
            break
        img = sample["image"]
        if img is None:
            continue
        out_path = AI_DIR / f"diffusiondb_{i:05d}.png"
        if not out_path.exists():
            img.save(out_path)
        count += 1
    print(f"[DiffDB] Saved {count} AI images to {AI_DIR}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 0 — Dataset Preparation")
    print("=" * 60)
    download_coco_val(N_PER_CLASS)
    download_diffusiondb(N_PER_CLASS)
    real_count = len(list(REAL_DIR.glob("*.jpg")))
    ai_count   = len(list(AI_DIR.glob("*.png")) + list(AI_DIR.glob("*.jpg")))
    print(f"\n✓ Dataset ready: {real_count} real, {ai_count} AI images")
    print(f"  Real dir : {REAL_DIR}")
    print(f"  AI dir   : {AI_DIR}")
