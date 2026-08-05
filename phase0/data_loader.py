"""
Phase 0 — Data Loader
=====================
Loads a small fixed slice of GenImage (real + AI images) for the pilot.
Handles the GenImage folder structure:
    {generator}/train/ai/    → AI-generated
    {generator}/train/nature/ → real (ImageNet)
Also supports loading from extracted directories.
"""

import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ─── ImageNet-normalised pre-processing (same as CLIP & NPR) ────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# CLIP uses its own normalisation — kept separate for the semantic stream
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


class PilotDataset(Dataset):
    """
    A small, fixed image dataset for Phase 0.
    Labels: 0 = real (nature), 1 = AI-generated (ai)
    """

    def __init__(self, real_paths: list, ai_paths: list,
                 transform=None, seed: int = 42):
        random.seed(seed)
        self.samples = (
            [(p, 0) for p in real_paths] +
            [(p, 1) for p in ai_paths]
        )
        random.shuffle(self.samples)
        self.transform = transform or TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)


def _collect_images(folder: Path, exts=(".jpg", ".jpeg", ".png", ".JPEG"),
                    max_n: Optional[int] = None) -> list:
    paths = [p for p in sorted(folder.rglob("*"))
             if p.suffix.lower() in exts]
    if max_n and len(paths) > max_n:
        random.shuffle(paths)
        paths = paths[:max_n]
    return paths


def build_pilot_dataset(data_root: str,
                        n_real: int = 300,
                        n_ai: int = 300,
                        seed: int = 42,
                        transform=None) -> PilotDataset:
    """
    Build a balanced pilot dataset from a GenImage-style root.
    Scans for all images under real_dir and ai_dir.
    """
    root = Path(data_root)
    real_dir = root / "nature"
    ai_dir   = root / "ai"

    if not real_dir.exists() or not ai_dir.exists():
        raise FileNotFoundError(
            f"Expected {real_dir} and {ai_dir}. "
            "Check that the data is organised in GenImage format."
        )

    random.seed(seed)
    real_paths = _collect_images(real_dir, max_n=n_real)
    ai_paths   = _collect_images(ai_dir,   max_n=n_ai)

    print(f"[Dataset] real={len(real_paths)}, ai={len(ai_paths)}")
    return PilotDataset(real_paths, ai_paths, transform=transform, seed=seed)


def build_dataset_from_dirs(real_dir: str, ai_dir: str,
                             n_real: int = 300, n_ai: int = 300,
                             seed: int = 42, transform=None) -> PilotDataset:
    """
    Build from explicitly specified real and ai directories.
    """
    random.seed(seed)
    real_paths = _collect_images(Path(real_dir), max_n=n_real)
    ai_paths   = _collect_images(Path(ai_dir),   max_n=n_ai)
    print(f"[Dataset] real={len(real_paths)}, ai={len(ai_paths)}")
    return PilotDataset(real_paths, ai_paths, transform=transform, seed=seed)
