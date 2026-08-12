"""
Phase 1 — GenImage Data Loader
================================
Supports the full GenImage dataset layout:

  <data_root>/
    <generator_name>/
      train/ (or test/)
        ai/    ← AI-generated images (label=1)
        nature/ ← Real ImageNet images  (label=0)

Known GenImage generators (8 total):
  midjourney, stable_diffusion_v_1_4, stable_diffusion_v_1_5,
  wukong, VQDM, ADM, glide, biggan

Usage:
  dataset = build_genimage_dataset(
      data_root       = '/path/to/GenImage',
      generators      = None,    # None = all 8 generators
      split           = 'test',
      n_per_generator = 1250,    # draws 1250 real + 1250 AI per generator
      seed            = 42,
  )

For a pilot subset (no download required), a local-directory mode is also
provided that mirrors the Phase 0 API:
  dataset = build_dataset_from_dirs(real_dir, ai_dir, n_real, n_ai, seed)
"""

import os
import random
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Standard transforms (CLIP / ResNet-50 compatible 224×224)
# ──────────────────────────────────────────────────────────────────────────────
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _NORMALIZE,
])

# ──────────────────────────────────────────────────────────────────────────────
# GenImage known generators
# ──────────────────────────────────────────────────────────────────────────────
GENIMAGE_GENERATORS = [
    "midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "wukong",
    "VQDM",
    "ADM",
    "glide",
    "biggan",
]


# ──────────────────────────────────────────────────────────────────────────────
# Core dataset primitive
# ──────────────────────────────────────────────────────────────────────────────
class ImageListDataset(Dataset):
    """
    A flat image-list dataset. Each sample is (image_tensor, label, metadata).
    metadata dict carries: {'path': str, 'generator': str, 'label': int}
    """

    def __init__(self, image_paths: List[str], labels: List[int],
                 generator: str = "unknown",
                 transform=TRANSFORM_EVAL):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.generator = generator
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Return black image for corrupted files (logged downstream)
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label, {"path": path, "generator": self.generator, "label": label}


def _collect_images(directory: str, exts=(".jpg", ".jpeg", ".png", ".webp")):
    """Recursively collect all image file paths from a directory."""
    paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in exts:
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def _balanced_sample(paths: List[str], n: int, rng: random.Random) -> List[str]:
    """Draw exactly n paths, with replacement if necessary."""
    if len(paths) == 0:
        raise ValueError("Directory is empty — no images found.")
    if n <= len(paths):
        return rng.sample(paths, n)
    # With replacement for rare case where n > available
    return [rng.choice(paths) for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# GenImage multi-generator dataset builder
# ──────────────────────────────────────────────────────────────────────────────
def build_genimage_dataset(
    data_root: str,
    generators: Optional[List[str]] = None,
    split: str = "test",
    n_per_generator: int = 1250,
    seed: int = 42,
    transform=TRANSFORM_EVAL,
) -> ConcatDataset:
    """
    Build a balanced GenImage evaluation dataset.

    Args:
        data_root:        Root path of the GenImage directory.
        generators:       List of generator names to include (None = all 8).
        split:            'train' or 'test' subfolder.
        n_per_generator:  Number of real AND AI images per generator.
                          Total N = n_per_generator * 2 * len(generators).
        seed:             Random seed for reproducible sampling.
        transform:        torchvision transform pipeline.

    Returns:
        ConcatDataset of ImageListDatasets (one per generator).

    Expected directory layout:
        <data_root>/<generator>/<split>/ai/   ← AI images   (label=1)
        <data_root>/<generator>/<split>/nature/ ← Real images (label=0)
    """
    root = Path(data_root)
    gens = generators or GENIMAGE_GENERATORS
    rng = random.Random(seed)

    sub_datasets = []
    for gen in gens:
        ai_dir = root / gen / split / "ai"
        nat_dir = root / gen / split / "nature"

        if not ai_dir.exists():
            raise FileNotFoundError(
                f"GenImage AI directory not found: {ai_dir}\n"
                f"Expected layout: <data_root>/<generator>/<split>/ai/"
            )
        if not nat_dir.exists():
            raise FileNotFoundError(
                f"GenImage nature directory not found: {nat_dir}\n"
                f"Expected layout: <data_root>/<generator>/<split>/nature/"
            )

        ai_paths = _collect_images(str(ai_dir))
        nat_paths = _collect_images(str(nat_dir))

        ai_sample = _balanced_sample(ai_paths, n_per_generator, rng)
        nat_sample = _balanced_sample(nat_paths, n_per_generator, rng)

        paths = nat_sample + ai_sample
        labels = [0] * n_per_generator + [1] * n_per_generator

        # Shuffle together to break ordering bias
        combined = list(zip(paths, labels))
        rng.shuffle(combined)
        paths, labels = zip(*combined)

        sub_datasets.append(
            ImageListDataset(list(paths), list(labels),
                             generator=gen, transform=transform)
        )
        print(f"  Generator '{gen}': {n_per_generator} real + {n_per_generator} AI = {n_per_generator * 2} total")

    total = sum(len(d) for d in sub_datasets)
    print(f"GenImage dataset: {len(sub_datasets)} generators, {total} total samples (seed={seed})")
    return ConcatDataset(sub_datasets), sub_datasets  # also return per-generator list


# ──────────────────────────────────────────────────────────────────────────────
# Phase 0-compatible local directory builder (unchanged API for compatibility)
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset_from_dirs(
    real_dir: str,
    ai_dir: str,
    n_real: int = 150,
    n_ai: int = 150,
    seed: int = 42,
    transform=TRANSFORM_EVAL,
) -> ImageListDataset:
    """
    Phase 0-compatible builder for a flat real/AI split from two directories.
    Used for pilot runs and local testing without the full GenImage download.
    """
    rng = random.Random(seed)
    real_paths = _collect_images(real_dir)
    ai_paths = _collect_images(ai_dir)

    real_sample = _balanced_sample(real_paths, n_real, rng)
    ai_sample = _balanced_sample(ai_paths, n_ai, rng)

    paths = real_sample + ai_sample
    labels = [0] * n_real + [1] * n_ai

    combined = list(zip(paths, labels))
    rng.shuffle(combined)
    paths, labels = zip(*combined)

    return ImageListDataset(list(paths), list(labels), generator="local", transform=transform)
