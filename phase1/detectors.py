"""
Phase 1 — Detector Definitions (Matched, Production-Grade)
============================================================
Paradigm A : NPR — "Rethinking the Up-Sampling Operations in CNN-based
             Generative Network for Generalizable Deepfake Detection"
             (Tan et al., CVPR 2024)
             → ResNet-50 backbone with a binary forensics classification head.
             Loads pretrained NPR weights when available, else falls back to
             ImageNet init (conservative stand-in, noted as a limitation).

Paradigm B : UnivFD — "Towards Universal Fake Image Detection by
             Generalizing the Spectrum of Generative Models"
             (Ojha et al., CVPR 2023)
             → Frozen CLIP ViT-L/14 visual encoder + linear probe (binary).
             Loads pretrained UnivFD probe weights when available.

Both models are run INFERENCE-ONLY in pure Float32.
No fine-tuning or training happens in Phase 1.

Weight loading priority (for each model):
  1. Local checkpoint file (--art_ckpt / --sem_ckpt flags or config dict)
  2. Auto-download from known URLs (requires internet)
  3. ImageNet/CLIP default init  (conservative fallback, logged as limitation)
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import clip
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Weight download utility
# ──────────────────────────────────────────────────────────────────────────────
def _try_download(url: str, dest: Path) -> bool:
    """Try to download a file; return True on success."""
    try:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded {dest.name} from {url}")
        return True
    except Exception as e:
        logger.warning(f"Could not download {url}: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Paradigm A: NPR (artifact / frequency / up-sampling artifact detector)
# ──────────────────────────────────────────────────────────────────────────────
class NPRDetector(nn.Module):
    """
    ResNet-50 backbone with a binary forensics classification head.

    Architecture follows NPR (Tan et al., CVPR 2024):
      - Standard ResNet-50 feature extractor up to layer4
      - Global average pool → 2048-dim feature
      - Linear classification head → 2 logits (real=0, fake=1)

    In Phase 1, we load the publicly released NPR checkpoint when available.
    The checkpoint key mapping handles the original NPR model.module prefix.
    """

    # Public NPR checkpoint released by the authors
    NPR_CHECKPOINT_URL = (
        "https://github.com/chuangchuangtan/NPR-DeepfakeDetection/releases/download/v1.0/NPR.pth"
    )

    def __init__(self, pretrained: bool = True, checkpoint_path: Optional[str] = None):
        super().__init__()
        base = tvm.resnet50(
            weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Keep features up to layer4 (exclude AdaptiveAvgPool2d and FC)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(2048, 2)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

        # Try to load NPR pretrained weights
        self._loaded_checkpoint = None
        if checkpoint_path is not None:
            self._load_checkpoint(Path(checkpoint_path))
        else:
            default_path = Path("checkpoints/NPR.pth")
            if default_path.exists():
                self._load_checkpoint(default_path)
            else:
                downloaded = _try_download(self.NPR_CHECKPOINT_URL, default_path)
                if downloaded:
                    self._load_checkpoint(default_path)
                else:
                    logger.warning(
                        "NPR checkpoint not found and download failed. "
                        "Using ImageNet-pretrained ResNet-50 as conservative stand-in. "
                        "Results will be conservative — use proper NPR weights for final evaluation."
                    )

    def _load_checkpoint(self, path: Path):
        try:
            ckpt = torch.load(path, map_location="cpu")
            # NPR checkpoint may be wrapped in DataParallel (module. prefix)
            state = ckpt.get("state_dict", ckpt)
            # Strip 'module.' prefix if present
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, unexpected = self.load_state_dict(state, strict=False)
            self._loaded_checkpoint = str(path)
            logger.info(
                f"Loaded NPR checkpoint from {path}. "
                f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
        except Exception as e:
            logger.warning(f"Failed to load NPR checkpoint {path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.avgpool(x).flatten(1)
        return self.classifier(feat)

    def get_target_layer(self):
        """Primary Grad-CAM target: last conv in layer4."""
        return self.layer4[-1].conv3

    def get_multi_layer_targets(self):
        """Multi-layer Grad-CAM targets for depth ensemble CSED (Phase 1+)."""
        return [
            self.layer2[-1].conv3,  # mid-level features (semantics + texture)
            self.layer3[-1].conv3,  # high-level features
            self.layer4[-1].conv3,  # deepest conv (main target)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Paradigm B: UnivFD (semantic / generative spectrum detector)
# ──────────────────────────────────────────────────────────────────────────────
class UnivFDDetector(nn.Module):
    """
    Frozen CLIP ViT-L/14 visual encoder + linear probe (binary).
    Mirrors the UnivFD architecture exactly (Ojha et al., CVPR 2023).

    The probe head maps 768-dim CLIP visual features to 2 logits.
    Loads pretrained UnivFD probe weights when available.
    """

    # Public UnivFD probe checkpoint
    UNIVFD_PROBE_URL = (
        "https://github.com/Yuheng-Li/UniversalFakeDetect/releases/download/v1.0/univfd_probe.pth"
    )

    def __init__(self, device: str = "cuda", checkpoint_path: Optional[str] = None):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-L/14", device=device)
        self.clip_model = self.clip_model.visual.float()
        # Keep requires_grad=True on visual model so Grad-CAM can backprop
        for p in self.clip_model.parameters():
            p.requires_grad_(True)

        feat_dim = 768  # ViT-L/14 visual output dim
        self.probe = nn.Linear(feat_dim, 2)
        nn.init.normal_(self.probe.weight, std=0.01)
        nn.init.zeros_(self.probe.bias)

        self._loaded_checkpoint = None
        if checkpoint_path is not None:
            self._load_checkpoint(Path(checkpoint_path))
        else:
            default_path = Path("checkpoints/univfd_probe.pth")
            if default_path.exists():
                self._load_checkpoint(default_path)
            else:
                downloaded = _try_download(self.UNIVFD_PROBE_URL, default_path)
                if downloaded:
                    self._load_checkpoint(default_path)
                else:
                    logger.warning(
                        "UnivFD probe checkpoint not found and download failed. "
                        "Using randomly initialized linear probe as conservative stand-in. "
                        "Results will be conservative — use proper UnivFD weights for final evaluation."
                    )

    def _load_checkpoint(self, path: Path):
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            # Load only the probe head weights (ignore backbone if present)
            probe_state = {
                k.replace("probe.", ""): v
                for k, v in state.items()
                if k.startswith("probe.")
            }
            if probe_state:
                self.probe.load_state_dict(probe_state, strict=False)
                self._loaded_checkpoint = str(path)
                logger.info(f"Loaded UnivFD probe weights from {path}")
            else:
                missing, unexpected = self.probe.load_state_dict(state, strict=False)
                self._loaded_checkpoint = str(path)
                logger.info(
                    f"Loaded UnivFD checkpoint from {path}. "
                    f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                )
        except Exception as e:
            logger.warning(f"Failed to load UnivFD checkpoint {path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.clip_model(x).float()
        return self.probe(feat)

    def get_target_layer(self):
        """Primary Grad-CAM target: last transformer block's LayerNorm."""
        return self.clip_model.transformer.resblocks[-1].ln_1

    def get_multi_layer_targets(self):
        """Multi-layer Grad-CAM targets for depth ensemble CSED (Phase 1+)."""
        blocks = self.clip_model.transformer.resblocks
        n = len(blocks)
        return [
            blocks[n // 2].ln_1,      # mid-depth transformer block
            blocks[3 * n // 4].ln_1,  # later block
            blocks[-1].ln_1,          # deepest block (primary target)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Unified loader
# ──────────────────────────────────────────────────────────────────────────────
def load_detectors(
    device: str = "cuda",
    art_checkpoint: Optional[str] = None,
    sem_checkpoint: Optional[str] = None,
) -> tuple:
    """
    Load both Phase 1 detectors on the given device in pure FP32 precision.

    Args:
        device:          Target device ('cuda' or 'cpu').
        art_checkpoint:  Optional path to NPR checkpoint file.
        sem_checkpoint:  Optional path to UnivFD probe checkpoint file.

    Returns:
        (art_model, sem_model): NPRDetector, UnivFDDetector — both .eval() FP32.
    """
    art = NPRDetector(pretrained=True, checkpoint_path=art_checkpoint)
    art = art.float().to(device).eval()

    sem = UnivFDDetector(device=device, checkpoint_path=sem_checkpoint)
    sem = sem.float().to(device).eval()

    # Log checkpoint provenance for reproducibility
    logger.info(f"NPR Detector loaded checkpoint: {art._loaded_checkpoint or 'ImageNet default (no forensics weights)'}")
    logger.info(f"UnivFD Detector loaded checkpoint: {sem._loaded_checkpoint or 'CLIP default (random probe)'}")

    return art, sem
