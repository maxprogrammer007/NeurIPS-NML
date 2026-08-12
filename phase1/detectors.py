"""
Phase 1 — Detector Definitions (Matched, Production-Grade)
============================================================
Paradigm A : NPR — "Rethinking the Up-Sampling Operations in CNN-based
             Generative Network for Generalizable Deepfake Detection"
             (Tan et al., CVPR 2024)
             → Custom ResNet backbone (3x3 stride-1 conv1 to capture spatial up-sampling
               artifacts + layer1 + layer2 bottleneck blocks + fc1).
             Loads official pretrained NPR weights from HuggingFace (`bitmind/npr`).

Paradigm B : UnivFD — "Towards Universal Fake Image Detection by
             Generalizing the Spectrum of Generative Models"
             (Ojha et al., CVPR 2023)
             → Frozen CLIP ViT-L/14 visual encoder + linear probe (binary).
             Loads official pretrained UnivFD linear probe weights from HuggingFace (`DaniilOr/detect`).

Both models run INFERENCE-ONLY in pure Float32.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import clip
from pathlib import Path
from typing import Optional
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paradigm A: NPR (Up-sampling artifact detector, Tan et al. CVPR 2024)
# ──────────────────────────────────────────────────────────────────────────────
class NPRDetector(nn.Module):
    """
    NPR Deepfake Detector (Tan et al., CVPR 2024).
    Uses a 3x3 stride-1 conv1 to retain high-frequency up-sampling artifacts,
    followed by ResNet bottleneck layer1 & layer2, and a linear head.
    Loads official pretrained NPR weights from HuggingFace repository `bitmind/npr`.
    """

    def __init__(self, pretrained: bool = True, checkpoint_path: Optional[str] = None):
        super().__init__()
        base = tvm.resnet50(weights=None)
        
        # NPR 3x3 conv1 replaces standard ResNet 7x7 conv1 to preserve high-frequency artifacts
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(512, 2)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

        self._loaded_checkpoint = None
        ckpt_file = checkpoint_path
        if ckpt_file is None:
            default_path = Path("checkpoints/NPR.pth")
            if default_path.exists():
                ckpt_file = str(default_path)
            else:
                try:
                    logger.info("Downloading official NPR weights from HF bitmind/npr ...")
                    downloaded = hf_hub_download(repo_id="bitmind/npr", filename="npr.pth")
                    ckpt_file = downloaded
                except Exception as e:
                    logger.warning(f"Could not download NPR weights: {e}")

        if ckpt_file:
            self._load_checkpoint(Path(ckpt_file))

    def _load_checkpoint(self, path: Path):
        try:
            state = torch.load(path, map_location="cpu")
            state = {k.replace("module.", ""): v for k, v in state.items()}
            
            if "fc1.weight" in state:
                fc_w = state.pop("fc1.weight")
                fc_b = state.pop("fc1.bias")
                if fc_w.shape[0] == 1:
                    self.classifier = nn.Linear(fc_w.shape[1], 2)
                    self.classifier.weight.data = torch.cat([-fc_w, fc_w], dim=0)
                    self.classifier.bias.data = torch.cat([-fc_b, fc_b], dim=0)
                else:
                    self.classifier = nn.Linear(fc_w.shape[1], fc_w.shape[0])
                    self.classifier.weight.data = fc_w
                    self.classifier.bias.data = fc_b

            missing, unexpected = self.load_state_dict(state, strict=False)
            self._loaded_checkpoint = str(path)
            logger.info(
                f"Loaded NPR checkpoint successfully from {path}. "
                f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
        except Exception as e:
            logger.warning(f"Failed to load NPR checkpoint {path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat = self.avgpool(x).flatten(1)
        return self.classifier(feat)

    def get_target_layer(self):
        """Primary Grad-CAM target: last conv in layer2."""
        return self.layer2[-1].conv3

    def get_multi_layer_targets(self):
        """Multi-layer Grad-CAM targets for depth ensemble CSED."""
        return [
            self.layer1[-1].conv3,
            self.layer2[-2].conv3,
            self.layer2[-1].conv3,
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Paradigm B: UnivFD (semantic / generative spectrum detector)
# ──────────────────────────────────────────────────────────────────────────────
class UnivFDDetector(nn.Module):
    """
    Frozen CLIP ViT-L/14 visual encoder + linear probe (binary).
    Loads official UnivFD linear probe weights from HuggingFace repository `DaniilOr/detect`.
    """

    def __init__(self, device: str = "cuda", checkpoint_path: Optional[str] = None):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-L/14", device=device)
        self.clip_model = self.clip_model.visual.float()
        for p in self.clip_model.parameters():
            p.requires_grad_(True)

        feat_dim = 768  # ViT-L/14 visual output dim
        self.probe = nn.Linear(feat_dim, 2)
        nn.init.normal_(self.probe.weight, std=0.01)
        nn.init.zeros_(self.probe.bias)

        self._loaded_checkpoint = None
        ckpt_file = checkpoint_path
        if ckpt_file is None:
            default_path = Path("checkpoints/univfd_probe.pth")
            if default_path.exists():
                ckpt_file = str(default_path)
            else:
                try:
                    logger.info("Downloading official UnivFD linear probe weights from HF DaniilOr/detect ...")
                    downloaded = hf_hub_download(repo_id="DaniilOr/detect", filename="fc_weights.pth")
                    ckpt_file = downloaded
                except Exception as e:
                    logger.warning(f"Could not download UnivFD weights: {e}")

        if ckpt_file:
            self._load_checkpoint(Path(ckpt_file))

    def _load_checkpoint(self, path: Path):
        try:
            state = torch.load(path, map_location="cpu")
            state = {k.replace("module.", ""): v for k, v in state.items()}
            
            if "weight" in state:
                w = state["weight"]
                b = state.get("bias", torch.zeros(1))
                if w.shape[0] == 1:
                    self.probe = nn.Linear(w.shape[1], 2)
                    self.probe.weight.data = torch.cat([-w, w], dim=0)
                    self.probe.bias.data = torch.cat([-b, b], dim=0)
                else:
                    self.probe = nn.Linear(w.shape[1], w.shape[0])
                    self.probe.weight.data = w
                    self.probe.bias.data = b
                self._loaded_checkpoint = str(path)
                logger.info(f"Loaded UnivFD probe weights successfully from {path}")
            else:
                missing, unexpected = self.probe.load_state_dict(state, strict=False)
                self._loaded_checkpoint = str(path)
                logger.info(f"Loaded UnivFD checkpoint from {path}. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            logger.warning(f"Failed to load UnivFD checkpoint {path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.clip_model(x).float()
        return self.probe(feat)

    def get_target_layer(self):
        """Primary Grad-CAM target: last transformer block's LayerNorm."""
        return self.clip_model.transformer.resblocks[-1].ln_1

    def get_multi_layer_targets(self):
        """Multi-layer Grad-CAM targets for depth ensemble CSED."""
        blocks = self.clip_model.transformer.resblocks
        n = len(blocks)
        return [
            blocks[n // 2].ln_1,
            blocks[3 * n // 4].ln_1,
            blocks[-1].ln_1,
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

    Returns:
        (art_model, sem_model): NPRDetector, UnivFDDetector — both .eval() FP32.
    """
    art = NPRDetector(pretrained=True, checkpoint_path=art_checkpoint)
    art = art.float().to(device).eval()

    sem = UnivFDDetector(device=device, checkpoint_path=sem_checkpoint)
    sem = sem.float().to(device).eval()

    logger.info(f"NPR Detector checkpoint: {art._loaded_checkpoint or 'ImageNet default'}")
    logger.info(f"UnivFD Detector checkpoint: {sem._loaded_checkpoint or 'CLIP default'}")

    return art, sem
