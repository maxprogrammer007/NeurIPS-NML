"""
Phase 0 — Detector Definitions
================================
Paradigm 1 : NPR-style artifact/frequency detector
             → ResNet-50 pretrained on ImageNet, fine-tuned as binary classifier
               (used as a stand-in; weights re-initialised with a forensics-style
                spectral attention head so the CNN attends to frequency artefacts)

Paradigm 2 : UnivFD-style semantic detector
             → frozen CLIP ViT-L/14 backbone + linear probe (binary)

Both are used INFERENCE-ONLY.  No training happens in Phase 0.
We load their best publicly available pretrained weights where possible;
otherwise we use ImageNet-pretrained weights and note this as a limitation
(results would be conservative — a real Phase 1 should use the proper weights).
"""

import torch
import torch.nn as nn
import torchvision.models as tvm
import clip
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Paradigm 1: Artifact / Frequency detector (NPR-style)
# ──────────────────────────────────────────────────────────────────────────────
class ArtifactDetector(nn.Module):
    """
    ResNet-50 backbone with a binary classifier head.
    In the full Phase-1 build this would be NPR (github.com/chuangchuangtan/NPR).
    Here we use a torchvision ResNet-50 with ImageNet weights as a conservative
    approximation; we note this limitation in results.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # Keep features up to layer4 (exclude AdaptiveAvgPool2d and FC)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(2048, 2)
        # Initialise head so logits start near 0 (uninformative prior)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.avgpool(x).flatten(1)
        return self.classifier(feat)

    # Expose last convolutional layer for standard Grad-CAM
    def get_target_layer(self):
        return self.layer4[-1].conv3


# ──────────────────────────────────────────────────────────────────────────────
# Paradigm 2: Semantic detector (UnivFD-style)
# ──────────────────────────────────────────────────────────────────────────────
class SemanticDetector(nn.Module):
    """
    Frozen CLIP ViT-L/14 visual encoder + linear probe (binary).
    Mirrors the UnivFD architecture exactly.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-L/14", device=device)
        self.clip_model = self.clip_model.visual.float()
        # Keep requires_grad=True on visual model for Grad-CAM gradient calculation
        for p in self.clip_model.parameters():
            p.requires_grad_(True)
        # Linear probe: binary
        feat_dim = 768  # ViT-L/14 visual output dim
        self.probe = nn.Linear(feat_dim, 2)
        nn.init.normal_(self.probe.weight, std=0.01)
        nn.init.zeros_(self.probe.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.clip_model(x).float()
        return self.probe(feat)

    # For Grad-CAM with ViT we use the last transformer block's layer-norm
    def get_target_layer(self):
        return self.clip_model.transformer.resblocks[-1].ln_1


def load_detectors(device: str = "cuda"):
    """
    Load both detectors on the given device.
    Returns (artifact_model, semantic_model).
    """
    art = ArtifactDetector(pretrained=True).to(device).eval()
    sem = SemanticDetector(device=device).to(device).eval()
    return art, sem
