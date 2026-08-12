"""
Verification script: FP32 vs FP16 Grad-CAM Degeneracy Check
------------------------------------------------------------
Tests whether flat Grad-CAM maps (std < 1e-5) are genuine physical dying-ReLU phenomena
or numerical precision underflow artifacts caused by FP16 Tensor Core execution.
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('.')
from detectors import load_detectors
from csed import get_gradcam_map
from data_loader import build_dataset_from_dirs
from attacks import pgd_adaptive_csed

def verify_fp32_degeneracy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    art_model, sem_model = load_detectors(device)
    art_layer = art_model.get_target_layer()
    sem_layer = sem_model.get_target_layer()
    
    # Force full FP32 float precision
    art_model.float().eval()
    sem_model.float().eval()
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = build_dataset_from_dirs('./data/nature', './data/ai', n_real=150, n_ai=150, seed=seed)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    images, labels = [], []
    for img, lbl in loader:
        images.append(img)
        labels.append(lbl)
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    
    print("\nGenerating PGD-T1 (30 steps, eps=8/255) in FULL FP32 precision...")
    # Generate PGD-T1 (lambda=0.0) in full FP32 without autocast
    adv_imgs_fp32 = []
    for i in range(0, len(images), 4):
        img_b = images[i:i+4]
        lbl_b = labels[i:i+4]
        # Run in pure FP32 (no autocast)
        adv_b = pgd_adaptive_csed(
            art_model, sem_model,
            art_layer, sem_layer,
            img_b, lbl_b,
            eps=8/255, alpha=2/255, steps=30, lam=0.0, device=device,
            use_feature_proxy=True
        )
        adv_imgs_fp32.append(adv_b.cpu())
    adv_imgs_fp32 = torch.cat(adv_imgs_fp32, 0)
    
    # Evaluate Grad-CAM in full FP32
    print("Computing Grad-CAM maps in FP32...")
    fp32_stds = []
    fp32_deg_count = 0
    
    for idx in range(len(adv_imgs_fp32)):
        img_adv = adv_imgs_fp32[idx:idx+1].to(device)
        lbl_val = int(labels[idx])
        
        # Pure FP32 Grad-CAM
        ma = get_gradcam_map(art_model, art_layer, img_adv[0], lbl_val, False)
        std_val = float(np.std(ma))
        fp32_stds.append(std_val)
        if std_val < 1e-5:
            fp32_deg_count += 1
            
    fp32_deg_rate = fp32_deg_count / len(adv_imgs_fp32)
    print(f"\n--- FP32 DEGENERACY VERIFICATION RESULTS ---")
    print(f"Total Samples Tested: {len(adv_imgs_fp32)}")
    print(f"FP32 Degenerate Flat-Map Count: {fp32_deg_count} / {len(adv_imgs_fp32)}")
    print(f"FP32 Degeneracy Rate: {fp32_deg_rate*100:.2f}%")
    print(f"Min Spatial Std in FP32: {min(fp32_stds):.8f}")
    print(f"Max Spatial Std in FP32: {max(fp32_stds):.8f}")

if __name__ == '__main__':
    verify_fp32_degeneracy()
