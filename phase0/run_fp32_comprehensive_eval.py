"""
Phase 0 FP32 Comprehensive Evaluation Script
---------------------------------------------
Executes locked evaluation in full FP32 float precision across all 7 conditions:
1. Clean Baseline
2. FGSM (eps=8/255)
3. PGD-T1 (lambda=0.0, 30 steps)
4. T2 Sweep (lambda in {0.1, 1.0, 10.0, 50.0, 100.0}, 30 steps)

Integrates:
- Pure FP32 Grad-CAM computation (prevents FP16 numerical underflow over-filtering)
- Independent per-sample degeneracy and label-flip verification
- 2x2 selection bias matrix (Degen x Label-Flip)
- Filtered AUC and Flipped + Filtered AUC with 95% bootstrapped CIs (B=1000)
"""

import sys
import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('.')
from detectors import load_detectors
from csed import get_gradcam_map, compute_csed
from data_loader import build_dataset_from_dirs
from attacks import pgd_adaptive_csed
import torchattacks

def bootstrap_auc_ci(y_true, y_score, B=1000, seed=42):
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return (float('nan'), float('nan'))
    rng = np.random.RandomState(seed)
    boot_aucs = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if len(boot_aucs) == 0:
        return (float('nan'), float('nan'))
    return (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5)))

class JointEnsembleModel(torch.nn.Module):
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
    def forward(self, x):
        return (self.m1(x) + self.m2(x)) / 2.0

def run_fp32_comprehensive_eval():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load models in pure FP32 float precision
    art_model, sem_model = load_detectors(device)
    art_model = art_model.float().eval()
    sem_model = sem_model.float().eval()
    art_layer = art_model.get_target_layer()
    sem_layer = sem_model.get_target_layer()
    joint_model = JointEnsembleModel(art_model, sem_model).to(device).float().eval()
    
    # 2. Lock dataset & seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = build_dataset_from_dirs('./data/nature', './data/ai', n_real=150, n_ai=150, seed=seed)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    images, labels = [], []
    for img, lbl in loader:
        images.append(img)
        labels.append(lbl)
    images = torch.cat(images, 0).float()
    labels = torch.cat(labels, 0)
    N = len(images)
    print(f"Dataset locked in FP32: Total={N} (Real=150, AI=150)")
    
    # 3. Clean Baseline Evaluation in FP32
    print("\n--- Evaluating Clean Baseline (FP32) ---")
    clean_cos, clean_js = [], []
    clean_deg_a, clean_deg_b = 0, 0
    clean_preds = []
    
    for idx in range(N):
        img_t = images[idx:idx+1].to(device)
        lbl_val = int(labels[idx])
        with torch.no_grad():
            logits = joint_model(img_t)
            pred = int(logits.argmax(dim=1).item())
        clean_preds.append(pred)
        
        # Grad-CAM in full FP32
        ma = get_gradcam_map(art_model, art_layer, img_t[0], lbl_val, False)
        mb = get_gradcam_map(sem_model, sem_layer, img_t[0], lbl_val, True)
        
        if np.std(ma) < 1e-5: clean_deg_a += 1
        if np.std(mb) < 1e-5: clean_deg_b += 1
        
        d = compute_csed(ma, mb)
        clean_cos.append(d['cosine'])
        clean_js.append(d['js'])
        
    clean_cos = np.array(clean_cos)
    clean_js = np.array(clean_js)
    clean_preds = np.array(clean_preds)
    clean_acc = float((clean_preds == labels.numpy()).mean())
    print(f"Clean Joint Classifier Accuracy: {clean_acc*100:.1f}%")
    print(f"Clean Degeneracy (FP32): ResNet={clean_deg_a/N*100:.2f}%, CLIP={clean_deg_b/N*100:.2f}%")
    
    configs = [
        ('FGSM (eps=8/255)', 'fgsm', None),
        ('PGD-T1 (lambda=0.0, 30 steps)', 'pgd_t1', 0.0),
        ('T2 Sweep lambda=0.1 (30 steps)', 't2', 0.1),
        ('T2 Sweep lambda=1.0 (30 steps)', 't2', 1.0),
        ('T2 Sweep lambda=10.0 (30 steps)', 't2', 10.0),
        ('T2 Sweep lambda=50.0 (30 steps)', 't2', 50.0),
        ('T2 Sweep lambda=100.0 (30 steps)', 't2', 100.0),
    ]
    
    full_results = {}
    y_true_all = np.concatenate([np.zeros(N), np.ones(N)])
    
    for name, mode, lam in configs:
        print(f"\n--- Running Attack in FP32: {name} ---")
        adv_imgs_list = []
        
        if mode == 'fgsm':
            atk = torchattacks.FGSM(joint_model, eps=8/255)
            for i in range(0, N, 4):
                adv_b = atk(images[i:i+4].to(device), labels[i:i+4].to(device))
                adv_imgs_list.append(adv_b.cpu())
                torch.cuda.empty_cache()
        elif mode in ('pgd_t1', 't2'):
            for i in tqdm(range(0, N, 4), desc=f"Generating {name}"):
                img_batch = images[i:i+4]
                lbl_batch = labels[i:i+4]
                adv_batch = pgd_adaptive_csed(
                    art_model, sem_model,
                    art_layer, sem_layer,
                    img_batch, lbl_batch,
                    eps=8/255, alpha=2/255, steps=30, lam=lam, device=device
                )
                adv_imgs_list.append(adv_batch.cpu())
                torch.cuda.empty_cache()
                
        adv_imgs = torch.cat(adv_imgs_list, 0)
        
        # Analyze metrics across N samples in pure FP32
        cos_adv, js_adv = [], []
        deg_a_count, deg_b_count = 0, 0
        is_deg_flags = []
        is_flipped_flags = []
        
        for idx in tqdm(range(N), desc=f"FP32 Grad-CAM {name}"):
            img_adv_t = adv_imgs[idx:idx+1].to(device)
            lbl_val = int(labels[idx])
            
            with torch.no_grad():
                logits = joint_model(img_adv_t)
                adv_pred = int(logits.argmax(dim=1).item())
            is_flipped = bool(adv_pred != lbl_val)
            is_flipped_flags.append(is_flipped)
            
            # Pure FP32 Grad-CAM map calculation
            ma = get_gradcam_map(art_model, art_layer, img_adv_t[0], lbl_val, False)
            mb = get_gradcam_map(sem_model, sem_layer, img_adv_t[0], lbl_val, True)
            
            is_deg_a = bool(np.std(ma) < 1e-5)
            is_deg_b = bool(np.std(mb) < 1e-5)
            if is_deg_a: deg_a_count += 1
            if is_deg_b: deg_b_count += 1
            is_deg = is_deg_a or is_deg_b
            is_deg_flags.append(is_deg)
            
            d = compute_csed(ma, mb)
            cos_adv.append(d['cosine'])
            js_adv.append(d['js'])
            
        cos_adv = np.array(cos_adv)
        js_adv = np.array(js_adv)
        is_deg_flags = np.array(is_deg_flags)
        is_flipped_flags = np.array(is_flipped_flags)
        
        asr = float(is_flipped_flags.mean())
        n_flipped = int(is_flipped_flags.sum())
        total_deg_rate = float(is_deg_flags.mean())
        deg_rate_a = deg_a_count / N
        deg_rate_b = deg_b_count / N
        
        # 2x2 Cross Tabulation (FP32): Degeneracy x Label-Flip
        deg_and_flipped = int((is_deg_flags & is_flipped_flags).sum())
        deg_and_unflipped = int((is_deg_flags & (~is_flipped_flags)).sum())
        valid_and_flipped = int(((~is_deg_flags) & is_flipped_flags).sum())
        valid_and_unflipped = int(((~is_deg_flags) & (~is_flipped_flags)).sum())
        
        # 1. Unfiltered AUC
        scores_unfilt = np.concatenate([clean_cos, cos_adv])
        auc_unfilt = float(roc_auc_score(y_true_all, scores_unfilt))
        ci_unfilt = bootstrap_auc_ci(y_true_all, scores_unfilt)
        
        # 2. Filtered AUC (Valid clean vs Valid adv in FP32)
        valid_mask_adv = ~is_deg_flags
        valid_cos_adv = cos_adv[valid_mask_adv]
        valid_clean_cos = clean_cos
        y_true_filt = np.concatenate([np.zeros(len(valid_clean_cos)), np.ones(len(valid_cos_adv))])
        scores_filt = np.concatenate([valid_clean_cos, valid_cos_adv])
        auc_filt = float(roc_auc_score(y_true_filt, scores_filt))
        ci_filt = bootstrap_auc_ci(y_true_filt, scores_filt)
        
        # 3. Flipped + Filtered AUC (Real Attacker's View in FP32)
        valid_flipped_mask = (~is_deg_flags) & is_flipped_flags
        n_valid_flipped = int(valid_flipped_mask.sum())
        if n_valid_flipped > 0:
            valid_flipped_cos = cos_adv[valid_flipped_mask]
            y_true_val_flip = np.concatenate([np.zeros(len(clean_cos)), np.ones(n_valid_flipped)])
            scores_val_flip = np.concatenate([clean_cos, valid_flipped_cos])
            auc_val_flip = float(roc_auc_score(y_true_val_flip, scores_val_flip))
            ci_val_flip = bootstrap_auc_ci(y_true_val_flip, scores_val_flip)
        else:
            auc_val_flip = float('nan')
            ci_val_flip = (float('nan'), float('nan'))
            
        full_results[name] = {
            'asr_pct': round(asr * 100, 2),
            'n_flipped': n_flipped,
            'total_deg_rate_pct': round(total_deg_rate * 100, 2),
            'deg_rate_a_pct': round(deg_rate_a * 100, 2),
            'deg_rate_b_pct': round(deg_rate_b * 100, 2),
            'cross_tab_2x2': {
                'deg_and_flipped': deg_and_flipped,
                'deg_and_unflipped': deg_and_unflipped,
                'valid_and_flipped': valid_and_flipped,
                'valid_and_unflipped': valid_and_unflipped
            },
            'auc_unfiltered': round(auc_unfilt, 4),
            'ci_unfiltered': [round(ci_unfilt[0], 4), round(ci_unfilt[1], 4)],
            'auc_filtered': round(auc_filt, 4),
            'ci_filtered': [round(ci_filt[0], 4), round(ci_filt[1], 4)],
            'auc_flipped_filtered': round(auc_val_flip, 4) if not np.isnan(auc_val_flip) else None,
            'ci_flipped_filtered': [round(ci_val_flip[0], 4), round(ci_val_flip[1], 4)] if not np.isnan(ci_val_flip[0]) else None,
            'n_valid_flipped': n_valid_flipped
        }

    with open('./results/phase0_fp32_comprehensive_summary.json', 'w') as f:
        json.dump(full_results, f, indent=2)
        
    print("\n================ FP32 COMPREHENSIVE SUMMARY ================")
    print(json.dumps(full_results, indent=2))

if __name__ == '__main__':
    run_fp32_comprehensive_eval()
