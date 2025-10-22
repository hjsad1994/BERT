# Loss Function Configurations - Quick Comparison

## 🎯 All Configurations Available

```
multi_label/
├── config_multi.yaml          # Focal + Contrastive (Original - 95.99% F1) ⭐
├── config_focal_only.yaml     # Focal ONLY (Baseline) 🔹
├── config_ghm.yaml            # GHM ONLY (No Contrastive) 🔥
└── config_ghm_full.yaml       # GHM + Contrastive (Best Expected) 🚀
```

---

## 📊 Quick Comparison Table

| Config | Loss Type | Contrastive | Expected F1 | Training Time | Complexity | Use Case |
|--------|-----------|-------------|-------------|---------------|------------|----------|
| **config_focal_only.yaml** | Focal | ❌ No | 93-95% | Baseline | ⭐ Simple | Baseline/Production |
| **config_multi.yaml** | Focal | ✅ Yes (5%) | 95.99% | +5% | ⭐⭐ Medium | Current Best |
| **config_ghm.yaml** | GHM-C | ❌ No | 96.0-96.5% | +5% | ⭐⭐ Medium | Production |
| **config_ghm_full.yaml** | GHM-C | ✅ Yes (5%) | 96.5-97% | +10% | ⭐⭐⭐ Complex | Research/SOTA |

---

## 🚀 Quick Start Commands

### 1. Focal ONLY (Baseline)

```bash
# Verify
python test_focal_only.py

# Train
TRAIN_FOCAL_ONLY.bat

# Or manually
python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 15
```

**Output:** `multi_label/models/multilabel_focal_only/`

---

### 2. Focal + Contrastive (Current Best)

```bash
# Train
python multi_label\train_multilabel_focal_contrastive.py --epochs 15

# Default uses config_multi.yaml
```

**Output:** `multi_label/models/multilabel_focal_contrastive/`
**Result:** 95.99% F1 (proven)

---

### 3. GHM ONLY (No Contrastive)

```bash
# Verify
python test_ghm_only.py

# Train
TRAIN_GHM_ONLY.bat

# Or manually
python multi_label\train_multilabel_ghm_contrastive.py --config multi_label/config_ghm.yaml --epochs 15
```

**Output:** `multi_label/models/multilabel_ghm_only/`

---

### 4. GHM + Contrastive (Best Expected)

**Note:** Requires updating `config_ghm.yaml`:

```yaml
# Edit config_ghm.yaml:
classification_weight: 0.95   # From 1.0
contrastive_weight: 0.05      # From 0.0
use_contrastive: true         # From false
output_dir: "multi_label/models/multilabel_ghm_contrastive"
```

Then:
```bash
python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
```

---

## 🔍 Detailed Settings

### 1. config_focal_only.yaml (Baseline)

```yaml
multi_label:
  loss_type: "focal"
  focal_weight: 1.0           # 100% Focal
  contrastive_weight: 0.0     # 0% Contrastive
  use_contrastive: false
  
  focal_gamma: 2.0
  focal_alpha: "auto"
```

**Purpose:** Pure baseline, no extras

---

### 2. config_multi.yaml (Current Best)

```yaml
multi_label:
  loss_type: "focal"  # Note: Uses Focal, not GHM
  focal_weight: 0.95           # 95% Focal
  contrastive_weight: 0.05     # 5% Contrastive
  use_contrastive: true
  
  focal_gamma: 2.0
  contrastive_temperature: 0.1
```

**Purpose:** Current production config (95.99% F1)

---

### 3. config_ghm.yaml (GHM Only)

```yaml
multi_label:
  loss_type: "ghm"
  classification_weight: 1.0   # 100% GHM-C
  contrastive_weight: 0.0      # 0% Contrastive
  use_contrastive: false
  
  ghm_bins: 10
  ghm_momentum: 0.75
```

**Purpose:** Test GHM-C without contrastive complexity

---

### 4. GHM + Contrastive (To Create)

To test best possible setup, modify `config_ghm.yaml`:

```yaml
multi_label:
  loss_type: "ghm"
  classification_weight: 0.95  # 95% GHM-C
  contrastive_weight: 0.05     # 5% Contrastive
  use_contrastive: true
  
  ghm_bins: 10
  ghm_momentum: 0.75
  contrastive_temperature: 0.1
```

---

## 📈 Expected Performance Hierarchy

```
Focal only (baseline)         93-95%      ← Simplest
    ↓ +2%
Focal + Contrastive          95.99%       ← Current
    ↓ +0.5%
GHM only                     96.0-96.5%   ← Better loss function
    ↓ +0.5%
GHM + Contrastive            96.5-97%     ← Best (more complex)
```

---

## 🎯 Decision Guide

### Choose **Focal only** if:
- ✅ Need baseline for comparison
- ✅ Want simplest approach
- ✅ Don't need max performance
- ✅ Quick prototyping

### Choose **Focal + Contrastive** if:
- ✅ Need proven results (95.99% F1)
- ✅ Want good balance
- ✅ Currently working well
- ✅ Don't want to experiment

### Choose **GHM only** if:
- ✅ Want better than Focal
- ✅ Don't want contrastive complexity
- ✅ Production system (simpler = better)
- ✅ Faster training than GHM+Contr

### Choose **GHM + Contrastive** if:
- ✅ Need absolute best F1
- ✅ Research/paper work
- ✅ Training time doesn't matter
- ✅ Want to push limits

---

## 📝 Testing Protocol

### To compare all approaches:

```bash
# 1. Baseline (Focal only)
python test_focal_only.py
TRAIN_FOCAL_ONLY.bat

# 2. Current best (Focal + Contrastive) - Already have 95.99%
# No need to retrain

# 3. GHM only
python test_ghm_only.py
TRAIN_GHM_ONLY.bat

# 4. GHM + Contrastive (optional)
# Edit config_ghm.yaml to enable contrastive
python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
```

**Total time:** ~3-4 hours for all experiments

---

## 📊 Results Comparison Template

After training, fill this in:

```
Method                   | F1 Score | Improvement | Training Time
-------------------------|----------|-------------|---------------
Focal only (baseline)    | __.__% | Baseline    | 75 min
Focal + Contrastive      | 95.99% | +X.XX%      | 79 min (proven)
GHM only                 | __.__% | +X.XX%      | 79 min
GHM + Contrastive        | __.__% | +X.XX%      | 83 min

Conclusion:
- Best simple approach: ________
- Best performance: ________
- Best balance: ________
```

---

## 🔧 Troubleshooting

### Config not loading?

```bash
# Check config exists
dir multi_label\config_*.yaml

# Test config
python test_focal_only.py   # For focal only
python test_ghm_only.py     # For GHM only
```

### Training using wrong config?

```bash
# Always specify config explicitly
python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 15

# Check which config is being used (look at console output)
```

### Results in wrong directory?

Check `output_dir` in config:
```yaml
# Each config has different output dir
config_focal_only.yaml  → multilabel_focal_only
config_multi.yaml       → multilabel_focal_contrastive
config_ghm.yaml         → multilabel_ghm_only
```

---

## 📚 Documentation

- **Focal only guide:** `multi_label/FOCAL_ONLY_GUIDE.md`
- **GHM only guide:** `multi_label/GHM_ONLY_GUIDE.md`
- **Full comparison:** `multi_label/LOSS_FUNCTIONS_COMPARISON.md`
- **Quick upgrade:** `multi_label/QUICK_LOSS_UPGRADE_GUIDE.md`

---

## ✅ Summary

### Files Created:

```
Configs:
✅ config_focal_only.yaml  (Focal 100%, Contr 0%)
✅ config_multi.yaml       (Focal 95%, Contr 5%) ← Current
✅ config_ghm.yaml         (GHM 100%, Contr 0%)

Test Scripts:
✅ test_focal_only.py
✅ test_ghm_only.py

Batch Files:
✅ TRAIN_FOCAL_ONLY.bat
✅ TRAIN_GHM_ONLY.bat

Guides:
✅ FOCAL_ONLY_GUIDE.md
✅ GHM_ONLY_GUIDE.md
✅ LOSS_COMPARISON_CONFIGS.md (this file)
```

### Quick Commands:

```bash
# Focal only (baseline)
TRAIN_FOCAL_ONLY.bat

# GHM only
TRAIN_GHM_ONLY.bat

# Current best (Focal+Contrastive)
python multi_label\train_multilabel_focal_contrastive.py --epochs 15
```

---

**Choose your approach based on your needs: Simple, Proven, or Best!** 🚀
