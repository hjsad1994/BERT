# Focal Loss ONLY (No Contrastive) - Training Guide

## Overview

**Pure Focal Loss baseline** - No contrastive learning, no extra complexity.

This is the **simplest approach** to establish a baseline for comparison.

---

## Configuration

**File:** `multi_label/config_focal_only.yaml`

### Key Settings:

```yaml
multi_label:
  loss_type: "focal"            # Standard Focal Loss
  
  # NO CONTRASTIVE
  focal_weight: 1.0             # 100% Focal
  contrastive_weight: 0.0       # 0% Contrastive (DISABLED)
  use_contrastive: false        # Disabled
  
  # Focal Loss parameters
  focal_gamma: 2.0              # Focusing parameter
  focal_alpha: "auto"           # Auto class weights
```

---

## Quick Start

### 1. Verify Config

```bash
cd D:\BERT
python test_focal_only.py
```

**Expected output:**
```
Testing Focal Loss Configuration (No Contrastive)

[OK] Focal weight is 1.0 (100%)
[OK] Contrastive weight is 0.0 (disabled)
[OK] use_contrastive is False (disabled)
[OK] Focal gamma: 2.0

[SUCCESS] Config is set correctly for Focal Loss ONLY
```

---

### 2. Train (One-Click)

```bash
cd D:\BERT
TRAIN_FOCAL_ONLY.bat
```

---

### 3. Train (Manual)

```bash
cd D:\BERT
python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 15
```

---

## Expected Results

### Comparison Table:

```
Method                   | F1 Score    | Training Time | Complexity
-------------------------|-------------|---------------|------------
Focal only (baseline)    | ~93-95%     | Baseline      | Simplest
Focal + Contrastive      | 95.99%      | +5% slower    | Medium
GHM only                 | 96.0-96.5%  | +5% slower    | Medium
GHM + Contrastive        | 96.5-97%    | +10% slower   | Complex
```

### Why Run This?

**Purpose:**
1. ✅ Establish pure baseline
2. ✅ Measure impact of contrastive learning
3. ✅ Compare with advanced methods
4. ✅ Verify simple approach works

**This tells you:**
- How much does contrastive help? (Focal vs Focal+Contr)
- How much does GHM help? (Focal vs GHM)
- Is complexity worth it? (Simple vs Complex)

---

## Output Location

**Directory:** `multi_label/models/multilabel_focal_only/`

**Files:**
```
multilabel_focal_only/
├── best_model.pt
├── checkpoint_epoch_*.pt
├── test_results_focal_contrastive.json  ← Check this!
└── training_logs/
    ├── epoch_losses_*.csv
    └── batch_losses_*.csv
```

---

## Analysis After Training

### Compare Results:

```bash
# Focal only (baseline)
type multi_label\models\multilabel_focal_only\test_results_focal_contrastive.json

# Focal + Contrastive (previous best)
type multi_label\models\multilabel_focal_contrastive\test_results_focal_contrastive.json

# Calculate improvement:
# Improvement = (Focal+Contr) - (Focal only)
# This tells you how much contrastive learning helps!
```

### Example Analysis:

```
Focal only:         93.5% F1
Focal+Contrastive:  95.99% F1
Improvement:        +2.49% F1  ← Contrastive contribution!

Conclusion: Contrastive learning adds significant value
```

---

## Tuning Focal Loss

If baseline is poor (<90% F1), try adjusting:

### 1. Focal Gamma (Focusing Parameter)

```yaml
# Default
focal_gamma: 2.0

# More focus on hard examples (for extreme imbalance)
focal_gamma: 3.0

# Less focus (for balanced data)
focal_gamma: 1.5
```

### 2. Learning Rate

```yaml
# Default
learning_rate: 2.0e-5

# Higher (faster convergence, may be unstable)
learning_rate: 3.0e-5

# Lower (more stable, slower)
learning_rate: 1.0e-5
```

### 3. Training Epochs

```bash
# More epochs for better convergence
python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 20
```

---

## Troubleshooting

### Issue: Very Low F1 (<90%)

**Possible causes:**
1. Data issue
2. Model not converging
3. Hyperparameters wrong

**Solutions:**

1. **Check data:**
```bash
# Verify data files exist
dir multi_label\data\*.csv

# Check sample counts
python -c "import pandas as pd; print(pd.read_csv('multi_label/data/train_multilabel_balanced.csv').shape)"
```

2. **Increase training:**
```bash
python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 20
```

3. **Adjust gamma:**
```yaml
focal_gamma: 3.0  # More focus on hard examples
```

---

### Issue: Not Converging

**Symptom:**
```
Epoch 1: Loss = 0.8
Epoch 5: Loss = 0.75
Epoch 10: Loss = 0.73  ← Still high
```

**Solutions:**

1. **Lower learning rate:**
```yaml
learning_rate: 1.0e-5  # From 2.0e-5
```

2. **More warmup:**
```yaml
warmup_ratio: 0.1  # From 0.06
```

3. **Check class weights:**
```python
# Add to training script to debug
print("Class weights:", train_dataset.get_label_weights())
```

---

## Training Output Example

```
================================================================================
Multi-Label ABSA Training with Focal + Contrastive + LOGGING
================================================================================

Using config: multi_label/config_focal_only.yaml

Loss settings:
   Loss type: FOCAL
   Focal weight: 1.0        ← 100% Focal
   Contrastive weight: 0.0   ← DISABLED
   Focal gamma: 2.0
   Focal alpha: auto

Training setup:
   Epochs: 15
   Output dir: multi_label/models/multilabel_focal_only

================================================================================
Epoch 1/15
================================================================================
Training: 100%|████| 498/498 [04:15<00:00, loss=0.32, focal=0.32, contr=0.00]
                                                                        ↑
                                                                Should be 0!

Train Losses:
   Total: 0.3200
   Focal: 0.3200        ← Same as total
   Contrastive: 0.0000   ← Disabled

Validating...
Overall Metrics:
   F1 Score: 61.50%

...

================================================================================
Epoch 15/15
================================================================================
Overall Metrics:
   F1 Score: 93.5%      ← Baseline result

Testing Best Model
   Test F1: 93.5%

This is your BASELINE!
Compare with Focal+Contrastive (95.99%) to see improvement from contrastive learning.
```

---

## Why Focal Only Might Be Lower

**Expected:**
```
Focal only:         ~93-95% F1
Focal+Contrastive:  ~96% F1
Difference:         ~1-3% F1
```

**Reason:**
- Focal Loss only optimizes classification
- Contrastive Loss adds representation learning
- Better representations → Better generalization → Higher F1

**This is GOOD!** It proves contrastive learning works.

---

## Summary

### What This Config Does:

```yaml
# BEFORE (Focal + Contrastive):
focal_weight: 0.95
contrastive_weight: 0.05
use_contrastive: true

# AFTER (Focal only):
focal_weight: 1.0
contrastive_weight: 0.0
use_contrastive: false
```

### Commands:

```bash
# 1. Verify
python test_focal_only.py

# 2. Train (one-click)
TRAIN_FOCAL_ONLY.bat

# 3. Or train manually
python multi_label\train_multilabel_focal_contrastive.py --config multi_label/config_focal_only.yaml --epochs 15
```

### Expected Timeline:

- Verify: 2 seconds
- Train: 75 minutes (15 epochs)
- Compare: 5 minutes
- **Total: ~80 minutes**

---

## Comparison Checklist

After training Focal only, you can compare:

- [ ] Focal only vs Focal+Contrastive
  - Shows contrastive learning impact
  
- [ ] Focal only vs GHM only
  - Shows GHM-C improvement over Focal
  
- [ ] Simple (Focal) vs Complex (GHM+Contrastive)
  - Shows if complexity is worth it

---

## Next Steps

### After Focal Only Training:

1. **Check baseline F1:**
```bash
type multi_label\models\multilabel_focal_only\test_results_focal_contrastive.json
```

2. **Compare with others:**
```bash
# Focal only
type multi_label\models\multilabel_focal_only\test_results_focal_contrastive.json

# Focal+Contrastive
type multi_label\models\multilabel_focal_contrastive\test_results_focal_contrastive.json

# GHM only
type multi_label\models\multilabel_ghm_only\test_results_ghm_contrastive.json
```

3. **Calculate improvements:**
```
Focal only:         X% F1
Focal+Contrastive:  Y% F1 → Improvement: (Y-X)%
GHM only:           Z% F1 → Improvement: (Z-X)%
```

4. **Decide best approach:**
- If baseline is good (>94%), may not need complexity
- If gap is large (>2%), complexity is worth it
- Balance simplicity vs performance

---

## Final Notes

### This is the BASELINE

All other methods should beat this baseline:
- ✅ Focal+Contrastive: Should be ~2% better
- ✅ GHM only: Should be ~2-3% better
- ✅ GHM+Contrastive: Should be ~3-4% better

If they don't beat baseline, something is wrong!

### When to Use Focal Only

**Good for:**
- Quick prototyping
- Establishing baseline
- Simple production systems
- When you don't need max F1
- Limited compute resources

**Not good for:**
- SOTA results
- Research papers
- When every 0.1% F1 matters
- Competitions

---

**This is your starting point for all comparisons!** 🎯
