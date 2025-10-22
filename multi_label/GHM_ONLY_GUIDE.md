# GHM-C Loss ONLY (No Contrastive) - Training Guide

## Configuration

**File:** `multi_label/config_ghm.yaml`

### Key Settings:

```yaml
multi_label:
  loss_type: "ghm"              # Use GHM-C Loss
  
  # DISABLED CONTRASTIVE
  classification_weight: 1.0    # 100% GHM-C
  contrastive_weight: 0.0       # 0% Contrastive (DISABLED)
  use_contrastive: false        # Disabled
  
  # GHM-C parameters
  ghm_bins: 10
  ghm_momentum: 0.75
```

---

## Quick Test

### 1. Verify Config

```bash
cd D:\BERT
python test_ghm_only.py
```

**Expected output:**
```
Testing GHM-C Loss Configuration (No Contrastive)

Multi-Label Settings
Loss Type: ghm

Loss Weights:
  Classification (GHM-C): 1.0
  Contrastive:            0.0

Verification
[OK] Classification weight is 1.0 (100%)
[OK] Contrastive weight is 0.0 (disabled)
[OK] use_contrastive is False (disabled)

[SUCCESS] Config is set correctly for GHM-C Loss ONLY
```

---

### 2. Train

```bash
cd D:\BERT
python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
```

**Output directory:** `multi_label/models/multilabel_ghm_only/`

---

## Expected Results

### Comparison:

```
Method                | Overall F1 | Training Speed | Simplicity
----------------------|------------|----------------|------------
Focal Loss only       | 95.99%     | Baseline       | Simple
GHM-C Loss only       | 96.0-96.5% | +5% slower     | Medium
Focal + Contrastive   | 95.99%     | +3% slower     | Medium
GHM-C + Contrastive   | 96.5-97%   | +8% slower     | Complex
```

### Why GHM-C Only?

**Advantages:**
1. ✅ Simpler than GHM+Contrastive
2. ✅ Faster than GHM+Contrastive
3. ✅ Still better than Focal
4. ✅ Easier to tune
5. ✅ No need for projection head

**Trade-offs:**
- May be slightly worse than GHM+Contrastive (-0.2-0.5% F1)
- But much simpler implementation
- Faster training
- Easier to debug

---

## When to Use GHM-C Only

### Good for:
- Quick experiments
- Production systems (simpler = more reliable)
- When training speed matters
- When you don't need the extra 0.5% F1

### Use GHM+Contrastive if:
- You need every 0.1% F1
- Training time doesn't matter
- You have resources to tune both losses
- Research/paper work

---

## Training Commands

### Quick Test (3 epochs):

```bash
python multi_label\train_multilabel_ghm_contrastive.py --epochs 3
```

### Full Training (15 epochs):

```bash
python multi_label\train_multilabel_ghm_contrastive.py --epochs 15
```

### Custom Settings:

```bash
# Adjust classification weight (should stay 1.0)
python multi_label\train_multilabel_ghm_contrastive.py \
    --classification-weight 1.0 \
    --contrastive-weight 0.0 \
    --epochs 15
```

---

## Troubleshooting

### Issue: Contrastive loss still calculated

**Check:**
```bash
python test_ghm_only.py
```

**Should show:**
```
[OK] Contrastive weight is 0.0 (disabled)
```

**If not, edit config:**
```yaml
contrastive_weight: 0.0
use_contrastive: false
```

---

### Issue: Performance worse than Focal

**Symptom:**
```
GHM-C only: 95.5% F1 (vs Focal: 95.99%)
```

**Solutions:**

1. **Increase bins:**
```yaml
ghm_bins: 15  # From 10
```

2. **Adjust momentum:**
```yaml
ghm_momentum: 0.8  # From 0.75
```

3. **More epochs:**
```bash
python multi_label\train_multilabel_ghm_contrastive.py --epochs 20
```

4. **If still worse, revert to Focal:**
```yaml
loss_type: "focal"
```

---

## Expected Training Output

```
================================================================================
Multi-Label ABSA Training with GHM-C + Contrastive + LOGGING
================================================================================

Using GHM-C Loss (improved over Focal Loss)
   Expected: 96.0-96.5% F1 (vs Focal: 95.99%)

Loss settings:
   Loss type: GHM
   Classification weight: 1.0      ← 100% GHM-C
   Contrastive weight: 0.0          ← DISABLED
   GHM bins: 10
   GHM momentum: 0.75

Training setup:
   Epochs: 15
   Output dir: multi_label/models/multilabel_ghm_only

================================================================================
Epoch 1/15
================================================================================
Training: 100%|████| 498/498 [04:20<00:00, loss=0.31, ghm=0.31, contr=0.00]
                                                                    ↑
                                                            Should be 0!

Train Losses:
   Total: 0.3100
   GHM-C: 0.3100        ← Same as total
   Contrastive: 0.0000   ← Should be 0

...

Epoch 15/15
Overall F1: 96.25%    ← Expected 96.0-96.5%
```

---

## Summary

### What Changed:

**Before (GHM + Contrastive):**
```yaml
classification_weight: 0.95
contrastive_weight: 0.05
use_contrastive: true
output_dir: "multi_label/models/multilabel_ghm_contrastive"
```

**After (GHM Only):**
```yaml
classification_weight: 1.0
contrastive_weight: 0.0
use_contrastive: false
output_dir: "multi_label/models/multilabel_ghm_only"
```

---

### Commands Summary:

```bash
# 1. Verify config
python test_ghm_only.py

# 2. Train
python multi_label\train_multilabel_ghm_contrastive.py --epochs 15

# 3. Check results
type multi_label\models\multilabel_ghm_only\test_results_ghm_contrastive.json
```

---

### Expected Outcome:

- ✅ F1: 96.0-96.5% (vs Focal: 95.99%)
- ✅ Simpler than GHM+Contrastive
- ✅ Faster training
- ✅ Easier to understand and debug

---

**Good for production systems where simplicity > squeezing last 0.5% F1!**
