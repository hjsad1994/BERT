# ⚡ QUICK GUIDE: Upgrade From Focal Loss

## 🎯 TL;DR

**Current:** Focal Loss (95.99% F1) ✅  
**Upgrade to:** GHM-C Loss (Expected: 96.5-97% F1) 🚀  
**Effort:** 10 minutes code change  
**Risk:** Low (easy rollback)

---

## 📁 Files Created

```
multi_label/losses/
├── __init__.py                  # Package init
├── ghm_loss.py                  # GHM-C Loss implementation ⭐
├── unified_focal_loss.py        # Unified Focal Loss implementation
├── LOSS_FUNCTIONS_COMPARISON.md # Detailed comparison
└── QUICK_LOSS_UPGRADE_GUIDE.md  # This file
```

---

## 🚀 Option 1: Quick Test (5 minutes)

Test if GHM-C Loss works better:

```python
# Create test script: test_ghm_vs_focal.py

import torch
from multi_label.losses import MultiLabelGHM_Loss
from utils import FocalLoss

# Your current data
batch_size = 32
logits = torch.randn(batch_size, 11, 3)
targets = torch.randint(0, 3, (batch_size, 11))

# Test 1: Focal Loss
focal_loss_fn = FocalLoss(alpha=None, gamma=2.0, reduction='mean')
focal_total = 0
for i in range(11):
    focal_total += focal_loss_fn(logits[:, i, :], targets[:, i])
focal_loss = focal_total / 11

# Test 2: GHM-C Loss
ghm_loss_fn = MultiLabelGHM_Loss(num_aspects=11, num_sentiments=3, bins=10, momentum=0.75)
ghm_loss = ghm_loss_fn(logits, targets)

print(f"Focal Loss: {focal_loss.item():.4f}")
print(f"GHM-C Loss: {ghm_loss.item():.4f}")
print(f"Difference: {abs(focal_loss.item() - ghm_loss.item()):.4f}")
```

Run:
```bash
cd D:\BERT
python test_ghm_vs_focal.py
```

---

## 🔧 Option 2: Full Training (30 minutes)

### Step 1: Update config

```yaml
# File: multi_label/config_multi.yaml

multi_label:
  # Loss function type
  loss_type: "ghm"  # "focal" or "ghm" or "unified"
  
  # GHM-C Loss settings (if loss_type = "ghm")
  ghm_bins: 10
  ghm_momentum: 0.75
  
  # Focal Loss settings (if loss_type = "focal") - keep for fallback
  focal_weight: 0.95
  contrastive_weight: 0.05
  focal_gamma: 2.0
  
  # Contrastive settings (shared)
  contrastive_temperature: 0.1
  contrastive_base_weight: 0.1
```

### Step 2: Update training script

```python
# File: train_multilabel_focal_contrastive.py
# Add at top:
from multi_label.losses import MultiLabelGHM_Loss

# Around line 325, replace:
# focal_loss_fn = FocalLoss(alpha=None, gamma=focal_gamma, reduction='mean')

# With:
loss_type = multi_label_config.get('loss_type', 'focal')

if loss_type == 'ghm':
    # GHM-C Loss
    ghm_bins = multi_label_config.get('ghm_bins', 10)
    ghm_momentum = multi_label_config.get('ghm_momentum', 0.75)
    classification_loss_fn = MultiLabelGHM_Loss(
        num_aspects=11,
        num_sentiments=3,
        bins=ghm_bins,
        momentum=ghm_momentum,
        loss_weight=1.0
    )
    print(f"   Using GHM-C Loss (bins={ghm_bins}, momentum={ghm_momentum})")
elif loss_type == 'focal':
    # Focal Loss (original)
    classification_loss_fn = FocalLoss(alpha=None, gamma=focal_gamma, reduction='mean')
    print(f"   Using Focal Loss (gamma={focal_gamma})")
else:
    raise ValueError(f"Unknown loss_type: {loss_type}")
```

### Step 3: Update loss calculation

```python
# In combined_focal_contrastive_loss function (around line 40)
# Rename to: combined_classification_contrastive_loss

def combined_classification_contrastive_loss(logits, embeddings, labels, 
                                  classification_loss_fn,  # Changed name
                                  contrastive_loss_fn, 
                                  weights=None, 
                                  classification_weight=0.95,  # Changed name
                                  contrastive_weight=0.05):
    """
    Combined loss: Classification Loss (Focal or GHM-C) + Contrastive Loss
    """
    batch_size, num_aspects, num_sentiments = logits.shape
    
    # Check if it's GHM-C Loss (multi-label version)
    if isinstance(classification_loss_fn, MultiLabelGHM_Loss):
        # GHM-C Loss handles all aspects at once
        classification_loss = classification_loss_fn(logits, labels)
    else:
        # Focal Loss: per-aspect (original logic)
        classification_loss = 0
        for i in range(num_aspects):
            aspect_logits = logits[:, i, :]
            aspect_labels = labels[:, i]
            
            if weights is not None:
                aspect_weights = weights[i]
                loss_fn_i = FocalLoss(alpha=aspect_weights, gamma=2.0, reduction='mean')
            else:
                loss_fn_i = classification_loss_fn
            
            loss = loss_fn_i(aspect_logits, aspect_labels)
            classification_loss += loss
        
        classification_loss = classification_loss / num_aspects
    
    # Contrastive loss (unchanged)
    contr_loss = contrastive_loss_fn(embeddings, labels)
    
    # Combined loss
    total_loss = classification_weight * classification_loss + contrastive_weight * contr_loss
    
    return total_loss, classification_loss, contr_loss
```

### Step 4: Train

```bash
# Test with GHM-C
python multi_label\train_multilabel_focal_contrastive.py --epochs 15

# Compare with Focal (if needed)
# Edit config: loss_type: "focal"
python multi_label\train_multilabel_focal_contrastive.py --epochs 15
```

---

## 📊 Expected Results

### Epoch-by-Epoch Comparison

```
Epoch | Focal Loss F1 | GHM-C Loss F1 | Difference
------|---------------|---------------|------------
  1   | 61.32%        | 62-63%        | +0.7-1.7%
  3   | 89.28%        | 89.5-90%      | +0.2-0.7%
  7   | 95.39%        | 95.6-96%      | +0.2-0.6%
 13   | 95.99%        | 96.3-97%      | +0.3-1.0%
```

### Per-Aspect Improvements

**Biggest gains expected for hard aspects:**
- Design: 93.21% → 94-95% (+0.8-1.8%)
- Price: 95.84% → 96.5-97% (+0.7-1.2%)
- Performance: 95.07% → 95.5-96% (+0.4-0.9%)

---

## ⚠️ Troubleshooting

### Issue 1: Import Error

```python
ModuleNotFoundError: No module named 'multi_label.losses'
```

**Fix:**
```bash
# Make sure __init__.py exists
cd D:\BERT\multi_label\losses
ls __init__.py  # Should exist

# Or use relative import in training script:
import sys
sys.path.insert(0, 'multi_label')
from losses import MultiLabelGHM_Loss
```

### Issue 2: Loss Not Decreasing

```
Epoch 1: GHM Loss = 0.8
Epoch 2: GHM Loss = 0.9  # Going up!
```

**Fix:**
```python
# Increase momentum for more stability
ghm_momentum = 0.85  # Instead of 0.75

# Or decrease bins
ghm_bins = 8  # Instead of 10
```

### Issue 3: Worse Performance Than Focal

```
GHM-C F1: 95.5% (worse than Focal 95.99%)
```

**Fix:**
```yaml
# Adjust classification vs contrastive weight
classification_weight: 0.97  # Instead of 0.95
contrastive_weight: 0.03     # Instead of 0.05

# Or switch back to Focal
loss_type: "focal"
```

---

## 🎓 Understanding the Difference

### Focal Loss Behavior

```
Sample Difficulty:    Easy    Medium    Hard
Focal Loss Weight:    Low     Medium    High
                      ↓        ↓         ↓
Focus:              Ignore  Balance  Emphasize

Problem: Fixed strategy, doesn't adapt during training
```

### GHM-C Loss Behavior

```
Training Phase:       Early   Middle    Late
Easy samples:         Many    Medium    Few
Hard samples:         Few     Medium    Many
Outliers:            Some     Some     Some

GHM-C adapts weights dynamically based on gradient density!

Epoch 1: Focus on hard examples (many easy examples dilute signal)
Epoch 5: Balance all (distribution evening out)
Epoch 13: Subtle tuning (all examples contributing)

Result: Better learning curve, fewer wasted updates
```

---

## 💡 Pro Tips

1. **Start with defaults:**
   - bins = 10
   - momentum = 0.75
   - Don't tune unless performance is worse

2. **Monitor gradient distributions:**
   ```python
   # Add logging in training loop
   print(f"Gradient stats: min={grad.min():.4f}, max={grad.max():.4f}, mean={grad.mean():.4f}")
   ```

3. **Compare after 3 epochs:**
   - If GHM-C Loss F1 > Focal by epoch 3 → Continue
   - If GHM-C Loss F1 < Focal by epoch 3 → Switch back

4. **Use tensorboard:**
   ```python
   from torch.utils.tensorboard import SummaryWriter
   writer = SummaryWriter('runs/ghm_experiment')
   
   # Log both losses
   writer.add_scalar('Loss/classification', classification_loss, epoch)
   writer.add_scalar('Loss/contrastive', contr_loss, epoch)
   ```

---

## ✅ Checklist

Before training with GHM-C:

- [ ] Files created in `multi_label/losses/`
- [ ] Config updated with `loss_type: "ghm"`
- [ ] Training script modified
- [ ] Tested import: `from multi_label.losses import MultiLabelGHM_Loss`
- [ ] Baseline results saved (Focal: 95.99% F1)
- [ ] Disk space for new checkpoints
- [ ] GPU available

After first epoch:

- [ ] Loss decreasing (should be ~0.3-0.4)
- [ ] F1 score reasonable (>60%)
- [ ] No errors in logs
- [ ] Gradient stats look normal

After full training:

- [ ] Compare F1 scores (Focal vs GHM-C)
- [ ] Check per-aspect improvements
- [ ] Analyze hard aspects (Design, Price)
- [ ] Decide: Keep GHM-C or revert to Focal

---

## 🚀 Quick Commands

```bash
# Test GHM-C implementation
cd D:\BERT\multi_label\losses
python ghm_loss.py

# Create test comparison script
cd D:\BERT
cat > test_ghm_vs_focal.py << 'EOF'
import torch
from multi_label.losses import MultiLabelGHM_Loss
from utils import FocalLoss

logits = torch.randn(32, 11, 3)
targets = torch.randint(0, 3, (32, 11))

ghm = MultiLabelGHM_Loss(11, 3, bins=10, momentum=0.75)
print(f"GHM-C Loss: {ghm(logits, targets).item():.4f}")
EOF

python test_ghm_vs_focal.py

# Train with GHM-C
python multi_label\train_multilabel_focal_contrastive.py --epochs 15

# Monitor training
tensorboard --logdir=runs
```

---

## 📚 Further Reading

- Full comparison: `multi_label/LOSS_FUNCTIONS_COMPARISON.md`
- GHM-C paper: https://arxiv.org/abs/1811.05181
- Implementation: `multi_label/losses/ghm_loss.py`

---

**Good luck! Reach for 96-97% F1! 🚀**
