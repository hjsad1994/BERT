# ✅ Multi-Label Training Script - Focal Loss Ready

## 🎯 Confirmation: train_multilabel.py ĐANG SỬ DỤNG FOCAL LOSS

---

## ✅ Code Verification

### 1. **Import Focal Loss** (Line 23)
```python
from focal_loss_multilabel import MultilabelFocalLoss, calculate_global_alpha
```
✅ **CONFIRMED**: Focal Loss được import

---

### 2. **Setup Focal Loss** (Lines 242-294)
```python
# =====================================================================
# SETUP FOCAL LOSS
# =====================================================================
print("🔥 Setting up Focal Loss...")

focal_config = config.get('multi_label', {})
use_focal_loss = focal_config.get('use_focal_loss', True)
focal_gamma = focal_config.get('focal_gamma', 2.0)
focal_alpha_config = focal_config.get('focal_alpha', 'auto')

if focal_alpha_config == 'auto':
    alpha = calculate_global_alpha(train_dataset, sentiment_to_idx)
elif isinstance(focal_alpha_config, list):
    alpha = focal_alpha_config
elif focal_alpha_config is None:
    alpha = [1.0, 1.0, 1.0]

focal_loss_fn = MultilabelFocalLoss(
    alpha=alpha,
    gamma=focal_gamma,
    num_aspects=11
)
```
✅ **CONFIRMED**: Focal Loss được khởi tạo với 3 modes (auto/custom/equal)

---

### 3. **Train Epoch với Focal Loss** (Lines 31-60)
```python
def train_epoch(model, dataloader, optimizer, scheduler, device, focal_loss_fn):
    """Train for one epoch"""
    model.train()
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask)
        
        # Loss (Focal Loss)  ← Line 48
        loss = focal_loss_fn(logits, labels)  ✅ USING FOCAL LOSS!
        
        # Backward
        loss.backward()
```
✅ **CONFIRMED**: Training loop SỬ DỤNG `focal_loss_fn()`

---

### 4. **Config Support** (config_multi.yaml)
```yaml
multi_label:
  use_focal_loss: true      # ✅ ENABLED
  focal_gamma: 2.0          # ✅ CONFIGURED
  focal_alpha: "auto"       # ✅ AUTO MODE
```
✅ **CONFIRMED**: Config hỗ trợ focal loss

---

## 📊 How It Works

### Flow:
```
1. Read config
   ↓
2. Load training data
   ↓
3. Calculate alpha (auto/custom/equal)
   ↓
4. Create MultilabelFocalLoss(alpha, gamma)
   ↓
5. Training loop:
     logits = model(input)           [batch, 11, 3]
     loss = focal_loss_fn(logits, labels)  ← FOCAL LOSS HERE!
     loss.backward()
   ↓
6. Evaluate & save
```

---

## 🔥 Focal Loss Details

### Global Alpha (Same for all 11 aspects):
```python
# Input
logits: [batch, 11 aspects, 3 sentiments]
labels: [batch, 11 aspects]

# Processing
for aspect in range(11):
    aspect_logits = logits[:, aspect, :]  # [batch, 3]
    aspect_labels = labels[:, aspect]      # [batch]
    
    # Focal Loss với SAME alpha
    probs = softmax(aspect_logits)
    class_probs = probs[labels]
    focal_weight = (1 - class_probs) ** gamma
    alpha_weight = alpha[labels]
    ce_loss = cross_entropy(aspect_logits, labels)
    
    loss += alpha_weight * focal_weight * ce_loss

return loss / 11  # Average over aspects
```

---

## ✅ Verification Checklist

- [x] **Import**: `from focal_loss_multilabel import MultilabelFocalLoss`
- [x] **Config**: `use_focal_loss: true`
- [x] **Initialize**: `focal_loss_fn = MultilabelFocalLoss(...)`
- [x] **Use**: `loss = focal_loss_fn(logits, labels)`
- [x] **Pass to train**: `train_epoch(..., focal_loss_fn)`
- [x] **Support 3 modes**: auto / custom / equal

**STATUS**: ✅ **ALL CHECKS PASSED**

---

## 🎯 Expected Output During Training

```
================================================================================
🔥 Setting up Focal Loss...
================================================================================

📊 Calculating global alpha weights...

   Total aspect-sentiment pairs: 35,000

   Sentiment distribution:
     positive  : 20,000 (57.14%)
     negative  : 10,000 (28.57%)
     neutral   :  5,000 (14.29%)

   Calculated alpha (inverse frequency):
     positive  : 0.5833
     negative  : 1.1667
     neutral   : 2.3333

✓ MultilabelFocalLoss initialized:
   Alpha: [0.5833, 1.1667, 2.3333]
   Gamma: 2.0
   Num aspects: 11

✓ Focal Loss ready:
   Gamma: 2.0
   Alpha: [0.5833, 1.1667, 2.3333]
   Mode: Global (same alpha for all 11 aspects)

================================================================================
Starting Training
================================================================================

Epoch 1/5
================================================================================
Training: 100%|████████| 250/250 [02:30<00:00, loss: 0.4523]

Train Loss: 0.4523
```

---

## 📚 Files Involved

1. ✅ **`focal_loss_multilabel.py`** - Focal Loss implementation
2. ✅ **`train_multilabel.py`** - Training script (USING FOCAL LOSS)
3. ✅ **`config_multi.yaml`** - Config with focal_alpha: "auto"

---

## 🎓 3 Alpha Modes

### Mode 1: AUTO (Recommended)
```yaml
focal_alpha: "auto"
```
→ Tự động tính từ data (inverse frequency)

### Mode 2: CUSTOM
```yaml
focal_alpha: [0.8, 1.0, 2.5]  # [positive, negative, neutral]
```
→ User-defined weights

### Mode 3: EQUAL
```yaml
focal_alpha: null
```
→ Equal weights [1.0, 1.0, 1.0]

---

## 🚀 Ready to Train

```bash
cd D:\BERT
python multi_label\train_multilabel.py --config multi_label\config_multi.yaml --epochs 5 --output-dir multi_label\models\focal_loss_model
```

**Expected**:
- ✅ Focal Loss initialized
- ✅ Alpha calculated from data
- ✅ Training với focal loss
- ✅ Better handling of imbalanced classes

---

## ✅ Summary

| Component | Status | Details |
|-----------|--------|---------|
| **Focal Loss Class** | ✅ | `focal_loss_multilabel.py` |
| **Import** | ✅ | Line 23 |
| **Setup** | ✅ | Lines 242-294 |
| **Usage** | ✅ | Line 48 in train_epoch |
| **Config** | ✅ | `use_focal_loss: true` |
| **Alpha Modes** | ✅ | auto/custom/equal |

**STATUS**: ✅ **READY TO USE**

---

**Conclusion**: `train_multilabel.py` ĐANG SỬ DỤNG FOCAL LOSS với global alpha cho tất cả 11 aspects!
