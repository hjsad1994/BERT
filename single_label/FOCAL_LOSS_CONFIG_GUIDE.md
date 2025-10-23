# 🔥 Focal Loss Configuration Guide

## 📋 Overview

Focal Loss được thiết kế để xử lý **class imbalance** bằng cách:
1. **Gamma (γ)**: Focus vào hard examples (samples khó phân loại)
2. **Alpha (α)**: Weight classes khác nhau (minority classes được tăng trọng số)

---

## ⚙️ Configuration Options

### In `config_single.yaml`:

```yaml
single_label:
  use_focal_loss: true      # Enable/disable Focal Loss
  focal_gamma: 2.0          # Focusing parameter
  focal_alpha: "auto"       # Class weights (3 options)
```

---

## 🎯 Option 1: AUTO (Recommended)

**Config**:
```yaml
focal_alpha: "auto"
```

**Behavior**:
- Tự động tính alpha từ **inverse frequency** của classes
- Formula: `alpha[i] = total_samples / (num_classes * class_i_count)`

**Example**:
```
Class distribution:
  positive: 7,000 samples (56%)
  negative: 4,000 samples (32%)
  neutral:  1,500 samples (12%)

Calculated alpha:
  positive: 0.60  (majority class → low weight)
  negative: 1.00
  neutral:  2.67  (minority class → high weight)
```

**When to use**:
- ✅ Default choice
- ✅ Khi có class imbalance
- ✅ Muốn tự động adapt theo data

---

## 🎨 Option 2: USER-DEFINED

**Config**:
```yaml
focal_alpha: [1.0, 1.5, 2.0]  # [positive, negative, neutral]
```

**Behavior**:
- Dùng alpha weights được chỉ định
- Thứ tự: `[weight_positive, weight_negative, weight_neutral]`

**Example**:
```yaml
# Tăng trọng số cho neutral (minority class)
focal_alpha: [1.0, 1.2, 3.0]

# Equal weights cho negative và neutral, giảm positive
focal_alpha: [0.5, 1.0, 1.0]
```

**When to use**:
- ✅ Khi muốn control chính xác weights
- ✅ Khi biết class nào quan trọng hơn
- ✅ Fine-tuning performance

---

## ⚖️ Option 3: EQUAL (No Class Weighting)

**Config**:
```yaml
focal_alpha: null
```

**Behavior**:
- Dùng equal weights `[1.0, 1.0, 1.0]`
- Chỉ dùng gamma (focusing parameter)
- KHÔNG weight theo class frequency

**When to use**:
- ✅ Khi classes đã balanced (sau oversampling)
- ✅ Khi chỉ muốn focus vào hard examples (gamma only)
- ✅ Baseline comparison

---

## 📊 Comparison

| **Option** | **Alpha** | **Use Case** | **Pros** | **Cons** |
|------------|-----------|--------------|----------|----------|
| **AUTO** | Inverse frequency | Class imbalance | Automatic, adaptive | May over-weight rare classes |
| **USER-DEFINED** | Custom `[w1,w2,w3]` | Fine-tuning | Full control | Requires domain knowledge |
| **EQUAL** | `null` → `[1,1,1]` | Balanced data | Simple, no bias | Ignores class imbalance |

---

## 🔢 Gamma Parameter

```yaml
focal_gamma: 2.0  # Recommended default
```

**Effect**:
- `gamma = 0`: Standard CrossEntropyLoss (no focusing)
- `gamma = 1`: Mild focusing
- `gamma = 2`: **Recommended** (Lin et al. paper)
- `gamma = 5`: Strong focusing on very hard examples

**Rule of thumb**:
- Start with `gamma = 2.0`
- Increase if model overfits easy examples
- Decrease if model struggles on all samples

---

## 💡 Examples

### Example 1: Default (AUTO with Gamma=2.0)

```yaml
single_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: "auto"
```

**Output during training**:
```
🎯 Alpha weights mode: AUTO (inverse frequency)

   Calculated alpha weights:
   positive   (class 0): 0.5982
   negative   (class 1): 1.0473
   neutral    (class 2): 2.7891

✓ Focal Loss created:
   Gamma: 2.0 (focusing parameter)
   Alpha: [0.5982, 1.0473, 2.7891]
✓ Focal Loss sẽ focus vào hard examples và handle class imbalance
```

---

### Example 2: Custom Weights

```yaml
single_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: [0.8, 1.0, 2.5]  # Boost neutral more
```

**Output**:
```
🎯 Alpha weights mode: USER-DEFINED

   Using custom alpha weights:
   positive   (class 0): 0.8000
   negative   (class 1): 1.0000
   neutral    (class 2): 2.5000
```

---

### Example 3: Equal Weights (No Class Weighting)

```yaml
single_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: null
```

**Output**:
```
🎯 Alpha weights mode: EQUAL (no class weighting)

   Using equal weights: [1.0, 1.0, 1.0]
```

---

### Example 4: Disable Focal Loss

```yaml
single_label:
  use_focal_loss: false
  # focal_gamma and focal_alpha will be ignored
```

**Uses standard CrossEntropyLoss instead**

---

## 🧪 Experimentation Guide

### Scenario 1: Severe Class Imbalance

**Problem**: Positive=80%, Negative=15%, Neutral=5%

**Solution**:
```yaml
focal_alpha: "auto"      # Auto-weight minority classes
focal_gamma: 2.0         # Focus on hard examples
```

---

### Scenario 2: After Oversampling

**Problem**: Classes now balanced, but still some hard examples

**Solution**:
```yaml
focal_alpha: null        # Equal weights (already balanced)
focal_gamma: 2.0         # Keep focusing on hard examples
```

---

### Scenario 3: Neutral Class Most Important

**Problem**: Neutral is rare but critical for business

**Solution**:
```yaml
focal_alpha: [1.0, 1.0, 3.0]  # Boost neutral class
focal_gamma: 2.0
```

---

### Scenario 4: Baseline Comparison

**Setup**: Test Focal Loss vs Standard Loss

```yaml
# Experiment 1: Standard Loss
use_focal_loss: false

# Experiment 2: Focal Loss (auto)
use_focal_loss: true
focal_alpha: "auto"

# Experiment 3: Focal Loss (equal)
use_focal_loss: true
focal_alpha: null
```

---

## 📈 Expected Results

### With AUTO Alpha:

```
Before Focal Loss (Standard CrossEntropy):
  Positive F1: 0.92  ← Overfits majority
  Negative F1: 0.78
  Neutral F1:  0.45  ← Poor on minority

After Focal Loss (AUTO alpha):
  Positive F1: 0.90  ← Slight decrease
  Negative F1: 0.82  ← Improved
  Neutral F1:  0.68  ← Much better! (+0.23)
  
Overall F1: 0.80 → 0.83 (+0.03)
```

---

## ⚠️ Common Mistakes

### ❌ Mistake 1: Wrong Alpha Format

```yaml
# WRONG
focal_alpha: [1.0, 2.0]  # Only 2 values (need 3!)

# CORRECT
focal_alpha: [1.0, 1.5, 2.0]  # 3 values for 3 classes
```

---

### ❌ Mistake 2: Too High Gamma

```yaml
# WRONG
focal_gamma: 10.0  # Too high, model can't learn easy examples

# CORRECT
focal_gamma: 2.0   # Standard value
```

---

### ❌ Mistake 3: Using Equal Weights with Imbalanced Data

```yaml
# SUBOPTIMAL for imbalanced data
focal_alpha: null  # Equal weights when 80/15/5 split

# BETTER
focal_alpha: "auto"  # Auto-weight minority classes
```

---

## 🎓 References

**Original Paper**:
- Lin et al. (2017). "Focal Loss for Dense Object Detection"
- Formula: `FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)`

**Key Insights**:
- `(1 - p_t)^γ`: Down-weights easy examples (high confidence)
- `α_t`: Balances class importance
- Works well with `γ = 2.0` in practice

---

## ✅ Quick Reference

```yaml
# Recommended for most cases
single_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: "auto"

# For balanced data (after oversampling)
single_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: null

# For custom control
single_label:
  use_focal_loss: true
  focal_gamma: 2.0
  focal_alpha: [w_pos, w_neg, w_neu]

# Disable (use standard CrossEntropy)
single_label:
  use_focal_loss: false
```

---

**Status**: ✅ **READY TO USE**

Config đã được update, code đã hỗ trợ 3 modes của focal_alpha!
