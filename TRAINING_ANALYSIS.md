# Phân Tích Training Log - Tại Sao Accuracy Thấp?

## 📊 Training Metrics Summary

| Epoch | Val Accuracy | Val F1 | Val Loss | Best? |
|-------|--------------|--------|----------|-------|
| **1** | **93.35%** | **93.29%** | 0.1135 | ✅ BEST |
| 2 | 90.34% | 90.48% | 0.1073 | ❌ (F1 giảm) |
| 3 | 91.49% | 91.54% | 0.1080 | ❌ |
| 4 | 91.68% | 91.68% | 0.1386 | ❌ (Loss tăng) |

**Test Set Performance:**
- Accuracy: **91.36%**
- F1 Score: **91.33%**

---

## 🔍 Vấn Đề Chính

### 1. **Epoch 1 TỐT NHẤT nhưng Test Thấp Hơn** ⚠️

```
Epoch 1 Validation: 93.35% accuracy, 93.29% F1 ← BEST
Test Set:          91.36% accuracy, 91.33% F1 ← 2% GAP!
```

**Lý do:**
- **Overfitting nhẹ**: Model fit val set tốt hơn test set
- **Distribution shift**: Val và test có phân bố hơi khác
- **Luck factor**: Epoch 1 có thể "may mắn" với val set

---

### 2. **Model Giảm Performance Sau Epoch 1** 📉

```
Epoch 1 → Epoch 2: 93.29% → 90.48% F1 (-2.81%)
Epoch 2 → Epoch 3: 90.48% → 91.54% F1 (+1.06%)
Epoch 3 → Epoch 4: 91.54% → 91.68% F1 (+0.14%)
```

**Điều này cho thấy:**
- ✅ Model đã hội tụ ở epoch 1
- ❌ Training thêm làm GIẢM performance
- ⚠️ Có thể đang **overfitting** hoặc **catastrophic forgetting**

---

### 3. **Training Loss vs Validation Loss** 📈

```
Epoch 1: train_loss ~ 0.15, eval_loss = 0.1135 (good gap)
Epoch 4: train_loss ~ 0.05, eval_loss = 0.1386 (bad gap!)
```

**Train loss giảm từ 0.15 → 0.05**
**Val loss TĂNG từ 0.1135 → 0.1386**

→ **Clear sign of OVERFITTING!** 🔴

---

### 4. **Checkpoint Được Chọn Là Đúng** ✅

```
Best checkpoint: checkpoint-9334-e1 (Epoch 1)
Loaded: epoch 1 model với 93.29% val F1
Test:   91.36% accuracy
```

Model **ĐÃ LOAD đúng** checkpoint tốt nhất (epoch 1).
Vấn đề không phải ở việc chọn checkpoint!

---

## 🎯 Tại Sao Test Accuracy Thấp Hơn Val?

### Gap: 93.35% (val) → 91.36% (test) = -1.99%

**Lý do 1: Val/Test Distribution Difference** ⭐⭐⭐⭐⭐
```
Validation set có thể "dễ hơn" test set
- Ít sample khó
- Phân bố aspect/sentiment khác
- Random split không hoàn hảo
```

**Lý do 2: Overfitting to Validation Set** ⭐⭐⭐
```
Model optimization dựa trên val set
→ Có thể học "quirks" của val set
→ Không generalize tốt cho test
```

**Lý do 3: Small Dataset Effect** ⭐⭐⭐
```
Dataset ~20k samples
Val set ~1.5k, Test set ~1.5k
→ High variance in metrics
→ 1-2% gap là NORMAL
```

**Lý do 4: Batch Size 16 Overfitting** ⭐⭐⭐
```
Batch 16 → High gradient noise
Có thể converge quá nhanh ở epoch 1
Sau đó bắt đầu memorize training set
```

---

## 📉 Detailed Performance Breakdown

### Per Class Performance (Test Set)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| **Positive** | 92.01% | 92.14% | 92.08% | 700 |
| **Negative** | 91.38% | 93.54% | 92.45% | 635 |
| **Neutral** | 89.15% | 82.89% | 85.91% | 228 |

**Problem:** 🔴
- **Neutral class** có performance thấp nhất
- Recall: 82.89% (worst!)
- F1: 85.91% (kéo tổng F1 xuống)

**Neutral class chiếm 228/1563 = 14.6% test set**
→ Ảnh hưởng đáng kể đến weighted F1!

---

## 🔬 Root Cause Analysis

### Hypothesis 1: Model Peaked at Epoch 1 ✅

**Evidence:**
- Epoch 1: Best val F1 (93.29%)
- Epoch 2-4: Degrading or stagnant
- Train loss keeps decreasing → overfitting

**Conclusion:**
Model đã tìm được optimal point ở epoch 1.
Training thêm chỉ làm overfit.

---

### Hypothesis 2: Batch Size 16 Converged Too Fast ✅

**Evidence:**
- Batch 16 = 1,250 updates/epoch
- Very frequent updates
- Reached good performance in just 1 epoch

**Research backing:**
> "Small batch sizes converge faster but may overfit quicker"

---

### Hypothesis 3: Neutral Class Distribution Gap ✅

**Evidence:**
- Neutral: 82.89% recall (worst)
- Positive: 92.14% recall (best)
- Negative: 93.54% recall (excellent)

**Reason:**
- Neutral samples harder to classify
- May have different distribution in val vs test
- Less training samples (minority class)

---

## 💡 Giải Pháp

### Solution 1: **Increase Batch Size to 32** ⭐⭐⭐⭐⭐

```yaml
per_device_train_batch_size: 32
gradient_accumulation_steps: 2
num_train_epochs: 3  # Reduce to 3 (enough!)
```

**Why:**
- Smoother convergence
- Less prone to quick overfitting
- Better generalization
- Still reasonable training time

**Expected:**
- Val F1: 92-93% (similar to now)
- Test F1: 92-92.5% (+1% improvement)
- Better val/test consistency

---

### Solution 2: **More Regularization** ⭐⭐⭐⭐

```yaml
weight_decay: 0.02           # Increase from 0.01
dropout: 0.15                # Add dropout
label_smoothing: 0.1         # Add label smoothing
```

**Why:**
- Prevent overfitting
- Better generalization
- Reduce val/test gap

---

### Solution 3: **Better Data Split** ⭐⭐⭐

```python
# Stratified split by aspect AND sentiment
from sklearn.model_selection import train_test_split

# Ensure val/test have similar distribution
train, val, test = stratified_split(data, 
                                    stratify_by=['aspect', 'sentiment'])
```

**Why:**
- Ensure val and test have same distribution
- Reduce metric variance
- More reliable validation

---

### Solution 4: **Early Stopping Earlier** ⭐⭐⭐

```yaml
early_stopping_patience: 1    # Stop after 1 epoch no improvement
num_train_epochs: 5           # Keep max at 5
```

**Why:**
- Model already peaked at epoch 1
- Training more = overfitting
- Save time and get better model

---

### Solution 5: **Ensemble or Average Checkpoints** ⭐⭐

```python
# Average weights from epoch 1, 3, 4
model_ensemble = average_checkpoints([
    'checkpoint-9334-e1',
    'checkpoint-9149-e3', 
    'checkpoint-9168-e4'
])
```

**Why:**
- Smoother predictions
- Better generalization
- Reduce variance

---

## 🎯 Recommended Next Steps

### Priority 1: **Change Batch Size to 32** ⭐⭐⭐⭐⭐

```bash
# Edit config.yaml
per_device_train_batch_size: 32
per_device_eval_batch_size: 64
gradient_accumulation_steps: 2
num_train_epochs: 3  # Reduce to 3

# Retrain
python train.py
```

**Expected improvement:** +0.5-1% test F1

---

### Priority 2: **Add Regularization**

```yaml
weight_decay: 0.02
warmup_ratio: 0.1  # More warmup
```

**Expected improvement:** +0.3-0.5% test F1

---

### Priority 3: **Focus on Neutral Class**

```python
# Add class weights for Focal Loss
class_weights = {
    'positive': 1.0,
    'negative': 1.0,
    'neutral': 1.5  # Higher weight for neutral
}
```

**Expected improvement:** +1-2% neutral F1

---

## 📊 Current vs Expected Performance

| Metric | Current | With Changes | Improvement |
|--------|---------|--------------|-------------|
| **Val F1** | 93.29% | 92.5-93% | Slight decrease OK |
| **Test F1** | 91.33% | 92-93% | **+0.7-1.7%** ✅ |
| **Val-Test Gap** | 1.99% | 0.5-1% | **Better consistency** ✅ |
| **Training Time** | 6 min | 9-12 min | Acceptable |

---

## ✅ Conclusion

### Tại Sao Test Accuracy "Chỉ" 91.36%?

1. ✅ **Model đã peak ở epoch 1** (93.29% val F1)
2. ❌ **Batch 16 converge quá nhanh** → overfit sau epoch 1
3. ⚠️ **Val/Test distribution khác nhau** → 2% gap
4. 🔴 **Neutral class yếu** (82.89% recall)
5. ❌ **Training thêm làm giảm performance** → overfitting

### Accuracy 91.36% Có Tốt Không?

**Góc nhìn tích cực:**
- ✅ 91.36% là **RẤT TỐT** cho ABSA task
- ✅ State-of-the-art thường ~92-94%
- ✅ Gap với val chỉ 2% (acceptable)

**Góc nhìn cải thiện:**
- ⚠️ Có thể lên 92-93% với batch size tốt hơn
- ⚠️ Neutral class cần attention
- ⚠️ Model overfit sau epoch 1

---

## 🚀 Action Items

**Ngay lập tức:**
1. ✅ Chấp nhận 91.36% (tốt rồi!)
2. 🔄 Hoặc retrain với batch 32 (improve thêm 1%)

**Nếu muốn improve:**
```bash
# 1. Edit config.yaml
per_device_train_batch_size: 32
num_train_epochs: 3

# 2. Retrain
python train.py

# Expected: 92-93% test F1
```

**Final verdict:**
🎉 **91.36% là performance TỐT!** Có thể improve thêm 1% nếu cần.
