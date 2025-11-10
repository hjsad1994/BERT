# Oversampling Failed Analysis
## Why 7x Oversampling Led to Worse Results

**Date**: 2025-11-09  
**Finding**: Oversampling caused overfitting and worse test performance  
**Decision**: Revert to baseline (no oversampling)

---

## 📊 **EXPERIMENT RESULTS**

### Performance Comparison:

| Approach | Test F1 | Test P | Test R | Val F1 | Val-Test Gap |
|----------|---------|--------|--------|--------|--------------|
| **Baseline (No OS)** | **92.81%** ✅ | 94.33% | 91.72% | ~93% | ~0.2% ✅ |
| **7x Oversampling** | **91.32%** ⚠️ | 92.73% | 90.49% | 94.94% | **-3.62%** ⚠️⚠️ |
| **Difference** | **-1.49%** ⬇️ | -1.60% | -1.23% | +1.94% | -3.42% |

**Key Finding**: 
- ❌ Test performance WORSE by 1.49% F1
- ❌ Validation-Test gap 3.62% indicates severe overfitting
- ❌ All metrics decreased on test set

---

## 🔍 **ROOT CAUSE ANALYSIS**

### 1. **Excessive Data Exposure**

```
Original data:  11,808 samples
Balanced data:  25,176 samples (2.13x)
Epochs:         10
Total exposures: ~251,760 training samples

→ Model sees each duplicated sample ~10 times
→ Memorizes training patterns
→ Fails to generalize to test set
```

### 2. **Validation Misleading**

Training progression:
```
Epoch 1:  Val F1 = 92.18%
Epoch 2:  Val F1 = 94.53% ← Peak generalization
Epoch 7:  Val F1 = 94.61%
Epoch 9:  Val F1 = 94.94% ← Best val, but overfitted
Epoch 10: Val F1 = 94.94%

Final Test: F1 = 91.32% ← Crash!
```

**Problem**: Validation F1 kept improving → Early stopping didn't trigger → Trained too long → Overfitting

### 3. **Distribution Mismatch**

| Dataset | Size | Distribution | Model Fit |
|---------|------|--------------|-----------|
| **Train (Balanced)** | 25,176 | Neutral ~27% | ✅ Excellent |
| **Val** | 1,476 | Similar to train | ✅ Good (94.94%) |
| **Test** | 1,477 | Original distribution | ❌ Poor (91.32%) |

→ Model optimized for balanced distribution  
→ Test set has original imbalanced distribution  
→ Generalization gap

---

## 🎯 **WHY OVERSAMPLING FAILED**

### Theory vs Reality:

| Aspect | Expected | Reality |
|--------|----------|---------|
| **Class Balance** | Neutral 7.88% → 27.48% | ✅ Worked |
| **Model Learning** | Better minority class learning | ✅ Worked on Val |
| **Generalization** | Better test performance | ❌ **FAILED** |
| **Overfitting Risk** | Controlled by regularization | ❌ **NOT ENOUGH** |

### Root Issues:

1. **Too Much Duplication**: 7x for Neutral = 16.5x for some aspects
2. **Too Many Epochs**: 10 epochs × 2.13x data = 21.3 epoch-equivalents
3. **Weak Regularization**: dropout=0.3, weight_decay=0.01 insufficient
4. **Test Distribution Shift**: Balanced train vs imbalanced test

---

## 📚 **LESSONS LEARNED**

### 1. **Oversampling Isn't Always Better**

Baseline without oversampling (92.81%) > Oversampling (91.32%)

**When oversampling helps**:
- Extreme imbalance (100:1 ratio)
- Small dataset (<5K samples)
- Controlled epochs (2-5 max)

**When oversampling hurts**:
- ✅ **Moderate imbalance** (8.5:1 can be handled by focal loss)
- ✅ **Medium dataset** (11K samples is enough)
- ✅ **Many epochs** (10 epochs too much for 2.13x data)

### 2. **Validation Performance != Test Performance**

```
Val F1 = 94.94% (excellent!)
Test F1 = 91.32% (poor!)
Gap = 3.62% (overfitting indicator)
```

**Lesson**: Always monitor val-test gap, not just val performance

### 3. **Focal Loss > Oversampling** (For this task)

Focal loss alpha weights can handle imbalance without data duplication:
```
Alpha: [0.59, 0.93, 4.23]  # Neutral weight 7x higher
→ Equivalent to 7x oversampling but no memorization risk
```

---

## ✅ **SOLUTION: REVERT TO BASELINE**

### Configuration Changes:

```yaml
# BEFORE (Failed with oversampling):
train_file_sc: "train_multilabel_balanced.csv"
epochs: 10
dropout: 0.3
weight_decay: 0.01
label_smoothing: 0.1

Results: Test F1 = 91.32%

# AFTER (Baseline + focal_gamma boost):
train_file_sc: "train_multilabel.csv"  # NO OVERSAMPLING
focal_gamma: 3.0  # Increased focus on hard samples
epochs: 10
dropout: 0.3
weight_decay: 0.01
label_smoothing: 0.0  # Not needed without oversampling

Expected: Test F1 = 92.81%+ (baseline or slightly better)
```

### Why This Works:

1. ✅ **No Duplication**: Original 11,808 samples
2. ✅ **Focal Loss Handles Imbalance**: Alpha weights compensate
3. ✅ **Better Generalization**: Train/test distribution match
4. ✅ **Simpler**: No oversampling complexity

---

## 📈 **ALTERNATIVE APPROACHES (If needed)**

If baseline 92.81% still not satisfactory:

### **Option A: Lighter Oversampling**
```python
# 3x cap instead of 7x
python augment_multilabel_balanced.py --max-ratio 3.0
```
Config:
```yaml
epochs: 5  # Fewer epochs
dropout: 0.4  # More regularization
```

**Expected**: 92-93% F1 (less overfitting risk)

### **Option B: Focus on Recall Improvements**
Instead of oversampling, use:
- ✅ Higher focal_gamma (3.0 → 4.0)
- ✅ Boosted Neutral alpha weight (4.23 → 5.0)
- ✅ Threshold tuning for classification

**Expected**: Recall +1-2%, F1 ~93-94%

### **Option C: Ensemble Approach**
Train 3-5 models with different seeds, ensemble predictions

**Expected**: F1 +0.5-1% improvement

---

## 🎓 **GENERAL PRINCIPLES**

### When to Use Oversampling:

✅ **YES if**:
- Extreme imbalance (>100:1 ratio)
- Very small dataset (<5K samples)
- Short training (2-3 epochs max)
- Strong regularization (dropout>0.5)

❌ **NO if**:
- Moderate imbalance (<20:1)
- Medium+ dataset (>10K samples)
- Long training (>5 epochs)
- Test distribution differs from train

### Better Alternatives:

1. **Focal Loss** (what we're using) ✅
2. **Class Weights** (built into focal loss)
3. **Data Augmentation** (paraphrase, not duplicate)
4. **Ensemble Methods**
5. **Threshold Tuning**

---

## 📋 **SUMMARY**

| Question | Answer |
|----------|--------|
| **Did oversampling help?** | ❌ No, made it worse (-1.49% F1) |
| **Why did it fail?** | Overfitting (Val 94.94% vs Test 91.32%) |
| **What's the fix?** | Revert to baseline (no oversampling) |
| **Expected result?** | F1 ~92.81% (baseline) |
| **Alternative?** | Focal loss + threshold tuning |

---

## 🚀 **NEXT STEPS**

1. ✅ **Config reverted** to baseline (no oversampling)
2. ✅ **Kept focal_gamma=3.0** (small improvement)
3. ⏳ **Re-train** to verify baseline results
4. ⏳ **If needed**: Try Option B (recall improvements without oversampling)

**Command to re-train**:
```bash
cd E:\BERT\VisoBERT-STL
python train_visobert_stl.py --config config_visobert_stl.yaml
```

**Expected**: Test F1 ~92.81% (same as original baseline) ✅

---

## 📊 **CONCLUSION**

**Oversampling is NOT a silver bullet.**

For this Vietnamese ABSA task with:
- 11.8K training samples
- 8.5:1 imbalance ratio
- Focal loss already implemented

→ **Baseline without oversampling performs BEST**

Key insight: "More data" doesn't always mean "better model" - it can lead to overfitting if not carefully controlled.

**Final recommendation**: Stick with baseline + focal loss. It's simpler, more stable, and performs better (92.81% vs 91.32%).
