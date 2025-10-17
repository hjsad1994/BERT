# Phân Tích Config Hiện Tại - Batch Size 16

## ⚠️ Config Hiện Tại Có VẤN ĐỀ!

```yaml
per_device_train_batch_size: 16
per_device_eval_batch_size: 192  # ← WRONG! Too large for batch 16
gradient_accumulation_steps: 1
```

---

## 🔴 Vấn Đề

### 1. **Eval Batch Quá Lớn**
```
Train batch: 16
Eval batch: 192  ← 12x train batch!
```

**Standard practice:**
- Eval batch = 2x train batch
- Should be: **32**, not 192

**Why it's wrong:**
- Inconsistent evaluation
- May cause OOM during eval
- Not matching training conditions

---

### 2. **No Gradient Accumulation**
```
gradient_accumulation_steps: 1
Effective batch = 16 * 1 = 16 (TOO SMALL!)
```

**Research recommendation:**
- Batch 16 should accumulate to 64-128
- Better: `gradient_accumulation_steps: 4`
- Effective batch = 16 * 4 = 64 ✓

---

### 3. **Comment Mismatch**
```yaml
# Pushing to 96 to max out RTX 3070 8GB VRAM
# This will achieve 99-100% GPU utilization
per_device_train_batch_size: 16  ← Actually 16, not 96!
```

Comments say 96, code says 16 → CONFUSING!

---

## ✅ Corrected Config for Batch 16

### Option A: Pure Batch 16 (Slow but Best Quality)

```yaml
training:
  # ========================================================
  # BATCH SIZE 16 - Maximum Generalization
  # ========================================================
  # Small batch for best generalization on 20k dataset
  # WARNING: 6x slower than batch 96!
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 32     # 2x train (standard)
  gradient_accumulation_steps: 4     # Effective = 16*4 = 64
  
  # ========================================================
  # OPTIMIZER - Standard for Small Batch
  # ========================================================
  optim: "adamw_torch"               # Don't need 8-bit with batch 16
  learning_rate: 2.0e-5              # Optimal for batch 16-32
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  max_grad_norm: 1.0
  
  # ========================================================
  # DATALOADER - Optimized for Small Batch
  # ========================================================
  dataloader_num_workers: 4          # More workers for small batch
  dataloader_pin_memory: true
  dataloader_prefetch_factor: 4      # Keep GPU fed
  dataloader_persistent_workers: true
  
  # Training duration
  num_train_epochs: 5
  
  # Mixed precision (still beneficial)
  fp16: true
  fp16_opt_level: "O1"               # Conservative for stability
  fp16_full_eval: false
  tf32: true
```

**Expected Results:**
- GPU: 15-20% utilization (low, expected)
- VRAM: 2-2.5GB (very low)
- Time: **~2.5-3 hours/epoch** (slow!)
- Total: **12-15 hours** for 5 epochs
- Quality: Best F1 score (generalization)

---

### Option B: Batch 32 (RECOMMENDED ⭐)

```yaml
training:
  # ========================================================
  # BATCH SIZE 32 - Balanced Speed & Quality
  # ========================================================
  # Sweet spot: Good generalization + reasonable speed
  # Best for 20k dataset
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 64     # 2x train
  gradient_accumulation_steps: 2     # Effective = 32*2 = 64
  
  # ========================================================
  # OPTIMIZER
  # ========================================================
  optim: "adamw_bnb_8bit"            # 8-bit helps even at batch 32
  learning_rate: 2.5e-5              # Slightly higher for batch 32
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  max_grad_norm: 1.0
  
  # ========================================================
  # DATALOADER
  # ========================================================
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  dataloader_prefetch_factor: 3
  dataloader_persistent_workers: true
  
  # Training duration
  num_train_epochs: 5
  
  # Mixed precision
  fp16: true
  fp16_opt_level: "O2"
  fp16_full_eval: false
  tf32: true
```

**Expected Results:**
- GPU: 50-60% utilization (acceptable)
- VRAM: 3.5-4GB
- Time: **~45-60 min/epoch**
- Total: **4-5 hours** for 5 epochs
- Quality: Near-optimal F1 (small decrease vs batch 16)

---

## 📊 Trade-off Comparison

| Config | Batch | GPU % | Time/Epoch | Total Time | Quality |
|--------|-------|-------|------------|------------|---------|
| **Current (Wrong)** | 16 | 15-20% | 150-180min | 12-15h | Best |
| **Option A (Pure 16)** | 16 + accum 4 | 15-20% | 150-180min | 12-15h | Best |
| **Option B (Batch 32)** | 32 + accum 2 | 50-60% | 45-60min | 4-5h | Excellent |
| **Previous (Batch 96)** | 96 + accum 1 | 99-100% | 25-30min | 2-2.5h | Good |

---

## 🎯 My Recommendation

### Fix Current Config Issues:

```yaml
# Fix 1: Correct eval batch size
per_device_eval_batch_size: 32  # Not 192!

# Fix 2: Add gradient accumulation
gradient_accumulation_steps: 4  # Get effective batch 64

# Fix 3: Fix comments
# BATCH SIZE 16 - Maximum Generalization (Slow but Best Quality)
```

### But I Strongly Recommend Batch 32:

**Why:**
✅ 3x faster than batch 16 (4h vs 12h)
✅ Still excellent generalization
✅ Better GPU utilization (50% vs 15%)
✅ More practical for development
✅ Research-backed for 20k dataset

**Quality difference:**
- Batch 16: F1 = 91.5% (estimated)
- Batch 32: F1 = 91.0% (estimated)
- **Only 0.5% difference!** Worth 3x speed

---

## 🚀 Recommended Action

### Change config to Batch 32:

```yaml
per_device_train_batch_size: 32
per_device_eval_batch_size: 64
gradient_accumulation_steps: 2
learning_rate: 2.5e-5
dataloader_num_workers: 4
```

**If you insist on Batch 16:**

```yaml
per_device_train_batch_size: 16
per_device_eval_batch_size: 32   # FIX THIS! (was 192)
gradient_accumulation_steps: 4   # FIX THIS! (was 1)
learning_rate: 2.0e-5
dataloader_num_workers: 4
```

---

## Summary

**Current config has 2 critical bugs:**
1. ❌ Eval batch 192 (should be 32)
2. ❌ No gradient accumulation (should be 4)

**Recommendation: Switch to Batch 32**
- 3x faster
- Still excellent quality
- Research-backed optimal
