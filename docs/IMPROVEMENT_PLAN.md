# Kế Hoạch Cải Thiện Độ Chính Xác Từ 91.06%

Dựa trên phân tích log training: `training_logs/training_log_20251010_225807.txt`

---

## 📊 Phân Tích Kết Quả Hiện Tại

### ✅ Điểm Mạnh:
- **Overall Accuracy**: 91.06% (khá tốt)
- **Negative class**: Precision=0.96, Recall=0.94, **F1=0.95** ✅ (1192 samples)
- **Positive class**: Precision=0.89, Recall=0.92, **F1=0.91** ✅ (716 samples)

### ❌ Vấn Đề Nghiêm Trọng:

#### 1. **Neutral Class Performance Rất Thấp**
```
Neutral: Precision=0.48, Recall=0.48, F1=0.48 ❌ (chỉ 106 samples)
```
- Chỉ dự đoán đúng ~50% samples neutral
- Nguyên nhân: **Severe imbalance** (11.08x)
- Neutral chỉ chiếm 5.3% trong training data

#### 2. **Overfitting Rõ Ràng**
```
Epoch 1: eval_loss=0.233, acc=91.3%
Epoch 2: eval_loss=0.207, acc=91.7% ← BEST MODEL
Epoch 3: eval_loss=0.214, acc=89.7% ↓ 
Epoch 4: eval_loss=0.356, acc=91.8% ← LOSS TĂNG 70%!
```
- Eval loss tăng từ 0.207 → 0.356 (tăng 70%)
- Model đang học thuộc training data sau epoch 2
- Early stopping KHÔNG hoạt động (đáng lẽ dừng ở epoch 2)

---

## 🎯 Các Bước Cải Thiện (Theo Thứ Tự Ưu Tiên)

### ✅ **1. BẬT LẠI OVERSAMPLING** (Ưu tiên CAO NHẤT) - ĐÃ FIX

**Vấn đề:** Neutral class quá ít (106 samples = 5.3%)

**Giải pháp đã áp dụng:**
```yaml
# train.py
- Oversample neutral từ 501 → 1,665 samples (30% of majority)
- Imbalance ratio giảm từ 11.08x → ~3.3x
```

**Kết quả mong đợi:** F1 neutral tăng từ 0.48 → 0.65-0.75

---

### ✅ **2. GIẢM SỐ EPOCHS** - ĐÃ FIX

**Vấn đề:** Training 4 epochs nhưng best model ở epoch 2

**Giải pháp đã áp dụng:**
```yaml
# config.yaml
num_train_epochs: 5 → 3
```

**Lý do:** 
- Epoch 2: eval_loss=0.207 (lowest)
- Epoch 3-4: eval_loss tăng (overfitting)
- Dừng ở epoch 3 để tránh lãng phí thời gian

---

### 🔄 **3. TĂNG DROPOUT** (Nếu vẫn overfit)

Nếu sau khi áp dụng 1+2 mà vẫn overfit:

```yaml
# config.yaml - CHƯA ÁP DỤNG, test xem bước 1+2 có đủ không
model:
  hidden_dropout_prob: 0.2  # Tăng từ 0.1 (default)
  attention_probs_dropout_prob: 0.2
```

---

### 📊 **4. ĐIỀU CHỈNH LEARNING RATE** (Tùy chọn)

Nếu kết quả chưa tốt:

**Option A: Giảm thêm một chút**
```yaml
learning_rate: 1.5e-5 → 1.2e-5
```
→ Học chậm hơn, ổn định hơn

**Option B: Tăng warmup**
```yaml
warmup_ratio: 0.1 → 0.15
```
→ Model học ổn định hơn ở đầu training

---

### 🔥 **5. TĂNG FOCAL LOSS GAMMA** (Tùy chọn)

Nếu neutral vẫn thấp sau oversampling:

```python
# utils.py - FocalLoss
gamma = 2.0 → 2.5  # Focus nhiều hơn vào hard examples (neutral)
```

---

## 📝 Checklist Các Thay Đổi Đã Áp Dụng

- ✅ Bật lại oversampling với 30% ratio
- ✅ Giảm epochs từ 5 → 3
- ✅ Giữ nguyên learning_rate = 1.5e-5
- ✅ Giữ nguyên weight_decay = 0.05
- ✅ Early stopping đã có (patience=2)

---

## 🚀 Bước Tiếp Theo

### 1. **Chạy Training Mới**
```bash
python train.py
```

### 2. **Theo Dõi Metrics**

**Chú ý các chỉ số sau:**

```
✅ Mục tiêu cải thiện:
- Neutral F1: 0.48 → >0.65 (tăng 35%+)
- Overall Accuracy: 91.06% → 92-93%
- Eval loss không tăng sau epoch 2

⚠️ Dấu hiệu xấu (cần điều chỉnh tiếp):
- Neutral F1 vẫn <0.60
- Eval loss vẫn tăng ở epoch 3
- Accuracy không cải thiện
```

### 3. **So Sánh Kết Quả**

| Metric | Trước | Mục tiêu | Thực tế |
|--------|-------|----------|---------|
| **Overall Acc** | 91.06% | 92-93% | ___ |
| **Positive F1** | 0.91 | ~0.91 | ___ |
| **Negative F1** | 0.95 | ~0.95 | ___ |
| **Neutral F1** | **0.48** | **>0.65** | ___ |
| **Best Epoch** | 2 | 2-3 | ___ |
| **Eval Loss** | Tăng | Không tăng | ___ |

---

## 📈 Dự Đoán Kết Quả

**Kịch bản lạc quan:**
```
- Neutral F1: 0.48 → 0.70 (tăng 46%)
- Overall Accuracy: 91.06% → 92.5%
- Không overfit (eval loss ổn định)
```

**Kịch bản thực tế:**
```
- Neutral F1: 0.48 → 0.62-0.68 (tăng 30-42%)
- Overall Accuracy: 91.06% → 91.8-92.2%
- Overfit nhẹ hơn
```

**Nếu kết quả không đạt → Áp dụng bước 3-5**

---

## ⚠️ Lưu Ý Quan Trọng

### 1. **Đừng Kỳ Vọng Quá Cao Vào Neutral**
- Neutral class vốn khó phân biệt (không positive, không negative)
- Dataset chỉ có 106 test samples → variance cao
- F1 = 0.65-0.70 là **rất tốt** cho class này

### 2. **Trade-off Có Thể Xảy Ra**
- Cải thiện neutral có thể làm giảm nhẹ positive/negative
- Overall accuracy có thể tăng nhẹ hoặc giữ nguyên
- **Mục tiêu: Cân bằng 3 classes, không chỉ maximize overall accuracy**

### 3. **Overfitting Vẫn Là Mối Quan Tâm**
- Oversampling tăng duplicates → có thể tăng overfitting
- Cần theo dõi eval loss cẩn thận
- Nếu overfit nặng → áp dụng bước 3 (tăng dropout)

---

## 📚 Tài Liệu Tham Khảo

- `OVERSAMPLING_FOCAL_LOSS.md`: Giải thích chi tiết về chiến lược
- `CHECKPOINT_MANAGEMENT.md`: Cách load best model
- `FIX_PREDICT_CRASH.md`: Xử lý memory issues

---

## 🔄 Nếu Vẫn Chưa Đạt Mục Tiêu

### Plan B: Thu Thập Thêm Data
- Annotate thêm 200-300 samples neutral
- Tăng từ 501 → 700-800 samples
- Đây là giải pháp **tốt nhất** nhưng tốn công

### Plan C: Data Augmentation
- Synonym replacement cho neutral samples
- Back-translation (Vi→En→Vi)
- Paraphrasing

### Plan D: Ensemble
- Train nhiều models với random seeds khác nhau
- Voting hoặc averaging predictions
- Thường tăng 0.5-1% accuracy

---

## ✅ Tóm Tắt

**Đã làm:**
1. ✅ Bật oversampling (neutral 30%)
2. ✅ Giảm epochs (5 → 3)

**Chạy ngay:**
```bash
python train.py
```

**Theo dõi:**
- Neutral F1 (mục tiêu: >0.65)
- Eval loss (không tăng)
- Overall accuracy (mục tiêu: >92%)

**Nếu chưa đủ → Áp dụng bước 3-5 trong plan**

Good luck! 🚀
