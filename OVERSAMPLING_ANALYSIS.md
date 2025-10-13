# Phân Tích: Tại Sao Oversampling Không Hiệu Quả

## Kết Quả Thực Tế

### Training với Oversampling (20251013_221133)

| Epoch | Accuracy | F1 (Overall) | Eval Loss | Vấn Đề |
|-------|----------|--------------|-----------|---------|
| 1 | 89.63% | 0.9028 | 0.1198 | Baseline |
| 2 | 89.63% | 0.9027 | 0.1422 | Loss ↑ 19% |
| 3 | 90.73% | 0.9100 | 0.1874 | Loss ↑ 32% |
| 4 | 92.05% | 0.9213 | 0.2410 | **OVERFITTING** |

**🚨 Dấu hiệu Overfitting rõ ràng:**
- F1 tăng nhưng Loss tăng liên tục
- Model đang memorize training data

---

## Nguyên Nhân Oversampling Thất Bại

### 1. Random Duplicate - Không Tạo Diversity
```python
# Cách hiện tại (oversampling_utils.py)
oversampled = class_df.sample(n=n_samples, replace=True, random_state=random_state)
```

**Vấn đề:**
- Chỉ copy y nguyên samples
- Model thấy cùng 1 sample nhiều lần → học thuộc
- Không có variation/noise để generalize

### 2. Oversample Quá Nhiều
```
neutral: 1,069 → 2,830 samples (+1,761)
Tăng 265% so với gốc!
```

**Tác động:**
- Model focus quá nhiều vào neutral class
- Training time lâu hơn
- Overfitting nghiêm trọng

### 3. Thiếu Data Augmentation
- Không có SMOTE (Synthetic Minority Over-sampling)
- Không có text augmentation (synonym replacement, back-translation)
- Không có mixup/cutmix

### 4. Không Track Per-Class Metrics
- Mục tiêu: cải thiện Neutral class (F1=0.48)
- Nhưng chỉ track overall F1
- Không biết Neutral class có cải thiện thật không

---

## Giải Pháp Đề Xuất

### ✅ Giải pháp 1: Smart Oversampling với Class Weights (Đơn giản, hiệu quả)

**Thay vì duplicate samples, tăng loss weight:**

```python
# train.py
from sklearn.utils.class_weight import compute_class_weight

# Tính class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Áp dụng vào Focal Loss (đã có)
# Alpha weights tự động balance theo frequency
```

**Ưu điểm:**
- Không duplicate → tránh overfitting
- Model focus nhiều hơn vào minority class
- Training nhanh hơn (data size không tăng)

---

### ✅ Giải pháp 2: Moderate Oversampling + Early Stopping

**Giảm tỷ lệ oversample:**

```python
# Thay vì 30% of majority (2,830 samples)
# → Chỉ 15-20% (1,698-2,264 samples)

target_counts = {
    'neutral': int(majority_count * 0.15),  # 15% thay vì 30%
    'positive': class_counts['positive'],
    'negative': class_counts['negative']
}
```

**Kết hợp Early Stopping chặt chẽ hơn:**

```yaml
# config.yaml
training:
  early_stopping_patience: 1  # Giảm từ 2 → 1
  num_train_epochs: 3  # Giảm từ 4 → 3
  metric_for_best_model: "eval_loss"  # Dùng loss thay vì F1
```

---

### ✅ Giải pháp 3: Text Augmentation (Nâng cao)

**Tạo synthetic samples thay vì duplicate:**

```python
# augmentation.py
import nlpaug.augmenter.word as naw

# 1. Synonym Replacement (tiếng Việt)
aug_synonym = naw.SynonymAug(aug_src='ppdb', model_path='vi')

# 2. Back Translation
aug_back_trans = naw.BackTranslationAug(
    from_model_name='VietAI/envit5-translation',
    to_model_name='VietAI/envit5-translation'
)

# 3. Random Insertion/Deletion
def augment_text(text, num_aug=2):
    augmented = []
    for _ in range(num_aug):
        augmented.append(aug_synonym.augment(text))
    return augmented

# Áp dụng chỉ cho minority class
if label == 'neutral':
    augmented_texts = augment_text(text, num_aug=2)
```

---

### ✅ Giải pháp 4: SMOTE cho Text (Embedding-based)

**Tạo synthetic samples trong embedding space:**

```python
from imblearn.over_sampling import SMOTE

# 1. Encode texts thành embeddings
embeddings = model.encode(texts)

# 2. Apply SMOTE
smote = SMOTE(sampling_strategy='auto', k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(embeddings, labels)

# 3. Decode embeddings → text (hoặc train trực tiếp trên embeddings)
```

---

### ✅ Giải pháp 5: Track Per-Class Metrics

**Thêm custom metric callback:**

```python
# focal_loss_trainer.py
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Overall metrics
    overall_f1 = f1_score(labels, preds, average='weighted')
    
    # Per-class metrics
    per_class_f1 = f1_score(labels, preds, average=None)
    
    return {
        'f1': overall_f1,
        'f1_neutral': per_class_f1[2],  # neutral = class 2
        'f1_positive': per_class_f1[0],
        'f1_negative': per_class_f1[1],
    }
```

---

## Khuyến Nghị Thực Hiện

### 🎯 Phương án Nhanh (1-2 giờ)

1. **Tắt Oversampling, tăng Class Weights**
   ```yaml
   # config.yaml - Tắt oversampling
   training:
     use_oversampling: false
   ```

2. **Giảm epochs xuống 3**
3. **Track per-class F1**
4. **Compare với baseline**

---

### 🎯 Phương án Tốt Nhất (1-2 ngày)

1. **Giữ Moderate Oversampling (15% thay vì 30%)**
2. **Thêm Text Augmentation cho Neutral class**
3. **Early stopping với patience=1**
4. **Track per-class metrics**
5. **Test nhiều strategies:**
   - No oversampling + class weights
   - Moderate oversampling + augmentation
   - SMOTE (nếu có thời gian)

---

## So Sánh Strategies

| Strategy | Overfitting Risk | Implementation Cost | Expected Improvement |
|----------|------------------|---------------------|----------------------|
| No oversampling + class weights | Low | Low | +2-3% Neutral F1 |
| Moderate oversampling (15%) | Medium | Low | +3-5% Neutral F1 |
| Text augmentation | Low | Medium | +5-7% Neutral F1 |
| SMOTE | Medium | High | +4-6% Neutral F1 |

---

## Next Steps

1. **Tắt oversampling, test baseline với class weights only**
2. **So sánh F1 per-class**
3. **Nếu không cải thiện, thử moderate oversampling (15%)**
4. **Nếu vẫn overfit, implement text augmentation**

---

## Tài Liệu Tham Khảo

- [SMOTE for Text Classification](https://arxiv.org/abs/1812.04718)
- [Text Augmentation Techniques](https://github.com/makcedward/nlpaug)
- [Class Imbalance in NLP](https://aclanthology.org/2020.emnlp-main.438/)
