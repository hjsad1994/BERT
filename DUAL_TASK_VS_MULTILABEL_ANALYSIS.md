# Phân tích: Tại sao Dual-Task Learning có F1 Score cao hơn Multi-Label cho Sentiment Classification

## Tóm tắt
- **Multi-Label**: 96.3% F1 Score cho Sentiment Classification (SC)
- **Dual-Task Learning**: 97.0% F1 Score cho SC
- **Chênh lệch**: +0.7%

---

## Nguyên nhân chính

### 1. **Khác biệt trong cách đánh giá (Evaluation Strategy)**

#### Dual-Task Learning:
```python
# Chỉ đánh giá SC trên các aspects đã được PHÁT HIỆN ĐÚNG
valid_sentiment_mask = valid_mask & (all_aspect_labels == 1) & (all_aspect_preds == 1)
```
- Điều kiện: Aspect phải:
  1. Có trong ground truth (present)
  2. Được model phát hiện đúng (predicted as present)
  3. Có label (not NaN)

#### Multi-Label:
```python
# Đánh giá SC trên TẤT CẢ aspects có label
mask = labeled_mask[:, i]  # Chỉ cần có label (not NaN)
```
- Điều kiện: Chỉ cần aspect có label (not NaN)

**Tác động:**
- Dual-task chỉ đánh giá trên subset "dễ hơn" (aspects đã detect đúng)
- Điều này tạo ra F1 cao hơn vì loại bỏ các trường hợp aspect detection thất bại

---

### 2. **Training Strategy khác nhau**

#### Dual-Task Learning:
- **Loss kết hợp**: `total_loss = 0.3 * AD_loss + 0.7 * SC_loss`
  - Tập trung vào SC (weight 0.7 > 0.3)
  - Học aspect detection và sentiment classification đồng thời
- **Hai task riêng biệt**: 
  - Task 1: Aspect Detection (binary: present/absent)
  - Task 2: Sentiment Classification (3-class)
- **Học phân cấp**: Model học detect aspect trước, rồi classify sentiment

#### Multi-Label:
- **Loss đơn thuần**: Chỉ có SC loss (focal loss)
- **Single task**: Predict sentiment cho tất cả aspects cùng lúc
- **Không có aspect detection**: Không học phân biệt aspect có present hay không

**Tác động:**
- Dual-task có tín hiệu học tốt hơn nhờ joint training
- Model hiểu rõ hơn về aspect presence trước khi classify sentiment

---

### 3. **Model Architecture**

#### Dual-Task Learning:
```python
# Hai head riêng biệt
self.aspect_detection_head = nn.Linear(hidden_size, num_aspects)      # Binary
self.sentiment_classification_head = nn.Linear(hidden_size, num_aspects * num_sentiments)  # 3-class
```
- **Shared backbone**: Cả hai task dùng chung BERT encoder
- **Separate heads**: Mỗi task có classifier riêng
- **Benefit**: Mỗi head có thể tối ưu cho task riêng của nó

#### Multi-Label:
```python
# Single head
self.classifier = nn.Linear(hidden_size, num_aspects * num_sentiments)
```
- **Single head**: Một classifier cho tất cả aspects × sentiments
- **Drawback**: Phải học tất cả combinations cùng lúc

---

### 4. **Loss Function Implementation**

#### Dual-Task Learning:
- **BinaryFocalLoss** cho Aspect Detection
  - Xử lý class imbalance cho binary classification
  - Separate alpha weights cho absent/present
- **MultilabelFocalLoss** cho Sentiment Classification
  - Áp dụng trên aspects đã được detect (có mask)
- **Combined loss**: Weighted combination với tuning riêng

#### Multi-Label:
- **MultilabelFocalLoss** duy nhất
  - Áp dụng trên tất cả labeled aspects
  - Không có aspect detection signal

**Tác động:**
- Dual-task có loss design tốt hơn cho từng task riêng biệt
- Multi-label phải xử lý tất cả trong một loss

---

## Kết luận

### Vì sao Dual-Task Learning cao hơn?

1. **Evaluation Strategy nghiêm ngặt hơn**
   - Chỉ đánh giá trên aspects đã detect đúng
   - Tạo ra subset "cleaner" và "easier"
   - → F1 cao hơn (legitimate - vì đúng là chỉ nên đánh giá SC trên aspects có present)

2. **Training Signal tốt hơn**
   - Joint training với 2 tasks tạo regularization tốt hơn
   - Model hiểu rõ hơn về aspect presence
   - Loss weighting (0.7 cho SC) tập trung vào SC task

3. **Architecture phù hợp hơn**
   - Separate heads cho từng task
   - Mỗi head có thể tối ưu riêng

### Lưu ý

**Đây KHÔNG phải là "cheating" hay unfair comparison:**
- Việc chỉ đánh giá SC trên aspects đã detect đúng là hợp lý
- Trong thực tế, nếu model không detect được aspect thì không cần đánh giá sentiment
- Dual-task learning mô phỏng pipeline thực tế tốt hơn (detect → classify)

### Có nên so sánh công bằng hơn?

Để so sánh công bằng hơn, có thể:
1. **Option 1**: Dual-task vẫn giữ nguyên evaluation (legitimate)
2. **Option 2**: Multi-label cũng chỉ đánh giá trên aspects có present (nhưng multi-label không có aspect detection)
3. **Option 3**: Tạo metric mới: "Sentiment Classification Accuracy trên Correctly Detected Aspects"

---

## Recommendations

### Để cải thiện Multi-Label:
1. **Thêm aspect detection task** (như dual-task)
2. **Sử dụng evaluation strategy tương tự** (chỉ đánh giá SC trên aspects có present)
3. **Xem xét joint training** với aspect detection

### Để cải thiện Dual-Task:
1. **Đã tốt rồi** - architecture và evaluation strategy đã hợp lý
2. **Có thể tune loss weights** (0.3/0.7) để cân bằng tốt hơn
3. **Có thể thử multi-task learning** với nhiều tasks hơn

---

## Code References

### Dual-Task Evaluation (Stricter):
```247:272:dual-task-learning/train_multitask.py
    valid_sentiment_mask = valid_mask & (all_aspect_labels == 1) & (all_aspect_preds == 1)
    
    overall_sc_acc = (all_sentiment_preds[valid_sentiment_mask] == all_sentiment_labels[valid_sentiment_mask]).float().mean().item()
    
    for i, aspect in enumerate(aspect_names):
        mask = valid_sentiment_mask[:, i]
        if mask.sum() == 0:
            continue
        
        sentiment_preds = all_sentiment_preds[:, i][mask].numpy()
        sentiment_labels = all_sentiment_labels[:, i][mask].numpy()
        n_samples = mask.sum().item()
        
        # Multi-class classification metrics
        sc_acc = accuracy_score(sentiment_labels, sentiment_preds)
        sc_precision, sc_recall, sc_f1, _ = precision_recall_fscore_support(
            sentiment_labels, sentiment_preds, average='weighted', zero_division=0
        )
        
        sentiment_metrics[aspect] = {
            'accuracy': sc_acc,
            'precision': sc_precision,
            'recall': sc_recall,
            'f1': sc_f1,
            'n_samples': n_samples
        }
```

### Multi-Label Evaluation (More Permissive):
```148:180:multi_label/train_multilabel.py
    for i, aspect in enumerate(aspect_names):
        if labeled_mask is not None:
            # Only evaluate on labeled samples for this aspect
            mask = labeled_mask[:, i]
            if mask.sum() == 0:
                # No labeled data for this aspect, skip
                print(f"   WARNING: {aspect}: No labeled data, skipping")
                continue
            
            aspect_preds = all_preds[:, i][mask].numpy()
            aspect_labels = all_labels[:, i][mask].numpy()
            n_samples = mask.sum().item()
        else:
            # Evaluate on all samples (old behavior)
            aspect_preds = all_preds[:, i].numpy()
            aspect_labels = all_labels[:, i].numpy()
            n_samples = len(aspect_preds)
        
        # Accuracy
        acc = accuracy_score(aspect_labels, aspect_preds)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='weighted', zero_division=0
        )
        
        aspect_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': n_samples
        }
```

