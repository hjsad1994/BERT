# 🔍 ERROR ANALYSIS GUIDE

## 📖 Hướng Dẫn Phân Tích Lỗi Model

---

## 🎯 MỤC ĐÍCH

Error Analysis giúp bạn:

1. ✅ **Tìm patterns** trong các predictions sai
2. ✅ **Phát hiện aspects yếu** cần cải thiện
3. ✅ **Phân tích confusion** giữa các sentiment classes
4. ✅ **Tìm hard cases** (câu khó) để cải thiện data
5. ✅ **Đề xuất cải thiện** cụ thể dựa trên phân tích

---

## 🚀 CÁCH SỬ DỤNG

### Bước 1: Đảm bảo có predictions

Kiểm tra xem đã có file `test_predictions.csv` chưa:

```bash
# Nếu chưa có, chạy inference trước
python analyze_results.py
```

Hoặc nếu đã train xong, file này sẽ tự động được tạo.

---

### Bước 2: Chạy Error Analysis

```bash
python error_analysis.py
```

**Output:**
- Terminal sẽ hiển thị phân tích chi tiết
- Các files được lưu trong folder `error_analysis_results/`

---

### Bước 3: Xem kết quả

```bash
cd error_analysis_results
```

**Files được tạo:**

```
error_analysis_results/
├── aspect_error_analysis.csv          ← Error rate theo aspect
├── sentiment_error_analysis.csv       ← Error rate theo sentiment
├── confusion_patterns.csv             ← Confusion pairs chi tiết
├── hard_cases.csv                     ← Các câu khó nhất
├── improvement_suggestions.txt        ← Đề xuất cải thiện
├── error_analysis_report.txt          ← Report tổng hợp
├── aspect_error_rates.png             ← Visualization
├── confusion_matrix.png               ← Confusion matrix
└── sentiment_error_rates.png          ← Error rates by sentiment
```

---

## 📊 PHÂN TÍCH CHI TIẾT

### 1. Aspect Error Analysis

**File:** `aspect_error_analysis.csv`

**Nội dung:**

```
Aspect          Total   Correct  Errors   Accuracy   Error Rate
═══════════════════════════════════════════════════════════════
Performance     120     98       22       81.67%     18.33%  ← Yếu!
Neutral         65      48       17       73.85%     26.15%  ← Rất yếu!
Camera          95      88       7        92.63%     7.37%   ← Tốt!
Battery         110     105      5        95.45%     4.55%   ← Rất tốt!
```

**Cách đọc:**
- **Error Rate cao** (>15%): Aspect yếu, cần cải thiện
- **Error Rate thấp** (<10%): Aspect mạnh
- **Errors column**: Số lượng predictions sai

**Action:**
1. Tìm aspects có Error Rate > 15%
2. Xem `hard_cases.csv` để hiểu tại sao sai
3. Cân nhắc:
   - Thu thập thêm data cho aspects yếu
   - Review labeling quality
   - Thêm keywords/features đặc trưng

---

### 2. Sentiment Error Analysis

**File:** `sentiment_error_analysis.csv`

**Nội dung:**

```
Sentiment    Total   Correct  Errors   Accuracy   Error Rate
═════════════════════════════════════════════════════════════
Negative     666     625      41       93.84%     6.16%   ← Tốt
Positive     426     398      28       93.43%     6.57%   ← Tốt
Neutral      65      48       17       73.85%     26.15%  ← Yếu!
```

**Vấn đề phổ biến:**

**Neutral class yếu nhất vì:**
- ❌ Imbalanced data (chỉ 5.7%)
- ❌ Subjective labeling
- ❌ Boundary không rõ ràng

**Đã áp dụng:**
- ✅ Focal Loss (γ=2.0)
- ✅ Oversampling (305 → 1,244)
- ✅ Class weights

**Có thể cải thiện thêm:**
- 🔧 Tăng oversampling ratio (40% → 50%)
- 🔧 Tăng Focal Loss gamma (2.0 → 3.0)
- 🔧 SMOTE thay vì random oversampling
- 🔧 Thu thập thêm neutral samples chất lượng

---

### 3. Confusion Patterns

**File:** `confusion_patterns.csv`

**Ví dụ:**

```
CONFUSION PAIRS (Nhầm gì thành gì):

True         →  Predicted    Count   % of Errors
════════════════════════════════════════════════
neutral      →  positive     25      29.1%  ← Confusion nhiều nhất!
neutral      →  negative     18      20.9%
positive     →  neutral      12      14.0%
negative     →  neutral      10      11.6%
positive     →  negative     8       9.3%
negative     →  positive     7       8.1%
```

**Cách đọc:**

**Pattern 1: `neutral → positive` (29.1%)**
- Model có **positive bias**
- Các neutral samples bị nhầm thành positive
- **Nguyên nhân:** Data imbalance, positive class nhiều hơn
- **Giải pháp:**
  - Tăng alpha weight cho neutral trong Focal Loss
  - Review các neutral samples có từ tích cực
  - Thêm negative keywords cho neutral detection

**Pattern 2: `positive → negative` hoặc `negative → positive`**
- Confusion **NGHIÊM TRỌNG** (ngược hoàn toàn!)
- **Nguyên nhân có thể:**
  - Sarcasm/irony ("Tốt lắm! Pin hết sau 2 tiếng")
  - Context phức tạp
  - Labeling errors
- **Giải pháp:**
  - Manual review các cases này
  - Check data quality
  - Xem xét thêm context features

---

### 4. Confusion by Aspect

**Ví dụ:**

```
CONFUSION PATTERNS BY ASPECT:

Performance:
  • neutral → positive: 8 cases
  • positive → neutral: 5 cases
  • negative → neutral: 3 cases

Camera:
  • neutral → negative: 4 cases
  • positive → neutral: 2 cases

Battery:
  • neutral → positive: 6 cases
```

**Insights:**

**Performance aspect:**
- Neutral bị nhầm positive nhiều nhất
- → Có thể từ "mượt", "nhanh" gây positive bias
- → Cần distinguish giữa "mượt" (positive) vs "bình thường" (neutral)

**Camera aspect:**
- Neutral bị nhầm negative
- → Có thể từ "tạm", "ok" bị hiểu nhầm
- → Cần balance positive/negative keywords

---

### 5. Hard Cases

**File:** `hard_cases.csv`

**Format:**

```csv
sentence,aspect,true_sentiment,predicted_sentiment,confusion_type
"Pin trâu nhưng không ấn tượng",Battery,neutral,positive,neutral → positive
"Camera tạm ổn cho tầm giá",Camera,neutral,negative,neutral → negative
"Máy mượt nhưng không xuất sắc",Performance,neutral,positive,neutral → positive
```

**Cách phân tích:**

#### Case 1: "Pin trâu nhưng không ấn tượng"

**Aspect:** Battery  
**True:** Neutral  
**Predicted:** Positive  

**Tại sao sai:**
- Từ "trâu" = positive keyword mạnh
- "Không ấn tượng" = neutral/negative BUT model focus "trâu"
- → Model thiên về positive vì "trâu" quá mạnh

**Cách fix:**
- Thêm training samples với pattern "X tốt nhưng Y"
- Teach model về adversative conjunctions ("nhưng", "tuy nhiên")
- Balance attention mechanism

#### Case 2: "Camera tạm ổn cho tầm giá"

**Aspect:** Camera  
**True:** Neutral  
**Predicted:** Negative  

**Tại sao sai:**
- "Tạm ổn" = neutral phrase NHƯNG có negative connotation
- "Cho tầm giá" = qualifying phrase
- → Model focus "tạm" = not good enough = negative

**Cách fix:**
- Label clarification: "tạm ổn" là neutral hay negative?
- Add more context-aware samples
- Teach model về qualifying phrases

---

## 💡 ĐỀ XUẤT CẢI THIỆN

### Priority 1: Aspects Yếu (High Error Rate)

**Nếu có aspects error rate > 15%:**

```
📍 Performance (Error Rate: 18.33%)
   Action:
   1. Thu thập thêm ~44 samples (2x số errors)
   2. Review labeling quality
   3. Thêm keywords: "mượt", "lag", "giật", "nhanh", "chậm"
   4. Check confusion patterns: neutral → positive?
```

**Implementation:**

```python
# 1. Check current data
performance_samples = df[df['aspect'] == 'Performance']
print(f"Current: {len(performance_samples)} samples")

# 2. Check sentiment distribution
print(performance_samples['sentiment'].value_counts())

# 3. Identify if imbalanced → apply oversampling
# 4. Collect more data từ specific sources
```

---

### Priority 2: Neutral Class

**Problem:** Error rate = 26.15%

**Đã làm:**
- ✅ Focal Loss (γ=2.0)
- ✅ Oversampling (305 → 1,244, 40% of majority)
- ✅ Alpha weights: [0.54, 0.35, 3.52]

**Cần làm thêm:**

#### Option A: Tăng Oversampling

```python
# config.yaml hoặc trong train.py
# Current: 40% of majority
target_neutral_count = int(majority_count * 0.4)

# New: 50% of majority
target_neutral_count = int(majority_count * 0.5)
```

#### Option B: Tăng Focal Loss Gamma

```python
# train.py, line ~316
# Current: gamma = 2.0
gamma = 2.0

# New: gamma = 3.0 (focus hơn vào hard examples)
gamma = 3.0
```

#### Option C: SMOTE (Synthetic Oversampling)

```bash
pip install imbalanced-learn

# Tạo synthetic samples thay vì duplicate
from imblearn.over_sampling import SMOTE
```

---

### Priority 3: Confusion Patterns

**Pattern:** `neutral → positive` (29.1% of errors)

**Root cause:** Positive bias

**Solutions:**

#### 1. Điều chỉnh Class Weights

```python
# train.py
# Current alpha (auto-calculated)
alpha = [0.54, 0.35, 3.52]  # [pos, neg, neutral]

# Manual adjustment: Boost neutral hơn nữa
alpha = [0.54, 0.35, 4.50]  # Tăng neutral weight
```

#### 2. Threshold Adjustment

```python
# test_sentiment_smart.py
# Add confidence threshold adjustment for neutral

if predicted_sentiment == 'neutral':
    # Require higher confidence for neutral
    if confidence < 0.75:  # Instead of 0.70
        continue
```

#### 3. Balanced Sampling

```python
# Ensure equal representation during training
from torch.utils.data import WeightedRandomSampler

# Calculate sample weights
class_counts = Counter(train_df['sentiment'])
weights = [1.0/class_counts[s] for s in train_df['sentiment']]

sampler = WeightedRandomSampler(weights, len(weights))

# Use in DataLoader
train_loader = DataLoader(train_dataset, sampler=sampler, ...)
```

---

### Priority 4: Data Quality

**Action items:**

#### 1. Review Labeling Guidelines

```
Positive: Rõ ràng tốt, không có nhược điểm đáng kể
  ✓ "Pin trâu lắm"
  ✓ "Camera đẹp"
  
Negative: Rõ ràng tệ, không có ưu điểm đáng kể
  ✓ "Pin tệ quá"
  ✓ "Camera mờ"
  
Neutral: Trung bình HOẶC có cả ưu và nhược điểm
  ✓ "Pin tạm ổn"
  ✓ "Camera bình thường"
  ✓ "Pin tốt nhưng camera tệ" (nếu nói về cả 2)
```

#### 2. Inter-Annotator Agreement

```python
# Check labeling consistency
# Có thể dùng Cohen's Kappa, Fleiss' Kappa

from sklearn.metrics import cohen_kappa_score

# If có multiple annotators
kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
print(f"Kappa: {kappa:.3f}")

# Kappa > 0.8: Excellent
# Kappa 0.6-0.8: Good
# Kappa < 0.6: Need improvement
```

#### 3. Hard Cases Manual Review

```bash
# Mở file hard_cases.csv
# Manual review từng case
# Re-label nếu cần
# Add to training data với label mới
```

---

## 📈 MONITORING IMPROVEMENTS

### After Implementing Changes

**1. Re-train model:**

```bash
python train.py
```

**2. Re-run Error Analysis:**

```bash
python error_analysis.py
```

**3. Compare metrics:**

```python
# Before:
Neutral Error Rate: 26.15%
Confusion (neutral → positive): 29.1%

# After (Expected):
Neutral Error Rate: 18-20%  ← Target
Confusion (neutral → positive): 20-22%  ← Reduced
```

**4. Track progress:**

```bash
# Create comparison report
echo "Iteration 1: Neutral Error Rate = 26.15%" >> progress.txt
echo "Iteration 2: Neutral Error Rate = 20.30%" >> progress.txt
```

---

## 🎯 SUCCESS METRICS

### Good Performance:

```
✓ Overall Accuracy: > 90%
✓ Per-aspect Error Rate: < 10%
✓ Neutral Class Error Rate: < 15%
✓ Confusion Rate: < 20% for any pair
```

### Current Status:

```
✓ Overall Accuracy: 91.1%  ✅
✗ Worst Aspect: 18.33%     ⚠️  Need improvement
✗ Neutral: 26.15%          ❌  Need significant improvement
✓ Best Aspect: 4.55%       ✅  Excellent
```

---

## 🛠️ ADVANCED TECHNIQUES

### 1. Error-Driven Data Augmentation

```python
# Generate similar samples to hard cases
from nlpaug.augmenter.word import SynonymAug

aug = SynonymAug(aug_src='wordnet')

# For each hard case
for sentence in hard_cases['sentence']:
    # Generate variations
    augmented = aug.augment(sentence, n=3)
    
    # Label manually
    # Add to training data
```

### 2. Active Learning

```python
# Select most uncertain samples for human labeling
uncertainty = []

for sample in unlabeled_data:
    probs = model.predict_proba(sample)
    entropy = -sum(p * log(p) for p in probs)
    uncertainty.append((sample, entropy))

# Sort by entropy (high = uncertain)
most_uncertain = sorted(uncertainty, key=lambda x: -x[1])[:100]

# Send to human annotators
```

### 3. Ensemble Methods

```python
# Train multiple models với different seeds/configs
models = [
    train_model(seed=42),
    train_model(seed=123),
    train_model(seed=456),
]

# Ensemble predictions
def ensemble_predict(sentence, aspect):
    predictions = [m.predict(sentence, aspect) for m in models]
    # Majority voting hoặc averaging
    return most_common(predictions)
```

---

## 📚 TÀI LIỆU THAM KHẢO

1. **Error Analysis:**
   - Manning & Schütze "Foundations of Statistical NLP"
   - Wu et al. "Errudite: Scalable Error Analysis" (ACL 2019)

2. **Class Imbalance:**
   - He & Garcia "Learning from Imbalanced Data" (2009)
   - Chawla et al. "SMOTE" (2002)

3. **Model Improvement:**
   - Howard & Ruder "Universal Language Model Fine-tuning" (2018)
   - Devlin et al. "BERT" (2018)

---

## 🎉 KẾT LUẬN

**Error Analysis giúp:**

1. ✅ **Hiểu** model đang sai ở đâu
2. ✅ **Phát hiện** patterns trong errors
3. ✅ **Ưu tiên** aspects/classes cần cải thiện
4. ✅ **Đề xuất** actions cụ thể
5. ✅ **Track** progress qua iterations

**Next Steps:**

```bash
# 1. Run Error Analysis
python error_analysis.py

# 2. Review results
cd error_analysis_results
cat improvement_suggestions.txt

# 3. Implement improvements
# - Adjust hyperparameters
# - Collect more data
# - Fix labeling issues

# 4. Re-train
python train.py

# 5. Compare
python error_analysis.py

# 6. Iterate until satisfied!
```

---

**🚀 Good luck improving your model!**
