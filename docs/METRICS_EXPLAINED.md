# Giải Thích Các Metrics và Loss Functions

Hướng dẫn chi tiết về các chỉ số đánh giá model trong Machine Learning

---

## 📊 1. CÁC METRICS CƠ BẢN

### 🎯 **Confusion Matrix (Ma Trận Nhầm Lẫn)**

Đây là nền tảng để tính tất cả metrics khác:

```
                    Dự đoán
                Positive  Negative
Thực tế  Pos  |   TP    |   FN    |
         Neg  |   FP    |   TN    |
```

**Giải thích:**
- **TP (True Positive)**: Dự đoán Positive, thực tế Positive ✅
- **TN (True Negative)**: Dự đoán Negative, thực tế Negative ✅
- **FP (False Positive)**: Dự đoán Positive, thực tế Negative ❌ (Type I Error)
- **FN (False Negative)**: Dự đoán Negative, thực tế Positive ❌ (Type II Error)

---

### 📈 **Accuracy (Độ Chính Xác)**

**Định nghĩa:** Tỷ lệ dự đoán đúng trên tổng số dự đoán

**Công thức:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Ví dụ từ model của bạn:**
```
Accuracy = 91.06% = 1834 đúng / 2014 tổng
```

**Khi nào dùng:**
- ✅ Classes cân bằng (positive ≈ negative ≈ neutral)
- ❌ Classes imbalance (như model của bạn: neutral chỉ 5%)

**Vấn đề:**
```python
# Ví dụ: 100 samples, 95 positive, 5 negative
# Model dự đoán TẤT CẢ là positive
# → Accuracy = 95% (cao!)
# → Nhưng không bao giờ detect được negative! (vô dụng)
```

---

### 🎯 **Precision (Độ Chính Xác Của Dự Đoán Positive)**

**Định nghĩa:** Trong những cái mô hình dự đoán Positive, bao nhiêu % thực sự Positive?

**Công thức:**
```
Precision = TP / (TP + FP)
```

**Ví dụ thực tế:**
```
Model dự đoán 100 câu là "positive"
→ 90 câu thực sự positive (TP = 90)
→ 10 câu thực ra negative (FP = 10)
→ Precision = 90/100 = 0.90 = 90%
```

**Từ log của bạn:**
```
Positive: precision=0.8934 
→ Khi model nói "positive", đúng 89.34%
```

**Khi nào quan trọng:**
- 🚨 **Spam filter**: Không muốn email quan trọng bị đánh dấu spam
- 🔍 **Search engine**: Kết quả tìm kiếm phải chính xác
- ⚖️ **Legal**: Không muốn buộc tội người vô tội

**Câu hỏi quan trọng:** *"Nếu model nói Positive, tôi có thể tin được không?"*

---

### 🎣 **Recall (Độ Phủ / Khả Năng Tìm Ra)**

**Định nghĩa:** Trong tất cả samples thực sự Positive, mô hình tìm được bao nhiêu %?

**Công thức:**
```
Recall = TP / (TP + FN)
```

**Ví dụ thực tế:**
```
Có 100 câu thực sự "positive"
→ Model tìm được 92 câu (TP = 92)
→ Bỏ sót 8 câu (FN = 8)
→ Recall = 92/100 = 0.92 = 92%
```

**Từ log của bạn:**
```
Positive: recall=0.9246
→ Với 716 câu positive thực tế, model tìm được 92.46%
```

**Khi nào quan trọng:**
- 🏥 **Medical diagnosis**: KHÔNG được bỏ sót bệnh nhân ung thư
- 🔐 **Security**: PHẢI phát hiện hết virus/malware
- 🚨 **Fire alarm**: Thà báo động nhầm, không được bỏ sót cháy thật

**Câu hỏi quan trọng:** *"Model có bỏ sót nhiều không?"*

---

### ⚖️ **Precision vs Recall Trade-off**

```
         HIGH PRECISION          HIGH RECALL
         (Chính xác)            (Toàn diện)
              
   Dự đoán ít nhưng chắc    Dự đoán nhiều để không bỏ sót
              |                      |
              v                      v
        Ít False Positive      Ít False Negative
```

**Ví dụ:**

**Spam Filter:**
```
High Precision (Conservative):
- Chỉ đánh dấu spam khi 99% chắc chắn
- Ít email quan trọng bị nhầm spam
- Nhưng nhiều spam lọt vào inbox

High Recall (Aggressive):
- Đánh dấu spam khi nghi ngờ 50%
- Bắt được hầu hết spam
- Nhưng nhiều email quan trọng bị nhầm spam
```

**Trade-off:**
- Tăng Precision → Giảm Recall
- Tăng Recall → Giảm Precision

→ Cần **cân bằng** → Dùng **F1 Score**!

---

### 🏆 **F1 Score (Harmonic Mean của Precision và Recall)**

**Định nghĩa:** Trung bình điều hòa của Precision và Recall

**Công thức:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Tại sao dùng Harmonic Mean (không phải Average)?**

```
Ví dụ 1: Precision=100%, Recall=0%
→ Average = (100+0)/2 = 50% (có vẻ oke?)
→ F1 = 2×(100×0)/(100+0) = 0% ✅ (phản ánh đúng: model vô dụng)

Ví dụ 2: Precision=90%, Recall=90%
→ Average = 90%
→ F1 = 2×(90×90)/(90+90) = 90% ✅ (cân bằng tốt)

Ví dụ 3: Precision=80%, Recall=100%
→ Average = 90%
→ F1 = 2×(80×100)/(180) = 88.9% (phạt khi không cân bằng)
```

**Từ log của bạn:**
```
Positive: precision=0.8934, recall=0.9246
→ F1 = 2×(0.8934×0.9246)/(0.8934+0.9246) = 0.9087

Neutral: precision=0.4766, recall=0.4811
→ F1 = 2×(0.4766×0.4811)/(0.4766+0.4811) = 0.4789 ❌ (RẤT THẤP)
```

**Khi nào dùng:**
- ✅ Muốn cân bằng giữa Precision và Recall
- ✅ Classes imbalance
- ✅ So sánh models

**Giải thích đơn giản:**
> F1 Score là **"điểm trung bình nghiêm khắc"** của Precision và Recall.  
> Nếu 1 trong 2 thấp → F1 thấp  
> Cả 2 cao → F1 cao

---

## 📊 2. MACRO vs WEIGHTED AVERAGE

### 🎨 **Macro Average** (Trung Bình Đơn Giản)

**Công thức:**
```
Macro F1 = (F1_positive + F1_negative + F1_neutral) / 3
```

**Ví dụ từ model của bạn:**
```
Positive F1 = 0.9087
Negative F1 = 0.9508  
Neutral F1  = 0.4789

Macro F1 = (0.9087 + 0.9508 + 0.4789) / 3 = 0.7795
```

**Đặc điểm:**
- ✅ Mỗi class có **trọng số bằng nhau** (không quan tâm số lượng samples)
- ✅ Phát hiện class yếu (neutral F1=0.48 kéo Macro xuống)
- ❌ Không phản ánh overall performance

**Khi nào dùng:**
- ✅ Mỗi class quan trọng như nhau (medical: mỗi bệnh đều quan trọng)
- ✅ Muốn cải thiện class yếu
- ✅ Classes imbalance (như bạn: neutral 5%, negative 59%)

**Ý nghĩa:**
> "Mô hình làm tốt **trung bình** trên từng class"  
> Nếu Macro thấp → có class rất yếu cần cải thiện

---

### ⚖️ **Weighted Average** (Trung Bình Có Trọng Số)

**Công thức:**
```
Weighted F1 = Σ(F1_class × số_samples_class) / tổng_samples
```

**Ví dụ từ model của bạn:**
```
Positive: F1=0.9087, samples=716  → 0.9087 × 716 = 650.6
Negative: F1=0.9508, samples=1192 → 0.9508 × 1192 = 1133.4
Neutral:  F1=0.4789, samples=106  → 0.4789 × 106 = 50.8

Weighted F1 = (650.6 + 1133.4 + 50.8) / 2014 = 0.9110
```

**Đặc điểm:**
- ✅ Gần với overall accuracy (91.10% ≈ 91.06%)
- ✅ Phản ánh performance trên toàn dataset
- ❌ Che giấu class yếu (neutral chỉ 5% → ít ảnh hưởng)

**Khi nào dùng:**
- ✅ Classes không quan trọng như nhau (VD: spam nhiều hơn important email)
- ✅ Production: quan tâm overall performance
- ❌ KHÔNG dùng khi muốn cải thiện class hiếm

**Ý nghĩa:**
> "Mô hình làm tốt **trên toàn bộ dataset**"  
> Nếu Weighted cao nhưng Macro thấp → có class hiếm bị bỏ rơi

---

### 🔍 **So Sánh Macro vs Weighted**

| Metric | Giá Trị | Ý Nghĩa |
|--------|---------|---------|
| **Macro Avg** | 0.7795 | Model trung bình 78% trên từng class (thấp vì neutral kéo xuống) |
| **Weighted Avg** | 0.9110 | Model đúng 91% trên toàn dataset (cao vì neutral chiếm ít) |

**Phân tích model của bạn:**
```
Weighted (0.91) >> Macro (0.78)
→ Model tốt trên positive/negative (chiếm 95% data)
→ Nhưng YẾU trên neutral (chỉ 5% data)
→ Cần cải thiện neutral!
```

---

## 📉 3. LOSS FUNCTIONS (Hàm Mất Mát)

### 🎯 **Loss Là Gì?**

**Định nghĩa:**  
Loss = Độ "sai" của model trên training data. Model học bằng cách **minimize loss**.

```
High Loss → Model dự đoán sai nhiều → Học chưa tốt
Low Loss  → Model dự đoán gần đúng → Học tốt
```

---

### 📊 **1. Cross-Entropy Loss** (Loss Cơ Bản)

**Công thức (cho 1 sample):**
```
Loss = -log(P_correct_class)

Nếu model tự tin đúng → P cao → Loss thấp
Nếu model không chắc → P thấp → Loss cao
```

**Ví dụ:**
```python
Sample: "Pin tốt" → Label: Positive

Trường hợp 1: Model dự đoán
  P(positive) = 0.9 → Loss = -log(0.9) = 0.105 ✅ (thấp)
  
Trường hợp 2: Model dự đoán  
  P(positive) = 0.3 → Loss = -log(0.3) = 1.204 ❌ (cao)
  
Trường hợp 3: Model hoàn toàn sai
  P(positive) = 0.01 → Loss = -log(0.01) = 4.605 ❌❌ (rất cao)
```

**Đặc điểm:**
- ✅ Standard cho classification
- ❌ Không xử lý class imbalance tốt

---

### 🔥 **2. Focal Loss** (Xử Lý Imbalance)

**Model của bạn đang dùng Focal Loss!**

**Công thức:**
```
Focal Loss = -α × (1 - P)^γ × log(P)

α (alpha): class weight (neutral cao, positive/negative thấp)
γ (gamma): focusing parameter (thường = 2)
```

**Cách hoạt động:**

```python
# Từ log của bạn:
Alpha weights:
  positive: 0.9360
  negative: 0.5644  
  neutral:  6.2515  ← Cao gấp 11x!

Gamma = 2.0

Ví dụ 1: Easy sample (model tự tin đúng)
  P = 0.95, class = neutral
  → Focal Loss = 6.2515 × (1-0.95)^2 × log(0.95)
                = 6.2515 × 0.0025 × 0.051
                = 0.0008 ✅ (rất thấp - bỏ qua easy samples)

Ví dụ 2: Hard sample (model không chắc)  
  P = 0.60, class = neutral
  → Focal Loss = 6.2515 × (1-0.60)^2 × log(0.60)
                = 6.2515 × 0.16 × 0.51
                = 0.51 ❌ (cao - focus vào hard samples)
```

**Ưu điểm:**
- ✅ **Alpha**: Tăng weight cho class hiếm (neutral)
- ✅ **Gamma**: Focus vào hard examples, bỏ qua easy examples
- ✅ Xử lý class imbalance tốt

**Tại sao cần:**
```
Neutral chỉ 5% data (501/9396)
→ Cross-Entropy Loss: Model học positive/negative nhiều hơn
→ Focal Loss: Buộc model chú ý đến neutral (alpha=6.25)
```

---

### 📈 **3. Eval Loss vs Training Loss**

**Training Loss:**
- Độ sai của model trên **training data**
- Giảm dần qua các epochs

**Eval Loss:**
- Độ sai của model trên **validation data** (data model chưa thấy)
- Đánh giá khả năng generalize

**Từ log của bạn:**
```
Epoch 1: train_loss=?, eval_loss=0.233  
Epoch 2: train_loss=?, eval_loss=0.207 ← BEST (thấp nhất)
Epoch 3: train_loss=?, eval_loss=0.214 ↑ (tăng nhẹ)
Epoch 4: train_loss=?, eval_loss=0.356 ↑↑ (tăng 70%!)

→ OVERFITTING! Model học thuộc training, không generalize
```

**Giải thích:**
```
Epoch 1-2: 
  Train loss ↓, Eval loss ↓
  → Model đang học tốt ✅

Epoch 3-4:
  Train loss ↓↓, Eval loss ↑↑  
  → Model học thuộc training data (overfitting) ❌
  → Cần dừng lại ở epoch 2!
```

---

## 📊 4. KHI NÀO DÙNG METRIC NÀO?

### 🎯 **Use Case: Sentiment Analysis (Model Của Bạn)**

| Mục tiêu | Metric | Tại sao |
|----------|--------|---------|
| **Overall performance** | Accuracy, Weighted F1 | Đánh giá tổng quan |
| **Mỗi class quan trọng như nhau** | Macro F1 | Phát hiện class yếu (neutral) |
| **Cải thiện class neutral** | F1-neutral, Recall-neutral | Focus vào class cụ thể |
| **Detect overfitting** | Eval Loss | Theo dõi loss tăng |

**Khuyến nghị cho bạn:**
```
Ưu tiên 1: Macro F1 (phát hiện neutral yếu)
Ưu tiên 2: F1 per class (neutral, positive, negative)
Ưu tiên 3: Eval Loss (tránh overfitting)
```

---

### 🏥 **Use Case: Medical Diagnosis**

```
Bệnh: Ung thư (hiếm, 2%), Không ung thư (98%)

Metric quan trọng: RECALL trên class ung thư
→ KHÔNG được bỏ sót bệnh nhân ung thư (FN = 0)
→ Có thể chấp nhận báo động nhầm (FP cao ổn)

Tệ nhất: Recall-cancer = 70% 
→ Bỏ sót 30% bệnh nhân → CHẾT!
```

---

### 🔍 **Use Case: Search Engine**

```
Query: "python tutorials"

Metric quan trọng: PRECISION@10 (top 10 results)
→ 10 kết quả đầu PHẢI chính xác
→ Có thể bỏ sót một số kết quả (Recall thấp ổn)

Tệ nhất: Precision = 20%
→ 8/10 kết quả không liên quan → User rời đi
```

---

### 🚨 **Use Case: Spam Filter**

```
2 loại lỗi:
1. False Positive: Email quan trọng → Spam folder ❌❌
2. False Negative: Spam → Inbox ❌

Metric quan trọng: PRECISION trên class spam
→ Chỉ đánh dấu spam khi chắc chắn
→ Có thể để một ít spam lọt (FN) vào inbox
```

---

## 🎯 5. ÁP DỤNG VÀO MODEL CỦA BẠN

### 📊 **Phân Tích Kết Quả Hiện Tại**

```
              precision  recall  f1-score  support

    positive     0.8934   0.9246    0.9087      716
    negative     0.9614   0.9404    0.9508     1192
     neutral     0.4766   0.4811    0.4789      106  ← VẤN ĐỀ

    accuracy                         0.9106     2014
   macro avg     0.7771   0.7820    0.7795     2014  ← Thấp vì neutral
weighted avg     0.9117   0.9106    0.9110     2014  ← Cao vì neutral ít
```

### 🔍 **Giải Thích Chi Tiết**

**1. Positive Class:**
```
Precision = 0.89 → Khi model nói "positive", đúng 89%
Recall = 0.92    → Tìm được 92% câu positive thực tế
F1 = 0.91        → Cân bằng tốt ✅

Ý nghĩa: Model làm TỐT trên positive
```

**2. Negative Class:**
```
Precision = 0.96 → Khi model nói "negative", đúng 96%
Recall = 0.94    → Tìm được 94% câu negative thực tế  
F1 = 0.95        → Rất tốt ✅

Ý nghĩa: Model làm XUẤT SẮC trên negative (class lớn nhất)
```

**3. Neutral Class:**
```
Precision = 0.48 → Khi model nói "neutral", CHỈ đúng 48% ❌
Recall = 0.48    → Chỉ tìm được 48% câu neutral thực tế ❌
F1 = 0.48        → RẤT THẤP ❌

Ý nghĩa: Model gần như ĐOÁN MÒ trên neutral (50-50)
Nguyên nhân: Neutral chỉ 106 samples (5.3%) → quá ít!
```

**4. Overall Metrics:**
```
Accuracy = 91.06%
→ Nhìn có vẻ tốt, nhưng...

Macro F1 = 0.78 (thấp)
→ Phát hiện ra neutral RẤT YẾU

Weighted F1 = 0.91 (cao)
→ Che giấu neutral (vì neutral chiếm ít)

KẾT LUẬN:
→ Model tốt trên positive/negative (95% data)
→ Model YẾU trên neutral (5% data)
→ CẦN cải thiện neutral!
```

---

### 🎯 **Mục Tiêu Cải Thiện**

| Metric | Hiện tại | Mục tiêu | Cách đạt |
|--------|----------|----------|----------|
| **Neutral F1** | 0.48 | **0.65-0.70** | Oversampling 30% |
| **Macro F1** | 0.78 | **0.83-0.85** | Cải thiện neutral |
| **Weighted F1** | 0.91 | **0.92** | Tự động tăng khi neutral tốt |
| **Eval Loss** | Tăng ở epoch 4 | **Không tăng** | Giảm epochs: 5→3 |

---

## 📚 TÓM TẮT

### 🎯 **Metrics - Khi Nào Dùng**

| Metric | Mục Đích | Khi Nào Dùng |
|--------|----------|--------------|
| **Accuracy** | Tỷ lệ đúng tổng thể | Classes cân bằng |
| **Precision** | "Nếu model nói Positive, tin được không?" | Không muốn False Positive (spam, legal) |
| **Recall** | "Model có bỏ sót không?" | Không muốn False Negative (medical, security) |
| **F1 Score** | Cân bằng Precision & Recall | Classes imbalance, so sánh models |
| **Macro Avg** | Trung bình trên mỗi class | Mỗi class quan trọng như nhau, phát hiện class yếu |
| **Weighted Avg** | Weighted theo số samples | Quan tâm overall performance |
| **Loss** | Độ sai của model | Detect overfitting, theo dõi training |

---

### 🔥 **Loss Functions**

| Loss | Đặc Điểm | Khi Dùng |
|------|----------|----------|
| **Cross-Entropy** | Standard loss | Classes cân bằng |
| **Focal Loss** | Alpha (class weight) + Gamma (focus hard) | Classes imbalance ✅ (Model của bạn) |

---

### ⚡ **Quick Reference**

```python
# Confusion Matrix
TP = True Positive   (dự đoán đúng positive)
TN = True Negative   (dự đoán đúng negative)
FP = False Positive  (dự đoán sai thành positive)
FN = False Negative  (dự đoán sai thành negative)

# Metrics
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)              # Dự đoán positive có chính xác không?
Recall    = TP / (TP + FN)              # Tìm được bao nhiêu % positive thực tế?
F1        = 2 × Precision × Recall / (Precision + Recall)

# Average
Macro Avg    = Trung bình đơn giản trên mỗi class
Weighted Avg = Trung bình có trọng số theo số samples
```

---

## 🎓 TÀI LIỆU THAM KHẢO

### 📖 **Đọc Thêm**
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [F1 Score Explained](https://en.wikipedia.org/wiki/F-score)
- [Focal Loss Paper](https://arxiv.org/abs/1708.02002)
- [Imbalanced Learning](https://imbalanced-learn.org/)

### 🎯 **Tip Cuối**

> **"Không có metric nào là hoàn hảo"**
> 
> - Accuracy cao không có nghĩa model tốt (nếu imbalance)
> - F1 cao trên class này, thấp trên class kia
> - Luôn nhìn NHIỀU metrics cùng lúc
> - Hiểu business context để chọn metric phù hợp
> 
> **Model của bạn:**
> - Accuracy = 91% (tốt)  
> - Nhưng Neutral F1 = 48% (tệ)
> - → Cần cải thiện neutral!

---

Hy vọng giúp bạn hiểu rõ hơn về các metrics! 🚀
