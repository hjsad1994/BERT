# 🎯 CÁCH MÔ HÌNH HỌC CÁC KHÍA CẠNH (ASPECTS)

## 📚 MỤC LỤC
1. [Tổng quan](#tổng-quan)
2. [Quá trình chuẩn bị dữ liệu](#quá-trình-chuẩn-bị-dữ-liệu)
3. [Cấu trúc input của model](#cấu-trúc-input-của-model)
4. [Quá trình training](#quá-trình-training)
5. [Quá trình inference](#quá-trình-inference)
6. [Ví dụ chi tiết](#ví-dụ-chi-tiết)

---

## 📖 TỔNG QUAN

### ❓ Câu hỏi: Model học aspect như thế nào?

**Câu trả lời ngắn gọn:**

Model **KHÔNG** tự động phát hiện aspect trong câu. Thay vào đó:

1. **Mỗi aspect được xem như một context riêng biệt**
2. **Model học mối quan hệ giữa câu + aspect → sentiment**
3. **Format input: `[CLS] câu [SEP] aspect [SEP]`**

Đây là **Aspect-Based Sentiment Analysis (ABSA)**, không phải aspect extraction!

---

## 🔄 QUÁ TRÌNH CHUẨN BỊ DỮ LIỆU

### 1. Dữ liệu gốc (Multi-label format)

File: `dataset.csv`

```
┌─────────────────────────────────────────────────────────────────────┐
│ data                          │ Battery  │ Camera   │ Performance  │
├───────────────────────────────┼──────────┼──────────┼──────────────┤
│ Pin tốt nhưng camera tệ       │ Positive │ Negative │              │
│ Máy nhanh, pin trâu           │ Positive │          │ Positive     │
│ Màn hình đẹp, giá hợp lý      │          │          │              │
└─────────────────────────────────────────────────────────────────────┘
```

**Đặc điểm:**
- 1 câu có nhiều aspect (multi-label)
- Mỗi aspect có sentiment riêng
- Nhiều aspect có thể rỗng (không được đề cập)

---

### 2. Chuyển đổi sang Single-label format

**Script:** `prepare_data.py`

**Quá trình:**

```python
# Input: 1 dòng với nhiều aspects
"Pin tốt nhưng camera tệ" | Battery: Positive | Camera: Negative

# Output: Nhiều samples (1 sample/aspect)
Sample 1: "Pin tốt nhưng camera tệ" + "Battery" → Positive
Sample 2: "Pin tốt nhưng camera tệ" + "Camera"  → Negative
```

**Kết quả:**

```
┌──────────────────────────────────┬──────────┬────────────┐
│ sentence                         │ aspect   │ sentiment  │
├──────────────────────────────────┼──────────┼────────────┤
│ Pin tốt nhưng camera tệ          │ Battery  │ positive   │
│ Pin tốt nhưng camera tệ          │ Camera   │ negative   │
│ Máy nhanh, pin trâu              │ Battery  │ positive   │
│ Máy nhanh, pin trâu              │ Performance │ positive│
└──────────────────────────────────┴──────────┴────────────┘
```

**Code thực hiện:**

```python
# prepare_data.py - Hàm convert_to_single_label()
for idx, row in df.iterrows():
    sentence = row['data']
    
    # Lặp qua TỪNG aspect column
    for aspect in aspect_columns:
        sentiment_value = row[aspect]
        
        # Bỏ qua aspect rỗng (không được đề cập)
        if pd.isna(sentiment_value):
            continue
        
        # Tạo 1 sample mới: sentence + aspect + sentiment
        absa_samples.append({
            'sentence': sentence,
            'aspect': aspect,
            'sentiment': normalized_sentiment
        })
```

**Thống kê:**
```
Original dataset:    4,021 rows (multi-label)
                        ↓
After conversion:    7,713 samples (single-label)
                        ↓
                Train: 5,399 samples
                  Val: 1,157 samples
                 Test: 1,157 samples
```

---

## 🎨 CẤU TRÚC INPUT CỦA MODEL

### Format Input: Sentence + Aspect Pair

**Model sử dụng:** ViSoBERT (Vietnamese BERT)

**Input format:**
```
[CLS] sentence [SEP] aspect [SEP]
```

### Ví dụ cụ thể:

#### Input 1:
```
Sentence: "Pin tốt nhưng camera tệ"
Aspect:   "Battery"

Tokenized input:
[CLS] Pin tốt nhưng camera tệ [SEP] Battery [SEP]
```

#### Input 2:
```
Sentence: "Pin tốt nhưng camera tệ"  (CÙng câu!)
Aspect:   "Camera"                    (Khác aspect!)

Tokenized input:
[CLS] Pin tốt nhưng camera tệ [SEP] Camera [SEP]
```

**→ Cùng 1 câu nhưng khác aspect = khác training sample!**

---

### Code thực hiện (ABSADataset):

```python
# utils.py - Class ABSADataset
def __getitem__(self, idx):
    row = self.dataframe.iloc[idx]
    
    sentence = str(row['sentence'])
    aspect = str(row['aspect'])
    label = int(row['label_id'])  # 0=positive, 1=negative, 2=neutral
    
    # Format: [CLS] sentence [SEP] aspect [SEP]
    encoding = self.tokenizer(
        sentence,      # Text A
        aspect,        # Text B
        add_special_tokens=True,  # Thêm [CLS], [SEP]
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoding['input_ids'],        # Token IDs
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding['token_type_ids'], # 0=sentence, 1=aspect
        'labels': torch.tensor(label)                  # Ground truth
    }
```

---

### Cách BERT xử lý:

```
[CLS] Pin tốt nhưng camera tệ [SEP] Battery [SEP]
  ↓     ↓   ↓    ↓     ↓     ↓   ↓    ↓       ↓   ↓
Token:  CLS Pin tốt  nhưng cam  tệ  SEP  Bat   SEP
Seg:    0   0   0     0     0    0   0    1    1    ← Segment IDs
Attn:   1   1   1     1     1    1   1    1    1    ← Attention mask
```

**Segment IDs (token_type_ids):**
- `0`: Thuộc sentence (text A)
- `1`: Thuộc aspect (text B)

**→ Model học được context của cả sentence VÀ aspect!**

---

## 🎓 QUÁ TRÌNH TRAINING

### Bước 1: Forward Pass

```
Input:  [CLS] Pin tốt [SEP] Battery [SEP]
           ↓
    ViSoBERT Encoder (12 layers)
           ↓
    [CLS] representation (768-dim)
           ↓
    Classification Head (Linear layer)
           ↓
    Logits: [2.1, -1.5, 0.3]  ← 3 classes: positive, negative, neutral
           ↓
    Softmax
           ↓
    Probabilities: [0.85, 0.05, 0.10]
```

### Bước 2: Loss Calculation

**Loss function:** Focal Loss (xử lý class imbalance)

```python
# Focal Loss = -α(1-pt)^γ * log(pt)
# α: Class weights (neutral class được boost)
# γ: Focusing parameter (γ=2.0)

Ground truth: positive (label=0)
Predicted:    [0.85, 0.05, 0.10]

Loss = FocalLoss(predicted, ground_truth)
     = -α[0] * (1-0.85)^2 * log(0.85)
     = Low loss (prediction correct!)
```

### Bước 3: Backpropagation

```
Loss → Gradients → Update weights
  ↓
Model learns:
  1. "Pin tốt" + Battery → Positive (✓)
  2. "camera tệ" + Camera → Negative (✓)
  3. "Pin tốt" + Camera → ??? (confusing context)
```

---

### Model học gì?

**Model học mối quan hệ:**

```
sentence + aspect → sentiment
```

**Không phải:**
- ❌ Tự động tìm aspect trong câu
- ❌ Extract aspect từ text
- ❌ Phân loại sentiment toàn câu

**Mà là:**
- ✅ Hiểu context của câu
- ✅ Focus vào aspect được chỉ định
- ✅ Phân loại sentiment cho aspect đó

---

### Ví dụ Model học:

#### Training sample 1:
```
Input:  "Pin trâu lắm, rất hài lòng" + "Battery"
Label:  Positive
→ Model học: "Pin trâu" + Battery = Positive
```

#### Training sample 2:
```
Input:  "Pin trâu nhưng camera tệ" + "Battery"
Label:  Positive
→ Model học: Cần focus "Pin trâu" phần đầu, ignore "camera tệ"
```

#### Training sample 3:
```
Input:  "Pin trâu nhưng camera tệ" + "Camera"
Label:  Negative
→ Model học: Cần focus "camera tệ" phần sau, ignore "Pin trâu"
```

**→ Model học được ATTENTION mechanism để focus đúng phần liên quan!**

---

## 🔮 QUÁ TRÌNH INFERENCE

### Kịch bản: Test với câu mới

**Input:** "Sản phẩm tốt, pin trâu lắm"

**Bước 1:** Phải test với TỪNG aspect riêng biệt

```python
# test_sentiment_smart.py
aspects = ['Battery', 'Camera', 'Performance', ...]

for aspect in aspects:
    # Create input
    input_text = f"[CLS] {sentence} [SEP] {aspect} [SEP]"
    
    # Tokenize
    inputs = tokenizer(sentence, aspect, ...)
    
    # Predict
    outputs = model(**inputs)
    logits = outputs.logits
    probs = softmax(logits)
    
    # Get prediction
    predicted_class = argmax(probs)
    confidence = max(probs)
    
    print(f"{aspect}: {predicted_class} ({confidence:.2%})")
```

### Kết quả:

```
┌──────────────┬───────────┬────────────┐
│ Aspect       │ Sentiment │ Confidence │
├──────────────┼───────────┼────────────┤
│ Battery      │ Positive  │ 95.2%      │  ← Có "pin trâu"
│ Camera       │ Positive  │ 62.3%      │  ← General "tốt"
│ Performance  │ Positive  │ 58.7%      │  ← General "tốt"
│ Display      │ Positive  │ 51.2%      │  ← General "tốt"
│ Price        │ Positive  │ 49.8%      │  ← Not mentioned
└──────────────┴───────────┴────────────┘
```

**Vấn đề:** Model predict TẤT CẢ aspects (kể cả không được đề cập)!

---

### Giải pháp: Smart Inference (Aspect Relevance Detection)

**File:** `test_sentiment_smart.py`

**Idea:** Chỉ hiển thị aspects THỰC SỰ được đề cập

```python
# Bước 1: Keyword matching
def is_aspect_relevant(sentence, aspect):
    keywords = ASPECT_KEYWORDS[aspect]
    
    # Check if any keyword present
    for keyword in keywords:
        if keyword in sentence.lower():
            return True
    
    return False

# Bước 2: Confidence filtering
MIN_CONFIDENCE = 0.70  # Threshold

# Bước 3: Filter results
relevant_aspects = []
for aspect, sentiment, confidence in predictions:
    if is_aspect_relevant(sentence, aspect) and confidence >= MIN_CONFIDENCE:
        relevant_aspects.append((aspect, sentiment, confidence))
```

**Kết quả sau khi filter:**

```
┌──────────────┬───────────┬────────────┐
│ Aspect       │ Sentiment │ Confidence │
├──────────────┼───────────┼────────────┤
│ Battery      │ Positive  │ 95.2%      │  ✓ Relevant!
│ General      │ Positive  │ 89.1%      │  ✓ "tốt"
└──────────────┴───────────┴────────────┘
```

**→ Chỉ hiển thị aspects được đề cập!**

---

## 📝 VÍ DỤ CHI TIẾT

### Ví dụ 1: Câu đơn giản

**Input:** "Pin tốt"

#### Training data tạo ra:

```
Sample 1: sentence="Pin tốt", aspect="Battery", sentiment=positive
Sample 2: sentence="Pin tốt", aspect="Camera", sentiment=??? (không có)
...
```

**→ Chỉ có Battery được label, các aspect khác bỏ qua trong data prep!**

#### Inference:

```python
# Test với Battery
Input: [CLS] Pin tốt [SEP] Battery [SEP]
→ Prediction: Positive (95%)  ✓ Correct!

# Test với Camera
Input: [CLS] Pin tốt [SEP] Camera [SEP]
→ Prediction: Positive (52%)  ✗ Wrong! (nhưng low confidence)
```

**Smart filter:**
- Battery: ✓ Hiển thị (có keyword "pin")
- Camera: ✗ Ẩn (không có keyword)

---

### Ví dụ 2: Câu phức tạp

**Input:** "Pin trâu nhưng camera tệ, màn hình đẹp"

#### Training data:

```
Sample 1: "Pin trâu nhưng camera tệ, màn hình đẹp" + Battery → Positive
Sample 2: "Pin trâu nhưng camera tệ, màn hình đẹp" + Camera → Negative
Sample 3: "Pin trâu nhưng camera tệ, màn hình đẹp" + Display → Positive
```

#### Model học cách attention:

```
Sentence: Pin trâu nhưng camera tệ , màn hình đẹp
Aspect:   Battery

Attention weights (simplified):
  Pin:    0.35  ← High attention!
  trâu:   0.28  ← High attention!
  nhưng:  0.05
  camera: 0.03
  tệ:     0.02
  ...

→ Model focuses on "Pin trâu" for Battery aspect!
```

```
Sentence: Pin trâu nhưng camera tệ , màn hình đẹp
Aspect:   Camera

Attention weights:
  Pin:    0.03
  trâu:   0.02
  nhưng:  0.05
  camera: 0.38  ← High attention!
  tệ:     0.32  ← High attention!
  ...

→ Model focuses on "camera tệ" for Camera aspect!
```

---

### Ví dụ 3: Câu không đề cập aspect

**Input:** "Sản phẩm tốt"

#### Inference:

```
Test với Battery:
Input: [CLS] Sản phẩm tốt [SEP] Battery [SEP]
→ Prediction: Positive (45%)  ← Low confidence!

Test với Camera:
Input: [CLS] Sản phẩm tốt [SEP] Camera [SEP]
→ Prediction: Positive (47%)  ← Low confidence!

Test với General:
Input: [CLS] Sản phẩm tốt [SEP] General [SEP]
→ Prediction: Positive (92%)  ← High confidence!
```

**Smart filter:**
- Battery: ✗ Ẩn (không có keyword + low confidence)
- Camera: ✗ Ẩn (không có keyword + low confidence)
- General: ✓ Hiển thị (có keyword "sản phẩm", "tốt" + high confidence)

---

## 🎯 TÓM TẮT CÁCH MODEL HỌC ASPECTS

### 1. Chuẩn bị dữ liệu:
```
Multi-label → Single-label
1 câu với nhiều aspects → Nhiều samples (1 sample/aspect)
```

### 2. Training:
```
Input:  [CLS] sentence [SEP] aspect [SEP]
Model:  ViSoBERT learns sentence+aspect → sentiment
Output: 3 logits (positive, negative, neutral)
```

### 3. Học gì:
```
✓ Context understanding (hiểu ngữ cảnh câu)
✓ Attention mechanism (focus vào phần liên quan)
✓ Aspect-specific sentiment (sentiment riêng cho aspect)
✗ KHÔNG tự động phát hiện aspect
✗ KHÔNG extract aspect từ text
```

### 4. Inference:
```
1. Test với TỪNG aspect riêng biệt
2. Get predictions cho tất cả aspects
3. Filter bằng:
   - Keyword matching (aspect có được đề cập?)
   - Confidence threshold (prediction có chắc chắn?)
4. Chỉ hiển thị relevant aspects
```

### 5. Kỹ thuật đặc biệt:
```
✓ Focal Loss: Xử lý class imbalance (neutral chỉ 5.7%)
✓ Oversampling: Tăng neutral samples (305 → 1,244)
✓ Smart inference: Keyword-based relevance detection
✓ Checkpoint naming: Dễ tìm best model (checkpoint-92)
```

---

## 🔍 DEEP DIVE: BERT ATTENTION MECHANISM

### Cách BERT học focus vào aspect:

```
Input: [CLS] Pin trâu nhưng camera tệ [SEP] Battery [SEP]

Layer 1 (Low-level):
  ↓
  Learns word relationships: "Pin" relates to "Battery"

Layer 6 (Mid-level):
  ↓
  Learns context: "trâu" is positive indicator for "Battery"

Layer 12 (High-level):
  ↓
  [CLS] representation encodes:
    - Sentence context
    - Aspect focus (Battery)
    - Sentiment polarity
  ↓
Classification Head
  ↓
Prediction: Positive
```

### Attention visualization (simplified):

```
Query: Battery aspect
Keys:  Words in sentence

Attention scores:
  [CLS]    ██░░░░░░░░ 0.08
  Pin      ████████░░ 0.42  ← High!
  trâu     ██████░░░░ 0.35  ← High!
  nhưng    ██░░░░░░░░ 0.05
  camera   █░░░░░░░░░ 0.03  ← Low (not relevant)
  tệ       █░░░░░░░░░ 0.02  ← Low (not relevant)
  [SEP]    █░░░░░░░░░ 0.02
  Battery  █░░░░░░░░░ 0.03

→ Model attends to "Pin trâu" when aspect is Battery!
```

---

## 📊 SỐ LIỆU THỐNG KÊ

### Dataset:
```
Original:    4,021 rows (multi-label)
Converted:   7,713 samples (single-label)
Average:     1.92 aspects/sentence

Split:
  Train:       5,399 samples (70%)
  Validation:  1,157 samples (15%)
  Test:        1,157 samples (15%)
```

### Aspects:
```
Total aspects: 13
  - Battery, Camera, Performance, Display, Design
  - Software, Packaging, Price, Warranty
  - Shop_Service, Shipping, General, Others
```

### Sentiment distribution:
```
Negative:    57.6%  (4,445 samples)
Positive:    36.7%  (2,833 samples)
Neutral:      5.7%  (435 samples)  ← Imbalanced!
```

### After Oversampling:
```
Negative:    47.8%  (3,111 samples)  ← Keep
Positive:    30.5%  (1,983 samples)  ← Keep
Neutral:     19.1%  (1,244 samples)  ← Oversampled from 305
Total:       6,338 samples
```

---

## 💡 KEY INSIGHTS

### 1. Tại sao dùng format này?

**Lý do:**
- ✅ Cho phép model học context-aware sentiment
- ✅ Cùng câu có thể có sentiment khác nhau cho aspects khác nhau
- ✅ Tận dụng BERT's contextualized embeddings
- ✅ Không cần aspect extraction module riêng

### 2. Ưu điểm:

- ✅ **Accuracy cao:** 91.1% (baseline), ~92-93% (with Focal Loss)
- ✅ **Flexible:** Dễ thêm/bớt aspects
- ✅ **Robust:** Xử lý được câu phức tạp
- ✅ **Context-aware:** Hiểu được từ nào liên quan aspect nào

### 3. Nhược điểm:

- ❌ **Phải test nhiều lần:** Mỗi aspect = 1 forward pass
- ❌ **Cần keyword matching:** Để filter aspects không liên quan
- ❌ **Không tự động extract:** Phải biết trước list aspects

### 4. Khi nào dùng approach này?

**Phù hợp khi:**
- ✓ Biết trước list aspects cần phân tích
- ✓ Cần phân tích sentiment chi tiết cho từng aspect
- ✓ Có labeled data cho aspects
- ✓ Câu văn có thể đề cập nhiều aspects với sentiments khác nhau

**Không phù hợp khi:**
- ✗ Cần tự động discover aspects mới
- ✗ Aspects không được định nghĩa trước
- ✗ Chỉ cần overall sentiment

---

## 🎓 KIẾN THỨC BỔ SUNG

### ABSA vs. Sentiment Analysis

| Aspect | ABSA (Project này) | Standard SA |
|--------|-------------------|-------------|
| **Input** | Sentence + Aspect | Sentence only |
| **Output** | Aspect-specific sentiment | Overall sentiment |
| **Complexity** | High | Low |
| **Use case** | Product reviews (chi tiết) | General sentiment |

### Format alternatives:

**Format 1 (Project này):**
```
[CLS] sentence [SEP] aspect [SEP]
```

**Format 2 (Concat):**
```
[CLS] aspect: sentence [SEP]
Example: [CLS] Battery: Pin tốt quá [SEP]
```

**Format 3 (Prompt):**
```
[CLS] What is the sentiment for {aspect}? {sentence} [SEP]
```

**→ Format 1 được chứng minh hiệu quả nhất cho ABSA!**

---

## 📚 TÀI LIỆU THAM KHẢO

1. **BERT Paper:**
   - Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)

2. **ABSA:**
   - Pontiki et al. "SemEval-2014 Task 4: Aspect Based Sentiment Analysis"

3. **ViSoBERT:**
   - "5CD-AI/Vietnamese-Sentiment-visobert" (HuggingFace model)

4. **Focal Loss:**
   - Lin et al. "Focal Loss for Dense Object Detection" (2017)

---

## 🎉 KẾT LUẬN

**Model KHÔNG tự động phát hiện aspects!**

**Thay vào đó:**

1. ✅ Nhận input: Sentence + Aspect cố định
2. ✅ Học mối quan hệ: (Sentence, Aspect) → Sentiment
3. ✅ Sử dụng attention để focus vào phần liên quan
4. ✅ Predict sentiment cho aspect đó

**Inference:**
- Test với TẤT CẢ aspects (14 aspects)
- Filter bằng keyword matching + confidence
- Chỉ hiển thị aspects relevant

**Kết quả:**
- Accuracy: 91-93%
- F1 Score: 90-93%
- Neutral F1: 80-85% (sau Focal Loss + Oversampling)

---

**🚀 Project hoàn chỉnh và ready to use!**
