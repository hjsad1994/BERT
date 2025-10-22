# ViSoBERT + Focal Loss + Contrastive Loss - Kiến trúc Chi tiết

## 🎯 TRẢ LỜI CÂU HỎI:

> "2 loss này chạy cùng lúc hay riêng lẻ? Focal trước hay Contrastive trước?"

### ✅ **CHẠY CÙNG LÚC (SIMULTANEOUSLY)!**

**KHÔNG phải:**
- ❌ Focal trước, rồi Contrastive
- ❌ Contrastive trước, rồi Focal
- ❌ Chạy 2 lần forward pass riêng lẻ

**MÀ LÀ:**
- ✅ 1 lần forward pass → 2 outputs (logits + embeddings)
- ✅ 2 losses tính CÙNG LÚC từ 2 outputs đó
- ✅ Kết hợp lại thành 1 loss duy nhất
- ✅ 1 lần backprop update TẤT CẢ weights

---

## 📊 KIẾN TRÚC MODEL

### **Toàn cảnh:**

```
INPUT TEXT ("Pin tot camera xau")
    |
    v
┌─────────────────────────┐
│   ViSoBERT ENCODER      │  ← Shared encoder
│   [batch, 768]          │
└─────────────────────────┘
          |
          v
    ┌─────┴─────┐  ← CHIA 2 NHÁNH (PARALLEL)
    |           |
    v           v
┌─────────┐ ┌──────────────┐
│ CLASS   │ │ PROJECTION   │
│ HEAD    │ │ HEAD         │
│         │ │              │
│ Logits  │ │ Embeddings   │
│[b,11,3] │ │ [b, 256]     │
└─────────┘ └──────────────┘
    |           |
    v           v
┌─────────┐ ┌──────────────┐
│ FOCAL   │ │ CONTRASTIVE  │  ← 2 LOSSES (PARALLEL)
│ LOSS    │ │ LOSS         │
└─────────┘ └──────────────┘
    |           |
    └─────┬─────┘
          v
    ┌─────────────┐
    │ COMBINED    │  ← 0.8*Focal + 0.2*Contr
    │ LOSS        │
    └─────────────┘
          |
          v
    ┌─────────────┐
    │ BACKPROP    │  ← Update TẤT CẢ weights
    └─────────────┘
```

---

## 🔄 FORWARD PASS FLOW (Step-by-Step)

### **Step 1: Input**
```python
text = "Pin tot camera xau"
input_ids = tokenizer(text)  # [batch_size, seq_len]
```

### **Step 2: Encoder (ViSoBERT)**
```python
hidden_states = visobert(input_ids, attention_mask)
pooled_output = hidden_states[:, 0, :]  # [CLS] token
# Shape: [batch_size, 768]
```

### **Step 3: PARALLEL - Two Branches**

**Branch A: Classification Head (cho Focal Loss)**
```python
# Dense layer
x = self.dense(pooled_output)  # [batch, 512]
x = F.relu(x)
x = self.dropout(x)

# Output layer
logits = self.out(x)  # [batch, 11, 3]
# 11 aspects × 3 sentiments
```

**Branch B: Projection Head (cho Contrastive Loss)**
```python
# Projection
embeddings = self.projection(pooled_output)  # [batch, 256]

# L2 normalize for contrastive learning
embeddings = F.normalize(embeddings, p=2, dim=1)
```

**⚠️ QUAN TRỌNG:** Cả 2 branches này chạy CÙNG LÚC trong 1 forward pass!

### **Step 4: Calculate Losses (SIMULTANEOUSLY)**

**Focal Loss (từ logits):**
```python
focal_loss = 0
for i in range(11):  # Mỗi aspect
    aspect_logits = logits[:, i, :]  # [batch, 3]
    aspect_labels = labels[:, i]     # [batch]
    
    loss = focal_loss_fn(aspect_logits, aspect_labels)
    focal_loss += loss

focal_loss = focal_loss / 11  # Average
```

**Contrastive Loss (từ embeddings):**
```python
contr_loss = contrastive_loss_fn(embeddings, labels)
# Pulls similar samples together
# Pushes different samples apart
```

**⚠️ QUAN TRỌNG:** 2 losses này tính CÙNG LÚC, KHÔNG phải tuần tự!

### **Step 5: Combine**
```python
# Weighted combination
total_loss = 0.8 * focal_loss + 0.2 * contr_loss
# (weights từ config.yaml)
```

### **Step 6: Backpropagation**
```python
# Single backprop
total_loss.backward()  # Updates ALL weights:
                       # - ViSoBERT encoder
                       # - Classification head
                       # - Projection head

optimizer.step()  # Apply updates
```

---

## ⏱️ TIMELINE (trong 1 iteration)

```
Time    |  Action
--------|--------------------------------------------------
0ms     |  Load batch (input_ids, labels)
        |
5ms     |  Forward: ViSoBERT encoder
        |     ↓
10ms    |  ┌──────────────┬──────────────┐
        |  │ Class head   │ Proj head    │  (PARALLEL)
15ms    |  │ → logits     │ → embeddings │
        |  └──────────────┴──────────────┘
        |     ↓               ↓
20ms    |  ┌──────────────┬──────────────┐
        |  │ Focal loss   │ Contr loss   │  (PARALLEL)
25ms    |  └──────────────┴──────────────┘
        |     ↓
30ms    |  Combine: total_loss
        |     ↓
50ms    |  Backprop: total_loss.backward()
        |     ↓
70ms    |  optimizer.step()
        |
        |  DONE! (Total: ~70ms per batch)
```

**Nếu chạy tuần tự (WRONG):**
```
Would need: ~100-120ms (slower!)
Because: 
  - Forward 1: Focal (50ms)
  - Backprop 1: Focal (30ms)
  - Forward 2: Contrastive (50ms)
  - Backprop 2: Contrastive (30ms)
  = 160ms total
```

---

## 💡 TẠI SAO CHẠY CÙNG LÚC?

### **1. Hiệu quả (Efficiency)**
- ✅ 1 forward pass thay vì 2
- ✅ 1 backprop thay vì 2
- ✅ Nhanh hơn ~2x

### **2. Học tốt hơn (Better Learning)**
- ✅ Model học CẢ classification VÀ representation cùng lúc
- ✅ Gradient từ 2 losses bổ sung cho nhau
- ✅ Encoder được optimize cho CẢ 2 tasks

### **3. Implementation đơn giản**
```python
# Chỉ cần 1 forward pass:
logits, embeddings = model(input_ids, attention_mask, 
                          return_embeddings=True)

# Tính cả 2 losses:
focal_loss = focal_loss_fn(logits, labels)
contr_loss = contr_loss_fn(embeddings, labels)

# Combine:
total_loss = 0.8 * focal_loss + 0.2 * contr_loss

# Single backprop:
total_loss.backward()
```

---

## 🔍 CHI TIẾT IMPLEMENTATION

### **Model Forward:**

```python
class MultiLabelViSoBERTFocalContrastive(nn.Module):
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        # 1. Encoder
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.pooler_output  # [batch, 768]
        
        # 2a. Classification branch
        x = self.dense(pooled)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.out(x)  # [batch, 11, 3]
        
        if return_embeddings:
            # 2b. Projection branch
            embeddings = self.projection(pooled)  # [batch, 256]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return logits, embeddings
        
        return logits
```

### **Training Loop:**

```python
for batch in dataloader:
    # 1. Forward (BOTH branches)
    logits, embeddings = model(input_ids, attention_mask, 
                              return_embeddings=True)
    
    # 2. Calculate BOTH losses (SIMULTANEOUSLY)
    focal_loss = focal_loss_fn(logits, labels)
    contr_loss = contr_loss_fn(embeddings, labels)
    
    # 3. Combine
    total_loss = 0.8 * focal_loss + 0.2 * contr_loss
    
    # 4. Single backprop
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

---

## 📊 LỢI ÍCH CỦA TỪNG LOSS

### **Focal Loss (từ Logits)**
**Mục đích:** Classification tốt hơn
```
Input:  Logits [batch, 11, 3]
Output: Loss scalar

Làm gì:
- Focus vào hard examples (sai nhiều)
- Giảm weight cho easy examples (đúng rồi)
- Handle class imbalance

Ví dụ:
  Easy:  Predicted [0.9, 0.05, 0.05], True=0
         → Weight thấp (0.1)
  
  Hard:  Predicted [0.4, 0.3, 0.3], True=0
         → Weight cao (2.5)
         → Model focus học cái này
```

### **Contrastive Loss (từ Embeddings)**
**Mục đích:** Representation learning
```
Input:  Embeddings [batch, 256]
Output: Loss scalar

Làm gì:
- Pull similar samples together
- Push different samples apart
- Organize embedding space

Ví dụ:
  Sample A: "Pin tot" → embedding [0.1, 0.2, ...]
  Sample B: "Pin ngon" → embedding [0.12, 0.22, ...]
  Sample C: "Pin te" → embedding [-0.5, -0.4, ...]
  
  Contrastive loss:
    - Pull A & B closer (cùng positive)
    - Push A & C apart (khác sentiment)
```

---

## 🎯 VỊ TRÍ THAM GIA

### **Trong Forward Pass:**

```
INPUT
  ↓
ENCODER (ViSoBERT)
  ↓
  ├─→ Classification Head → LOGITS → FOCAL LOSS
  │
  └─→ Projection Head → EMBEDDINGS → CONTRASTIVE LOSS
```

**Cả 2 branches đều:**
- ✅ Bắt đầu sau encoder
- ✅ Chạy parallel (không phải sequential)
- ✅ Produce outputs cùng lúc
- ✅ Losses tính cùng lúc

### **Trong Training:**

```
Epoch 1:
  Batch 1: ViSoBERT → [Logits, Embeddings] → [Focal, Contr] → Combined → Backprop
  Batch 2: ViSoBERT → [Logits, Embeddings] → [Focal, Contr] → Combined → Backprop
  ...
  
Epoch 2:
  Batch 1: ViSoBERT → [Logits, Embeddings] → [Focal, Contr] → Combined → Backprop
  ...
```

**Mỗi batch:**
- ✅ 1 forward pass
- ✅ 2 losses calculated simultaneously
- ✅ 1 backprop

---

## 📈 KẾT QUẢ

### **Focal Loss Only:**
```
Focus: Classification
Result: ~95.0% F1
Problem: Representation không tốt
```

### **Contrastive Loss Only:**
```
Focus: Representation
Result: ~95.4% F1
Problem: Classification không optimal
```

### **Focal + Contrastive (Combined):**
```
Focus: BOTH Classification + Representation
Result: ~96.0% F1
Benefit: Best of both worlds! ✅
```

---

## ✅ TÓM TẮT

| Câu hỏi | Trả lời |
|---------|---------|
| Chạy cùng lúc hay riêng lẻ? | **CÙNG LÚC** |
| Focal trước hay Contrastive trước? | **KHÔNG có trước/sau - PARALLEL** |
| Mấy lần forward pass? | **1 LẦN** |
| Mấy lần backprop? | **1 LẦN** |
| Tham gia giai đoạn nào? | **SAU Encoder, trước Backprop** |
| Có chậm hơn single loss không? | **KHÔNG - vẫn nhanh tương đương** |

---

## 🎓 KEY INSIGHTS:

1. **Model có 2 heads (branches) từ shared encoder**
   - Classification head → Logits
   - Projection head → Embeddings

2. **Single forward pass produces BOTH outputs**
   - Không phải 2 forward passes riêng
   - Efficient và nhanh

3. **2 losses calculated SIMULTANEOUSLY**
   - Focal từ logits
   - Contrastive từ embeddings
   - Kết hợp thành 1 loss

4. **Single backprop updates EVERYTHING**
   - Encoder weights
   - Classification head weights
   - Projection head weights
   - Tất cả learn cùng lúc

5. **Complementary learning**
   - Focal: Improve classification
   - Contrastive: Improve representation
   - Together: Better than either alone

---

**Xem visualizations:** 
- `model_architecture.png` - Kiến trúc đầy đủ
- `forward_pass_timeline.png` - Timeline chi tiết
- `sequential_vs_parallel.png` - So sánh sai vs đúng
