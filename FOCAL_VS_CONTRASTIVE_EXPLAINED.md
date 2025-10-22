# Focal Loss vs Contrastive Loss - Mối quan hệ

## 🎯 Câu hỏi:
> Khi Focal Loss giảm, Contrastive Loss phải giảm hay tăng?
> Tăng = đẩy xa, giảm = kéo về gần đúng không?

---

## ✅ Trả lời:

### **CẢ HAI LOSS ĐỀU GIẢM CÙNG NHAU trong quá trình training tốt!**

---

## 📊 Hiểu từng Loss:

### 1. **Focal Loss (Classification)**

**Mục đích:** Phân loại đúng sentiment cho từng aspect

**Giảm = TỐT:**
```
Focal Loss giảm → Model dự đoán đúng hơn

Epoch 1: Focal Loss = 0.800 → Sai nhiều
Epoch 5: Focal Loss = 0.350 → Đúng hơn ✅
Epoch 10: Focal Loss = 0.150 → Rất đúng ✅✅
```

**Tăng = TỆ:**
```
Focal Loss tăng → Model dự đoán sai nhiều hơn ❌
```

---

### 2. **Contrastive Loss (Representation Learning)**

**Mục đích:** Organize embeddings - kéo giống nhau lại gần, đẩy khác nhau ra xa

**Giảm = TỐT:**
```
Contrastive Loss giảm → Embeddings được organize tốt hơn

Epoch 1: Contr Loss = 0.800 → Chưa organize
         Similar samples: xa nhau
         Different samples: gần nhau
         
Epoch 5: Contr Loss = 0.500 → Organize tốt hơn ✅
         Similar samples: gần hơn
         Different samples: xa hơn
         
Epoch 10: Contr Loss = 0.300 → Rất tốt ✅✅
         Similar samples: rất gần
         Different samples: rất xa
```

**Công thức:**
```python
loss = -log(similar_pairs / all_pairs)

Loss cao → Similar pairs chưa gần nhau
Loss thấp → Similar pairs đã gần nhau ✅
```

---

## 🔄 Mối quan hệ trong Training:

### **Scenario Lý tưởng (Training tốt):**

```
Epoch | Focal Loss | Contr Loss | Ý nghĩa
------|------------|------------|------------------
  1   |   0.800    |   0.700    | Đầu training, cả 2 cao
  3   |   0.500    |   0.550    | ✅ Cả 2 giảm
  5   |   0.350    |   0.450    | ✅ Cả 2 giảm
  8   |   0.200    |   0.350    | ✅ Cả 2 giảm
 10   |   0.150    |   0.300    | ✅ Cả 2 thấp → TỐT!
```

**Kết luận:** CẢ HAI GIẢM = Model học tốt cả classification lẫn representation

---

### **Scenario Có vấn đề:**

#### **A. Focal giảm quá nhanh, Contrastive không giảm:**
```
Epoch | Focal Loss | Contr Loss | Vấn đề
------|------------|------------|------------------
  1   |   0.800    |   0.700    | 
  3   |   0.300    |   0.650    | Focal giảm nhanh
  5   |   0.150    |   0.600    | Contr không giảm ❌
  
→ Model overfitting classification, không học representation
→ Cần TĂNG contrastive_weight
```

#### **B. Contrastive giảm quá nhanh, Focal không giảm:**
```
Epoch | Focal Loss | Contr Loss | Vấn đề
------|------------|------------|------------------
  1   |   0.800    |   0.700    | 
  3   |   0.750    |   0.400    | Contr giảm nhanh
  5   |   0.700    |   0.300    | Focal không giảm ❌
  
→ Model học representation tốt nhưng không phân loại được
→ Cần TĂNG focal_weight
```

---

## 🧠 Hiểu sâu hơn: "Tăng = đẩy xa, Giảm = kéo gần"

### ❌ SAI LẦM phổ biến:

> "Contrastive Loss tăng = đẩy samples ra xa nhau"

### ✅ ĐÚNG:

**Contrastive Loss CAO:**
```
Loss cao = Chưa organize tốt
- Similar samples: VẪN Ở XA NHAU (chưa kéo gần được)
- Different samples: VẪN Ở GẦN NHAU (chưa đẩy xa được)
→ Cần training thêm để giảm loss
```

**Contrastive Loss THẤP:**
```
Loss thấp = Đã organize tốt
- Similar samples: ĐÃ ĐƯỢC KÉO GẦN ✅
- Different samples: ĐÃ ĐƯỢC ĐẨY XA ✅
→ Representation space tốt
```

---

## 📐 Công thức chi tiết:

### **Contrastive Loss:**
```python
# Simplified version
loss = -log(
    sum(similarity with positive pairs) /
    sum(similarity with all pairs)
)
```

**Khi loss GI��M:**
```
Before (Loss cao = 0.8):
  Sample A: [0.1, 0.2, ...]  (Battery=positive)
  Sample B: [0.9, 0.8, ...]  (Battery=positive)  ← Xa nhau!
  Similarity = 0.3  (thấp)
  
After training (Loss thấp = 0.3):
  Sample A: [0.1, 0.2, ...]  (Battery=positive)
  Sample B: [0.15, 0.25, ...]  (Battery=positive)  ← Gần nhau! ✅
  Similarity = 0.95  (cao)
  
→ Loss giảm VÌ similar samples được KÉO GẦN nhau
```

---

## 🎯 Trong project của bạn:

### **Current config:**
```yaml
focal_weight: 0.8         # 80% classification
contrastive_weight: 0.2   # 20% representation
```

### **Expected behavior:**

```
Good Training:
--------------
Epoch | Focal | Contr | F1    | Status
------|-------|-------|-------|--------
  1   | 0.65  | 0.64  | 88%   | Cả 2 cao
  3   | 0.45  | 0.58  | 92%   | ✅ Cả 2 giảm
  5   | 0.35  | 0.52  | 94%   | ✅ Cả 2 giảm
  8   | 0.28  | 0.48  | 95.5% | ✅ Cả 2 giảm
 10   | 0.25  | 0.45  | 96%   | ✅ Cả 2 thấp
 15   | 0.22  | 0.42  | 96.5% | ✅ Converged
```

**Nếu thấy:**
```
Bad Training 1:
--------------
Epoch | Focal | Contr | Problem
------|-------|-------|--------------------
  1   | 0.65  | 0.64  | 
  5   | 0.25  | 0.62  | ❌ Contr không giảm
  
→ Tăng contrastive_weight: 0.2 → 0.3


Bad Training 2:
--------------
Epoch | Focal | Contr | Problem
------|-------|-------|--------------------
  1   | 0.65  | 0.64  | 
  5   | 0.60  | 0.35  | ❌ Focal không giảm
  
→ Tăng focal_weight: 0.8 → 0.9
```

---

## 🔧 Cách điều chỉnh:

### **Rule of thumb:**

1. **Cả 2 loss giảm đều nhau:**
   ```
   ✅ Training tốt, giữ nguyên weights
   ```

2. **Focal giảm nhanh, Contrastive đứng yên:**
   ```
   ❌ Model overfitting classification
   → Tăng contrastive_weight
   → focal: 0.8 → 0.7
   → contrastive: 0.2 → 0.3
   ```

3. **Contrastive giảm nhanh, Focal đứng yên:**
   ```
   ❌ Model không học classification
   → Tăng focal_weight
   → focal: 0.8 → 0.9
   → contrastive: 0.2 → 0.1
   ```

4. **Cả 2 loss không giảm:**
   ```
   ❌ Learning rate quá thấp hoặc quá cao
   → Check learning rate
   → Check data quality
   ```

---

## 📊 Visualization:

### **Loss giảm = Representation space improve:**

```
Before Training (High Contrastive Loss):
========================================

   Battery=Pos    Battery=Neg
      🔴            🔵
         🔵      🔴
      🔴    🔵
         🔴

→ Mixed up, chưa tách biệt


After Training (Low Contrastive Loss):
======================================

   Battery=Pos          Battery=Neg
   
   🔴 🔴 🔴              🔵 🔵 🔵
   🔴 🔴 🔴              🔵 🔵 🔵
   🔴 🔴 🔴              🔵 🔵 🔵

→ Clustered, tách biệt rõ ràng ✅
```

---

## ✅ TÓM TẮT:

| Câu hỏi | Trả lời |
|---------|---------|
| Focal giảm, Contrastive nên? | **Cũng GIẢM** (không phải tăng!) |
| Loss tăng = đẩy xa? | ❌ SAI - Loss cao = chưa organize tốt |
| Loss giảm = kéo gần? | ✅ ĐÚNG - Loss thấp = đã organize tốt |
| Ideal training? | **Cả 2 loss đều GIẢM đều nhau** |
| Trade-off? | Có - nếu weights không cân bằng |

---

## 🎓 Key Takeaway:

> **CẢ HAI LOSS ĐỀU GIẢM = MODEL HỌC TỐT**
> 
> - Focal Loss giảm → Classification tốt
> - Contrastive Loss giảm → Representation tốt
> - Không phải "cái này giảm thì cái kia tăng"
> - Là "cả hai cùng giảm" cho model tốt nhất!

---

**Analogy:**
```
Học toán (Focal) và Học văn (Contrastive)

❌ SAI: "Học toán giỏi thì văn phải dốt"
✅ ĐÚNG: "Học cả 2 đều giỏi → Học sinh xuất sắc"

→ Focal giảm + Contrastive giảm = Model xuất sắc!
```
