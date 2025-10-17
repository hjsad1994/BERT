# Chiến Lược Xử Lý Class Imbalance: Oversampling + Focal Loss

## Vấn đề

Dữ liệu ABSA thường bị **class imbalance** nghiêm trọng:
- Positive: 60%
- Negative: 35%
- **Neutral: 5%** ← Minority class!

Imbalance ratio: **12x** (majority/minority)

## Giải pháp

Kết hợp 2 phương pháp:

### 1. **Oversampling** (Data-level)
- Duplicate random samples từ minority class (neutral)
- Target: Neutral ít nhất **20%** của majority class
- Giảm imbalance ratio từ 12x → 5x

**Ví dụ:**
```
TRƯỚC oversampling:
- Positive: 6000 (60%)
- Negative: 3500 (35%)  
- Neutral:   500 (5%)   ← Quá ít!

SAU oversampling (20%):
- Positive: 6000 (55.3%)
- Negative: 3500 (32.3%)
- Neutral:  1200 (11.1%) ← Đã tăng, nhưng vừa phải!
```

**So sánh 20% vs 40%:**
| Strategy | Target Neutral | Imbalance After | Ưu điểm | Nhược điểm |
|----------|---------------|----------------|---------|------------|
| **20%** | 1,200 samples | 5.0x | Ít duplicate, giảm overfitting | Model thấy neutral ít hơn |
| **40%** | 2,400 samples | 2.5x | Model thấy neutral nhiều | Nhiều duplicate, có thể memorize |

### 2. **Focal Loss** (Loss-level)
- Tăng trọng số loss cho minority class
- Alpha weights: Inverse frequency của classes
- Gamma = 2.0: Focus vào hard examples

**QUAN TRỌNG:** Alpha weights phải tính trên **DỮ LIỆU GỐC** (BEFORE oversampling), không phải sau oversampling!

## Tại sao Alpha phải dùng Original Counts?

### ❌ SAI: Tính alpha trên dữ liệu ĐÃ oversampled

```python
# SAU oversampling
label_counts = Counter(train_df_oversampled['sentiment'])
# positive: 6000, negative: 3500, neutral: 2400

alpha = [total/(3*count) for count in label_counts.values()]
# Alpha: [0.66, 1.13, 1.65] ← Gần bằng nhau!
```

**Vấn đề:** 
- Oversampling đã làm cân bằng data
- Alpha weights trở nên gần bằng nhau
- Focal Loss **mất tác dụng**!

### ✅ ĐÚNG: Tính alpha trên dữ liệu GỐC

```python
# TRƯỚC oversampling
label_counts_original = Counter(train_df_original['sentiment'])
# positive: 6000, negative: 3500, neutral: 500

alpha = [total/(3*count) for count in label_counts_original.values()]
# Alpha: [0.56, 0.95, 6.67] ← Phản ánh đúng imbalance!
```

**Lợi ích:**
- Alpha neutral = 6.67 (cao hơn positive 12x!)
- Phản ánh đúng imbalance gốc
- Focal Loss và Oversampling **bổ trợ nhau**, không chồng chéo

## Cách 2 phương pháp hoạt động cùng nhau

| Phương pháp | Tác động | Mục đích |
|-------------|----------|----------|
| **Oversampling** | Tăng số lượng samples minority | Model thấy minority class nhiều hơn trong training |
| **Focal Loss** | Tăng trọng số loss minority | Model chú ý hơn đến minority samples khi tính loss |

→ **Kết hợp**: Model vừa thấy nhiều samples neutral (oversampling), vừa chú ý nhiều hơn (focal loss)

## Code Implementation

```python
# 1. LƯU LẠI class counts GỐC (TRƯỚC oversampling)
class_counts_original = Counter(train_df['sentiment'])

# 2. Thực hiện oversampling
train_df_oversampled = random_oversample(train_df, ...)

# 3. Tính alpha weights DỰA TRÊN GỐC
label_counts = class_counts_original  # ← Key point!
total = sum(label_counts.values())

alpha = []
for label, idx in label_map.items():
    count = label_counts.get(label, 1)
    alpha[idx] = total / (len(label_map) * count)  # Inverse frequency

# 4. Tạo Focal Loss với alpha từ GỐC
focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
```

## Kết quả mong đợi

- Model học tốt hơn trên minority class (neutral)
- F1 score cân bằng hơn giữa các classes
- Giảm overfitting nhờ regularization mạnh hơn từ class weights

## Tài liệu tham khảo

- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [Learning from Imbalanced Data](https://www.jair.org/index.php/jair/article/view/10302)
