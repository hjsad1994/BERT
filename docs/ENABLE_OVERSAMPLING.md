# Cách bật lại Oversampling

Oversampling hiện đang **DISABLED** (commented out) trong `train.py`.

## Để bật lại Oversampling:

1. Mở file `train.py`
2. Tìm đến section:
   ```python
   # ============================================================
   # OVERSAMPLING CODE - COMMENTED OUT (Uncomment to enable)
   # ============================================================
   ```

3. **Uncomment** toàn bộ code từ dòng:
   ```python
   # from oversampling_utils import random_oversample
   ```
   
   Đến dòng:
   ```python
   # ============================================================
   # END OVERSAMPLING CODE
   # ============================================================
   ```

4. Xóa dấu `#` ở đầu mỗi dòng để bật lại code

5. Cập nhật lại header message:
   ```python
   # Đổi từ:
   print("📈 OVERSAMPLING - DISABLED (Chỉ dùng Focal Loss)")
   
   # Thành:
   print("📈 OVERSAMPLING - Xử lý class imbalance...")
   ```

6. Cập nhật message trong phần Trainer:
   ```python
   # Đổi từ:
   print(f"   • Oversampling: DISABLED (không dùng)")
   
   # Thành:
   print(f"   • Oversampling: Tăng số lượng samples của minority class")
   ```

## Điều chỉnh ratio oversampling:

Trong code oversampling, tìm dòng:
```python
target_neutral_count = int(majority_count * 0.2)  # 20%
```

Có thể thay đổi:
- `0.2` → 20% of majority (ít oversampling, giảm overfitting)
- `0.3` → 30% of majority (trung bình)
- `0.4` → 40% of majority (nhiều oversampling)
- `0.5` → 50% of majority (rất nhiều oversampling)

## So sánh chiến lược:

| Chiến lược | Oversampling | Focal Loss | Khi nào dùng |
|-----------|--------------|------------|--------------|
| **Chỉ Focal Loss** | ❌ OFF | ✅ ON | Test xem Focal Loss có đủ mạnh không |
| **Oversampling 20%** | ✅ 20% | ✅ ON | Cân bằng giữa data và loss weighting |
| **Oversampling 40%** | ✅ 40% | ✅ ON | Imbalance nghiêm trọng (>10x) |

## Khuyến nghị:

1. **Bắt đầu với Focal Loss only** (hiện tại) → Xem kết quả
2. Nếu F1 neutral vẫn thấp → Bật Oversampling 20%
3. Nếu vẫn thấp → Tăng lên 30-40%
4. Theo dõi overfitting (eval loss tăng) → Giảm oversampling ratio

## Tại sao tắt Oversampling?

- Test xem **Focal Loss đơn lẻ có đủ mạnh** để xử lý imbalance không
- Giảm **duplicates** trong training data
- Giảm nguy cơ **memorization** (model học thuộc lòng các samples lặp lại)
- Model **generalize tốt hơn** nếu không có duplicates
- Focal Loss với alpha weights cao (6.67x cho neutral) có thể đủ mạnh
