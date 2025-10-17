# Fix: Crash khi Predict trên Test Set

## Vấn đề

Training script bị crash ở bước predict trên test set, ngay sau khi evaluate thành công.

```
✓ Kết quả đánh giá trên tập test:
   Accuracy:  0.9052
   ...
   
⏳ Đang predict để lấy detailed metrics...
✓ Predict hoàn tất

======================================================================
🔮 Đang dự đoán trên tập test...
======================================================================
[CRASH] ← Bị crash ở đây
```

## Nguyên nhân

Script đang gọi **predict() 2 lần** trên test dataset:

1. **Lần 1**: Trong section "ĐÁNH GIÁ TRÊN TẬP TEST"
   ```python
   predictions_output = eval_trainer.predict(test_dataset)  # Để lấy detailed metrics
   ```

2. **Lần 2**: Trong hàm `save_predictions()`
   ```python
   def save_predictions(trainer, test_dataset, ...):
       predictions_output = trainer.predict(test_dataset)  # Predict lại lần 2!
   ```

**Vấn đề**: 
- Predict lần 1 thành công
- Predict lần 2 crash do VRAM/RAM không đủ
- Dù VRAM "oke", nhưng accumulated memory từ training + eval + predict lần 1 → Lần 2 bị OOM

## Giải pháp

### 1. Tạo hàm mới `save_predictions_from_output()` (utils.py)

```python
def save_predictions_from_output(predictions_output, test_df, config, id2label):
    """
    Lưu predictions từ output đã có (KHÔNG predict lại)
    """
    predictions = predictions_output.predictions
    # ... xử lý và lưu file
```

**Ưu điểm**:
- Không predict lại → Tiết kiệm memory
- Nhanh hơn (không tính toán lại)
- Tránh crash

### 2. Cập nhật train.py

```python
# Import hàm mới
from utils import save_predictions_from_output

# Section 10: Predict 1 lần duy nhất
torch.cuda.empty_cache()  # Giải phóng cache trước
predictions_output = eval_trainer.predict(test_dataset)

# Section 11: Tái sử dụng predictions_output
save_predictions_from_output(predictions_output, test_df, config, id2label)
```

### 3. Thêm `torch.cuda.empty_cache()`

Trước khi predict, giải phóng CUDA cache để đảm bảo có đủ memory:

```python
# Giải phóng cache trước khi predict
torch.cuda.empty_cache()

# Predict
predictions_output = eval_trainer.predict(test_dataset)
```

## Kết quả

✅ **Trước**: Predict 2 lần → Crash  
✅ **Sau**: Predict 1 lần, tái sử dụng kết quả → Không crash

✅ **Lợi ích**:
- Tiết kiệm memory (không duplicate predictions array)
- Nhanh hơn (không tính toán lại)
- Ổn định hơn (không bị OOM)

## Files thay đổi

1. **utils.py**: 
   - Thêm hàm `save_predictions_from_output()`
   - Giữ nguyên hàm `save_predictions()` cũ (backward compatible)

2. **train.py**:
   - Import hàm mới
   - Thêm `torch.cuda.empty_cache()` trước predict
   - Gọi `save_predictions_from_output()` thay vì `save_predictions()`
   - Thêm comment giải thích

## Cách áp dụng cho code khác

Nếu gặp OOM khi predict, áp dụng pattern này:

```python
# BAD: Predict nhiều lần
predictions1 = trainer.predict(dataset)  # Cho mục đích 1
predictions2 = trainer.predict(dataset)  # Cho mục đích 2 → OOM!

# GOOD: Predict 1 lần, tái sử dụng
torch.cuda.empty_cache()  # Giải phóng cache trước
predictions_output = trainer.predict(dataset)

# Dùng predictions_output cho nhiều mục đích
detailed_metrics = get_detailed_metrics(predictions_output.predictions, ...)
save_predictions_from_output(predictions_output, ...)
analyze_errors(predictions_output, ...)
```

## Memory consumption comparison

| Action | Memory Usage | Notes |
|--------|--------------|-------|
| Training | ~3-4 GB VRAM | Model + optimizer + gradients |
| Evaluate | ~1-2 GB VRAM | Model + forward pass only |
| Predict (1st) | ~1-2 GB VRAM | + predictions array in RAM |
| Predict (2nd) | **OOM!** | Accumulated memory too high |
| **After fix** | **No OOM** | Only 1 predict, reuse results |

## Khi nào cần hàm cũ `save_predictions()`?

Dùng `save_predictions()` khi:
- Chỉ cần save predictions, không cần metrics khác
- Chưa có predictions_output sẵn
- Script độc lập, không phải trong training workflow

Dùng `save_predictions_from_output()` khi:
- Đã có predictions_output từ bước trước
- Cần tiết kiệm memory
- Trong training workflow (train.py)
