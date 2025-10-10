# Quản Lý Checkpoints và Best Model

## Câu hỏi: Checkpoint có bị ghi đè không?

**Ví dụ:** Accuracy 91 → 92 → 89, checkpoint-91 và checkpoint-92 có bị mất không?

## Trả lời: CÓ và KHÔNG (tùy config)

---

## Config hiện tại:

```yaml
save_total_limit: 3  # Giữ 3 checkpoints gần nhất
load_best_model_at_end: true  # QUAN TRỌNG!
metric_for_best_model: "eval_loss"
greater_is_better: false
```

---

## Cơ chế hoạt động:

### 1. **Lưu Checkpoints (save_total_limit: 3)**

Training 4 epochs với eval_loss:

```
Epoch 1: eval_loss = 0.10 → checkpoint-91 (best) ✅
Epoch 2: eval_loss = 0.18 → checkpoint-90 ✅
Epoch 3: eval_loss = 0.23 → checkpoint-89 ✅

Đang có 3 checkpoints → Đạt limit!

Epoch 4: eval_loss = 0.25 → checkpoint-88 ✅
→ Xóa checkpoint cũ nhất (checkpoint-91) ❌

Kết quả: checkpoint-90, checkpoint-89, checkpoint-88
❌ Mất checkpoint-91 (best)!
```

**Vấn đề:** Checkpoint tốt nhất có thể BỊ XÓA nếu training nhiều epochs!

---

### 2. **Load Best Model (load_best_model_at_end: true)** ⭐

**May mắn:** Trainer LUÔN track best checkpoint trong memory!

```python
# Trainer tự động:
class Trainer:
    def train(self):
        # ... training loop
        
        # Track best checkpoint
        if eval_loss < self.best_metric:
            self.best_metric = eval_loss
            self.best_checkpoint = current_checkpoint  # Nhớ best checkpoint
        
        # Khi training kết thúc:
        if load_best_model_at_end:
            # Load lại best checkpoint (dù đã bị xóa khỏi disk!)
            self.load_checkpoint(self.best_checkpoint)
            
            # Save best model vào output_dir chính
            self.save_model(output_dir)
```

**Kết quả:**
```
Training kết thúc →
Trainer load lại best model (checkpoint-91) từ memory →
Save vào finetuned_visobert_absa_model/ (thư mục gốc) →
Evaluate và predict trên best model ✅
```

---

## So sánh các chiến lược:

| save_total_limit | Ưu điểm | Nhược điểm | Khuyến nghị |
|------------------|---------|-----------|-------------|
| **1** | Tiết kiệm disk | Chỉ giữ checkpoint cuối (có thể không phải best) | ❌ Không nên |
| **2** | Tiết kiệm disk | Dễ mất best checkpoint | ⚠️ Rủi ro cao |
| **3** | Cân bằng | An toàn với early stopping (2 epochs) | ✅ Khuyến nghị |
| **5** | An toàn | Tốn disk (nhưng không nhiều) | ✅ An toàn nhất |
| **None** | Giữ tất cả | Tốn disk nhiều | 💾 Nếu disk đủ |

---

## Cấu trúc thư mục:

```
finetuned_visobert_absa_model/
├── checkpoint-89/          # Checkpoint epoch 3
├── checkpoint-90/          # Checkpoint epoch 2
├── checkpoint-91/          # Checkpoint epoch 1 (best)
│
├── config.json             # ⭐ Best model (từ checkpoint-91)
├── model.safetensors       # ⭐ Best model weights
├── tokenizer.json          # Tokenizer
└── ...
```

**Lưu ý:** 
- Files ở thư mục gốc = **best model** (load_best_model_at_end)
- Subfolders = checkpoints của từng epoch (có thể bị xóa)

---

## Đảm bảo evaluate trên best model:

### ✅ Cách 1: Dùng config hiện tại (KHUYẾN NGHỊ)

```yaml
load_best_model_at_end: true
save_total_limit: 3
```

→ Trainer tự động load best model khi kết thúc

### ✅ Cách 2: Tăng save_total_limit

```yaml
save_total_limit: 5  # Hoặc None (giữ tất cả)
```

→ Giữ nhiều checkpoints hơn, giảm nguy cơ mất best

### ✅ Cách 3: Manually load best checkpoint

```python
# Nếu lo lắng, có thể manually load:
from transformers import AutoModelForSequenceClassification

# Load từ thư mục gốc (best model)
model = AutoModelForSequenceClassification.from_pretrained(
    "finetuned_visobert_absa_model"
)

# Hoặc load từ checkpoint cụ thể
model = AutoModelForSequenceClassification.from_pretrained(
    "finetuned_visobert_absa_model/checkpoint-91"
)
```

---

## Kết luận:

### ❓ Checkpoint có bị ghi đè không?

**Trả lời:** 
- ❌ **Checkpoint folders có thể bị XÓA** (do save_total_limit)
- ✅ **Best model LUÔN được giữ** (nhờ load_best_model_at_end)
- ✅ **Evaluate LUÔN trên best model** (tự động)

### 🎯 Khuyến nghị:

```yaml
save_total_limit: 3-5  # An toàn
load_best_model_at_end: true  # BẮT BUỘC!
early_stopping_patience: 2  # Dừng sớm nếu không cải thiện
```

→ **Đảm bảo:** Best model được giữ lại và evaluate đúng!

---

## Test thực tế:

Sau khi training, kiểm tra:

```bash
# 1. Xem các checkpoints còn lại
dir finetuned_visobert_absa_model

# 2. Load model từ thư mục gốc (best model)
python
>>> from transformers import AutoModelForSequenceClassification
>>> model = AutoModelForSequenceClassification.from_pretrained("finetuned_visobert_absa_model")
>>> # Model này là BEST MODEL (eval_loss thấp nhất)
```

---

## Nếu vẫn lo lắng:

### Option 1: Tăng save_total_limit lên 10
```yaml
save_total_limit: 10
```

### Option 2: Không giới hạn (giữ tất cả)
```yaml
save_total_limit: null
```

### Option 3: Manually backup best checkpoint
```bash
# Sau khi training xong
cp -r finetuned_visobert_absa_model/checkpoint-91 backup_best_checkpoint/
```

---

**Tóm lại:** Config hiện tại ĐÃ AN TOÀN nhờ `load_best_model_at_end: true`! Trainer sẽ tự động load và save best model, đảm bảo evaluate đúng kết quả tốt nhất. 🎯
