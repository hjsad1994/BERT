# Fix: Checkpoint Renaming Conflict with load_best_model_at_end

## Vấn đề

Khi training với cả hai tính năng:
- `load_best_model_at_end: true` (trong config.yaml)
- Checkpoint renaming callback (trong train.py)

Sẽ xảy ra lỗi:
```
Could not locate the best model at finetuned_visobert_absa_model\checkpoint-454\pytorch_model.bin
```

## Nguyên nhân

1. **Checkpoint được lưu**: checkpoint-454 (step 454, 88.8% accuracy)
2. **Callback đổi tên ngay**: checkpoint-454 → checkpoint-88
3. **Trainer cần load best model**: Tìm checkpoint-454 nhưng không thấy (đã bị đổi tên!)
4. **Kết quả**: Best model không được load và lưu vào thư mục chính

## Timeline của sự cố

```
Training Epoch 2:
├─ Save checkpoint-454 ✅
├─ on_save callback triggers → Rename to checkpoint-88 ✅
└─ Trainer tracks: best_model_checkpoint = "checkpoint-454"

Training kết thúc:
├─ load_best_model_at_end tries to load "checkpoint-454" ❌
│  └─ Error: Checkpoint not found (đã bị rename!)
└─ Model không được lưu vào thư mục chính ❌
```

## Giải pháp đã áp dụng

### 1. Fix Immediate: Copy best model manually

```bash
cd finetuned_visobert_absa_model
cp checkpoint-88/config.json .
cp checkpoint-88/model.safetensors .
```

**Kết quả**: Model có thể load được ngay lập tức

### 2. Fix Long-term: Delay checkpoint renaming

Thay đổi `SimpleMetricCheckpointCallback` trong `checkpoint_renamer.py`:

**Trước (có vấn đề):**
```python
def on_save(self, args, state, control, **kwargs):
    # Rename ngay lập tức
    checkpoint_path.rename(new_path)  # ❌ Conflict!
```

**Sau (đã fix):**
```python
def on_save(self, args, state, control, **kwargs):
    # Chỉ track rename, chưa thực hiện
    self.pending_renames[old_name] = new_name  # ✅ Delay

def on_train_end(self, args, state, control, **kwargs):
    # Rename SAU KHI trainer đã load best model
    for old_name, new_name in self.pending_renames.items():
        old_path.rename(new_path)  # ✅ Safe now!
```

## Cách hoạt động của fix

```
Training Epoch 2:
├─ Save checkpoint-454 ✅
├─ on_save callback: Track pending rename (454 → 88) 📝
└─ Trainer tracks: best_model_checkpoint = "checkpoint-454"

Training kết thúc:
├─ load_best_model_at_end loads "checkpoint-454" ✅ (vẫn tồn tại!)
├─ Save best model to main directory ✅
└─ on_train_end: Now rename checkpoint-454 → checkpoint-88 ✅
```

## Kiểm tra fix đã hoạt động

### 1. Kiểm tra model files trong thư mục chính

```bash
ls finetuned_visobert_absa_model/*.json
ls finetuned_visobert_absa_model/*.safetensors
```

Phải thấy:
- `config.json` ✅
- `model.safetensors` ✅

### 2. Test load model

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "finetuned_visobert_absa_model"
)
print("Model loaded successfully!")
```

Nếu load được → Fix đã hoạt động ✅

### 3. Kiểm tra checkpoint folders

```bash
ls finetuned_visobert_absa_model/checkpoint-*
```

Phải thấy các folder đã được rename theo accuracy:
- `checkpoint-83/` (83% accuracy)
- `checkpoint-88/` (88% accuracy - BEST MODEL)
- `checkpoint-91/` (91% accuracy)

## Best Practices

### 1. Luôn dùng cả hai tính năng

```yaml
# config.yaml
load_best_model_at_end: true  # ✅ Bắt buộc
save_total_limit: 3-5          # ✅ Giữ một số checkpoints
```

```python
# train.py
from checkpoint_renamer import SimpleMetricCheckpointCallback
callback = SimpleMetricCheckpointCallback()  # ✅ Fixed version
trainer.add_callback(callback)
```

### 2. Verify sau khi training

```python
# Luôn kiểm tra model có load được không
from transformers import AutoModelForSequenceClassification
try:
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
```

### 3. Backup best checkpoint

Nếu lo lắng, tạo backup:
```bash
# Sau khi training xong
cp -r finetuned_visobert_absa_model best_model_backup/
```

## Kết luận

- ✅ **Fix ngay**: Copy model từ checkpoint-88
- ✅ **Fix lâu dài**: Update checkpoint_renamer.py (đã hoàn thành)
- ✅ **Verify**: Model load được từ thư mục chính
- ✅ **Prevent**: Checkpoint renaming chỉ xảy ra SAU khi best model đã được load

**Status**: Đã fix hoàn toàn! Training mới sẽ không gặp vấn đề này nữa.
