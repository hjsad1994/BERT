# Training với Logging và Visualization

## ✅ Đã thêm gì?

### 1. **Training Script với Logging**
File: `multi_label/train_multilabel_focal_contrastive_with_logging.py`

**Features mới:**
- ✅ Log losses mỗi epoch vào CSV
- ✅ Log losses mỗi batch vào CSV
- ✅ Lưu training history để visualize sau
- ✅ Timestamp cho mỗi training run
- ✅ Tương thích 100% với code cũ

### 2. **Visualization Script**
File: `multi_label/visualize_training_logs.py`

**Visualizations:**
- ✅ Total loss over epochs
- ✅ Focal vs Contrastive loss comparison
- ✅ Validation F1 score progression
- ✅ Learning rate schedule
- ✅ Loss reduction rates
- ✅ Batch-level losses (detailed)

---

## 🚀 Cách sử dụng:

### **Step 1: Train với Logging**

```bash
# Run training with logging
python multi_label\train_multilabel_focal_contrastive_with_logging.py

# Hoặc với custom config
python multi_label\train_multilabel_focal_contrastive_with_logging.py \
    --epochs 10 \
    --focal-weight 0.8 \
    --contrastive-weight 0.2
```

**Outputs:**
```
multi_label/models/multilabel_focal_contrastive/
├── best_model.pt
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── ...
├── test_results_focal_contrastive.json
└── training_logs/
    ├── epoch_losses_20250121_143052.csv  ← LOG per epoch
    └── batch_losses_20250121_143052.csv  ← LOG per batch
```

### **Step 2: Visualize Logs**

```bash
# Visualize epoch losses
python multi_label\visualize_training_logs.py \
    --epoch-log multi_label\models\multilabel_focal_contrastive\training_logs\epoch_losses_20250121_143052.csv

# Visualize batch losses (optional)
python multi_label\visualize_training_logs.py \
    --epoch-log multi_label\models\multilabel_focal_contrastive\training_logs\epoch_losses_20250121_143052.csv \
    --batch-log multi_label\models\multilabel_focal_contrastive\training_logs\batch_losses_20250121_143052.csv
```

**Outputs:**
```
training_logs/
├── training_losses_real_data.png  ← 4 subplots: losses, F1, LR
├── loss_reduction_rates.png       ← % reduction over time
└── batch_losses_real_data.png     ← Detailed batch losses
```

---

## 📊 Log File Format:

### **Epoch Losses CSV:**
```csv
epoch,train_loss,train_focal_loss,train_contrastive_loss,val_accuracy,val_f1,val_precision,val_recall,learning_rate
1,0.6542,0.3214,0.6402,0.8832,0.8856,0.8912,0.8832,2.0e-05
2,0.5123,0.2543,0.5832,0.9145,0.9187,0.9201,0.9145,1.8e-05
3,0.4012,0.1987,0.5102,0.9345,0.9389,0.9402,0.9345,1.5e-05
...
```

### **Batch Losses CSV:**
```csv
epoch,batch,total_loss,focal_loss,contrastive_loss
1,0,0.8543,0.4123,0.7865
1,1,0.7234,0.3567,0.7012
1,2,0.6891,0.3234,0.6543
...
```

---

## 📈 Example Visualizations:

### **Plot 1: Training Losses**
4 subplots:
1. **Total Loss** - Overall training loss
2. **Focal vs Contrastive** - Compare both losses
   - Shows if both are decreasing (GOOD)
   - Or if one is stuck (BAD)
3. **Validation F1** - F1 score progress
   - Target: 96%
   - Baseline: 95.5%
4. **Learning Rate** - LR schedule

### **Plot 2: Loss Reduction Rates**
- % reduction from starting point
- Shows how fast each loss is decreasing
- Compare Focal vs Contrastive reduction speed

### **Plot 3: Batch Losses (Optional)**
- Detailed batch-level view
- Helps identify unstable training
- Shows smoothed moving average

---

## 🎯 Interpreting Results:

### **Good Training:**
```
Focal Loss:    0.6500 → 0.2200 (↓ 66%)
Contr Loss:    0.6400 → 0.4200 (↓ 34%)
Val F1:        88.5%  → 96.2%  (↑ 7.7%)

Status: Both decreasing - GOOD!
```

### **Bad Training - Contrastive Stuck:**
```
Focal Loss:    0.6500 → 0.2200 (↓ 66%)
Contr Loss:    0.6400 → 0.6300 (↓ 2%)   ← STUCK!
Val F1:        88.5%  → 94.5%  (↑ 6.0%)

Status: Contrastive not decreasing
Action: Increase contrastive_weight: 0.2 → 0.3
```

### **Bad Training - Focal Stuck:**
```
Focal Loss:    0.6500 → 0.6200 (↓ 5%)   ← STUCK!
Contr Loss:    0.6400 → 0.3500 (↓ 45%)
Val F1:        88.5%  → 93.2%  (↑ 4.7%)

Status: Focal not decreasing
Action: Increase focal_weight: 0.8 → 0.9
```

---

## 🔧 So sánh Old vs New:

### **Old Training Script:**
```bash
python multi_label\train_multilabel_focal_contrastive.py

Outputs:
- Checkpoints (.pt)
- Final results (JSON)
- ❌ NO per-epoch logs
- ❌ NO visualization data
```

### **New Training Script:**
```bash
python multi_label\train_multilabel_focal_contrastive_with_logging.py

Outputs:
- Checkpoints (.pt)
- Final results (JSON)
- ✅ Epoch losses (CSV)
- ✅ Batch losses (CSV)
- ✅ Ready for visualization
```

**Lưu ý:** File cũ vẫn hoạt động bình thường, file mới thêm logging.

---

## 💡 Use Cases:

### **1. Verify Training Behavior:**
```bash
# Run training
python multi_label\train_multilabel_focal_contrastive_with_logging.py --epochs 5

# Visualize to check if both losses decrease
python multi_label\visualize_training_logs.py --epoch-log path/to/logs.csv

# If contrastive stuck → Adjust config
```

### **2. Compare Different Hyperparameters:**
```bash
# Run 1: 70/30 split
python ... --focal-weight 0.7 --contrastive-weight 0.3

# Run 2: 80/20 split
python ... --focal-weight 0.8 --contrastive-weight 0.2

# Compare visualizations side-by-side
```

### **3. Paper/Report:**
```bash
# Train with logging
python ... --epochs 15

# Generate publication-quality plots
python visualize_training_logs.py --epoch-log logs.csv

# Use PNG files in paper/presentation
```

---

## 📝 Batch Scripts:

### **Create: `train_and_visualize.bat`**

```batch
@echo off
REM Train with logging then visualize

echo Training with logging...
python multi_label\train_multilabel_focal_contrastive_with_logging.py ^
    --epochs 10 ^
    --focal-weight 0.8 ^
    --contrastive-weight 0.2

echo.
echo Finding latest log file...
for /f "delims=" %%i in ('dir /b /od multi_label\models\multilabel_focal_contrastive\training_logs\epoch_losses_*.csv') do set LATEST=%%i

echo.
echo Visualizing: %LATEST%
python multi_label\visualize_training_logs.py ^
    --epoch-log multi_label\models\multilabel_focal_contrastive\training_logs\%LATEST%

echo.
echo Done! Check training_logs folder for plots.
pause
```

---

## ✅ Benefits:

| Feature | Old Script | New Script |
|---------|-----------|-----------|
| Training | ✅ | ✅ |
| Checkpoints | ✅ | ✅ |
| Final results | ✅ | ✅ |
| Per-epoch logs | ❌ | ✅ |
| Batch logs | ❌ | ✅ |
| Visualization data | ❌ | ✅ |
| Analysis-ready | ❌ | ✅ |
| Debugging | Hard | Easy |
| Paper-ready plots | ❌ | ✅ |

---

## 🎓 Summary:

### **Câu hỏi:** "Có thể visualize bằng data thật không?"

### **Trả lời:** 
✅ **CÓ!** 

**Steps:**
1. Train với logging script mới
2. CSV logs được tạo tự động
3. Visualize từ CSV logs
4. Analyze training behavior với real data

**Files cần:**
- `train_multilabel_focal_contrastive_with_logging.py` - Training + logging
- `visualize_training_logs.py` - Visualization từ logs
- CSV logs - Được tạo tự động khi training

**Result:**
- ✅ Real data visualizations
- ✅ Analyze loss behavior
- ✅ Verify cả 2 losses có giảm không
- ✅ Publication-quality plots

---

**Try it now:**
```bash
# 1. Train with logging (3-5 epochs for quick test)
python multi_label\train_multilabel_focal_contrastive_with_logging.py --epochs 3

# 2. Find the log file
dir multi_label\models\multilabel_focal_contrastive\training_logs\*.csv

# 3. Visualize
python multi_label\visualize_training_logs.py --epoch-log [log_file_path]
```
