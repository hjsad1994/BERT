# 🚀 Quick Start Guide

## ⚠️ Important: Chạy Từ ROOT Folder!

**Luôn chạy từ `D:\BERT\`** (root folder), không phải từ trong single_label/ hay multi_label/

---

## 🎯 Single-Label Training (93% F1)

```bash
# Từ D:\BERT\
python single_label\train.py --config single_label\config_single.yaml
```

**Expected:**
- Training time: ~30 minutes
- F1 Score: 93%
- Logs: `single_label/training_logs/`
- Models: `single_label/models/`
- Results: `single_label/results/`

---

## 🎯 Multi-Label Training (96% F1) ⭐ RECOMMENDED

```bash
# Từ D:\BERT\
python multi_label\train_multilabel_focal_contrastive.py --epochs 8 --focal-weight 0.7 --contrastive-weight 0.3 --temperature 0.1
```

**Or quick start:**
```bash
# Từ D:\BERT\
train_focal_contrastive.bat
```

**Expected:**
- Training time: ~45 minutes
- F1 Score: 96.0-96.5%
- Logs: Console (redirect to `multi_label/training_logs/` if needed)
- Models: `multi_label/models/`
- Results: `multi_label/results/`

---

## 📁 File Paths

**All paths in config files are relative to ROOT folder (D:\BERT\)**

```yaml
# Correct (from root):
paths:
  train_file: "data/train.csv"
  validation_file: "data/validation.csv"
  output_dir: "single_label/models/"

# Incorrect:
paths:
  train_file: "../data/train.csv"  # ❌ Wrong!
```

---

## 🔧 Config Files

**Single-Label:**
```
single_label/config_single.yaml
```

**Multi-Label:**
```
multi_label/config_multi.yaml
```

**Both configs đã được fix để work từ root folder!**

---

## ✅ Complete Workflow

### **Single-Label:**
```bash
# Step 1: Ensure you're in root folder
cd D:\BERT

# Step 2: Train
python single_label\train.py --config single_label\config_single.yaml

# Step 3: Results
# models/finetuned_visobert_single/
# results/evaluation_report_single.txt
```

---

### **Multi-Label (Recommended):**
```bash
# Step 1: Ensure you're in root folder
cd D:\BERT

# Step 2: Train (Focal + Contrastive)
python multi_label\train_multilabel_focal_contrastive.py --epochs 8 --focal-weight 0.7 --contrastive-weight 0.3

# Step 3: Results
# multi_label/models/multilabel_focal_contrastive/
# test_results_focal_contrastive.json
```

---

## 🎯 Alternative Options

### **Multi-Label with Different Settings:**

**More classification focus:**
```bash
python multi_label\train_multilabel_focal_contrastive.py --epochs 10 --focal-weight 0.8 --contrastive-weight 0.2
```

**Very aggressive:**
```bash
python multi_label\train_multilabel_focal_contrastive.py --epochs 12 --focal-weight 0.9 --contrastive-weight 0.1
```

**Baseline (for comparison):**
```bash
python multi_label\train_multilabel.py --epochs 5
```

---

## ⚠️ Common Errors

### **Error: "Không tìm thấy file dữ liệu"**

**Cause:** Running from wrong folder or wrong path in config

**Solution:**
1. Make sure you're in `D:\BERT\` (root folder)
2. Config files should have `data/` not `../data/`
3. ✅ Already fixed in both config files!

---

### **Error: "ModuleNotFoundError"**

**Cause:** Missing dependencies

**Solution:**
```bash
pip install -r requirements.txt
```

---

### **Error: "CUDA out of memory"**

**Cause:** Batch size too large

**Solution:** Reduce batch size in config:
```yaml
training:
  per_device_train_batch_size: 16  # Reduce from 32
```

---

## 📊 Expected Results

| Method | Command | F1 Score | Time |
|--------|---------|----------|------|
| **Single-Label** | `python single_label\train.py ...` | 90-92% | ~30 min |
| **Multi-Label Baseline** | `python multi_label\train_multilabel.py ...` | 95.49% | ~35 min |
| **Multi-Label Focal+Contrastive** | `python multi_label\train_multilabel_focal_contrastive.py ...` | **96.0-96.5%** | **~45 min** |

---

## 📚 More Information

**Detailed guides:**
- `single_label/README.md` - Single-label complete guide
- `multi_label/README.md` - Multi-label complete guide
- `FINAL_COMMANDS.md` - All commands
- `FOCAL_CONTRASTIVE_ANALYSIS.md` - Method details

---

## ✅ Summary

**Always run from:** `D:\BERT\` (root folder)

**Single-Label:**
```bash
python single_label\train.py --config single_label\config_single.yaml
```

**Multi-Label (Recommended):**
```bash
python multi_label\train_multilabel_focal_contrastive.py --epochs 8 --focal-weight 0.7 --contrastive-weight 0.3
```

**Config files:** Already fixed to work from root!

🎯 **Ready to train!**
