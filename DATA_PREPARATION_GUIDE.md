# 📊 Data Preparation Guide

## 🎯 Overview

Dự án có 2 approaches với data formats khác nhau:

```
Single-Label:  One row per (sentence, aspect) pair
Multi-Label:   One row per sentence (all aspects as columns)
```

Mỗi approach có data folder riêng và scripts riêng.

---

## 📂 Folder Structure

```
D:\BERT\
├── dataset.csv                       (Original multi-label format)
│
├── single_label/
│   ├── data/                         Single-label data folder
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   ├── test.csv
│   │   ├── train_augmented_neutral_nhung.csv
│   │   └── README.md
│   ├── prepare_data.py               Convert to single-label
│   ├── augment_neutral_and_nhung.py  Augment data
│   └── prepare_and_augment.bat       Quick start script
│
└── multi_label/
    ├── data/                         Multi-label data folder
    │   ├── train_multilabel.csv
    │   ├── validation_multilabel.csv
    │   ├── test_multilabel.csv
    │   ├── train_multilabel_balanced.csv
    │   └── README.md
    ├── prepare_data_multilabel.py    Split dataset
    ├── augment_multilabel_balanced.py Balance data
    └── prepare_and_augment.bat       Quick start script
```

---

## 🔧 Single-Label Data Preparation

### **Manual Steps:**

**Step 1: Prepare Data**
```bash
# From D:\BERT\
python single_label/prepare_data.py
```
**Output:** `single_label/data/train.csv`, `validation.csv`, `test.csv`

**Step 2: Augment Data**
```bash
python single_label/augment_neutral_and_nhung.py
```
**Output:** `single_label/data/train_augmented_neutral_nhung.csv`

### **Quick Start (Recommended):**

```bash
# From D:\BERT\
single_label\prepare_and_augment.bat
```

**This will automatically:**
1. Convert dataset.csv to single-label format
2. Split into train/val/test (80/10/10)
3. Apply Neutral + "nhưng" augmentation
4. Create `train_augmented_neutral_nhung.csv`

### **Data Format:**

```csv
sentence,aspect,sentiment
"Pin tốt camera xấu",Battery,positive
"Pin tốt camera xấu",Camera,negative
"Pin tốt camera xấu",Performance,neutral
```

### **Augmentation Strategy:**

1. **Neutral Oversampling:** Balance Neutral with avg(Positive, Negative)
2. **"Nhưng" Oversampling:** x3 factor for adversative constructions
3. **Overlap Handling:** Intelligent handling of samples with both

**Expected Result:** ~10,000-12,000 training samples

---

## 🔧 Multi-Label Data Preparation

### **Manual Steps:**

**Step 1: Prepare Data**
```bash
# From D:\BERT\
python multi_label/prepare_data_multilabel.py
```
**Output:** `multi_label/data/train_multilabel.csv`, `validation_multilabel.csv`, `test_multilabel.csv`

**Step 2: Balance Data**
```bash
python multi_label/augment_multilabel_balanced.py
```
**Output:** `multi_label/data/train_multilabel_balanced.csv`

### **Quick Start (Recommended):**

```bash
# From D:\BERT\
multi_label\prepare_and_augment.bat
```

**This will automatically:**
1. Split dataset.csv into train/val/test (80/10/10)
2. Apply aspect-wise balanced oversampling
3. Create `train_multilabel_balanced.csv`

### **Data Format:**

```csv
text,Battery,Camera,Performance,Display,...
"Pin tốt camera xấu",Positive,Negative,Neutral,...
```

### **Balancing Strategy:**

**Aspect-wise Oversampling:**
- For each aspect: oversample minority sentiments to match majority
- Example: Battery has Negative=500, Positive=200, Neutral=100
  → Oversample Positive and Neutral to 500 each

**Result:**
- Original: 7,309 samples
- Balanced: 15,921 samples (+117.8%)
- Imbalance: 5.30x → 1.22x (77% improvement)

---

## 📊 Comparison

| Feature | Single-Label | Multi-Label |
|---------|-------------|-------------|
| **Format** | One row per (sentence, aspect) | One row per sentence |
| **Original Size** | ~80,000 samples | ~7,309 samples |
| **After Augmentation** | ~12,000 samples | ~15,921 samples |
| **Augmentation** | Neutral + "nhưng" | Aspect-wise balancing |
| **Training Time** | Longer (more samples) | Faster (fewer samples) |
| **Inference** | 11 forward passes | 1 forward pass |
| **Expected F1** | ~93% | ~96% |

---

## 🚀 Complete Workflow

### **For Single-Label:**

```bash
# 1. Prepare data
cd D:\BERT
single_label\prepare_and_augment.bat

# 2. Train
python single_label\train.py --config single_label\config_single.yaml

# 3. Analyze
bash single_label/test/run_analysis.sh
```

---

### **For Multi-Label:**

```bash
# 1. Prepare data
cd D:\BERT
multi_label\prepare_and_augment.bat

# 2. Train (Target: 96% F1)
python multi_label\train_multilabel_focal_contrastive.py --epochs 8 --focal-weight 0.7 --contrastive-weight 0.3

# Or quick start:
train_focal_contrastive.bat

# 3. Analyze
bash multi_label/test/run_analysis.sh
```

---

## 📁 Generated Files

### **Single-Label:**
```
single_label/data/
├── train.csv                              (~6,500 samples)
├── validation.csv                         (~800 samples)
├── test.csv                               (~800 samples)
├── train_augmented_neutral_nhung.csv      (~12,000 samples) ⭐
└── data_metadata.json
```

### **Multi-Label:**
```
multi_label/data/
├── train_multilabel.csv                   (7,309 samples)
├── validation_multilabel.csv              (914 samples)
├── test_multilabel.csv                    (914 samples)
├── train_multilabel_balanced.csv          (15,921 samples) ⭐
└── multilabel_metadata.json
```

**⭐ = Recommended for training**

---

## ⚙️ Configuration

### **Single-Label Config:**
```yaml
# single_label/config_single.yaml
paths:
  train_file: "single_label/data/train_augmented_neutral_nhung.csv"
  validation_file: "single_label/data/validation.csv"
  test_file: "single_label/data/test.csv"
```

### **Multi-Label Config:**
```yaml
# multi_label/config_multi.yaml
paths:
  train_file: "multi_label/data/train_multilabel_balanced.csv"
  validation_file: "multi_label/data/validation_multilabel.csv"
  test_file: "multi_label/data/test_multilabel.csv"
```

---

## 🔍 Troubleshooting

### **Problem: Data folders empty**

**Solution:**
```bash
# Single-label
single_label\prepare_and_augment.bat

# Multi-label
multi_label\prepare_and_augment.bat
```

### **Problem: Need to regenerate data**

**Solution:**
```bash
# Delete old files
rm single_label/data/*.csv
rm multi_label/data/*_multilabel*.csv

# Regenerate
single_label\prepare_and_augment.bat
multi_label\prepare_and_augment.bat
```

### **Problem: Training fails with file not found**

**Check:**
1. Run data preparation scripts first
2. Verify paths in config files
3. Make sure you're running from `D:\BERT\` directory

---

## ✅ Quick Checklist

### **Before Training Single-Label:**
- [ ] `single_label/data/train_augmented_neutral_nhung.csv` exists
- [ ] `single_label/data/validation.csv` exists
- [ ] `single_label/data/test.csv` exists
- [ ] Config points to correct files

### **Before Training Multi-Label:**
- [ ] `multi_label/data/train_multilabel_balanced.csv` exists
- [ ] `multi_label/data/validation_multilabel.csv` exists
- [ ] `multi_label/data/test_multilabel.csv` exists
- [ ] Config points to correct files

---

## 📝 Notes

- **Always run from root:** All scripts should be run from `D:\BERT\` directory
- **Separate data:** Single-label and multi-label use completely different formats
- **Augmentation recommended:** Both approaches benefit from data augmentation
- **Configs already set:** Default configs point to augmented data files
- **Quick start scripts:** Use `.bat` files for automatic preparation

---

## 🎯 Summary

| Approach | Quick Start Command | Expected F1 |
|----------|-------------------|-------------|
| Single-Label | `single_label\prepare_and_augment.bat` | 93% |
| Multi-Label | `multi_label\prepare_and_augment.bat` | 96% |

**Recommendation:** Use multi-label approach for better performance! 🚀
