# 📁 Project Structure - Vietnamese ABSA

## 🎯 Overview

**Two Approaches:**
1. **Single-Label** (Traditional) - 90-92% F1
2. **Multi-Label** (Novel - Focal+Contrastive) - 96%+ F1 ⭐

---

## 📂 Folder Structure

```
D:\BERT\
│
├── 📁 single_label/          (Traditional approach - 90-92% F1)
│   ├── train.py             (Main training)
│   ├── config_single.yaml   (Configuration)
│   ├── README.md            (Complete guide)
│   └── ...                  (utilities, analysis)
│
├── 📁 multi_label/           (Novel approach - 96% F1) ⭐
│   ├── train_multilabel_focal_contrastive.py  (Main training)
│   ├── config_multi.yaml    (Configuration)
│   ├── README.md            (Complete guide)
│   └── ...                  (models, utilities)
│
├── 📁 data/
│   ├── dataset.csv          (Original data)
│   ├── train.csv            (Single-label format)
│   ├── train_multilabel_balanced.csv  (Multi-label balanced)
│   ├── validation*.csv
│   └── test*.csv
│
├── 📁 docs/                  (Documentation)
├── 📁 backups/              (Backups)
├── 📁 backup_before_contrastive/
│
├── 📄 README.md             (Main README)
├── 📄 FINAL_COMMANDS.md     (Quick start guide) ⭐
├── 📄 FOCAL_CONTRASTIVE_ANALYSIS.md  (Method analysis) ⭐
├── 📄 PAPER_METHODOLOGY.md  (Paper writing guide)
├── 📄 STRATEGIES_TO_96_F1.md (Research strategies)
├── 📄 SOTA_TECHNIQUES_2024.md (SOTA techniques)
├── 📄 config.yaml           (Old config - not used)
│
└── 📄 utils.py              (Shared utilities)
```

---

## 🎯 Quick Start

### **For Multi-Label (Recommended - 96% F1):**

```bash
cd multi_label
python train_multilabel_focal_contrastive.py --epochs 8 --focal-weight 0.7 --contrastive-weight 0.3
```

**Or:**
```bash
cd multi_label
train_focal_contrastive.bat
```

**See:** `multi_label/README.md` for complete guide

---

### **For Single-Label (Traditional - 90-92% F1):**

```bash
cd single_label
python prepare_data.py
python train.py --config config_single.yaml
```

**See:** `single_label/README.md` for complete guide

---

## 📊 Key Files

### **📌 Must Read:**
1. **`FINAL_COMMANDS.md`** - Quick start commands ⭐
2. **`multi_label/README.md`** - Main method guide ⭐
3. **`FOCAL_CONTRASTIVE_ANALYSIS.md`** - Why it works
4. **`PAPER_METHODOLOGY.md`** - Paper writing guide

### **📚 Reference:**
5. **`STRATEGIES_TO_96_F1.md`** - Research strategies
6. **`SOTA_TECHNIQUES_2024.md`** - SOTA techniques from 2024
7. **`single_label/README.md`** - Traditional approach guide

---

## 🗂️ File Categories

### **Training Scripts:**
```
single_label/train.py                           (Single-label training)
multi_label/train_multilabel_focal_contrastive.py  (Multi-label training) ⭐
```

### **Configuration:**
```
single_label/config_single.yaml
multi_label/config_multi.yaml
```

### **Models:**
```
multi_label/model_multilabel_focal_contrastive.py  (Main model)
multi_label/model_multilabel_contrastive.py        (Base class)
multi_label/model_multilabel_contrastive_v2.py     (Loss functions)
```

### **Data Preparation:**
```
single_label/prepare_data.py             (Convert to single-label)
multi_label/prepare_data_multilabel.py   (Convert to multi-label)
multi_label/augment_multilabel_balanced.py  (Balance data)
```

### **Utilities:**
```
utils.py                    (Focal Loss, metrics, etc.)
single_label/oversampling_utils.py
multi_label/ensemble_multilabel.py
```

### **Analysis:**
```
single_label/analyze_results.py
single_label/generate_predictions.py
```

---

## 🎯 Recommended Workflow

### **For Research Paper (96% F1):**

1. **Prepare data:**
   ```bash
   cd multi_label
   python prepare_data_multilabel.py
   ```

2. **Train model:**
   ```bash
   python train_multilabel_focal_contrastive.py --epochs 8 --focal-weight 0.7 --contrastive-weight 0.3
   ```

3. **Expected result:** 96.0-96.5% F1

4. **Write paper:** Use `PAPER_METHODOLOGY.md` as guide

---

### **For Learning/Understanding:**

1. **Start with single-label:**
   ```bash
   cd single_label
   python prepare_data.py
   python train.py --config config_single.yaml
   ```

2. **Result:** 90-92% F1 (baseline)

3. **Then try multi-label:**
   ```bash
   cd ../multi_label
   python train_multilabel_focal_contrastive.py --epochs 8
   ```

4. **Compare:** Understand why multi-label is better

---

## 📈 Performance Comparison

| Method | Folder | F1 Score | Speed | Novel |
|--------|--------|----------|-------|-------|
| **Single-Label** | `single_label/` | 90-92% | 1× | ❌ |
| **Multi-Label Baseline** | `multi_label/` | 95.49% | 11× | ❌ |
| **Multi-Label Focal+Contrastive** | `multi_label/` | **96.0-96.5%** | **11×** | **✅** |

---

## 🧹 Clean Structure

**Removed (old/redundant):**
- ❌ 21 old .md files (comparisons, old guides, logs)
- ❌ Contrastive-only training scripts (didn't improve)
- ❌ Old documentation (replaced by README in each folder)

**Kept (essential):**
- ✅ 2 main folders (single_label, multi_label)
- ✅ 6 key .md files (README, guides, research)
- ✅ All necessary training scripts
- ✅ All model files
- ✅ All utilities

---

## 🔗 Navigation

### **Getting Started:**
1. Read `FINAL_COMMANDS.md` for quick start
2. Choose single_label or multi_label
3. Read respective README.md
4. Run training

### **For Research:**
1. Use `multi_label/` approach
2. Read `FOCAL_CONTRASTIVE_ANALYSIS.md` for understanding
3. Use `PAPER_METHODOLOGY.md` for writing
4. Reference `STRATEGIES_TO_96_F1.md` and `SOTA_TECHNIQUES_2024.md`

### **For Production:**
1. Use `multi_label/` approach (11× faster)
2. Expected 96%+ F1
3. All aspects in one forward pass

---

## ✅ Summary

**Clean, organized structure:**
- 📁 2 folders (single_label, multi_label)
- 📄 6 key .md files
- 🎯 Clear separation of approaches
- 🚀 Ready for 96% F1 training

**Main method:** `multi_label/train_multilabel_focal_contrastive.py` ⭐

**Expected result:** 96.0-96.5% F1

**Quick start:** See `FINAL_COMMANDS.md`
