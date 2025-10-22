# Multi-Label Test Scripts

## 📊 Scripts

### **1. analyze_results.py**

**Purpose:** Phân tích kết quả chi tiết theo từng aspect

**Features:**
- Confusion matrices cho từng aspect
- Bar charts về accuracy, precision, recall, F1
- Heatmap tổng hợp
- Báo cáo chi tiết dạng text và PNG
- Sample distribution analysis

**Usage:**
```bash
# From D:\BERT\
python multi_label\test\analyze_results.py
```

**Requirements:**
- `multi_label/results/test_predictions_multi.csv` (from training)

**Outputs:**
- `multi_label/analysis_results/*.png` (visualizations)
- `multi_label/analysis_results/detailed_analysis_report.txt`

---

### **2. error_analysis.py**

**Purpose:** Phân tích lỗi chi tiết để tìm patterns và cải thiện

**Features:**
- Tìm các predictions sai
- Phân tích theo aspect
- Phân tích theo sentiment
- Phân tích confusion patterns
- Tìm hard cases
- Đề xuất cải thiện

**Usage:**
```bash
# From D:\BERT\
python multi_label\test\error_analysis.py
```

**Requirements:**
- `multi_label/data/test_multilabel.csv` (ground truth)
- `multi_label/results/test_predictions_multi.csv` (predictions)

**Outputs:**
- `multi_label/error_analysis_results/aspect_error_analysis.csv`
- `multi_label/error_analysis_results/sentiment_error_analysis.csv`
- `multi_label/error_analysis_results/confusion_patterns.csv`
- `multi_label/error_analysis_results/hard_cases.csv`
- `multi_label/error_analysis_results/all_errors_detailed.csv`
- `multi_label/error_analysis_results/improvement_suggestions.txt`
- `multi_label/error_analysis_results/*.png` (visualizations)

---

## 🎯 Typical Workflow

### **After Training:**

**Option 1: Run Both Scripts at Once (Recommended)** 🚀
```bash
# From D:\BERT\
bash multi_label/test/run_analysis.sh
```
**This will run both analyze_results.py and error_analysis.py automatically!**

**Option 2: Run Scripts Individually**

1. **Run analyze_results.py:**
   ```bash
   python multi_label\test\analyze_results.py
   ```
   **Get:** Overall metrics, confusion matrices, visualizations

2. **Run error_analysis.py:**
   ```bash
   python multi_label\test\error_analysis.py
   ```
   **Get:** Detailed error analysis, hard cases, improvement suggestions

---

## 📊 Expected Outputs

### **Analysis Results:**
```
multi_label/analysis_results/
├── confusion_matrices_all_aspects.png
├── confusion_matrix_overall.png
├── metrics_comparison.png
├── accuracy_by_aspect.png
├── f1_score_by_aspect.png
├── precision_by_aspect.png
├── recall_by_aspect.png
├── sample_distribution.png
├── metrics_heatmap.png
├── summary_table.png
└── detailed_analysis_report.txt
```

### **Error Analysis Results:**
```
multi_label/error_analysis_results/
├── aspect_error_analysis.csv
├── sentiment_error_analysis.csv
├── confusion_patterns.csv
├── hard_cases.csv
├── all_errors_detailed.csv
├── errors_summary_by_aspect.csv
├── improvement_suggestions.txt
├── error_analysis_report.txt
├── aspect_error_rates.png
├── confusion_matrix.png
└── sentiment_error_rates.png
```

---

## ⚙️ Configuration

**Paths are pre-configured for multi-label:**
- Data: `multi_label/data/test_multilabel.csv`
- Predictions: `multi_label/results/test_predictions_multi.csv`
- Analysis output: `multi_label/analysis_results/`
- Error analysis output: `multi_label/error_analysis_results/`

**All paths are relative to project root (D:\BERT\)**

---

## 🔍 What to Look For

### **In analyze_results.py:**
- Overall F1 score (target: 96%+) 🎯
- Weakest aspects (lowest F1)
- Class imbalance in confusion matrix
- Aspect-wise performance variation

### **In error_analysis.py:**
- Error rate by aspect (target: < 5%)
- Most common confusion pairs
- Hard cases (sentences confused across multiple aspects)
- Improvement suggestions
- Focal Loss + Contrastive Learning effectiveness

---

## 📝 Notes

**Multi-Label Format:**
- Each row is one sentence with ALL 11 aspects
- Labels: 0=positive, 1=negative, 2=neutral
- One row per sentence (all aspects in columns)

**Novel Approach:**
- Using Focal Loss + Contrastive Learning
- Balanced data (15,921 samples)
- Target: 96%+ F1 score

**Run from root:** Always run from `D:\BERT\` directory

**Dependencies:**
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## ✅ Quick Test

**Verify scripts work:**
```bash
# From D:\BERT\
python multi_label\test\analyze_results.py
python multi_label\test\error_analysis.py
```

**Expected:** No errors, outputs created in respective folders

---

## 🎯 Performance Comparison

**Expected Results:**

```
Single-Label:  93.31% F1
Multi-Label:   96.0-96.5% F1 (target)

Improvement:   +2.7-3.2%
Method:        Focal Loss + Contrastive Learning
```

**Use these test scripts to verify results after training!** 🚀
