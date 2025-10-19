# Run Analysis Scripts Guide

Hướng dẫn sử dụng các script để chạy phân tích kết quả model.

## 📋 Scripts có sẵn

Có 3 script để chạy cả `analyze_results.py` và `tests/error_analysis.py` cùng lúc:

1. **`run_analysis.bat`** - Cho Windows Command Prompt (Khuyến nghị)
2. **`run_analysis.ps1`** - Cho Windows PowerShell (Có màu sắc đẹp hơn)
3. **`run_analysis.sh`** - Cho Linux/Mac/Git Bash

## 🚀 Cách chạy

### Option 1: Windows Command Prompt (CMD)

```cmd
run_analysis.bat
```

Hoặc double-click vào file `run_analysis.bat` trong Windows Explorer.

### Option 2: Windows PowerShell

```powershell
.\run_analysis.ps1
```

**Lưu ý:** Nếu gặp lỗi "cannot be loaded because running scripts is disabled", chạy lệnh sau với quyền Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Option 3: Git Bash / Linux / Mac

```bash
bash run_analysis.sh
```

Hoặc:

```bash
chmod +x run_analysis.sh
./run_analysis.sh
```

## 📊 Output

Script sẽ chạy 2 bước:

### Bước 1: `analyze_results.py`
Tạo các file trong thư mục `analysis_results/`:
- `confusion_matrices_all_aspects.png` - Grid confusion matrices
- `confusion_matrix_overall.png` - Overall confusion matrix
- `metrics_comparison.png` - So sánh metrics
- `accuracy_by_aspect.png` - Accuracy theo aspect
- `f1_score_by_aspect.png` - F1 score theo aspect
- `precision_by_aspect.png` - Precision theo aspect
- `recall_by_aspect.png` - Recall theo aspect
- `sample_distribution.png` - Phân bố samples
- `metrics_heatmap.png` - Heatmap metrics
- `summary_table.png` - Bảng tổng hợp
- `detailed_analysis_report.txt` - Báo cáo chi tiết

### Bước 2: `tests/error_analysis.py`
Tạo các file trong thư mục `error_analysis_results/`:
- `aspect_error_analysis.csv` - Phân tích lỗi theo aspect
- `sentiment_error_analysis.csv` - Phân tích lỗi theo sentiment
- `confusion_patterns.csv` - Patterns nhầm lẫn
- `hard_cases.csv` - Cases khó nhất
- `improvement_suggestions.txt` - Đề xuất cải thiện
- `error_analysis_report.txt` - Báo cáo lỗi
- `aspect_error_rates.png` - Error rates theo aspect
- `confusion_matrix.png` - Confusion matrix
- `sentiment_error_rates.png` - Error rates theo sentiment

## ✅ Requirements

Trước khi chạy script, đảm bảo đã có:

1. **`test_predictions.csv`** - Tạo bằng lệnh:
   ```bash
   python generate_test_predictions.py
   ```

2. **`data/test.csv`** - Tạo bằng lệnh:
   ```bash
   python prepare_data.py
   ```

3. **Trained model** - Phải có folder `finetuned_visobert_absa_model/`

## 🔍 Nếu gặp lỗi

### Error: test_predictions.csv not found
```bash
python generate_test_predictions.py
```

### Error: data/test.csv not found
```bash
python prepare_data.py
```

### Error: Model not found
```bash
python train.py
```

### PowerShell execution policy error
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 💡 Tips

- **Chạy riêng từng script** nếu cần debug:
  ```bash
  python analyze_results.py
  python tests/error_analysis.py
  ```

- **Xem kết quả nhanh**:
  ```bash
  # Windows
  explorer analysis_results
  explorer error_analysis_results
  
  # Linux/Mac
  open analysis_results/
  open error_analysis_results/
  ```

- **Chạy lại sau khi retrain**: Script tự động detect và overwrite kết quả cũ

## 📂 Cấu trúc thư mục sau khi chạy

```
D:/BERT/
├── analysis_results/           # Kết quả analyze_results.py
│   ├── *.png                   # Biểu đồ visualizations
│   └── *.txt                   # Báo cáo text
├── error_analysis_results/     # Kết quả error_analysis.py
│   ├── *.csv                   # Phân tích errors
│   ├── *.png                   # Biểu đồ errors
│   └── *.txt                   # Đề xuất improvements
├── run_analysis.bat            # Script Windows CMD
├── run_analysis.ps1            # Script PowerShell
└── run_analysis.sh             # Script Bash
```

## 🎯 Workflow hoàn chỉnh

```bash
# 1. Chuẩn bị data
python prepare_data.py

# 2. Train model
python train.py

# 3. Generate predictions
python generate_test_predictions.py

# 4. Run full analysis (sử dụng script)
run_analysis.bat              # Windows CMD
# hoặc
.\run_analysis.ps1            # PowerShell
# hoặc
bash run_analysis.sh          # Git Bash/Linux/Mac
```

Hoặc chỉ cần chạy `python train.py` vì nó sẽ tự động chạy `analyze_results.py` sau khi train xong!
