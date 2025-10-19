# Hướng dẫn cải thiện model cho câu có "nhưng"

## 📊 Vấn đề phát hiện:

Model hiện tại **yếu rõ rệt** trên câu có từ chuyển ý "nhưng":

- **Overall accuracy**: 91.34%
- **Accuracy trên câu "nhưng"**: 79.57%
- **Chênh lệch**: **-11.77%** ⚠️

**Lý do:**
- Câu có "nhưng" thường có sentiment đảo ngược ("Pin tốt nhưng camera tệ")
- Model chưa học đủ patterns này (chỉ 12.3% training data có "nhưng")
- Phần SAU "nhưng" thường quan trọng hơn nhưng model chưa nhận ra

## 🎯 Giải pháp: Data Augmentation

**Đã tạo sẵn 2 augmented training files:**

### Option 1: Basic Augmentation (Khuyến nghị) ⭐
- **File:** `data/train_augmented_nhung.csv`
- **Samples:** 15,542 (tăng +24.7%)
- **Samples có "nhưng":** 4,617 (29.7% của total)
- **Cách thức:** Oversample 3x các samples có "nhưng"

### Option 2: Advanced Augmentation (Thử nghiệm)
- **File:** `data/train_augmented_nhung_advanced.csv`
- **Samples:** 15,750 (tăng +26.4%)
- **Cách thức:** Oversample 3x các samples có từ chuyển ý (nhưng, tuy nhiên, mặc dù, song...)

## 🚀 Cách sử dụng:

### Bước 1: Update config.yaml

Mở file `config.yaml` và thay đổi:

```yaml
# Từ:
train_file: data/train.csv

# Thành (chọn 1 trong 2):
train_file: data/train_augmented_nhung.csv          # Option 1: Basic
# train_file: data/train_augmented_nhung_advanced.csv  # Option 2: Advanced
```

### Bước 2: Retrain model

```bash
python train.py
```

**Lưu ý:**
- Training time sẽ tăng ~25% do data lớn hơn
- Expected: 6-7 tiếng (với batch size 16)
- Có thể tăng batch size nếu GPU cho phép

### Bước 3: Test lại performance

Sau khi train xong:

```bash
# Generate predictions
python generate_test_predictions.py

# Analyze errors với "nhưng"
python analyze_nhung_errors.py
```

## 📈 Kết quả mong đợi:

**Trước augmentation:**
- Overall accuracy: 91.34%
- Accuracy trên "nhưng": 79.57%
- Gap: -11.77%

**Sau augmentation (dự đoán):**
- Overall accuracy: ~91-92% (giữ nguyên hoặc tăng nhẹ)
- Accuracy trên "nhưng": **~85-88%** (tăng +5-8%)
- Gap: ~-3-4% (thu hẹp đáng kể)

## 🔍 Các giải pháp khác (nếu augmentation chưa đủ):

### 1. Rule-based Post-processing
Thêm logic xử lý riêng cho câu có "nhưng":

```python
def adjust_prediction_for_nhung(sentence, aspect, predicted_sentiment):
    if 'nhưng' in sentence.lower():
        # Split tại vị trí "nhưng"
        parts = sentence.lower().split('nhưng')
        
        # Phần sau "nhưng" thường quan trọng hơn
        after_nhung = parts[1] if len(parts) > 1 else ""
        
        # Check aspect trong phần sau "nhưng"
        if aspect.lower() in after_nhung:
            # Phân tích lại phần này với weight cao hơn
            # hoặc đảo ngược prediction nếu cần
            pass
    
    return predicted_sentiment
```

### 2. Special Token [ADV]
Thêm special token để highlight từ chuyển ý:

```python
# Trước khi tokenize:
sentence = sentence.replace('nhưng', '[ADV] nhưng')
# "Pin tốt nhưng camera tệ" → "Pin tốt [ADV] nhưng camera tệ"
```

**Cần:**
- Thêm [ADV] vào tokenizer vocabulary
- Retrain từ đầu

### 3. Ensemble với model chuyên biệt
- Train model riêng chỉ trên data có "nhưng"
- Combine predictions với main model
- Weight: 70% main + 30% specialized

### 4. Context Attention
Tăng attention vào phần sau "nhưng":

```python
# Trong training, tăng weight cho tokens sau "nhưng"
# Sử dụng position-aware loss weighting
```

## 📁 Files đã tạo:

```
D:/BERT/
├── data/
│   ├── train.csv                               # Original
│   ├── train_augmented_nhung.csv              # Basic augmentation ⭐
│   └── train_augmented_nhung_advanced.csv     # Advanced augmentation
├── error_analysis_results/
│   ├── all_errors_detailed.csv                # Tất cả errors
│   └── nhung_errors_detailed.csv             # Chỉ errors có "nhưng"
├── analyze_nhung_errors.py                    # Script phân tích
├── augment_nhung_samples.py                   # Script augmentation
└── NHUNG_IMPROVEMENT_GUIDE.md                # File này
```

## 🔬 Phân tích chi tiết errors:

Xem file `error_analysis_results/nhung_errors_detailed.csv` để:
- Xem tất cả 38 errors có "nhưng"
- Phân tích confusion patterns
- Tìm patterns đặc biệt cần xử lý

Top confusion patterns:
1. **positive → negative** (11 cases, 28.9%)
   - Ví dụ: "Pin tốt nhưng hiệu năng tệ" (aspect: Battery)
   - Model dự đoán negative nhưng true label là positive
   - Lý do: Model focus sai phần (focus vào "tệ" thay vì "pin tốt")

2. **positive → neutral** (7 cases, 18.4%)
   - Model không chắc chắn về sentiment

3. **neutral → negative** (6 cases, 15.8%)
   - Model bị ảnh hưởng bởi từ tiêu cực sau "nhưng"

## 🎯 Quick Start:

```bash
# 1. Update config
# Sửa config.yaml: train_file: data/train_augmented_nhung.csv

# 2. Retrain
python train.py

# 3. Test
python analyze_nhung_errors.py

# 4. Compare results
# Xem improvement trong accuracy trên câu có "nhưng"
```

## 💡 Tips:

1. **Start với Basic Augmentation** (option 1) - đơn giản và hiệu quả
2. **Monitor training loss** - đảm bảo không bị overfit
3. **Compare với baseline** - so sánh accuracy trên "nhưng" sentences
4. **Iterate** - nếu chưa đủ, thử advanced hoặc combine với rule-based

## 📞 Troubleshooting:

**Q: Training quá lâu?**
- A: Tăng batch size trong config.yaml (nếu GPU cho phép)

**Q: Accuracy giảm sau augmentation?**
- A: Có thể bị overfit trên "nhưng" samples. Giảm oversample_factor từ 3 → 2

**Q: Vẫn yếu trên "nhưng" sau retrain?**
- A: Thử combine với rule-based post-processing hoặc special token [ADV]

## 📚 References:

- Error analysis: `error_analysis_results/nhung_errors_detailed.csv`
- Full analysis script: `analyze_nhung_errors.py`
- Augmentation script: `augment_nhung_samples.py`
- Training config: `config.yaml`

---

**Created:** $(date)
**Author:** Droid AI Assistant
**Version:** 1.0
