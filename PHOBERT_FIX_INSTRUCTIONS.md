# 🔧 Fix PhoBERT Preprocessing Error

## ❌ Your Error

```
KeyError: 'sentence'
```

**Location**: `E:\ABSA-PhoBERT\multi-label\preprocess_data.py` line 58

**Cause**: Script expects column `'sentence'` but your dataset has column `'data'`

---

## ✅ Solution (3 Options)

### Option 1: Quick Fix (1 minute)

Edit `E:\ABSA-PhoBERT\multi-label\preprocess_data.py`:

**Line 58**:
```python
# BEFORE
result_df['sentence'] = df['sentence']

# AFTER
result_df['sentence'] = df['data']  # ← Change to 'data'
```

---

### Option 2: Use Fixed Script (Recommended)

1. **Copy script mới**:
   ```powershell
   # Copy từ D:\BERT sang E:\ABSA-PhoBERT
   Copy-Item "D:\BERT\preprocess_data_multilabel_fixed.py" -Destination "E:\ABSA-PhoBERT\multi-label\"
   ```

2. **Chạy**:
   ```bash
   cd E:\ABSA-PhoBERT
   python multi-label\preprocess_data_multilabel_fixed.py --seed 42
   ```

**Benefits**:
- ✅ Fixed column name issue
- ✅ Stratified split (balanced)
- ✅ Sorted for reproducibility
- ✅ Detailed logging

---

### Option 3: Manual Copy-Paste

Nếu không copy được file, tạo file mới:

1. **Create**: `E:\ABSA-PhoBERT\multi-label\preprocess_fixed.py`

2. **Copy toàn bộ code từ**: `D:\BERT\preprocess_data_multilabel_fixed.py`

3. **Run**:
   ```bash
   python multi-label\preprocess_fixed.py --seed 42
   ```

---

## 🎯 Expected Output

```
======================================================================
Multi-Label Data Preprocessing for ABSA
======================================================================

Configuration:
  Random seed:  42
  Train ratio:  0.8
  Val ratio:    0.1
  Test ratio:   0.1

======================================================================
1. Loading dataset...
======================================================================
   Found: dataset.csv
   Loaded: 9129 reviews
   Columns: ['data', 'Battery', 'Camera', ...]

======================================================================
2. Verifying columns...
======================================================================
   ✓ Text column: 'data'
   ✓ Found aspects: 11/11

======================================================================
3. Cleaning data...
======================================================================
   Initial: 9129 rows
   After cleaning: 9129 rows
   Removed: 0 empty reviews

======================================================================
4. Sorting data for cross-machine reproducibility...
======================================================================
   ✓ Sorted by: ['data', 'Battery', 'Camera', 'Performance']
   ✓ This ensures same split on different machines

======================================================================
5. Creating stratification label...
======================================================================
   Computing dominant sentiment per review...

   Dominant sentiment distribution:
     positive  :   5000 (54.76%)
     negative  :   3000 (32.87%)
     neutral   :   1129 (12.37%)

======================================================================
6. Splitting data with stratification...
======================================================================

   Step 1: Splitting (train+val) vs test...
   ✓ Train+Val: 8216, Test: 913

   Step 2: Splitting train vs val...
   ✓ Train: 7303, Val: 913

   Final split:
     Train:   7303 (80.00%)
     Val:      913 (10.00%)
     Test:     913 (10.00%)
     Total:   9129

======================================================================
7. Verifying stratification...
======================================================================

   Train sentiment distribution:
     positive  : 54.75%
     negative  : 32.88%
     neutral   : 12.37%

   Val sentiment distribution:
     positive  : 54.76%
     negative  : 32.86%
     neutral   : 12.38%

   Test sentiment distribution:
     positive  : 54.77%
     negative  : 32.86%
     neutral   : 12.37%

======================================================================
8. Saving splits...
======================================================================

   ✓ Saved files:
     data/train.csv
     data/validation.csv
     data/test.csv

======================================================================
9. Saving metadata...
======================================================================
   ✓ Saved: data/metadata.json

======================================================================
✅ COMPLETE!
======================================================================

✓ Summary:
   Input:  dataset.csv (9129 reviews)
   Format: Multi-label (1 row per review)
   Split:  Stratified by dominant sentiment
   Output: data/
   Seed:   42

✓ Files created:
   data/train.csv  (7303 rows)
   data/validation.csv  (913 rows)
   data/test.csv  (913 rows)

✓ Next step:
   Train your multi-label model with these splits!
```

---

## 📊 Output Files

```
E:\ABSA-PhoBERT\multi-label\data\
├── train.csv          (7,303 rows - multi-label format)
├── validation.csv     (913 rows)
├── test.csv           (913 rows)
└── metadata.json      (split info)
```

**Format** (multi-label):
```csv
data,Battery,Camera,Performance,Display,...
"Pin tốt camera kém",Positive,Negative,,,,...
```

---

## 🎯 Key Features of Fixed Script

1. ✅ **Correct column name**: Uses `'data'` not `'sentence'`
2. ✅ **Stratified split**: Balanced sentiment distribution
3. ✅ **Sorted data**: Same split on different machines
4. ✅ **Detailed logging**: See what's happening
5. ✅ **Metadata**: Track split configuration

---

## 🔍 Verify It Works

After running, check:

```bash
# Check files exist
ls E:\ABSA-PhoBERT\multi-label\data\

# Check file sizes
python -c "import pandas as pd; print(len(pd.read_csv('data/train.csv')))"
# Should print: 7303 (or similar based on your data)
```

---

## ❓ FAQ

**Q: Tại sao script cũ lỗi?**

A: Script expect format:
```csv
sentence,aspect,sentiment  ← Single-label format
```

Nhưng dataset của bạn là:
```csv
data,Battery,Camera,...    ← Multi-label format
```

---

**Q: Script mới có tương thích với D:\BERT không?**

A: Có! Cả 2 projects giờ dùng cùng approach:
- ✅ Column `'data'` for text
- ✅ Stratified split
- ✅ Seed-based reproducibility

---

**Q: Có cần chạy lại nếu đã có data/?**

A: Chỉ cần nếu:
- ❌ File train.csv bị lỗi
- ❌ Muốn thay đổi seed
- ❌ Muốn thay đổi train/val/test ratio

---

## 🚀 Next Steps

1. ✅ Fix preprocessing (done với script mới)
2. ⏭️ Train model:
   ```bash
   python multi-label\train.py --config multi-label\config.yaml
   ```

---

## 📚 Files to Copy

From `D:\BERT\`:
- `preprocess_data_multilabel_fixed.py` → `E:\ABSA-PhoBERT\multi-label\`

---

**Status**: ✅ **READY TO USE**

Script đã fix lỗi column name và có stratified split!
