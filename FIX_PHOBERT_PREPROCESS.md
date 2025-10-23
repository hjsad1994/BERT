# 🔧 Fix PhoBERT preprocess_data.py

## ❌ Error

```
KeyError: 'sentence'
```

**Nguyên nhân**: 
- Script expect column `'sentence'` 
- Dataset có column `'data'`

---

## ✅ Solution

### Option 1: Fix Script (Quick)

Open `E:\ABSA-PhoBERT\multi-label\preprocess_data.py` và tìm dòng:

```python
result_df['sentence'] = df['sentence']  # Line 58
```

**Sửa thành**:
```python
result_df['sentence'] = df['data']  # Use 'data' column instead
```

---

### Option 2: Copy Updated Script from D:\BERT

Bạn có script đã update ở `D:\BERT\single_label\preprocess_data.py` với stratified split.

**Copy sang PhoBERT project**:

```bash
# Copy script
cp D:\BERT\single_label\preprocess_data.py E:\ABSA-PhoBERT\multi-label\

# Hoặc dùng PowerShell
Copy-Item "D:\BERT\single_label\preprocess_data.py" -Destination "E:\ABSA-PhoBERT\multi-label\"
```

**Rồi chạy**:
```bash
cd E:\ABSA-PhoBERT
python multi-label\preprocess_data.py --config multi-label\config.yaml
```

---

## 🎯 Recommended: Use Correct Script

Dataset của bạn format:
```csv
data,Battery,Camera,Performance,...
"Pin tốt camera kém",Positive,Negative,...
```

→ Đây là **MULTI-LABEL format** (1 row per review)

**Script cần**:
- Input: `dataset.csv` with column `'data'`
- Keep multi-label format (không convert sang single-label)
- Split with stratification

---

## 📝 Full Fix Script

Tôi tạo script mới đúng cho bạn:

```python
"""
Multi-Label Data Preprocessing for PhoBERT ABSA
Keeps multi-label format: 1 row per review with all aspects
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import yaml

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()
    seed = config['reproducibility']['data_split_seed']
    
    print("="*60)
    print("Multi-Label Data Preprocessing")
    print("="*60)
    print(f"Seed: {seed}")
    
    # Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv('dataset.csv', encoding='utf-8-sig')
    print(f"   Loaded: {len(df)} reviews")
    
    # Verify columns
    expected_cols = ['data', 'Battery', 'Camera', 'Performance', 
                     'Display', 'Design', 'Packaging', 'Price',
                     'Shop_Service', 'Shipping', 'General', 'Others']
    
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    print(f"   ✓ All columns present")
    
    # Sort for reproducibility across machines
    print("\n2. Sorting data for reproducibility...")
    df = df.sort_values(by=['data', 'Battery', 'Camera'], ignore_index=True)
    print(f"   ✓ Data sorted")
    
    # Split with stratification
    print("\n3. Splitting data (stratified)...")
    
    # Create stratify label (dominant sentiment)
    def get_dominant_sentiment(row):
        sentiments = []
        for col in expected_cols[1:]:  # Skip 'data'
            if pd.notna(row[col]) and str(row[col]).strip() != '':
                sentiments.append(str(row[col]).strip())
        
        if not sentiments:
            return 'neutral'
        
        from collections import Counter
        counter = Counter(sentiments)
        return counter.most_common(1)[0][0]
    
    df['stratify_temp'] = df.apply(get_dominant_sentiment, axis=1)
    
    # Split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=seed,
        stratify=df['stratify_temp'],
        shuffle=True
    )
    
    # Split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.11,  # 10% of total (10/90)
        random_state=seed,
        stratify=train_val_df['stratify_temp'],
        shuffle=True
    )
    
    # Remove stratify column
    train_df = train_df.drop('stratify_temp', axis=1)
    val_df = val_df.drop('stratify_temp', axis=1)
    test_df = test_df.drop('stratify_temp', axis=1)
    
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save
    print("\n4. Saving splits...")
    os.makedirs('data', exist_ok=True)
    
    train_df.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
    val_df.to_csv('data/validation.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('data/test.csv', index=False, encoding='utf-8-sig')
    
    print(f"   ✓ Saved to data/ folder")
    
    print("\n✓ Complete!")

if __name__ == '__main__':
    main()
```

**Lưu script này vào**: `E:\ABSA-PhoBERT\multi-label\preprocess_data_fixed.py`

**Chạy**:
```bash
cd E:\ABSA-PhoBERT
python multi-label\preprocess_data_fixed.py
```

---

## 🎯 Summary

**Lỗi**: Script expect `'sentence'` column, dataset có `'data'` column

**Fix**:
1. **Quick**: Sửa line 58: `df['sentence']` → `df['data']`
2. **Better**: Copy script updated từ D:\BERT
3. **Best**: Dùng script fixed ở trên (có stratification + sorting)

---

**Bạn muốn tôi tạo full script mới không?**
