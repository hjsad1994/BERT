import pandas as pd
import numpy as np

aspects = ['Battery', 'Camera', 'Performance', 'Display', 'Design', 
           'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General', 'Others']

# Check TEST set
print("=" * 80)
print("TEST SET - NaN vs Neutral Analysis")
print("=" * 80)

df_test = pd.read_csv('multi_label/data/test_multilabel.csv', encoding='utf-8-sig')

print(f"\nTotal reviews: {len(df_test)}")
print(f"\n{'Aspect':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10} {'NaN':<10}")
print("-" * 60)

for aspect in aspects:
    counts = df_test[aspect].value_counts()
    nan_count = df_test[aspect].isna().sum()
    
    pos = counts.get('Positive', 0)
    neg = counts.get('Negative', 0)
    neu = counts.get('Neutral', 0)
    
    print(f"{aspect:<15} {pos:<10} {neg:<10} {neu:<10} {nan_count:<10}")

# Sample some rows to see actual data
print("\n" + "=" * 80)
print("SAMPLE DATA (first 5 rows)")
print("=" * 80)
print(df_test[['data'] + aspects].head())

# Check if NaN is filled with 'Neutral'
print("\n" + "=" * 80)
print("Checking unique values per aspect:")
print("=" * 80)
for aspect in aspects:
    unique_vals = df_test[aspect].unique()
    print(f"{aspect}: {unique_vals}")
    
# Check TRAIN set
print("\n" + "=" * 80)
print("TRAIN SET (BALANCED) - NaN vs Neutral Analysis")
print("=" * 80)

df_train = pd.read_csv('multi_label/data/train_multilabel_balanced.csv', encoding='utf-8-sig')

print(f"\nTotal reviews: {len(df_train)}")
print(f"\n{'Aspect':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10} {'NaN':<10}")
print("-" * 60)

for aspect in aspects:
    counts = df_train[aspect].value_counts()
    nan_count = df_train[aspect].isna().sum()
    
    pos = counts.get('Positive', 0)
    neg = counts.get('Negative', 0)
    neu = counts.get('Neutral', 0)
    
    print(f"{aspect:<15} {pos:<10} {neg:<10} {neu:<10} {nan_count:<10}")
