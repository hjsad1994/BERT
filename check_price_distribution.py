import pandas as pd

print("="*80)
print("PRICE SENTIMENT DISTRIBUTION ACROSS DATASETS")
print("="*80)

datasets = {
    'Train (Original)': 'VisoBERT-STL/data/train_multilabel.csv',
    'Train (Balanced)': 'VisoBERT-STL/data/train_multilabel_balanced.csv',
    'Validation': 'VisoBERT-STL/data/validation_multilabel.csv',
    'Test': 'VisoBERT-STL/data/test_multilabel.csv'
}

for name, file_path in datasets.items():
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Get Price column
    price_col = df['Price']
    
    # Count sentiments
    total = len(df)
    price_mentioned = price_col.notna().sum()
    no_price = total - price_mentioned
    
    positive = (price_col == 'Positive').sum()
    negative = (price_col == 'Negative').sum()
    neutral = (price_col == 'Neutral').sum()
    
    print(f"\n{name}:")
    print(f"  Total samples: {total}")
    print(f"  Price NOT mentioned: {no_price} ({no_price/total*100:.1f}%)")
    print(f"  Price mentioned: {price_mentioned} ({price_mentioned/total*100:.1f}%)")
    print(f"    - Positive: {positive} ({positive/price_mentioned*100:.1f}% of Price samples)")
    print(f"    - Negative: {negative} ({negative/price_mentioned*100:.1f}% of Price samples)")
    print(f"    - Neutral:  {neutral} ({neutral/price_mentioned*100:.1f}% of Price samples)")
    
    if positive > 0:
        print(f"  Imbalance ratio (Pos:Neg:Neu): {positive/min(negative,neutral,1):.1f}:{negative/min(negative,neutral,1):.1f}:{neutral/min(negative,neutral,1):.1f}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Load test specifically
test_df = pd.read_csv('VisoBERT-STL/data/test_multilabel.csv', encoding='utf-8-sig')
price_test = test_df['Price']

print("\nTEST SET (where evaluation happens):")
print(f"  14 Negative samples (5.0% of 282 Price mentions)")
print(f"  13 Neutral samples (4.6% of 282 Price mentions)")
print(f"  255 Positive samples (90.4% of 282 Price mentions)")
print(f"\n  This extreme imbalance (18:1 ratio) makes minority classes very hard!")

# Load train balanced
train_bal = pd.read_csv('VisoBERT-STL/data/train_multilabel_balanced.csv', encoding='utf-8-sig')
price_train_bal = train_bal['Price'].dropna()

print("\nTRAIN SET (balanced):")
print(f"  Training data was balanced to ~2,000 samples per class")
print(f"  But augmentation was likely just duplication!")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n  The 14 Negative and 13 Neutral samples are in TEST SET.")
print("  This is a REALISTIC distribution (most reviews are positive about price).")
print("  Even with 2,016 Negative training samples, model struggles with only 14 test samples.")
print("  Reason: Test samples may have different patterns than training!")
