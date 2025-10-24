"""Check why Shipping has low accuracy"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
from collections import Counter

print("="*80)
print("CHECKING SHIPPING ACCURACY DISCREPANCY")
print("="*80)

# Load error details
errors_df = pd.read_csv('multi_label/error_analysis_results/all_errors_detailed.csv', 
                        encoding='utf-8-sig')

# Filter Shipping
ship_df = errors_df[errors_df['aspect'] == 'Shipping']
print(f"\nShipping errors: {len(ship_df)}")
print(f"Total Shipping samples (from error analysis): 153")
print(f"Correct: {153 - len(ship_df)} = {(153-len(ship_df))/153*100:.1f}%")
print(f"Errors: {len(ship_df)} = {len(ship_df)/153*100:.1f}%")

# Check confusion
print("\n" + "="*80)
print("CONFUSION PATTERNS - SHIPPING")
print("="*80)

confusion = Counter(zip(ship_df['sentiment'], ship_df['predicted_sentiment']))
print("\nTrue → Predicted:")
for (true, pred), count in confusion.most_common(10):
    print(f"  {true:<10} → {pred:<10} : {count:>3} errors")

# Sample errors
print("\n" + "="*80)
print("SAMPLE ERRORS (First 5)")
print("="*80)

for idx, row in ship_df.head(5).iterrows():
    print(f"\nText: {row['data'][:80]}...")
    print(f"  True: {row['sentiment']} → Pred: {row['predicted_sentiment']} ❌")

# Check test file to verify ground truth
print("\n" + "="*80)
print("VERIFY GROUND TRUTH IN TEST FILE")
print("="*80)

test_df = pd.read_csv('multi_label/data/test_multilabel.csv', encoding='utf-8-sig')
shipping_labeled = test_df['Shipping'].notna().sum()
shipping_total = len(test_df)

print(f"\nTest file:")
print(f"  Total samples: {shipping_total}")
print(f"  Shipping labeled: {shipping_labeled} ({shipping_labeled/shipping_total*100:.1f}%)")
print(f"  Shipping NaN: {shipping_total - shipping_labeled} ({(shipping_total-shipping_labeled)/shipping_total*100:.1f}%)")

# Check distribution
ship_dist = test_df['Shipping'].value_counts()
print(f"\nShipping distribution in test:")
for label, count in ship_dist.items():
    print(f"  {label}: {count}")

print("\n" + "="*80)
print("HYPOTHESIS")
print("="*80)

print("""
Training evaluation:  95.42% F1 (Shipping, 153 labeled samples)
Error analysis:       13.07% accuracy (20/153 correct)

HUGE DISCREPANCY! (~82 points difference!)

Possible causes:
1. ❓ Different evaluation methodology
2. ❓ Predictions file có vấn đề
3. ❓ Error analysis code có bug
4. ❓ Model output không consistent

Need to check:
- What does training evaluation actually measure?
- What does error analysis actually measure?
- Are they looking at the same data?
""")

print("="*80)
