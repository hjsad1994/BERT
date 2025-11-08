import pandas as pd

print("="*80)
print("CHECKING AUGMENTATION RESULTS")
print("="*80)

# Load original and balanced data
train_orig = pd.read_csv('VisoBERT-STL/data/train_multilabel.csv', encoding='utf-8-sig')
train_bal = pd.read_csv('VisoBERT-STL/data/train_multilabel_balanced.csv', encoding='utf-8-sig')

print(f"\nTotal samples:")
print(f"  Original: {len(train_orig)}")
print(f"  Balanced: {len(train_bal)}")
print(f"  Increase: +{len(train_bal) - len(train_orig)} (+{(len(train_bal)-len(train_orig))/len(train_orig)*100:.1f}%)")

# Check Price specifically
print(f"\n" + "="*80)
print("PRICE SENTIMENT DISTRIBUTION")
print("="*80)

price_orig = train_orig['Price'].value_counts()
price_bal = train_bal['Price'].value_counts()

print(f"\nOriginal (train_multilabel.csv):")
for sent in ['Positive', 'Negative', 'Neutral']:
    count = (train_orig['Price'] == sent).sum()
    total = train_orig['Price'].notna().sum()
    print(f"  {sent:10}: {count:5} ({count/total*100:5.1f}%)")

print(f"\nBalanced (train_multilabel_balanced.csv):")
for sent in ['Positive', 'Negative', 'Neutral']:
    count = (train_bal['Price'] == sent).sum()
    total = train_bal['Price'].notna().sum()
    print(f"  {sent:10}: {count:5} ({count/total*100:5.1f}%)")

print(f"\n" + "-"*80)
print("OVERSAMPLING EFFECT:")
print("-"*80)
print(f"{'Sentiment':<12} {'Original':<10} {'Balanced':<10} {'Factor':<10}")
print("-"*80)

for sent in ['Positive', 'Negative', 'Neutral']:
    orig = (train_orig['Price'] == sent).sum()
    bal = (train_bal['Price'] == sent).sum()
    factor = bal / orig if orig > 0 else 0
    print(f"{sent:<12} {orig:<10} {bal:<10} x{factor:.1f}")

# Check if balanced
pos_bal = (train_bal['Price'] == 'Positive').sum()
neg_bal = (train_bal['Price'] == 'Negative').sum()
neu_bal = (train_bal['Price'] == 'Neutral').sum()

print(f"\n" + "="*80)
print("BALANCE CHECK")
print("="*80)

max_count = max(pos_bal, neg_bal, neu_bal)
min_count = min(pos_bal, neg_bal, neu_bal)
imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

print(f"\nAfter oversampling:")
print(f"  Max: {max_count}")
print(f"  Min: {min_count}")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio < 1.5:
    print(f"\n  ✓ WELL BALANCED (ratio < 1.5x)")
elif imbalance_ratio < 2.0:
    print(f"\n  ~ REASONABLY BALANCED (ratio < 2.0x)")
else:
    print(f"\n  ✗ STILL IMBALANCED (ratio >= 2.0x)")

print(f"\nTarget: Negative = Positive")
print(f"  Positive: {pos_bal}")
print(f"  Negative: {neg_bal}")
print(f"  Difference: {abs(pos_bal - neg_bal)}")

if abs(pos_bal - neg_bal) / max(pos_bal, neg_bal) < 0.1:
    print(f"  ✓ ACHIEVED: Negative ≈ Positive (within 10%)")
else:
    print(f"  ~ CLOSE: Negative ≈ {neg_bal/pos_bal*100:.0f}% of Positive")

# Check Design distribution (should NOT be balanced)
print(f"\n" + "="*80)
print("DESIGN SENTIMENT DISTRIBUTION (Should NOT be balanced)")
print("="*80)

design_orig = train_orig['Design'].value_counts()
design_bal = train_bal['Design'].value_counts()

print(f"\nOriginal:")
for sent in ['Positive', 'Negative', 'Neutral']:
    count = (train_orig['Design'] == sent).sum()
    total = train_orig['Design'].notna().sum()
    print(f"  {sent:10}: {count:5} ({count/total*100:5.1f}%)")

print(f"\nBalanced:")
for sent in ['Positive', 'Negative', 'Neutral']:
    count = (train_bal['Design'] == sent).sum()
    total = train_bal['Design'].notna().sum()
    print(f"  {sent:10}: {count:5} ({count/total*100:5.1f}%)")

design_pos_bal = (train_bal['Design'] == 'Positive').sum()
design_neg_bal = (train_bal['Design'] == 'Negative').sum()
design_neu_bal = (train_bal['Design'] == 'Neutral').sum()

design_max = max(design_pos_bal, design_neg_bal, design_neu_bal)
design_min = min(design_pos_bal, design_neg_bal, design_neu_bal)
design_ratio = design_max / design_min if design_min > 0 else float('inf')

print(f"\nDesign imbalance ratio: {design_ratio:.2f}x")

if design_ratio < 1.5:
    print(f"  ⚠️ WARNING: Design is BALANCED (should be natural distribution!)")
    print(f"  This will cause distribution mismatch with test set")
else:
    print(f"  ✓ GOOD: Design keeps natural distribution")

print(f"\n" + "="*80)
print("CONCLUSION")
print("="*80)

print(f"\n1. Price Negative oversampling:")
if neg_bal >= pos_bal * 0.9:
    print(f"   ✓ SUCCESS: Negative ({neg_bal}) ≈ Positive ({pos_bal})")
else:
    print(f"   ~ PARTIAL: Negative ({neg_bal}) = {neg_bal/pos_bal*100:.0f}% of Positive ({pos_bal})")

print(f"\n2. Augmentation method:")
print(f"   Current: DUPLICATION (same reviews copied multiple times)")
print(f"   Issue: No diversity in patterns")
print(f"   Recommendation: Use back-translation or paraphrasing")

print(f"\n3. Design distribution:")
if design_ratio < 1.5:
    print(f"   ⚠️ PROBLEM: Design is balanced (ratio {design_ratio:.2f}x)")
    print(f"   This causes mismatch with test set (98% Positive)")
    print(f"   Solution: Exclude Design from augmentation")
else:
    print(f"   ✓ OK: Design keeps natural distribution")
