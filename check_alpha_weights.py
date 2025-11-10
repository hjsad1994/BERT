"""
Check alpha weights from both original and balanced data
"""
import pandas as pd
from collections import Counter

def calculate_alpha(file_path, aspects):
    """Calculate alpha weights from data"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # Collect all sentiments
    all_sentiments = []
    for aspect in aspects:
        if aspect in df.columns:
            sentiments = df[aspect].dropna()
            sentiments = sentiments.astype(str).str.strip().str.lower()
            all_sentiments.extend(sentiments.tolist())
    
    # Count
    counts = Counter(all_sentiments)
    total = sum(counts.values())
    
    print(f"\n   File: {file_path}")
    print(f"   Total: {total:,}")
    print(f"   Distribution:")
    for sent in ['positive', 'negative', 'neutral']:
        count = counts.get(sent, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"     {sent:10s}: {count:6,} ({pct:5.2f}%)")
    
    # Calculate alpha (inverse frequency)
    num_classes = 3
    alpha = []
    for sent in ['positive', 'negative', 'neutral']:
        count = max(counts.get(sent, 0), 1)
        weight = total / (num_classes * count)
        alpha.append(weight)
    
    print(f"   Alpha weights:")
    print(f"     positive: {alpha[0]:.4f}")
    print(f"     negative: {alpha[1]:.4f}")
    print(f"     neutral:  {alpha[2]:.4f}")
    
    return alpha

aspects = ['Battery', 'Camera', 'Performance', 'Display', 'Design',
           'Packaging', 'Price', 'Shop_Service', 'Shipping', 'General']

print("=" * 80)
print("ALPHA WEIGHTS COMPARISON")
print("=" * 80)

print("\n1. ORIGINAL DATA (train_multilabel.csv):")
alpha_old = calculate_alpha('VisoBERT-STL/data/train_multilabel.csv', aspects)

print("\n" + "=" * 80)
print("\n2. BALANCED DATA (train_multilabel_balanced.csv):")
alpha_new = calculate_alpha('VisoBERT-STL/data/train_multilabel_balanced.csv', aspects)

print("\n" + "=" * 80)
print("\nCOMPARISON:")
print(f"{'Sentiment':<12} {'Original':<12} {'Balanced':<12} {'Change':<12}")
print("-" * 50)
for i, sent in enumerate(['positive', 'negative', 'neutral']):
    change = alpha_new[i] - alpha_old[i]
    change_pct = (change / alpha_old[i] * 100) if alpha_old[i] > 0 else 0
    print(f"{sent:<12} {alpha_old[i]:<12.4f} {alpha_new[i]:<12.4f} {change_pct:+.1f}%")

print("\n" + "=" * 80)
print("\nIMPACT:")
print("  - Original: Neutral weight VERY HIGH (4.23) because only 7.88% of data")
print("  - Balanced: Neutral weight LOWER because oversampled to ~30% of data")
print("  - Result: More balanced training, better generalization")
print("=" * 80)
