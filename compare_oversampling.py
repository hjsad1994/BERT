"""Compare original vs balanced (oversampled) data"""
import pandas as pd

# Load data
orig = pd.read_csv('VisoBERT-STL/data/train_multilabel.csv', encoding='utf-8-sig')
bal = pd.read_csv('VisoBERT-STL/data/train_multilabel_balanced.csv', encoding='utf-8-sig')

print(f"{'='*80}")
print(f"OVERSAMPLING COMPARISON")
print(f"{'='*80}\n")

print(f"Original: {len(orig):,} samples")
print(f"Balanced: {len(bal):,} samples")
print(f"Oversampling ratio: {len(bal)/len(orig):.2f}x\n")

print(f"{'='*80}")
print(f"SENTIMENT DISTRIBUTION PER ASPECT")
print(f"{'='*80}\n")

aspects = ['Battery', 'Camera', 'Performance', 'Display', 'Design']

for aspect in aspects:
    print(f"{aspect}:")
    print(f"  Original:")
    orig_counts = orig[aspect].value_counts()
    for sent in ['Positive', 'Negative', 'Neutral']:
        count = orig_counts.get(sent, 0)
        print(f"    {sent}: {count}")
    
    print(f"  Balanced:")
    bal_counts = bal[aspect].value_counts()
    for sent in ['Positive', 'Negative', 'Neutral']:
        count = bal_counts.get(sent, 0)
        ratio = count / orig_counts.get(sent, 1) if sent in orig_counts else 0
        print(f"    {sent}: {count} ({ratio:.2f}x)")
    print()
