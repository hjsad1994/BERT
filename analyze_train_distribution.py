"""
Analyze sentiment distribution in training data
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

print("="*80)
print("ANALYZING TRAINING DATA DISTRIBUTION")
print("="*80)

# Load balanced training data
train_df = pd.read_csv('multi_label/data/train_multilabel_balanced.csv', encoding='utf-8-sig')
print(f"\nTotal sentences: {len(train_df):,}")

# Get aspect columns
aspect_cols = [col for col in train_df.columns if col != 'data']
print(f"Aspects: {len(aspect_cols)}")

# Count sentiments across all aspects
sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'NaN': 0}

for aspect in aspect_cols:
    for val in train_df[aspect]:
        if pd.isna(val):
            sentiment_counts['NaN'] += 1
        else:
            val_str = str(val).strip().capitalize()
            if val_str in sentiment_counts:
                sentiment_counts[val_str] += 1

# Calculate percentages
total_labels = sum(sentiment_counts.values())
print(f"\nTotal aspect-labels: {total_labels:,}")
print("\n" + "="*80)
print("SENTIMENT DISTRIBUTION")
print("="*80)

for sentiment, count in sorted(sentiment_counts.items(), key=lambda x: -x[1]):
    pct = count / total_labels * 100
    print(f"{sentiment:<12} {count:>8,} ({pct:>6.2f}%)")

# Calculate ratio
labeled_total = sentiment_counts['Positive'] + sentiment_counts['Negative'] + sentiment_counts['Neutral']
print(f"\n{'─'*80}")
print(f"Labeled:     {labeled_total:>8,} ({labeled_total/total_labels*100:>6.2f}%)")
print(f"Unlabeled:   {sentiment_counts['NaN']:>8,} ({sentiment_counts['NaN']/total_labels*100:>6.2f}%)")

# Check class balance among LABELED aspects only
print("\n" + "="*80)
print("BALANCE AMONG LABELED ASPECTS ONLY")
print("="*80)

pos_pct = sentiment_counts['Positive'] / labeled_total * 100
neg_pct = sentiment_counts['Negative'] / labeled_total * 100
neu_pct = sentiment_counts['Neutral'] / labeled_total * 100

print(f"Positive:    {sentiment_counts['Positive']:>8,} ({pos_pct:>6.2f}%)")
print(f"Negative:    {sentiment_counts['Negative']:>8,} ({neg_pct:>6.2f}%)")
print(f"Neutral:     {sentiment_counts['Neutral']:>8,} ({neu_pct:>6.2f}%)")

# Check if balanced
balance_ratio = max(pos_pct, neg_pct, neu_pct) / min(pos_pct, neg_pct, neu_pct)
print(f"\nBalance Ratio (max/min): {balance_ratio:.2f}x")

if balance_ratio > 1.5:
    print("⚠️  IMBALANCED! Largest class is >1.5x smallest class")
    print(f"   This causes {'Positive' if pos_pct == max(pos_pct, neg_pct, neu_pct) else 'Negative' if neg_pct == max(pos_pct, neg_pct, neu_pct) else 'Neutral'} BIAS!")
else:
    print("✅ BALANCED! All classes within 1.5x ratio")

# Per-aspect distribution
print("\n" + "="*80)
print("PER-ASPECT SENTIMENT DISTRIBUTION")
print("="*80)

for aspect in aspect_cols:
    aspect_data = train_df[aspect]
    
    # Count
    pos = (aspect_data == 'Positive').sum() + (aspect_data == 'positive').sum()
    neg = (aspect_data == 'Negative').sum() + (aspect_data == 'negative').sum()
    neu = (aspect_data == 'Neutral').sum() + (aspect_data == 'neutral').sum()
    nan_count = aspect_data.isna().sum()
    
    labeled = pos + neg + neu
    
    if labeled > 0:
        print(f"\n{aspect}:")
        print(f"  Pos: {pos:>4} ({pos/labeled*100:>5.1f}%)  "
              f"Neg: {neg:>4} ({neg/labeled*100:>5.1f}%)  "
              f"Neu: {neu:>4} ({neu/labeled*100:>5.1f}%)  "
              f"(Labeled: {labeled:>5}, NaN: {nan_count:>5})")
        
        # Check imbalance
        if neu > pos + neg:
            print(f"  ⚠️  NEUTRAL DOMINATES! ({neu/labeled*100:.1f}%)")

print("\n" + "="*80)
