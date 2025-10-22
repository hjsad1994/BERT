"""
Independent Augmentation Strategy
==================================
2 strategies riêng biệt, có thể overlap:

1. Oversample ALL Neutral samples (không quan tâm "nhưng")
   → Target: balance với Positive/Negative

2. Oversample ALL "nhưng" samples (không quan tâm sentiment)
   → Factor: x3 để model học tốt pattern phức tạp

Note: Samples vừa Neutral vừa có "nhưng" sẽ được oversample 2 lần
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import os
import numpy as np


def augment_neutral_and_nhung(
    train_file='single_label/data/train.csv',
    output_file='single_label/data/train_augmented_neutral_nhung.csv',
    neutral_target=3000,  # Target số lượng Neutral samples
    nhung_factor=3,       # Oversample "nhưng" samples x3
    overlap_strategy='max'  # 'max', 'neutral', 'nhung'
):
    """
    Independent augmentation: Neutral + "nhưng" WITHOUT OVERLAP
    
    Args:
        train_file: Input training file
        output_file: Output augmented file
        neutral_target: Target number of Neutral samples (None = auto balance)
        nhung_factor: Oversample factor for "nhưng" samples
        overlap_strategy: How to handle overlap samples
            - 'max': Use max(neutral_factor, nhung_factor)
            - 'neutral': Prioritize neutral oversample
            - 'nhung': Prioritize nhung oversample
    """
    print("="*80)
    print("🔄 INDEPENDENT AUGMENTATION: NEUTRAL + 'NHƯNG' (NO OVERLAP)")
    print("="*80)
    print("\nStrategy:")
    print(f"  1. Oversample Neutral (excluding overlap) → Target: ~{neutral_target} total")
    print(f"  2. Oversample 'nhưng' (excluding overlap) → Factor: x{nhung_factor}")
    print(f"  3. Overlap (Neutral + 'nhưng') → Strategy: '{overlap_strategy}'")
    print(f"  4. Note: Mỗi sample CHỈ được oversample 1 lần")
    
    os.chdir('D:/BERT')
    
    # Load data
    if not os.path.exists(train_file):
        print(f"\n❌ File không tồn tại: {train_file}")
        return
    
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    print(f"\n📊 Original data: {len(df)} samples")
    
    # Analyze original distribution
    print(f"\n📊 Original Sentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sent, count in sentiment_counts.items():
        print(f"  {sent:>10}: {count:>6} ({count/len(df)*100:>5.1f}%)")
    
    # Identify EXCLUSIVE groups (no overlap)
    is_neutral = df['sentiment'] == 'neutral'
    has_nhung = df['sentence'].str.contains('nhưng', case=False, na=False)
    
    # Group A: Overlap (Neutral + "nhưng")
    group_a_overlap = df[is_neutral & has_nhung].copy()
    
    # Group B: Only Neutral (no "nhưng")
    group_b_neutral_only = df[is_neutral & ~has_nhung].copy()
    
    # Group C: Only "nhưng" (not Neutral)
    group_c_nhung_only = df[~is_neutral & has_nhung].copy()
    
    # Group D: Baseline (no Neutral, no "nhưng")
    group_d_baseline = df[~is_neutral & ~has_nhung].copy()
    
    print(f"\n📊 EXCLUSIVE Groups (no overlap):")
    print(f"  Group A (Neutral + 'nhưng'):     {len(group_a_overlap):>5} ({len(group_a_overlap)/len(df)*100:>5.1f}%)")
    print(f"  Group B (Neutral only):          {len(group_b_neutral_only):>5} ({len(group_b_neutral_only)/len(df)*100:>5.1f}%)")
    print(f"  Group C ('Nhưng' only):          {len(group_c_nhung_only):>5} ({len(group_c_nhung_only)/len(df)*100:>5.1f}%)")
    print(f"  Group D (Baseline):              {len(group_d_baseline):>5} ({len(group_d_baseline)/len(df)*100:>5.1f}%)")
    print(f"  Total:                           {len(df):>5} (should match)")
    
    # Verify no overlap
    total_check = len(group_a_overlap) + len(group_b_neutral_only) + len(group_c_nhung_only) + len(group_d_baseline)
    assert total_check == len(df), f"Groups don't add up: {total_check} != {len(df)}"
    
    # Calculate oversample factors
    if neutral_target is None:
        # Auto: balance với average của Positive & Negative
        pos_count = (df['sentiment'] == 'positive').sum()
        neg_count = (df['sentiment'] == 'negative').sum()
        neutral_target = int((pos_count + neg_count) / 2)
        print(f"\n💡 Auto-calculated Neutral target: {neutral_target} (avg of Pos & Neg)")
    
    # Total current Neutral samples
    total_neutral = len(group_a_overlap) + len(group_b_neutral_only)
    neutral_factor = neutral_target / total_neutral if total_neutral > 0 else 1
    
    # Determine oversample factor for Group A (overlap)
    if overlap_strategy == 'max':
        group_a_factor = max(neutral_factor, nhung_factor)
    elif overlap_strategy == 'neutral':
        group_a_factor = neutral_factor
    elif overlap_strategy == 'nhung':
        group_a_factor = nhung_factor
    else:
        group_a_factor = max(neutral_factor, nhung_factor)
    
    print(f"\n🔄 Augmentation Plan:")
    print(f"  Group A (overlap): x{group_a_factor:.2f} (strategy: {overlap_strategy})")
    print(f"  Group B (Neutral only): x{neutral_factor:.2f}")
    print(f"  Group C ('Nhưng' only): x{nhung_factor}")
    print(f"  Group D (Baseline): x1")
    
    # Oversample each group
    print(f"\n🔄 Oversampling each group...")
    
    # Helper function
    def oversample_group(group, factor):
        if factor <= 1:
            return group.copy()
        n_full = int(factor)
        remainder = factor - n_full
        result = pd.concat([group] * n_full, ignore_index=True)
        if remainder > 0:
            n_extra = int(len(group) * remainder)
            if n_extra > 0:
                extra = group.sample(n=n_extra, random_state=42, replace=False)
                result = pd.concat([result, extra], ignore_index=True)
        return result
    
    # Group A: Overlap
    oversampled_a = oversample_group(group_a_overlap, group_a_factor)
    print(f"  Group A: {len(group_a_overlap)} → {len(oversampled_a)} (x{group_a_factor:.2f})")
    
    # Group B: Neutral only
    oversampled_b = oversample_group(group_b_neutral_only, neutral_factor)
    print(f"  Group B: {len(group_b_neutral_only)} → {len(oversampled_b)} (x{neutral_factor:.2f})")
    
    # Group C: "nhưng" only
    oversampled_c = oversample_group(group_c_nhung_only, nhung_factor)
    print(f"  Group C: {len(group_c_nhung_only)} → {len(oversampled_c)} (x{nhung_factor})")
    
    # Group D: Baseline (no change)
    oversampled_d = group_d_baseline.copy()
    print(f"  Group D: {len(group_d_baseline)} → {len(oversampled_d)} (x1)")
    
    # Combine all groups
    print(f"\n🔄 Combining all groups...")
    augmented_df = pd.concat([
        oversampled_a,
        oversampled_b,
        oversampled_c,
        oversampled_d
    ], ignore_index=True)
    
    # Shuffle
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Augmented data: {len(augmented_df)} samples")
    print(f"✓ Increase: +{len(augmented_df) - len(df)} samples (+{(len(augmented_df) - len(df))/len(df)*100:.1f}%)")
    
    # Save
    augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved to: {output_file}")
    
    # Detailed statistics
    print(f"\n{'='*80}")
    print("📊 AUGMENTED DATA STATISTICS")
    print(f"{'='*80}")
    
    # Sentiment distribution
    print(f"\n📊 Sentiment Distribution:")
    augmented_sentiment = augmented_df['sentiment'].value_counts()
    for sent, count in augmented_sentiment.items():
        print(f"  {sent:>10}: {count:>6} ({count/len(augmented_df)*100:>5.1f}%)")
    
    # Calculate imbalance ratio
    max_sent = augmented_sentiment.max()
    min_sent = augmented_sentiment.min()
    imbalance_ratio = max_sent / min_sent
    print(f"\n⚖️  Imbalance Ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2:
        print(f"   ⚠️  HIGH IMBALANCE (> 2x)")
    elif imbalance_ratio > 1.5:
        print(f"   ⚠️  MODERATE IMBALANCE (> 1.5x)")
    else:
        print(f"   ✓  BALANCED (< 1.5x)")
    
    # "nhưng" distribution
    nhung_augmented = augmented_df[augmented_df['sentence'].str.contains('nhưng', case=False, na=False)]
    print(f"\n📊 'Nhưng' Samples:")
    print(f"  Total: {len(nhung_augmented)} ({len(nhung_augmented)/len(augmented_df)*100:.1f}%)")
    nhung_sent_dist = nhung_augmented['sentiment'].value_counts()
    for sent, count in nhung_sent_dist.items():
        print(f"    {sent:>10}: {count:>5} ({count/len(nhung_augmented)*100:>5.1f}%)")
    
    # Overlap statistics
    overlap_augmented = augmented_df[(augmented_df['sentiment'] == 'neutral') & 
                                      (augmented_df['sentence'].str.contains('nhưng', case=False, na=False))]
    print(f"\n📊 Overlap (Neutral + 'nhưng'):")
    print(f"  Before: {len(group_a_overlap)} ({len(group_a_overlap)/len(df)*100:.1f}%)")
    print(f"  After:  {len(overlap_augmented)} ({len(overlap_augmented)/len(augmented_df)*100:.1f}%)")
    print(f"  Factor: x{group_a_factor:.2f} (NO double counting)")
    
    # Aspect distribution
    print(f"\n📊 Top 10 Aspects:")
    aspect_dist = augmented_df['aspect'].value_counts().head(10)
    for asp, count in aspect_dist.items():
        print(f"  {asp:<15}: {count:>5} ({count/len(augmented_df)*100:>5.1f}%)")
    
    # Comparison table
    print(f"\n{'='*80}")
    print("📊 BEFORE vs AFTER COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Before':>12} {'After':>12} {'Change':>12}")
    print("-"*68)
    
    # Total samples
    print(f"{'Total Samples':<30} {len(df):>12} {len(augmented_df):>12} {len(augmented_df)-len(df):>+12}")
    
    # Sentiment counts
    for sent in ['positive', 'negative', 'neutral']:
        before = (df['sentiment'] == sent).sum()
        after = (augmented_df['sentiment'] == sent).sum()
        print(f"{sent.capitalize() + ' samples':<30} {before:>12} {after:>12} {after-before:>+12}")
    
    # "nhưng" samples
    before_nhung = df['sentence'].str.contains('nhưng', case=False, na=False).sum()
    after_nhung = augmented_df['sentence'].str.contains('nhưng', case=False, na=False).sum()
    print(f"{'Nhưng samples':<30} {before_nhung:>12} {after_nhung:>12} {after_nhung-before_nhung:>+12}")
    
    # "nhưng" + Neutral
    before_overlap = ((df['sentence'].str.contains('nhưng', case=False, na=False)) & 
                      (df['sentiment'] == 'neutral')).sum()
    after_overlap = ((augmented_df['sentence'].str.contains('nhưng', case=False, na=False)) & 
                     (augmented_df['sentiment'] == 'neutral')).sum()
    print(f"{'Nhưng + Neutral':<30} {before_overlap:>12} {after_overlap:>12} {after_overlap-before_overlap:>+12}")
    
    print(f"\n{'='*80}")
    print("🎯 NEXT STEPS")
    print(f"{'='*80}")
    print("\n1. Update config.yaml:")
    print(f"   train_file: {output_file}")
    print("\n2. Train model:")
    print("   python train.py")
    print("\n3. Expected improvements:")
    print("   • Neutral accuracy: increase (better class balance)")
    print("   • 'Nhưng' accuracy: increase (more training samples)")
    print("   • Overall F1: potentially +1-2%")
    print(f"\n{'='*80}\n")
    
    return augmented_df


def analyze_current_data(train_file='single_label/data/train.csv'):
    """Analyze current data distribution"""
    print("="*80)
    print("📊 CURRENT DATA ANALYSIS")
    print("="*80)
    
    os.chdir('D:/BERT')
    
    if not os.path.exists(train_file):
        print(f"\n❌ File không tồn tại: {train_file}")
        return
    
    df = pd.read_csv(train_file, encoding='utf-8-sig')
    
    print(f"\n📊 Total samples: {len(df)}")
    
    # Sentiment distribution
    print(f"\n📊 Sentiment Distribution:")
    sentiment_dist = df['sentiment'].value_counts()
    for sent, count in sentiment_dist.items():
        print(f"  {sent:>10}: {count:>6} ({count/len(df)*100:>5.1f}%)")
    
    # Calculate imbalance
    max_count = sentiment_dist.max()
    min_count = sentiment_dist.min()
    imbalance_ratio = max_count / min_count
    print(f"\n⚖️  Imbalance Ratio: {imbalance_ratio:.2f}x")
    
    # "nhưng" samples
    has_nhung = df['sentence'].str.contains('nhưng', case=False, na=False)
    nhung_df = df[has_nhung]
    
    print(f"\n📊 'Nhưng' Samples: {len(nhung_df)} ({len(nhung_df)/len(df)*100:.1f}%)")
    nhung_sent = nhung_df['sentiment'].value_counts()
    for sent, count in nhung_sent.items():
        print(f"  {sent:>10}: {count:>5} ({count/len(nhung_df)*100:>5.1f}%)")
    
    # Overlap
    overlap = df[(df['sentiment'] == 'neutral') & has_nhung]
    print(f"\n📊 Overlap (Neutral + 'nhưng'): {len(overlap)} ({len(overlap)/len(df)*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    neutral_count = (df['sentiment'] == 'neutral').sum()
    pos_count = (df['sentiment'] == 'positive').sum()
    neg_count = (df['sentiment'] == 'negative').sum()
    avg_pos_neg = (pos_count + neg_count) / 2
    
    suggested_neutral_target = int(avg_pos_neg)
    neutral_increase = suggested_neutral_target - neutral_count
    
    print(f"  1. Neutral target: ~{suggested_neutral_target} samples (+{neutral_increase})")
    print(f"     Reason: Balance với avg(Positive, Negative) = {avg_pos_neg:.0f}")
    
    nhung_x3 = len(nhung_df) * 3
    nhung_increase = nhung_x3 - len(nhung_df)
    print(f"\n  2. 'Nhưng' x3: {len(nhung_df)} → {nhung_x3} (+{nhung_increase})")
    print(f"     Reason: Đủ samples để học adversative patterns")
    
    total_increase = neutral_increase + nhung_increase
    overlap_double_count = len(overlap) * 2  # Counted in both strategies
    adjusted_increase = total_increase - overlap_double_count
    
    print(f"\n  3. Total increase estimate: ~+{adjusted_increase}-{total_increase} samples")
    print(f"     Note: {len(overlap)} overlap samples được count 2 lần")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    # Analyze current data
    print("\n" + "="*80)
    print("STEP 1: ANALYZE CURRENT DATA")
    print("="*80)
    analyze_current_data()
    
    # Run augmentation
    print("\n" + "="*80)
    print("STEP 2: INDEPENDENT AUGMENTATION")
    print("="*80)
    
    augment_neutral_and_nhung(
        train_file='single_label/data/train.csv',
        output_file='single_label/data/train_augmented_neutral_nhung.csv',
        neutral_target=3000,     # Target Neutral samples (None = auto balance)
        nhung_factor=3,          # "nhưng" oversample factor
        overlap_strategy='max'   # 'max', 'neutral', or 'nhung'
    )
    
    print("\n✅ COMPLETED!\n")
