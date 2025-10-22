"""
Aspect-wise Balanced Oversampling for Multi-Label ABSA

Strategy: For each aspect, oversample to match the max count
Example: Battery has negative=500, positive=200, neutral=100
         → Oversample positive and neutral to 500 each
"""

import pandas as pd
import numpy as np
from collections import Counter
import os

def analyze_imbalance(df, aspect_cols):
    """Analyze per-aspect sentiment distribution"""
    print("=" * 80)
    print("Analyzing Per-Aspect Imbalance")
    print("=" * 80)
    
    imbalance_info = {}
    
    print(f"\n{'Aspect':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10} {'Imbalance':<10}")
    print("-" * 65)
    
    for aspect in aspect_cols:
        counts = df[aspect].value_counts()
        
        pos = counts.get('Positive', 0)
        neg = counts.get('Negative', 0)
        neu = counts.get('Neutral', 0)
        
        # Calculate imbalance ratio (max / min)
        if pos > 0 and neg > 0 and neu > 0:
            max_count = max(pos, neg, neu)
            min_count = min(pos, neg, neu)
            imbalance_ratio = max_count / min_count
        else:
            imbalance_ratio = float('inf')
        
        print(f"{aspect:<15} {pos:<10} {neg:<10} {neu:<10} {imbalance_ratio:<10.2f}x")
        
        imbalance_info[aspect] = {
            'Positive': pos,
            'Negative': neg,
            'Neutral': neu,
            'max_count': max(pos, neg, neu) if (pos + neg + neu) > 0 else 0,
            'imbalance_ratio': imbalance_ratio
        }
    
    # Overall imbalance
    avg_imbalance = np.mean([info['imbalance_ratio'] for info in imbalance_info.values() 
                             if info['imbalance_ratio'] != float('inf')])
    print(f"\nAverage imbalance ratio: {avg_imbalance:.2f}x")
    
    return imbalance_info

def oversample_aspect_balanced(df, aspect_cols, seed=42):
    """
    Oversample each aspect to balance sentiments
    
    Strategy:
    1. For each aspect, find max sentiment count
    2. Oversample rows where that aspect has minority sentiments
    3. Each row can be duplicated multiple times for different aspects
    """
    print("\n" + "=" * 80)
    print("Oversampling Per-Aspect to Balance")
    print("=" * 80)
    
    np.random.seed(seed)
    
    # Analyze imbalance
    imbalance_info = analyze_imbalance(df, aspect_cols)
    
    # Calculate how many times to duplicate each row for each aspect
    print(f"\nCalculating oversample factors...")
    
    # Create augmented dataset
    augmented_rows = []
    
    for idx, row in df.iterrows():
        # Start with original row (always included once)
        augmented_rows.append(row.to_dict())
        
        # For each aspect, determine if this row should be oversampled
        for aspect in aspect_cols:
            sentiment = row[aspect]
            
            if pd.isna(sentiment):
                sentiment = 'Neutral'
            
            # Get target count and current count
            info = imbalance_info[aspect]
            max_count = info['max_count']
            current_count = info.get(sentiment, 0)
            
            if current_count == 0 or max_count == 0:
                continue
            
            # Calculate oversample factor for this sentiment
            oversample_factor = max_count / current_count
            
            # Duplicate this row with probability based on oversample factor
            # oversample_factor = 2.5 means duplicate 1-2 times (50% chance for 2nd)
            num_duplicates = int(oversample_factor - 1)  # -1 because already included once
            fractional_part = (oversample_factor - 1) - num_duplicates
            
            # Add full duplicates
            for _ in range(num_duplicates):
                augmented_rows.append(row.to_dict())
            
            # Add fractional duplicate with probability
            if np.random.random() < fractional_part:
                augmented_rows.append(row.to_dict())
    
    # Convert to DataFrame
    augmented_df = pd.DataFrame(augmented_rows)
    
    # Remove duplicates (same row duplicated by multiple aspects)
    print(f"\nBefore deduplication: {len(augmented_df)} rows")
    augmented_df = augmented_df.drop_duplicates()
    print(f"After deduplication: {len(augmented_df)} rows")
    
    return augmented_df

def oversample_simple_per_aspect(df, aspect_cols, seed=42):
    """
    Simple oversampling: For each aspect, duplicate rows to match max count
    
    More aggressive approach - better for severe imbalance
    """
    print("\n" + "=" * 80)
    print("Simple Per-Aspect Oversampling")
    print("=" * 80)
    
    np.random.seed(seed)
    
    # Analyze imbalance
    imbalance_info = analyze_imbalance(df, aspect_cols)
    
    # For each aspect, oversample minority classes
    all_augmented = [df.copy()]  # Start with original
    
    for aspect in aspect_cols:
        info = imbalance_info[aspect]
        max_count = info['max_count']
        
        print(f"\nProcessing {aspect}...")
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            current_count = info.get(sentiment, 0)
            
            if current_count == 0 or current_count >= max_count:
                continue
            
            # Get rows with this sentiment for this aspect
            mask = df[aspect] == sentiment
            sentiment_rows = df[mask]
            
            if len(sentiment_rows) == 0:
                continue
            
            # Calculate how many to add
            to_add = max_count - current_count
            
            # Sample with replacement
            sampled = sentiment_rows.sample(n=to_add, replace=True, random_state=seed)
            
            all_augmented.append(sampled)
            
            print(f"   {sentiment}: {current_count} -> {max_count} (+{to_add})")
    
    # Combine all
    augmented_df = pd.concat(all_augmented, ignore_index=True)
    
    print(f"\nTotal samples: {len(df)} -> {len(augmented_df)} (+{len(augmented_df) - len(df)})")
    
    return augmented_df

def main():
    # Configuration
    input_file = 'multi_label/data/train_multilabel.csv'
    output_file = 'multi_label/data/train_multilabel_balanced.csv'
    
    aspect_cols = [
        'Battery', 'Camera', 'Performance', 'Display', 'Design',
        'Packaging', 'Price', 'Shop_Service',
        'Shipping', 'General', 'Others'
    ]
    
    print("=" * 80)
    print("Multi-Label Aspect-Wise Balanced Oversampling")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"Loaded {len(df)} samples")
    
    # Analyze original imbalance
    print("\n" + "=" * 80)
    print("Original Data Distribution")
    print("=" * 80)
    imbalance_info = analyze_imbalance(df, aspect_cols)
    
    # Oversample
    print("\n" + "=" * 80)
    print("Applying Oversampling")
    print("=" * 80)
    
    # Method 1: Simple per-aspect oversampling (RECOMMENDED)
    augmented_df = oversample_simple_per_aspect(df, aspect_cols, seed=42)
    
    # Analyze augmented distribution
    print("\n" + "=" * 80)
    print("Augmented Data Distribution")
    print("=" * 80)
    augmented_info = analyze_imbalance(augmented_df, aspect_cols)
    
    # Calculate improvement
    print("\n" + "=" * 80)
    print("Imbalance Improvement")
    print("=" * 80)
    
    print(f"\n{'Aspect':<15} {'Before':<12} {'After':<12} {'Improvement':<15}")
    print("-" * 60)
    
    improvements = []
    for aspect in aspect_cols:
        before = imbalance_info[aspect]['imbalance_ratio']
        after = augmented_info[aspect]['imbalance_ratio']
        
        if before != float('inf') and after != float('inf'):
            improvement = ((before - after) / before) * 100
            improvements.append(improvement)
            print(f"{aspect:<15} {before:<12.2f}x {after:<12.2f}x {improvement:>12.1f}%")
    
    avg_improvement = np.mean(improvements) if improvements else 0
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    
    # Save
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved augmented data to: {output_file}")
    print(f"   Original samples: {len(df)}")
    print(f"   Augmented samples: {len(augmented_df)}")
    print(f"   Increase: +{len(augmented_df) - len(df)} (+{(len(augmented_df) - len(df)) / len(df) * 100:.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print(f"\nBalancing strategy: Per-aspect oversampling")
    print(f"   Each aspect's sentiments are balanced to max count")
    print(f"   Example: If Battery has Negative=500, Positive=200, Neutral=100")
    print(f"            -> Oversample Positive and Neutral to 500 each")
    
    print(f"\nResults:")
    print(f"   Average imbalance: {np.mean([i['imbalance_ratio'] for i in imbalance_info.values() if i['imbalance_ratio'] != float('inf')]):.2f}x")
    print(f"   -> {np.mean([i['imbalance_ratio'] for i in augmented_info.values() if i['imbalance_ratio'] != float('inf')]):.2f}x")
    print(f"   Improvement: {avg_improvement:.1f}%")
    
    print(f"\nNext step:")
    print(f"   Update config.yaml:")
    print(f"   train_file: data/train_multilabel_balanced.csv")
    print(f"   Then train: python train_multilabel.py --epochs 5")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
