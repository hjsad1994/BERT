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
import json
import yaml
import argparse
from typing import Optional
import random
from datetime import datetime

def set_all_seeds(seed):
    """Set random seed for all libraries to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # Note: pandas uses numpy's random state

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

def oversample_aspect_balanced(df, aspect_cols, seed=324):
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
    
    set_all_seeds(seed)
    
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

def oversample_moderate_per_aspect(df, aspect_cols, seed=324, target_ratio=0.6):
    """
    Moderate oversampling: Balance towards a target ratio instead of perfect balance
    
    Args:
        df: Input DataFrame
        aspect_cols: List of aspect columns
        seed: Random seed for reproducibility
        target_ratio: Target minimum/maximum ratio (e.g., 0.6 means minority should be at least 60% of majority)
                     Default: 0.6 (gentle balancing)
    
    Gentler approach - reduces overfitting risk
    """
    print("\n" + "=" * 80)
    print(f"Moderate Per-Aspect Oversampling (Target Ratio: {target_ratio:.1%})")
    print("=" * 80)
    
    set_all_seeds(seed)
    
    # Analyze imbalance
    imbalance_info = analyze_imbalance(df, aspect_cols)
    
    # For each aspect, oversample minority classes moderately
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
            
            # Calculate target count with target_ratio
            # Example: If max_count=1000, target_ratio=0.6
            #          target = 1000 * 0.6 = 600 (instead of matching 1000)
            # This gives gentler oversampling
            target_count = int(max_count * target_ratio)
            
            # Don't undersample - only oversample
            if target_count < current_count:
                target_count = current_count
            
            # Calculate how many to add
            to_add = target_count - current_count
            
            if to_add <= 0:
                continue
            
            # Calculate actual ratio for reporting
            actual_ratio = target_count / current_count if current_count > 0 else 1.0
            
            # Sample with replacement
            sampled = sentiment_rows.sample(n=to_add, replace=True, random_state=seed)
            
            all_augmented.append(sampled)
            
            print(f"   {sentiment}: {current_count} -> {target_count} (+{to_add}) [{actual_ratio:.2f}x]")
    
    # Combine all
    augmented_df = pd.concat(all_augmented, ignore_index=True)
    
    print(f"\nTotal samples: {len(df)} -> {len(augmented_df)} (+{len(augmented_df) - len(df)})")
    
    return augmented_df

def oversample_simple_per_aspect(df, aspect_cols, seed=324, max_ratio=7.0):
    """
    Simple oversampling: For each aspect, duplicate rows to match max count
    
    Args:
        df: Input DataFrame
        aspect_cols: List of aspect columns
        seed: Random seed for reproducibility
        max_ratio: Maximum oversampling ratio (e.g., 7.0 means max 7x duplication)
                   Default: 7.0 to prevent extreme overfitting
    
    More aggressive approach - better for severe imbalance
    """
    print("\n" + "=" * 80)
    print(f"Simple Per-Aspect Oversampling (Max Ratio: {max_ratio}x)")
    print("=" * 80)
    
    set_all_seeds(seed)
    
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
            
            # Calculate target count with max_ratio cap
            # Example: If max_count=1000 and current_count=100 (10x ratio)
            #          Cap at 7x: target = 100 * 7 = 700 instead of 1000
            ratio = max_count / current_count
            if ratio > max_ratio:
                target_count = int(current_count * max_ratio)
                capped = True
            else:
                target_count = max_count
                capped = False
            
            # Calculate how many to add
            to_add = target_count - current_count
            
            if to_add <= 0:
                continue
            
            # Sample with replacement
            sampled = sentiment_rows.sample(n=to_add, replace=True, random_state=seed)
            
            all_augmented.append(sampled)
            
            if capped:
                print(f"   {sentiment}: {current_count} -> {target_count} (+{to_add}) [CAPPED at {max_ratio}x]")
            else:
                print(f"   {sentiment}: {current_count} -> {target_count} (+{to_add})")
    
    # Combine all
    augmented_df = pd.concat(all_augmented, ignore_index=True)
    
    print(f"\nTotal samples: {len(df)} -> {len(augmented_df)} (+{len(augmented_df) - len(df)})")
    
    return augmented_df


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(config_path: Optional[str] = None, max_ratio: float = 7.0, moderate: bool = False, target_ratio: float = 0.5):
    # Configuration
    if config_path:
        config = load_config(config_path)
        input_file = config['paths']['train_file'].replace('_balanced', '')  # Get original train file
        
        # Get seed from config with fallback
        if 'reproducibility' in config:
            if 'oversampling_seed' in config['reproducibility']:
                seed = config['reproducibility']['oversampling_seed']
            elif 'seed' in config['reproducibility']:
                seed = config['reproducibility']['seed']
            else:
                seed = 324
        else:
            seed = 324
        
        # Get oversampling params from config if available
        if 'oversampling' in config:
            if 'max_ratio' in config['oversampling']:
                max_ratio = config['oversampling']['max_ratio']
            if 'moderate' in config['oversampling']:
                moderate = config['oversampling']['moderate']
            if 'target_ratio' in config['oversampling']:
                target_ratio = config['oversampling']['target_ratio']
        
        print(f"\n[Using config: {config_path}]")
        print(f"[Oversampling seed: {seed}]")
        if moderate:
            print(f"[Moderate oversampling: target_ratio={target_ratio:.1%}]")
        else:
            print(f"[Max oversampling ratio: {max_ratio}x]")
        set_all_seeds(seed)
        
        # Still save to all 4 directories
        output_files = [
            'BILSTM-MTL/data/train_multilabel_balanced.csv',
            'BILSTM-STL/data/train_multilabel_balanced.csv',
            'VisoBERT-MTL/data/train_multilabel_balanced.csv',
            'VisoBERT-STL/data/train_multilabel_balanced.csv'
        ]
    else:
        input_file = 'BILSTM-MTL/data/train_multilabel.csv'  # Use first directory as source
        output_files = [
            'BILSTM-MTL/data/train_multilabel_balanced.csv',
            'BILSTM-STL/data/train_multilabel_balanced.csv',
            'VisoBERT-MTL/data/train_multilabel_balanced.csv',
            'VisoBERT-STL/data/train_multilabel_balanced.csv'
        ]
        seed = 324
        
        print(f"\n[No config provided, using defaults]")
        print(f"[Default seed: {seed}]")
        print(f"[Will save to all 4 model directories]")
        set_all_seeds(seed)
    
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
    if moderate:
        print(f"Applying Moderate Oversampling (Target Ratio: {target_ratio:.1%})")
    else:
        print(f"Applying Oversampling (Max Ratio: {max_ratio}x)")
    print("=" * 80)
    
    # Choose oversampling method
    if moderate:
        # Method 2: Moderate oversampling (RECOMMENDED for avoiding overfitting)
        augmented_df = oversample_moderate_per_aspect(df, aspect_cols, seed=seed, target_ratio=target_ratio)
    else:
        # Method 1: Simple per-aspect oversampling with cap
        augmented_df = oversample_simple_per_aspect(df, aspect_cols, seed=seed, max_ratio=max_ratio)
    
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
    
    # Save to all 4 directories
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    # Ensure output_files is a list
    if isinstance(output_files, str):
        output_files = [output_files]
    
    for output_file in output_files:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\nSaved augmented data to: {output_file}")
        
        # Save metadata with seed information
        metadata_file = os.path.join(output_dir, 'multilabel_oversampling_metadata.json')
        metadata = {
            'oversampling_seed': seed,
            'strategy': 'moderate' if moderate else 'capped',
            'max_oversampling_ratio': max_ratio if not moderate else None,
            'target_ratio': target_ratio if moderate else None,
            'original_samples': len(df),
            'augmented_samples': len(augmented_df),
            'increase': len(augmented_df) - len(df),
            'increase_percentage': (len(augmented_df) - len(df)) / len(df) * 100 if len(df) > 0 else 0,
            'aspects': aspect_cols,
            'created_at': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   Metadata: {metadata_file}")
    
    print(f"\n   Original samples: {len(df)}")
    print(f"   Augmented samples: {len(augmented_df)}")
    print(f"   Increase: +{len(augmented_df) - len(df)} (+{(len(augmented_df) - len(df)) / len(df) * 100:.1f}%)")
    print(f"\n   Saved to {len(output_files)} directories:")
    for output_file in output_files:
        print(f"      - {output_file}")
    
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
    
    print(f"\nReproducibility:")
    print(f"   Oversampling seed: {seed}")
    print(f"   All random states set for reproducibility")
    
    print(f"\nNext step:")
    print(f"   Files saved to all 4 model directories:")
    print(f"   - BILSTM-MTL/data/train_multilabel_balanced.csv")
    print(f"   - BILSTM-STL/data/train_multilabel_balanced.csv")
    print(f"   - VisoBERT-MTL/data/train_multilabel_balanced.csv")
    print(f"   - VisoBERT-STL/data/train_multilabel_balanced.csv")
    print(f"   You can now train models in each directory.")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Aspect-wise balanced oversampling for multi-label ABSA'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional)'
    )
    parser.add_argument(
        '--max-ratio',
        type=float,
        default=7.0,
        help='Maximum oversampling ratio for capped mode (default: 7.0x)'
    )
    parser.add_argument(
        '--moderate',
        action='store_true',
        help='Use moderate oversampling (gentler, prevents overfitting)'
    )
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=0.5,
        help='Target min/max ratio for moderate mode (default: 0.5 = 50%%)'
    )
    args = parser.parse_args()
    
    main(config_path=args.config, max_ratio=args.max_ratio, moderate=args.moderate, target_ratio=args.target_ratio)
