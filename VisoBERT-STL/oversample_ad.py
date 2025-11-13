"""
Oversampling Script for Aspect Detection (AD) in VisoBERT-STL
=============================================================

Strategy: Binary classification oversampling for each aspect
- For each aspect, balance "mentioned" (1) vs "not mentioned" (0) samples
- AD task: Detect which aspects are mentioned in the text (binary classification)
- Each aspect is treated independently (multilabel)

Example:
  Battery: mentioned=500, not_mentioned=5000 -> oversample mentioned to 5000
  Camera: mentioned=800, not_mentioned=4700 -> oversample mentioned to 4700
"""

import pandas as pd
import numpy as np
from collections import Counter
import os
import json
import yaml
import argparse
from typing import Optional, Dict, List
import random
from datetime import datetime


def set_all_seeds(seed: int):
    """Set random seed for all libraries to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def analyze_ad_imbalance(df: pd.DataFrame, aspect_cols: List[str]) -> Dict:
    """
    Analyze binary imbalance for each aspect (mentioned vs not mentioned)
    
    Args:
        df: DataFrame with aspect columns
        aspect_cols: List of aspect column names
        
    Returns:
        Dict with imbalance information per aspect
    """
    print("=" * 80)
    print("Analyzing Aspect Detection (AD) Binary Imbalance")
    print("=" * 80)
    
    imbalance_info = {}
    
    print(f"\n{'Aspect':<15} {'Mentioned':<12} {'Not Mentioned':<15} {'Imbalance':<12} {'Mentioned %':<12}")
    print("-" * 80)
    
    for aspect in aspect_cols:
        # Count mentioned (not NaN) vs not mentioned (NaN)
        mentioned = df[aspect].notna().sum()
        not_mentioned = df[aspect].isna().sum()
        total = len(df)
        
        # Calculate imbalance ratio
        if mentioned > 0 and not_mentioned > 0:
            imbalance_ratio = max(mentioned, not_mentioned) / min(mentioned, not_mentioned)
        else:
            imbalance_ratio = float('inf')
        
        mentioned_pct = (mentioned / total) * 100 if total > 0 else 0
        
        print(f"{aspect:<15} {mentioned:<12} {not_mentioned:<15} {imbalance_ratio:<12.2f}x {mentioned_pct:<12.2f}%")
        
        imbalance_info[aspect] = {
            'mentioned': mentioned,
            'not_mentioned': not_mentioned,
            'total': total,
            'imbalance_ratio': imbalance_ratio,
            'mentioned_pct': mentioned_pct,
            'is_positive_minority': mentioned < not_mentioned  # Positive class (mentioned) is minority
        }
    
    # Overall statistics
    avg_imbalance = np.mean([info['imbalance_ratio'] for info in imbalance_info.values() 
                             if info['imbalance_ratio'] != float('inf')])
    avg_mentioned_pct = np.mean([info['mentioned_pct'] for info in imbalance_info.values()])
    
    print(f"\nAverage imbalance ratio: {avg_imbalance:.2f}x")
    print(f"Average mentioned percentage: {avg_mentioned_pct:.2f}%")
    
    return imbalance_info


def oversample_ad_balanced(
    df: pd.DataFrame,
    aspect_cols: List[str],
    seed: int = 42,
    strategy: str = 'moderate',
    target_ratio: float = 0.5,
    max_ratio: float = 5.0
) -> pd.DataFrame:
    """
    Oversample AD data to balance binary labels (mentioned vs not mentioned) for each aspect
    
    Args:
        df: Input DataFrame
        aspect_cols: List of aspect columns
        seed: Random seed for reproducibility
        strategy: 'moderate' (gentle) or 'aggressive' (full balance)
        target_ratio: For moderate strategy, target min/max ratio (e.g., 0.5 = 50%)
        max_ratio: Maximum oversampling ratio cap (e.g., 5.0 = 5x duplication)
        
    Returns:
        Oversampled DataFrame
    """
    print("\n" + "=" * 80)
    if strategy == 'moderate':
        print(f"Moderate AD Oversampling (Target Ratio: {target_ratio:.1%})")
    else:
        print("Aggressive AD Oversampling (Full Balance)")
    print("=" * 80)
    
    set_all_seeds(seed)
    
    # Analyze imbalance
    imbalance_info = analyze_ad_imbalance(df, aspect_cols)
    
    # Start with original data
    all_augmented = [df.copy()]
    
    # Process each aspect independently
    for aspect in aspect_cols:
        info = imbalance_info[aspect]
        mentioned_count = info['mentioned']
        not_mentioned_count = info['not_mentioned']
        
        # Determine which class is minority
        if mentioned_count == 0:
            print(f"\n{aspect}: WARNING - No mentioned samples, skipping")
            continue
        
        if not_mentioned_count == 0:
            print(f"\n{aspect}: WARNING - No not_mentioned samples, skipping")
            continue
        
        # Get rows where this aspect is mentioned
        mentioned_mask = df[aspect].notna()
        mentioned_rows = df[mentioned_mask]
        
        print(f"\nProcessing {aspect}...")
        print(f"   Mentioned: {mentioned_count}, Not mentioned: {not_mentioned_count}")
        
        if strategy == 'moderate':
            # Moderate: Balance to target_ratio
            # For multilabel: when adding n rows with aspect mentioned:
            #   - mentioned_final = mentioned_orig + n (increases)
            #   - not_mentioned_final = not_mentioned_orig (stays same, because new rows have this aspect mentioned)
            #   - total_final = total_orig + n
            # Target: mentioned_final / total_final = target_ratio
            # (mentioned_orig + n) / (total_orig + n) = target_ratio
            # mentioned_orig + n = target_ratio * (total_orig + n)
            # mentioned_orig + n = target_ratio * total_orig + target_ratio * n
            # mentioned_orig - target_ratio * total_orig = target_ratio * n - n
            # mentioned_orig - target_ratio * total_orig = n * (target_ratio - 1)
            # n = (mentioned_orig - target_ratio * total_orig) / (target_ratio - 1)
            
            total_orig = mentioned_count + not_mentioned_count
            target_mentioned_ratio = target_ratio
            
            # Calculate how many rows to add to achieve target ratio
            if target_mentioned_ratio < 1.0:
                n_rows_needed = (mentioned_count - target_mentioned_ratio * total_orig) / (target_mentioned_ratio - 1.0)
            else:
                n_rows_needed = 0
            
            # Only oversample if mentioned is minority
            if mentioned_count < not_mentioned_count and n_rows_needed > 0:
                to_add = int(n_rows_needed)
                
                # Apply max_ratio cap
                max_rows = int(mentioned_count * max_ratio) - mentioned_count
                if to_add > max_rows:
                    to_add = max_rows
                    capped = True
                else:
                    capped = False
                
                if to_add > 0:
                    # Sample with replacement
                    sampled = mentioned_rows.sample(n=to_add, replace=True, random_state=seed)
                    all_augmented.append(sampled)
                    
                    # Calculate expected final counts
                    expected_mentioned = mentioned_count + to_add
                    expected_not_mentioned = not_mentioned_count  # Stays same
                    expected_total = expected_mentioned + expected_not_mentioned
                    expected_ratio = expected_mentioned / expected_total if expected_total > 0 else 0
                    
                    if capped:
                        print(f"   Oversampled: +{to_add} rows [CAPPED at {max_ratio}x]")
                    else:
                        print(f"   Oversampled: +{to_add} rows")
                    print(f"   Expected final: mentioned={expected_mentioned}, not_mentioned={expected_not_mentioned}, ratio={expected_ratio:.1%}")
                else:
                    print(f"   No oversampling needed")
            else:
                print(f"   No oversampling needed (mentioned is majority or calculation invalid)")
                
        else:  # aggressive
            # Aggressive: Balance to achieve mentioned = not_mentioned
            # For multilabel: mentioned_final = mentioned_orig + n, not_mentioned_final = not_mentioned_orig (stays same)
            # Target: mentioned_final = not_mentioned_final
            # mentioned_orig + n = not_mentioned_orig
            # n = not_mentioned_orig - mentioned_orig
            
            if mentioned_count < not_mentioned_count:
                # Calculate rows needed for perfect balance
                n_rows_needed = not_mentioned_count - mentioned_count
                to_add = int(n_rows_needed)
                
                # Apply max_ratio cap
                max_rows = int(mentioned_count * max_ratio) - mentioned_count
                if to_add > max_rows:
                    to_add = max_rows
                    capped = True
                else:
                    capped = False
                
                if to_add > 0:
                    # Sample with replacement
                    sampled = mentioned_rows.sample(n=to_add, replace=True, random_state=seed)
                    all_augmented.append(sampled)
                    
                    # Calculate expected final counts
                    expected_mentioned = mentioned_count + to_add
                    expected_not_mentioned = not_mentioned_count  # Stays same
                    expected_imbalance = max(expected_mentioned, expected_not_mentioned) / min(expected_mentioned, expected_not_mentioned) if min(expected_mentioned, expected_not_mentioned) > 0 else float('inf')
                    
                    if capped:
                        print(f"   Oversampled: +{to_add} rows [CAPPED at {max_ratio}x]")
                    else:
                        print(f"   Oversampled: +{to_add} rows (targeting perfect balance)")
                    print(f"   Expected final: mentioned={expected_mentioned}, not_mentioned={expected_not_mentioned}, imbalance={expected_imbalance:.2f}x")
                else:
                    print(f"   No oversampling needed")
            else:
                print(f"   Mentioned is majority, no oversampling needed")
    
    # Combine all augmented data
    augmented_df = pd.concat(all_augmented, ignore_index=True)
    
    # Shuffle
    augmented_df = augmented_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"\nTotal samples: {len(df)} -> {len(augmented_df)} (+{len(augmented_df) - len(df)})")
    
    return augmented_df


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(
    config_path: Optional[str] = None,
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    strategy: str = 'moderate',
    target_ratio: float = 0.5,
    max_ratio: float = 5.0,
    seed: Optional[int] = None
):
    """
    Main function to run AD oversampling
    
    Args:
        config_path: Path to config YAML file (optional)
        input_file: Path to input CSV file (optional, overrides config)
        output_file: Path to output CSV file (optional, overrides config)
        strategy: 'moderate' or 'aggressive'
        target_ratio: Target ratio for moderate strategy (default: 0.5)
        max_ratio: Max oversampling ratio for aggressive strategy (default: 5.0)
        seed: Random seed (optional, uses config or default)
    """
    print("=" * 80)
    print("Aspect Detection (AD) Oversampling for VisoBERT-STL")
    print("=" * 80)
    
    # Load configuration
    if config_path:
        config = load_config(config_path)
        if input_file is None:
            input_file = config['paths']['ad_train_file']
        if seed is None:
            seed = config.get('reproducibility', {}).get('training_seed', 42)
        
        print(f"\n[Using config: {config_path}]")
        print(f"[Oversampling seed: {seed}]")
    else:
        if input_file is None:
            input_file = 'VisoBERT-STL/data/train_multilabel.csv'
        if seed is None:
            seed = 42
        
        print(f"\n[No config provided, using defaults]")
        print(f"[Oversampling seed: {seed}]")
    
    # Set output file
    if output_file is None:
        if strategy == 'moderate':
            output_file = input_file.replace('.csv', '_ad_balanced_moderate.csv')
        else:
            output_file = input_file.replace('.csv', '_ad_balanced_aggressive.csv')
    
    set_all_seeds(seed)
    
    # Aspect columns (11 aspects including Others)
    aspect_cols = [
        'Battery', 'Camera', 'Performance', 'Display', 'Design',
        'Packaging', 'Price', 'Shop_Service',
        'Shipping', 'General', 'Others'
    ]
    
    # Load data
    print(f"\n[Loading data from: {input_file}]")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"Loaded {len(df)} samples")
    
    # Analyze original imbalance
    print("\n" + "=" * 80)
    print("Original Data Distribution")
    print("=" * 80)
    original_imbalance = analyze_ad_imbalance(df, aspect_cols)
    
    # Apply oversampling
    print("\n" + "=" * 80)
    print(f"Applying {strategy.upper()} Oversampling")
    print("=" * 80)
    
    augmented_df = oversample_ad_balanced(
        df,
        aspect_cols,
        seed=seed,
        strategy=strategy,
        target_ratio=target_ratio,
        max_ratio=max_ratio
    )
    
    # Analyze augmented distribution
    print("\n" + "=" * 80)
    print("Augmented Data Distribution")
    print("=" * 80)
    augmented_imbalance = analyze_ad_imbalance(augmented_df, aspect_cols)
    
    # Calculate improvement
    print("\n" + "=" * 80)
    print("Imbalance Improvement")
    print("=" * 80)
    
    print(f"\n{'Aspect':<15} {'Before':<12} {'After':<12} {'Improvement':<15}")
    print("-" * 60)
    
    improvements = []
    for aspect in aspect_cols:
        before = original_imbalance[aspect]['imbalance_ratio']
        after = augmented_imbalance[aspect]['imbalance_ratio']
        
        if before != float('inf') and after != float('inf'):
            improvement = ((before - after) / before) * 100
            improvements.append(improvement)
            print(f"{aspect:<15} {before:<12.2f}x {after:<12.2f}x {improvement:>12.1f}%")
    
    avg_improvement = np.mean(improvements) if improvements else 0
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    
    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save augmented data
    augmented_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved oversampled data to: {output_file}")
    
    # Save metadata
    metadata_file = output_file.replace('.csv', '_metadata.json')
    metadata = {
        'oversampling_seed': seed,
        'strategy': strategy,
        'target_ratio': target_ratio if strategy == 'moderate' else None,
        'max_ratio': max_ratio if strategy == 'aggressive' else None,
        'original_samples': len(df),
        'augmented_samples': len(augmented_df),
        'increase': len(augmented_df) - len(df),
        'increase_percentage': (len(augmented_df) - len(df)) / len(df) * 100 if len(df) > 0 else 0,
        'aspects': aspect_cols,
        'original_imbalance': {k: v['imbalance_ratio'] for k, v in original_imbalance.items()},
        'augmented_imbalance': {k: v['imbalance_ratio'] for k, v in augmented_imbalance.items()},
        'created_at': datetime.now().isoformat()
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to: {metadata_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print(f"\nOversampling strategy: {strategy}")
    if strategy == 'moderate':
        print(f"   Target ratio: {target_ratio:.1%} (gentle balancing)")
    else:
        print(f"   Max ratio cap: {max_ratio}x (prevents extreme overfitting)")
    
    print(f"\nResults:")
    avg_before = np.mean([i['imbalance_ratio'] for i in original_imbalance.values() 
                         if i['imbalance_ratio'] != float('inf')])
    avg_after = np.mean([i['imbalance_ratio'] for i in augmented_imbalance.values() 
                        if i['imbalance_ratio'] != float('inf')])
    print(f"   Average imbalance: {avg_before:.2f}x -> {avg_after:.2f}x")
    print(f"   Improvement: {avg_improvement:.1f}%")
    print(f"   Samples: {len(df):,} -> {len(augmented_df):,} (+{len(augmented_df) - len(df):,})")
    
    print(f"\nReproducibility:")
    print(f"   Oversampling seed: {seed}")
    
    print(f"\nNext steps:")
    print(f"   1. Update config_visobert_stl.yaml:")
    print(f"      ad_train_file: \"{output_file}\"")
    print(f"   2. Run training: python train_visobert_stl.py")
    print(f"   3. Compare performance with/without oversampling")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Oversampling for Aspect Detection (AD) in VisoBERT-STL'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input CSV file (optional, overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output CSV file (optional, overrides config)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['moderate', 'aggressive'],
        default='moderate',
        help='Oversampling strategy: moderate (gentle) or aggressive (full balance)'
    )
    parser.add_argument(
        '--target-ratio',
        type=float,
        default=0.5,
        help='Target min/max ratio for moderate strategy (default: 0.5 = 50%%)'
    )
    parser.add_argument(
        '--max-ratio',
        type=float,
        default=5.0,
        help='Maximum oversampling ratio for aggressive strategy (default: 5.0x)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (optional, uses config or default)'
    )
    
    args = parser.parse_args()
    
    main(
        config_path=args.config,
        input_file=args.input,
        output_file=args.output,
        strategy=args.strategy,
        target_ratio=args.target_ratio,
        max_ratio=args.max_ratio,
        seed=args.seed
    )

