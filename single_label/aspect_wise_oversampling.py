"""
Aspect-Wise Oversampling Strategy
==================================
Research-backed oversampling cho ABSA

Strategy:
- Với mỗi aspect riêng biệt
- Tìm sentiment class có nhiều samples nhất
- Oversample các sentiment khác để bằng với max
- Example: Audio has neg=500, pos=400, neu=200 -> 500, 500, 500

References:
1. "The Impact of Oversampling and Undersampling on ABSA" (2024)
2. "Data oversampling and imbalanced datasets" (2024)
3. "SMOTE for imbalanced sentiment analysis" (2024)
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Optional
import json
import os
import yaml
import argparse


def aspect_wise_balance_oversample(
    df: pd.DataFrame,
    aspect_column: str = 'aspect',
    sentiment_column: str = 'sentiment',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Oversample data theo từng aspect riêng biệt
    
    Strategy:
    - Với mỗi aspect:
      1. Tìm sentiment có nhiều samples nhất (max_count)
      2. Oversample các sentiment khác lên max_count
      
    Ví dụ:
    Audio aspect:
      - negative: 500 (max)
      - positive: 400
      - neutral: 200
    -> Oversample to:
      - negative: 500 (unchanged)
      - positive: 500 (add 100 samples)
      - neutral: 500 (add 300 samples)
    
    Args:
        df: DataFrame với columns [sentence, aspect, sentiment, ...]
        aspect_column: Tên cột chứa aspect (default: 'aspect')
        sentiment_column: Tên cột chứa sentiment (default: 'sentiment')
        random_state: Random seed cho reproducibility
        
    Returns:
        DataFrame: Oversampled data
    """
    np.random.seed(random_state)
    
    print("\n" + "="*70)
    print("ASPECT-WISE OVERSAMPLING STRATEGY")
    print("="*70)
    print("\nStrategy: Balance sentiment for each aspect separately")
    print("- For each aspect: Find max sentiment count")
    print("- Oversample other sentiments up to max count")
    print("="*70)
    
    # Store original distribution
    original_total = len(df)
    original_dist = {}
    
    # Analyze original distribution per aspect
    print("\n[ORIGINAL DISTRIBUTION]:")
    print("-" * 70)
    
    aspects = df[aspect_column].unique()
    
    for aspect in sorted(aspects):
        aspect_data = df[df[aspect_column] == aspect]
        sentiment_counts = Counter(aspect_data[sentiment_column])
        original_dist[aspect] = dict(sentiment_counts)
        
        print(f"\n{aspect}:")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            pct = (count / len(aspect_data)) * 100 if len(aspect_data) > 0 else 0
            print(f"  {sentiment:10}: {count:4} samples ({pct:5.1f}%)")
        print(f"  {'Total':10}: {len(aspect_data):4} samples")
    
    # Oversample for each aspect
    print("\n" + "="*70)
    print("[OVERSAMPLING PROCESS]:")
    print("="*70)
    
    oversampled_dfs = []
    oversampling_info = {
        'strategy': 'aspect_wise_balance',
        'original_total': original_total,
        'aspects': {}
    }
    
    for aspect in sorted(aspects):
        print(f"\n[Processing: {aspect}]")
        print("-" * 50)
        
        # Get data for this aspect
        aspect_data = df[df[aspect_column] == aspect].copy()
        sentiment_counts = Counter(aspect_data[sentiment_column])
        
        # Find max count
        max_count = max(sentiment_counts.values())
        max_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        print(f"  Max sentiment: {max_sentiment} ({max_count} samples)")
        print(f"  Target: All sentiments -> {max_count} samples")
        
        # Oversample each sentiment to max_count
        aspect_oversampled = []
        added_counts = {}
        
        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_data = aspect_data[aspect_data[sentiment_column] == sentiment]
            current_count = len(sentiment_data)
            
            if current_count == 0:
                print(f"  WARNING: {sentiment}: 0 samples -> skipping")
                continue
            
            # Keep original samples
            aspect_oversampled.append(sentiment_data)
            
            # Oversample if needed
            if current_count < max_count:
                n_to_add = max_count - current_count
                
                # Random oversample with replacement
                # Use modulo to keep seed within valid range (0 to 2^32 - 1)
                seed = (random_state + abs(hash(aspect)) + abs(hash(sentiment))) % (2**32 - 1)
                oversampled = sentiment_data.sample(
                    n=n_to_add,
                    replace=True,
                    random_state=seed
                )
                aspect_oversampled.append(oversampled)
                
                added_counts[sentiment] = n_to_add
                print(f"  + {sentiment:10}: {current_count:4} -> {max_count:4} (+{n_to_add:4})")
            else:
                added_counts[sentiment] = 0
                print(f"  - {sentiment:10}: {current_count:4} (unchanged)")
        
        # Combine oversampled data for this aspect
        aspect_balanced = pd.concat(aspect_oversampled, ignore_index=True)
        oversampled_dfs.append(aspect_balanced)
        
        # Store info
        oversampling_info['aspects'][aspect] = {
            'original': dict(sentiment_counts),
            'target': max_count,
            'added': added_counts,
            'final_total': len(aspect_balanced)
        }
    
    # Combine all aspects
    df_oversampled = pd.concat(oversampled_dfs, ignore_index=True)
    
    # Shuffle
    df_oversampled = df_oversampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Summary
    print("\n" + "="*70)
    print("[OVERSAMPLING COMPLETED]")
    print("="*70)
    
    print(f"\n[SUMMARY]:")
    print(f"  Original total: {original_total:,} samples")
    print(f"  Final total:    {len(df_oversampled):,} samples")
    print(f"  Added:          {len(df_oversampled) - original_total:,} samples")
    print(f"  Increase:       {((len(df_oversampled) / original_total) - 1) * 100:.1f}%")
    
    # Final distribution
    print("\n[FINAL DISTRIBUTION]:")
    print("-" * 70)
    
    for aspect in sorted(aspects):
        aspect_data = df_oversampled[df_oversampled[aspect_column] == aspect]
        sentiment_counts = Counter(aspect_data[sentiment_column])
        
        print(f"\n{aspect}:")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            original = original_dist[aspect].get(sentiment, 0)
            diff = count - original
            print(f"  {sentiment:10}: {count:4} samples (was {original:4}, +{diff:4})")
        print(f"  {'Total':10}: {len(aspect_data):4} samples")
    
    # Save info
    oversampling_info['final_total'] = len(df_oversampled)
    oversampling_info['increase_pct'] = ((len(df_oversampled) / original_total) - 1) * 100
    
    return df_oversampled, oversampling_info


def analyze_aspect_sentiment_distribution(
    df: pd.DataFrame,
    aspect_column: str = 'aspect',
    sentiment_column: str = 'sentiment'
) -> Dict:
    """
    Phân tích phân bố sentiment cho từng aspect
    
    Returns:
        Dict: Distribution info per aspect
    """
    print("\n" + "="*70)
    print("ASPECT-SENTIMENT DISTRIBUTION ANALYSIS")
    print("="*70)
    
    distribution = {}
    aspects = df[aspect_column].unique()
    
    for aspect in sorted(aspects):
        aspect_data = df[df[aspect_column] == aspect]
        sentiment_counts = Counter(aspect_data[sentiment_column])
        
        total = len(aspect_data)
        max_count = max(sentiment_counts.values())
        min_count = min(sentiment_counts.values()) if sentiment_counts else 0
        
        distribution[aspect] = {
            'total': total,
            'sentiment_counts': dict(sentiment_counts),
            'max_count': max_count,
            'min_count': min_count,
            'imbalance_ratio': max_count / min_count if min_count > 0 else float('inf')
        }
        
        print(f"\n{aspect}:")
        print(f"  Total: {total} samples")
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            pct = (count / total) * 100 if total > 0 else 0
            print(f"  {sentiment:10}: {count:4} ({pct:5.1f}%)")
        print(f"  Imbalance ratio: {distribution[aspect]['imbalance_ratio']:.2f}x")
    
    return distribution


def save_oversampling_info(info: Dict, output_path: str = 'analysis_results/oversampling_info_aspect_wise.json'):
    """
    Lưu thông tin oversampling để visualization sau này
    """
    import os
    from datetime import datetime
    
    info['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Saved oversampling info to: {output_path}]")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main(config_path: Optional[str] = None):
    """
    Main function to run aspect-wise oversampling
    
    Args:
        config_path: Path to config YAML file (optional)
    """
    # Get script directory to handle paths correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load config if provided
    if config_path:
        config = load_config(config_path)
        train_path = config['paths']['train_file']
        output_filename = 'train_oversampled_aspect_wise.csv'
        output_dir = os.path.dirname(train_path)
        output_path = os.path.join(output_dir, output_filename)
        random_seed = config['reproducibility']['oversampling_seed']
        
        print(f"\n[Using config: {config_path}]")
        print(f"[Oversampling seed: {random_seed}]")
    else:
        # Default paths (backward compatibility)
        data_dir = os.path.join(script_dir, 'data')
        train_path = os.path.join(data_dir, 'train.csv')
        output_path = os.path.join(data_dir, 'train_oversampled_aspect_wise.csv')
        random_seed = 42
        
        print(f"\n[No config provided, using defaults]")
        print(f"[Default seed: {random_seed}]")
    
    # Load data
    print(f"\n[Loading data from: {train_path}]")
    train_df = pd.read_csv(train_path)
    
    print("Original data:")
    print(f"  Total samples: {len(train_df)}")
    
    # Analyze distribution
    distribution = analyze_aspect_sentiment_distribution(train_df)
    
    # Apply aspect-wise oversampling
    train_oversampled, info = aspect_wise_balance_oversample(
        train_df,
        random_state=random_seed
    )
    
    # Save
    train_oversampled.to_csv(output_path, index=False)
    print(f"\n[Saved oversampled data to: {output_path}]")
    
    # Save info
    info_path = os.path.join(script_dir, 'analysis_results', 'oversampling_info_aspect_wise.json')
    save_oversampling_info(info, info_path)
    
    print("\n" + "="*70)
    print("[DONE!]")
    print("="*70)
    print("\nNext steps:")
    print("1. Update config.yaml: train_file: 'data/train_oversampled_aspect_wise.csv'")
    print("2. Run: python train.py")
    print("3. Compare performance with/without oversampling")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Aspect-wise oversampling for ABSA training data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional, defaults to using script directory paths)'
    )
    args = parser.parse_args()
    
    main(config_path=args.config)
