"""
Multi-Label Data Preparation for Vietnamese ABSA
Split dataset.csv into train/val/test while keeping multi-label format
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import yaml
import argparse
from typing import Optional
from pathlib import Path

def load_and_validate_data(input_file):
    """Load and validate dataset.csv"""
    print("=" * 80)
    print("Multi-Label Data Preparation")
    print("=" * 80)
    
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"Loaded {len(df)} reviews")
    print(f"Columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['data']
    aspect_cols = ['Battery', 'Camera', 'Performance', 'Display', 'Design', 
                   'Packaging', 'Price', 'Shop_Service', 
                   'Shipping', 'General', 'Others']
    
    missing_cols = [col for col in required_cols + aspect_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    print(f"Validation passed")
    
    return df, aspect_cols

def analyze_distribution(df, aspect_cols):
    """Analyze sentiment distribution across aspects"""
    print("\n" + "=" * 80)
    print("Data Distribution Analysis")
    print("=" * 80)
    
    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    aspect_stats = {}
    
    for aspect in aspect_cols:
        counts = df[aspect].value_counts()
        aspect_stats[aspect] = counts.to_dict()
        
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            if sentiment in counts:
                sentiment_counts[sentiment] += counts[sentiment]
    
    print("\nOverall Sentiment Distribution:")
    total = sum(sentiment_counts.values())
    for sentiment, count in sentiment_counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"   {sentiment:8s}: {count:5d} ({pct:5.2f}%)")
    
    print("\nPer-Aspect Distribution:")
    print(f"{'Aspect':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10} {'Total':<10}")
    print("-" * 60)
    
    for aspect in aspect_cols:
        stats = aspect_stats.get(aspect, {})
        pos = stats.get('Positive', 0)
        neg = stats.get('Negative', 0)
        neu = stats.get('Neutral', 0)
        total = pos + neg + neu
        print(f"{aspect:<15} {pos:<10} {neg:<10} {neu:<10} {total:<10}")
    
    return aspect_stats

def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    """Split dataset into train/val/test with stratification"""
    print("\n" + "=" * 80)
    print("Splitting Dataset")
    print("=" * 80)
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
    
    # Create stratification label (combine sentiments for rough stratification)
    # Use most common sentiment as stratification key
    def get_dominant_sentiment(row):
        sentiments = []
        # Use the same 11 aspects as in config
        for col in ['Battery', 'Camera', 'Performance', 'Display', 'Design', 
                    'Packaging', 'Price', 'Shop_Service', 
                    'Shipping', 'General', 'Others']:
            if col in df.columns and not pd.isna(row[col]):
                sentiments.append(row[col])
        
        if not sentiments:
            return 'Neutral'
        
        # Count sentiments
        from collections import Counter
        counter = Counter(sentiments)
        return counter.most_common(1)[0][0]
    
    print("Creating stratification labels...")
    df['_stratify'] = df.apply(get_dominant_sentiment, axis=1)
    
    print(f"Stratification distribution:")
    print(df['_stratify'].value_counts())
    
    # Remove classes with < 2 samples for stratification
    stratify_counts = df['_stratify'].value_counts()
    classes_to_keep = stratify_counts[stratify_counts >= 2].index
    df_stratified = df[df['_stratify'].isin(classes_to_keep)].copy()
    
    if len(df_stratified) < len(df):
        print(f"Removed {len(df) - len(df_stratified)} samples with rare stratification labels")
    
    # First split: train + (val + test)
    train_df, temp_df = train_test_split(
        df_stratified, 
        test_size=(val_size + test_size), 
        random_state=seed,
        stratify=df_stratified['_stratify']
    )
    
    # Second split: val and test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=(1 - val_ratio), 
        random_state=seed,
        stratify=temp_df['_stratify']
    )
    
    # Remove stratification column
    train_df = train_df.drop('_stratify', axis=1)
    val_df = val_df.drop('_stratify', axis=1)
    test_df = test_df.drop('_stratify', axis=1)
    
    print(f"\nSplit completed:")
    print(f"   Train: {len(train_df):5d} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df):5d} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df):5d} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir='data'):
    """Save train/val/test splits"""
    print("\n" + "=" * 80)
    print("Saving Splits")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    train_file = os.path.join(output_dir, 'train_multilabel.csv')
    val_file = os.path.join(output_dir, 'validation_multilabel.csv')
    test_file = os.path.join(output_dir, 'test_multilabel.csv')
    
    train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_file, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
    
    print(f"Saved train to: {train_file}")
    print(f"Saved val to:   {val_file}")
    print(f"Saved test to:  {test_file}")
    
    # Save metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'split_ratio': {
            'train': len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
            'val': len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
            'test': len(test_df) / (len(train_df) + len(val_df) + len(test_df))
        },
        'aspects': ['Battery', 'Camera', 'Performance', 'Display', 'Design', 
                    'Packaging', 'Price', 'Shop_Service', 
                    'Shipping', 'General', 'Others'],
        'format': 'multi-label',
        'seed': 42
    }
    
    metadata_file = os.path.join(output_dir, 'multilabel_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved metadata to: {metadata_file}")
    
    return train_file, val_file, test_file

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_dataset_csv():
    """Find dataset.csv in the project root (D:\BERT\)"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Try parent directory first (if script is in dual-task-learning/)
    parent_dir = script_dir.parent
    dataset_in_parent = parent_dir / 'dataset.csv'
    
    # Try current directory (if script is run from D:\BERT\)
    dataset_in_current = script_dir / 'dataset.csv'
    
    if dataset_in_parent.exists():
        return str(dataset_in_parent)
    elif dataset_in_current.exists():
        return str(dataset_in_current)
    else:
        # Try to find it by going up until we find it
        current = script_dir
        for _ in range(3):  # Go up max 3 levels
            current = current.parent
            dataset = current / 'dataset.csv'
            if dataset.exists():
                return str(dataset)
        
        # If not found, return relative path as fallback
        return '../dataset.csv'


def get_script_directory():
    """Get the directory where this script is located (dual-task-learning/)"""
    return Path(__file__).parent.absolute()


def main(config_path: Optional[str] = None):
    # Get script directory (dual-task-learning/)
    script_dir = get_script_directory()
    
    # Configuration
    if config_path:
        # If config path is relative, make it relative to script directory
        if not os.path.isabs(config_path):
            config_path = str(script_dir / config_path)
        
        config = load_config(config_path)
        input_file = find_dataset_csv()  # Auto-detect dataset.csv location
        # Ensure output_dir is relative to script directory
        output_dir_from_config = config['paths']['train_file']
        if os.path.isabs(output_dir_from_config):
            output_dir = os.path.dirname(output_dir_from_config)
        else:
            # Resolve relative to script directory
            output_dir = str((script_dir / os.path.dirname(output_dir_from_config)).resolve())
        
        train_size = 0.8
        val_size = 0.1
        test_size = 0.1
        seed = config['reproducibility']['data_split_seed']
        
        print(f"\n[Using config: {config_path}]")
        print(f"[Data split seed: {seed}]")
        print(f"[Script directory: {script_dir}]")
        print(f"[Input: {input_file}]")
        print(f"[Output: {output_dir}/]")
    else:
        input_file = find_dataset_csv()  # Auto-detect dataset.csv location
        # Always output to script directory/data/
        output_dir = str(script_dir / 'data')
        train_size = 0.8
        val_size = 0.1
        test_size = 0.1
        seed = 42
        
        print(f"\n[No config provided, using defaults]")
        print(f"[Default seed: {seed}]")
        print(f"[Script directory: {script_dir}]")
        print(f"[Input: {input_file}]")
        print(f"[Output: {output_dir}/]")
    
    # Load data
    df, aspect_cols = load_and_validate_data(input_file)
    
    # Analyze distribution
    analyze_distribution(df, aspect_cols)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(
        df, 
        train_size=train_size, 
        val_size=val_size, 
        test_size=test_size,
        seed=seed
    )
    
    # Save splits
    train_file, val_file, test_file = save_splits(train_df, val_df, test_df, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"\nMulti-label data preparation completed!")
    print(f"\nFiles created:")
    print(f"   {train_file}")
    print(f"   {val_file}")
    print(f"   {test_file}")
    print(f"   {os.path.join(output_dir, 'multilabel_metadata.json')}")
    
    print(f"\nDataset statistics:")
    print(f"   Total reviews: {len(df)}")
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   Aspects: {len(aspect_cols)}")
    
    print(f"\nNext step:")
    print(f"   python train_multilabel.py")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare multi-label ABSA data: split into train/val/test'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional)'
    )
    args = parser.parse_args()
    
    main(config_path=args.config)
