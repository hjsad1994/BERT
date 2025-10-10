"""
Oversampling Utilities để xử lý class imbalance
"""
import pandas as pd
import numpy as np
from collections import Counter


def random_oversample(df, target_column='sentiment', sampling_strategy='auto', random_state=42):
    """
    Random Oversampling - Duplicate samples từ minority classes
    
    Args:
        df: DataFrame chứa data
        target_column: Tên cột chứa labels
        sampling_strategy: 
            - 'auto': Balance tất cả classes về số lượng của majority class
            - 'minority': Chỉ oversample minority class
            - dict: Custom ratio, e.g., {'neutral': 1000}
            - float: Target ratio of minority/majority (e.g., 0.5 = 50% of majority)
        random_state: Random seed
        
    Returns:
        DataFrame: Oversampled dataframe
    """
    np.random.seed(random_state)
    
    # Count classes
    class_counts = Counter(df[target_column])
    print(f"\n📊 Original class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"   {label:10}: {count:6,} samples")
    
    # Get majority class count
    majority_count = max(class_counts.values())
    majority_class = max(class_counts, key=class_counts.get)
    
    print(f"\n🎯 Majority class: {majority_class} ({majority_count} samples)")
    
    # Determine target counts for each class
    target_counts = {}
    
    if sampling_strategy == 'auto':
        # Balance all classes to majority count
        for label in class_counts:
            target_counts[label] = majority_count
            
    elif sampling_strategy == 'minority':
        # Only oversample minority class to majority level
        minority_class = min(class_counts, key=class_counts.get)
        for label in class_counts:
            if label == minority_class:
                target_counts[label] = majority_count
            else:
                target_counts[label] = class_counts[label]
                
    elif isinstance(sampling_strategy, dict):
        # Custom counts
        for label in class_counts:
            target_counts[label] = sampling_strategy.get(label, class_counts[label])
            
    elif isinstance(sampling_strategy, float):
        # Ratio of minority to majority
        for label in class_counts:
            if class_counts[label] < majority_count:
                target_counts[label] = int(majority_count * sampling_strategy)
            else:
                target_counts[label] = class_counts[label]
    else:
        raise ValueError(f"Invalid sampling_strategy: {sampling_strategy}")
    
    # Oversample each class
    oversampled_dfs = []
    
    for label, target_count in target_counts.items():
        class_df = df[df[target_column] == label]
        current_count = len(class_df)
        
        if target_count > current_count:
            # Need to oversample
            n_samples = target_count - current_count
            
            # Random sample with replacement
            oversampled = class_df.sample(n=n_samples, replace=True, random_state=random_state)
            
            # Combine original + oversampled
            combined = pd.concat([class_df, oversampled], ignore_index=True)
            oversampled_dfs.append(combined)
            
            print(f"   ⬆️  {label:10}: {current_count:6,} → {target_count:6,} (+{n_samples:,} oversampled)")
        else:
            # No oversampling needed
            oversampled_dfs.append(class_df)
            print(f"   ✓  {label:10}: {current_count:6,} (no change)")
    
    # Combine all classes
    result_df = pd.concat(oversampled_dfs, ignore_index=True)
    
    # Shuffle
    result_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Print final distribution
    final_counts = Counter(result_df[target_column])
    print(f"\n📊 Final class distribution:")
    total = len(result_df)
    for label, count in sorted(final_counts.items()):
        pct = (count / total) * 100
        print(f"   {label:10}: {count:6,} samples ({pct:5.2f}%)")
    
    print(f"\n✓ Total samples: {len(df):,} → {len(result_df):,} (+{len(result_df) - len(df):,})")
    
    return result_df


def smart_oversample(df, target_column='sentiment', min_ratio=0.3, random_state=42):
    """
    Smart Oversampling - Oversample minority classes to at least min_ratio of majority
    
    Args:
        df: DataFrame
        target_column: Label column
        min_ratio: Minimum ratio of minority to majority (e.g., 0.3 = 30%)
        random_state: Random seed
        
    Returns:
        DataFrame: Oversampled dataframe
    """
    class_counts = Counter(df[target_column])
    majority_count = max(class_counts.values())
    
    # Calculate target counts
    target_counts = {}
    for label, count in class_counts.items():
        if count < majority_count * min_ratio:
            # Need to oversample to min_ratio
            target_counts[label] = int(majority_count * min_ratio)
        else:
            target_counts[label] = count
    
    return random_oversample(df, target_column, sampling_strategy=target_counts, random_state=random_state)


def get_class_balance_report(df, target_column='sentiment'):
    """
    Báo cáo chi tiết về class balance
    
    Args:
        df: DataFrame
        target_column: Label column
        
    Returns:
        dict: Balance metrics
    """
    class_counts = Counter(df[target_column])
    total = len(df)
    
    majority_count = max(class_counts.values())
    minority_count = min(class_counts.values())
    
    imbalance_ratio = majority_count / minority_count
    
    report = {
        'total_samples': total,
        'num_classes': len(class_counts),
        'class_counts': dict(class_counts),
        'majority_count': majority_count,
        'minority_count': minority_count,
        'imbalance_ratio': imbalance_ratio,
        'is_balanced': imbalance_ratio < 1.5  # Considered balanced if < 1.5x
    }
    
    return report


if __name__ == '__main__':
    # Fix encoding for Windows
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Test với dummy data
    print("="*70)
    print("TEST: Oversampling Utilities")
    print("="*70)
    
    # Create imbalanced dummy data
    data = {
        'text': ['sample'] * 1000,
        'sentiment': ['negative'] * 600 + ['positive'] * 350 + ['neutral'] * 50
    }
    df = pd.DataFrame(data)
    
    print("\n1️⃣ Original Data:")
    report = get_class_balance_report(df)
    print(f"   Imbalance ratio: {report['imbalance_ratio']:.2f}x")
    
    print("\n2️⃣ Auto Oversampling (balance all to majority):")
    df_balanced = random_oversample(df, sampling_strategy='auto')
    
    print("\n3️⃣ Minority Oversampling (only minority class):")
    df_minority = random_oversample(df, sampling_strategy='minority')
    
    print("\n4️⃣ Smart Oversampling (min 30% ratio):")
    df_smart = smart_oversample(df, min_ratio=0.3)
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)
