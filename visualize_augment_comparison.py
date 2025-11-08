"""
Visualize Before/After Augmentation Comparison for Multi-Label ABSA
Creates visualizations comparing data distribution before and after augmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from typing import Optional
import yaml

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_imbalance_info(df, aspect_cols):
    """Calculate imbalance information for each aspect"""
    imbalance_info = {}
    
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
        
        imbalance_info[aspect] = {
            'Positive': pos,
            'Negative': neg,
            'Neutral': neu,
            'total': pos + neg + neu,
            'imbalance_ratio': imbalance_ratio
        }
    
    return imbalance_info

def plot_sentiment_distribution_comparison(before_info, after_info, aspect_cols, output_dir):
    """Plot side-by-side comparison of sentiment distribution"""
    n_aspects = len(aspect_cols)
    fig, axes = plt.subplots(n_aspects, 2, figsize=(16, 4 * n_aspects))
    fig.suptitle('Sentiment Distribution: Before vs After Augmentation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    sentiments = ['Positive', 'Negative', 'Neutral']
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
    
    for idx, aspect in enumerate(aspect_cols):
        # Before
        before_data = before_info[aspect]
        before_counts = [before_data['Positive'], before_data['Negative'], before_data['Neutral']]
        
        axes[idx, 0].bar(sentiments, before_counts, color=[colors[s] for s in sentiments], alpha=0.7)
        axes[idx, 0].set_title(f'{aspect} - Before\n(Imbalance: {before_info[aspect]["imbalance_ratio"]:.2f}x)', 
                              fontweight='bold')
        axes[idx, 0].set_ylabel('Count')
        axes[idx, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (sent, count) in enumerate(zip(sentiments, before_counts)):
            axes[idx, 0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
        
        # After
        after_data = after_info[aspect]
        after_counts = [after_data['Positive'], after_data['Negative'], after_data['Neutral']]
        
        axes[idx, 1].bar(sentiments, after_counts, color=[colors[s] for s in sentiments], alpha=0.7)
        axes[idx, 1].set_title(f'{aspect} - After\n(Imbalance: {after_info[aspect]["imbalance_ratio"]:.2f}x)', 
                              fontweight='bold')
        axes[idx, 1].set_ylabel('Count')
        axes[idx, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (sent, count) in enumerate(zip(sentiments, after_counts)):
            axes[idx, 1].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'sentiment_distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_imbalance_ratio_comparison(before_info, after_info, aspect_cols, output_dir):
    """Plot imbalance ratio comparison"""
    aspects = []
    before_ratios = []
    after_ratios = []
    
    for aspect in aspect_cols:
        before_ratio = before_info[aspect]['imbalance_ratio']
        after_ratio = after_info[aspect]['imbalance_ratio']
        
        if before_ratio != float('inf') and after_ratio != float('inf'):
            aspects.append(aspect)
            before_ratios.append(before_ratio)
            after_ratios.append(after_ratio)
    
    x = np.arange(len(aspects))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, before_ratios, width, label='Before', alpha=0.8, color='#e74c3c')
    bars2 = ax.bar(x + width/2, after_ratios, width, label='After', alpha=0.8, color='#2ecc71')
    
    ax.set_xlabel('Aspect', fontweight='bold')
    ax.set_ylabel('Imbalance Ratio (max/min)', fontweight='bold')
    ax.set_title('Imbalance Ratio Comparison: Before vs After Augmentation', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'imbalance_ratio_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_sample_count_comparison(before_info, after_info, aspect_cols, output_dir):
    """Plot total sample count comparison per aspect"""
    aspects = []
    before_totals = []
    after_totals = []
    
    for aspect in aspect_cols:
        aspects.append(aspect)
        before_totals.append(before_info[aspect]['total'])
        after_totals.append(after_info[aspect]['total'])
    
    x = np.arange(len(aspects))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, before_totals, width, label='Before', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, after_totals, width, label='After', alpha=0.8, color='#9b59b6')
    
    ax.set_xlabel('Aspect', fontweight='bold')
    ax.set_ylabel('Total Samples', fontweight='bold')
    ax.set_title('Total Sample Count Comparison: Before vs After Augmentation', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'sample_count_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_overall_statistics(before_df, after_df, before_info, after_info, aspect_cols, output_dir):
    """Plot overall statistics comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Overall Statistics: Before vs After Augmentation', 
                 fontsize=16, fontweight='bold')
    
    # 1. Total samples
    total_before = len(before_df)
    total_after = len(after_df)
    increase = total_after - total_before
    increase_pct = (increase / total_before * 100) if total_before > 0 else 0
    
    axes[0, 0].bar(['Before', 'After'], [total_before, total_after], 
                   color=['#e74c3c', '#2ecc71'], alpha=0.7)
    axes[0, 0].set_ylabel('Total Samples', fontweight='bold')
    axes[0, 0].set_title(f'Total Samples\nBefore: {total_before:,} | After: {total_after:,}\nIncrease: +{increase:,} ({increase_pct:.1f}%)', 
                        fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, val in enumerate([total_before, total_after]):
        axes[0, 0].text(i, val, f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Average imbalance ratio
    before_ratios = [info['imbalance_ratio'] for info in before_info.values() 
                     if info['imbalance_ratio'] != float('inf')]
    after_ratios = [info['imbalance_ratio'] for info in after_info.values() 
                    if info['imbalance_ratio'] != float('inf')]
    
    avg_before = np.mean(before_ratios) if before_ratios else 0
    avg_after = np.mean(after_ratios) if after_ratios else 0
    improvement = ((avg_before - avg_after) / avg_before * 100) if avg_before > 0 else 0
    
    axes[0, 1].bar(['Before', 'After'], [avg_before, avg_after], 
                   color=['#e74c3c', '#2ecc71'], alpha=0.7)
    axes[0, 1].set_ylabel('Average Imbalance Ratio', fontweight='bold')
    axes[0, 1].set_title(f'Average Imbalance Ratio\nBefore: {avg_before:.2f}x | After: {avg_after:.2f}x\nImprovement: {improvement:.1f}%', 
                        fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, val in enumerate([avg_before, avg_after]):
        axes[0, 1].text(i, val, f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')
    
    # 3. Overall sentiment distribution - Before
    before_sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for info in before_info.values():
        before_sentiments['Positive'] += info['Positive']
        before_sentiments['Negative'] += info['Negative']
        before_sentiments['Neutral'] += info['Neutral']
    
    axes[1, 0].bar(before_sentiments.keys(), before_sentiments.values(), 
                   color=['#2ecc71', '#e74c3c', '#95a5a6'], alpha=0.7)
    axes[1, 0].set_ylabel('Count', fontweight='bold')
    axes[1, 0].set_title('Overall Sentiment Distribution - Before', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, (sent, count) in enumerate(before_sentiments.items()):
        axes[1, 0].text(i, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall sentiment distribution - After
    after_sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for info in after_info.values():
        after_sentiments['Positive'] += info['Positive']
        after_sentiments['Negative'] += info['Negative']
        after_sentiments['Neutral'] += info['Neutral']
    
    axes[1, 1].bar(after_sentiments.keys(), after_sentiments.values(), 
                   color=['#2ecc71', '#e74c3c', '#95a5a6'], alpha=0.7)
    axes[1, 1].set_ylabel('Count', fontweight='bold')
    axes[1, 1].set_title('Overall Sentiment Distribution - After', fontweight='bold')
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, (sent, count) in enumerate(after_sentiments.items()):
        axes[1, 1].text(i, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'overall_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main(config_path: Optional[str] = None):
    """Main function to create visualizations"""
    print("=" * 80)
    print("Augmentation Comparison Visualization")
    print("=" * 80)
    
    # Determine input files
    if config_path:
        config = load_config(config_path)
        before_file = config['paths']['train_file'].replace('_balanced', '')
        after_file = config['paths']['train_file']
        output_dir = os.path.join(os.path.dirname(config['paths']['data_dir']), 'analysis_results')
    else:
        before_file = 'BILSTM-MTL/data/train_multilabel.csv'
        after_file = 'BILSTM-MTL/data/train_multilabel_balanced.csv'
        output_dir = 'multi_label/analysis_results'
    
    aspect_cols = [
        'Battery', 'Camera', 'Performance', 'Display', 'Design',
        'Packaging', 'Price', 'Shop_Service',
        'Shipping', 'General', 'Others'
    ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Before: {before_file}")
    print(f"  After:  {after_file}")
    
    before_df = pd.read_csv(before_file, encoding='utf-8-sig')
    after_df = pd.read_csv(after_file, encoding='utf-8-sig')
    
    print(f"  Before: {len(before_df):,} samples")
    print(f"  After:  {len(after_df):,} samples")
    
    # Calculate imbalance info
    print(f"\nCalculating imbalance information...")
    before_info = get_imbalance_info(before_df, aspect_cols)
    after_info = get_imbalance_info(after_df, aspect_cols)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    print(f"  Output directory: {output_dir}")
    
    plot_sentiment_distribution_comparison(before_info, after_info, aspect_cols, output_dir)
    plot_imbalance_ratio_comparison(before_info, after_info, aspect_cols, output_dir)
    plot_sample_count_comparison(before_info, after_info, aspect_cols, output_dir)
    plot_overall_statistics(before_df, after_df, before_info, after_info, aspect_cols, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    before_ratios = [info['imbalance_ratio'] for info in before_info.values() 
                     if info['imbalance_ratio'] != float('inf')]
    after_ratios = [info['imbalance_ratio'] for info in after_info.values() 
                    if info['imbalance_ratio'] != float('inf')]
    
    avg_before = np.mean(before_ratios) if before_ratios else 0
    avg_after = np.mean(after_ratios) if after_ratios else 0
    improvement = ((avg_before - avg_after) / avg_before * 100) if avg_before > 0 else 0
    
    print(f"\nTotal samples: {len(before_df):,} -> {len(after_df):,} (+{len(after_df) - len(before_df):,})")
    print(f"Average imbalance ratio: {avg_before:.2f}x -> {avg_after:.2f}x")
    print(f"Improvement: {improvement:.1f}%")
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("  - sentiment_distribution_comparison.png")
    print("  - imbalance_ratio_comparison.png")
    print("  - sample_count_comparison.png")
    print("  - overall_statistics.png")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize before/after augmentation comparison'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (optional)'
    )
    args = parser.parse_args()
    
    main(config_path=args.config)

