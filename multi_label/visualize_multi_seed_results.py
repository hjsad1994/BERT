"""
Visualize Multi-Seed Experiment Results
Creates plots showing mean and standard deviation across multiple training runs
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def plot_overall_metrics(df: pd.DataFrame, output_dir: str) -> None:
    """Plot overall metrics with error bars"""
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    metric_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        # Extract values for each seed
        values = df[metric] * 100
        seeds = df['seed']
        
        # Plot individual points
        ax.scatter(seeds, values, s=100, alpha=0.6, color='steelblue', zorder=3)
        
        # Plot mean line
        mean_val = values.mean()
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}%', zorder=2)
        
        # Plot std band
        std_val = values.std()
        ax.axhspan(mean_val - std_val, mean_val + std_val, 
                   alpha=0.2, color='red', label=f'±1 Std: {std_val:.2f}%')
        
        ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{name} (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{name} Across Seeds', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        y_min = max(0, mean_val - 3*std_val)
        y_max = min(100, mean_val + 3*std_val)
        ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'overall_metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_aspect_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Plot per-aspect metrics comparison"""
    # Find all aspects
    aspects = []
    for col in df.columns:
        if col.endswith('_f1') and not col.startswith('test_'):
            aspect = col.replace('_f1', '')
            aspects.append(aspect)
    
    aspects = sorted(aspects)
    
    # Prepare data for plotting
    aspect_means = []
    aspect_stds = []
    
    for aspect in aspects:
        f1_col = f'{aspect}_f1'
        if f1_col in df.columns:
            values = df[f1_col] * 100
            aspect_means.append(values.mean())
            aspect_stds.append(values.std())
        else:
            aspect_means.append(0)
            aspect_stds.append(0)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(aspects))
    bars = ax.bar(x, aspect_means, yerr=aspect_stds, capsize=5, 
                   color='steelblue', alpha=0.7, edgecolor='black')
    
    # Color code by performance
    for i, bar in enumerate(bars):
        if aspect_means[i] >= 80:
            bar.set_color('green')
        elif aspect_means[i] >= 70:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Aspect', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score (%) - Mean ± Std', fontsize=12, fontweight='bold')
    ax.set_title('Per-Aspect F1 Scores Across All Seeds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(aspect_means, aspect_stds)):
        ax.text(i, mean + std + 2, f'{mean:.1f}±{std:.1f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'per_aspect_f1_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_metric_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Create heatmap of all metrics across seeds"""
    # Select relevant columns
    metric_cols = [col for col in df.columns 
                   if any(x in col for x in ['accuracy', 'f1', 'precision', 'recall'])
                   and not col.startswith('best_val')]
    
    if not metric_cols:
        print("No metrics found for heatmap")
        return
    
    # Prepare data
    heatmap_data = df[metric_cols] * 100
    heatmap_data.index = [f"Seed {s}" for s in df['seed']]
    
    # Rename columns for better display
    heatmap_data.columns = [col.replace('_', ' ').title() for col in heatmap_data.columns]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': 'Score (%)'}, ax=ax)
    
    ax.set_title('All Metrics Heatmap Across Seeds', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Seeds', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_variance_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Analyze variance across different metrics"""
    # Calculate coefficient of variation (CV) for each metric
    metric_cols = [col for col in df.columns 
                   if any(x in col for x in ['accuracy', 'f1'])
                   and not col.startswith('best_val')]
    
    cv_data = []
    for col in metric_cols:
        values = df[col] * 100
        mean_val = values.mean()
        std_val = values.std()
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        
        cv_data.append({
            'metric': col.replace('_', ' ').title(),
            'mean': mean_val,
            'std': std_val,
            'cv': cv
        })
    
    cv_df = pd.DataFrame(cv_data).sort_values('cv', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(len(cv_df))
    bars = ax.barh(y_pos, cv_df['cv'], color='coral', alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cv_df['metric'])
    ax.set_xlabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    ax.set_title('Metric Variance Across Seeds (Lower = More Stable)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (cv, mean, std) in enumerate(zip(cv_df['cv'], cv_df['mean'], cv_df['std'])):
        ax.text(cv + 0.1, i, f'{cv:.2f}% (μ={mean:.1f}, σ={std:.1f})', 
                va='center', fontsize=8)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'variance_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main(results_dir: str):
    """Main visualization function"""
    print("=" * 80)
    print("Visualizing Multi-Seed Experiment Results")
    print("=" * 80)
    
    # Load individual results
    results_file = os.path.join(results_dir, 'individual_runs_results.csv')
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        return
    
    df = pd.read_csv(results_file, encoding='utf-8-sig')
    print(f"\nLoaded results for {len(df)} runs")
    print(f"Seeds: {sorted(df['seed'].tolist())}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    plot_overall_metrics(df, results_dir)
    plot_aspect_comparison(df, results_dir)
    plot_metric_heatmap(df, results_dir)
    plot_variance_analysis(df, results_dir)
    
    print("\n" + "=" * 80)
    print("Visualization Complete!")
    print("=" * 80)
    print(f"\nAll plots saved to: {results_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize multi-seed experiment results')
    parser.add_argument('--results-dir', type=str, 
                       default='multi_label/results/multi_seed_experiments',
                       help='Directory containing experiment results')
    
    args = parser.parse_args()
    main(args.results_dir)
