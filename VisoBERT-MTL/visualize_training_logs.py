"""
Visualize Training Logs from Real Data

Usage:
    python visualize_training_logs.py --log-file path/to/epoch_losses.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def plot_losses_from_csv(csv_path, output_dir='training_logs'):
    """Plot losses from training log CSV"""
    
    print(f"\nLoading logs from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} epochs")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ============ PLOT 1: Losses Over Time ============
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1.1: Total Loss
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss'], 'b-o', linewidth=2, markersize=8, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Total Training Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Annotate start and end
    start_loss = df['train_loss'].iloc[0]
    end_loss = df['train_loss'].iloc[-1]
    ax1.annotate(f'Start: {start_loss:.4f}', 
                xy=(df['epoch'].iloc[0], start_loss),
                xytext=(df['epoch'].iloc[0] + 1, start_loss + 0.05),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, color='blue', fontweight='bold')
    ax1.annotate(f'End: {end_loss:.4f}', 
                xy=(df['epoch'].iloc[-1], end_loss),
                xytext=(df['epoch'].iloc[-1] - 2, end_loss + 0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold')
    
    # Plot 1.2: Focal vs Contrastive
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'], df['train_focal_loss'], 'r-s', linewidth=2, markersize=8, label='Focal Loss')
    ax2.plot(df['epoch'], df['train_contrastive_loss'], 'g-^', linewidth=2, markersize=8, label='Contrastive Loss')
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Focal vs Contrastive Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Check if both decrease
    focal_decreasing = df['train_focal_loss'].iloc[-1] < df['train_focal_loss'].iloc[0]
    contr_decreasing = df['train_contrastive_loss'].iloc[-1] < df['train_contrastive_loss'].iloc[0]
    
    if focal_decreasing and contr_decreasing:
        status_text = 'Both decreasing - GOOD!'
        status_color = 'green'
    elif focal_decreasing and not contr_decreasing:
        status_text = 'Contrastive not decreasing - increase weight'
        status_color = 'red'
    elif not focal_decreasing and contr_decreasing:
        status_text = 'Focal not decreasing - increase weight'
        status_color = 'red'
    else:
        status_text = 'Both not decreasing - check config'
        status_color = 'red'
    
    ax2.text(0.5, 0.95, status_text,
            transform=ax2.transAxes, ha='center', va='top',
            fontsize=11, color=status_color, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Plot 1.3: Validation F1
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'], df['val_f1'] * 100, 'purple', marker='o', linewidth=2, markersize=8)
    ax3.axhline(y=96, color='green', linestyle='--', alpha=0.5, label='Target: 96%')
    ax3.axhline(y=95.5, color='orange', linestyle='--', alpha=0.5, label='Baseline: 95.5%')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Best F1
    best_f1 = df['val_f1'].max() * 100
    best_epoch = df.loc[df['val_f1'].idxmax(), 'epoch']
    ax3.annotate(f'Best: {best_f1:.2f}% (Epoch {int(best_epoch)})', 
                xy=(best_epoch, best_f1),
                xytext=(best_epoch - 2, best_f1 + 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    # Plot 1.4: Learning Rate
    if 'learning_rate' in df.columns:
        ax4 = axes[1, 1]
        ax4.plot(df['epoch'], df['learning_rate'], 'orange', marker='s', linewidth=2, markersize=8)
        ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning rate not logged',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'training_losses_real_data.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.show()
    
    # ============ PLOT 2: Loss Reduction Rate ============
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate % reduction
    focal_start = df['train_focal_loss'].iloc[0]
    focal_reduction = (df['train_focal_loss'] - focal_start) / focal_start * 100
    
    contr_start = df['train_contrastive_loss'].iloc[0]
    contr_reduction = (df['train_contrastive_loss'] - contr_start) / contr_start * 100
    
    ax.plot(df['epoch'], focal_reduction, 'r-o', linewidth=2, markersize=8, label='Focal Loss Reduction')
    ax.plot(df['epoch'], contr_reduction, 'g-s', linewidth=2, markersize=8, label='Contrastive Loss Reduction')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Reduction (%)', fontsize=12, fontweight='bold')
    ax.set_title('Loss Reduction from Starting Point', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Final reductions
    focal_final = focal_reduction.iloc[-1]
    contr_final = contr_reduction.iloc[-1]
    ax.text(0.02, 0.98, f'Focal reduced: {abs(focal_final):.1f}%\nContrastive reduced: {abs(contr_final):.1f}%',
           transform=ax.transAxes, va='top',
           fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'loss_reduction_rates.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()
    
    # ============ PRINT SUMMARY ============
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print(f"\nLoss Changes:")
    print(f"   Focal Loss:")
    print(f"      Start: {df['train_focal_loss'].iloc[0]:.4f}")
    print(f"      End:   {df['train_focal_loss'].iloc[-1]:.4f}")
    print(f"      Change: {focal_final:.1f}%")
    
    print(f"\n   Contrastive Loss:")
    print(f"      Start: {df['train_contrastive_loss'].iloc[0]:.4f}")
    print(f"      End:   {df['train_contrastive_loss'].iloc[-1]:.4f}")
    print(f"      Change: {contr_final:.1f}%")
    
    print(f"\nF1 Score:")
    print(f"      Start: {df['val_f1'].iloc[0]*100:.2f}%")
    print(f"      Best:  {best_f1:.2f}% (Epoch {int(best_epoch)})")
    print(f"      End:   {df['val_f1'].iloc[-1]*100:.2f}%")
    
    print(f"\nTraining Behavior:")
    if focal_decreasing and contr_decreasing:
        print(f"   GOOD: Both losses decreasing!")
    else:
        print(f"   WARNING: Check weight balance")
        if not focal_decreasing:
            print(f"      - Focal loss not decreasing -> increase focal_weight")
        if not contr_decreasing:
            print(f"      - Contrastive loss not decreasing -> increase contrastive_weight")
    
    print("\n" + "="*80)


def plot_batch_losses(csv_path, output_dir='training_logs', num_epochs_to_show=3):
    """Plot batch-level losses (first few epochs)"""
    
    print(f"\nLoading batch logs from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter to first few epochs for readability
    df_subset = df[df['epoch'] <= num_epochs_to_show]
    
    if len(df_subset) == 0:
        print("No batch data found")
        return
    
    print(f"Plotting {len(df_subset)} batches from {num_epochs_to_show} epochs")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: All losses
    ax1 = axes[0]
    for epoch in df_subset['epoch'].unique():
        epoch_data = df_subset[df_subset['epoch'] == epoch]
        x = epoch_data['batch'].values + (epoch - 1) * df_subset['batch'].max()
        ax1.plot(x, epoch_data['focal_loss'], 'r-', alpha=0.6, linewidth=1)
        ax1.plot(x, epoch_data['contrastive_loss'], 'g-', alpha=0.6, linewidth=1)
    
    ax1.plot([], [], 'r-', label='Focal Loss', linewidth=2)
    ax1.plot([], [], 'g-', label='Contrastive Loss', linewidth=2)
    ax1.set_xlabel('Batch (across epochs)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'Batch-Level Losses (First {num_epochs_to_show} Epochs)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    ax2 = axes[1]
    window = 50
    df_subset['focal_ma'] = df_subset['focal_loss'].rolling(window=window, min_periods=1).mean()
    df_subset['contr_ma'] = df_subset['contrastive_loss'].rolling(window=window, min_periods=1).mean()
    
    ax2.plot(df_subset.index, df_subset['focal_ma'], 'r-', linewidth=2, label=f'Focal (MA-{window})')
    ax2.plot(df_subset.index, df_subset['contr_ma'], 'g-', linewidth=2, label=f'Contrastive (MA-{window})')
    ax2.set_xlabel('Batch Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (Moving Average)', fontsize=12, fontweight='bold')
    ax2.set_title('Smoothed Batch Losses', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'batch_losses_real_data.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Training Logs')
    parser.add_argument('--epoch-log', type=str, help='Path to epoch losses CSV')
    parser.add_argument('--batch-log', type=str, help='Path to batch losses CSV')
    parser.add_argument('--output-dir', type=str, default='training_logs', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.epoch_log:
        if os.path.exists(args.epoch_log):
            plot_losses_from_csv(args.epoch_log, args.output_dir)
        else:
            print(f"Error: File not found: {args.epoch_log}")
    
    if args.batch_log:
        if os.path.exists(args.batch_log):
            plot_batch_losses(args.batch_log, args.output_dir)
        else:
            print(f"Error: File not found: {args.batch_log}")
    
    if not args.epoch_log and not args.batch_log:
        print("\n" + "="*80)
        print("NO LOGS PROVIDED")
        print("="*80)
        print("\nTo visualize training logs:")
        print("\n1. First, train with logging:")
        print("   python train_multilabel_focal_contrastive_with_logging.py")
        print("\n2. Then visualize:")
        print("   python visualize_training_logs.py --epoch-log path/to/epoch_losses.csv")
        print("\nExample:")
        print("   python visualize_training_logs.py \\")
        print("       --epoch-log multi_label/models/multilabel_ghm_only/training_logs/epoch_losses_20251021_212833.csv")


if __name__ == '__main__':
    main()
