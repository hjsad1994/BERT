"""
Create dummy training logs for testing visualization

Creates realistic-looking training logs with:
- Both losses decreasing (Good training)
- Focal decreasing, Contrastive stuck (Bad training scenario)
- Learning rate schedule
"""

import pandas as pd
import numpy as np
import os

def create_good_training_logs(output_dir='training_logs'):
    """Create logs where both losses decrease (GOOD)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    # Simulate 15 epochs
    epochs = np.arange(1, 16)
    
    # Focal loss: decreases from 0.65 to 0.22
    focal_loss = 0.65 * np.exp(-0.15 * (epochs - 1)) + 0.2 + np.random.normal(0, 0.01, len(epochs))
    
    # Contrastive loss: decreases from 0.64 to 0.42
    contr_loss = 0.64 * np.exp(-0.12 * (epochs - 1)) + 0.3 + np.random.normal(0, 0.01, len(epochs))
    
    # Total loss
    total_loss = 0.8 * focal_loss + 0.2 * contr_loss
    
    # Val F1: increases from 88% to 96%
    val_f1 = 0.88 + 0.08 * (1 - np.exp(-0.2 * (epochs - 1))) + np.random.normal(0, 0.005, len(epochs))
    val_f1 = np.clip(val_f1, 0.85, 0.97)
    
    # Other metrics
    val_acc = val_f1 - np.random.uniform(0, 0.01, len(epochs))
    val_precision = val_f1 + np.random.uniform(0, 0.01, len(epochs))
    val_recall = val_f1 - np.random.uniform(0, 0.005, len(epochs))
    
    # Learning rate: cosine schedule
    lr_start = 2e-5
    lr_end = 5e-7
    learning_rate = lr_end + (lr_start - lr_end) * 0.5 * (1 + np.cos(np.pi * (epochs - 1) / len(epochs)))
    
    # Create DataFrame
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': total_loss,
        'train_focal_loss': focal_loss,
        'train_contrastive_loss': contr_loss,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'learning_rate': learning_rate
    })
    
    # Save
    filename = os.path.join(output_dir, 'epoch_losses_good_training_dummy.csv')
    df.to_csv(filename, index=False)
    print(f"Created: {filename}")
    print(f"\nPreview:")
    print(df.head(10))
    print(f"\nTraining summary:")
    print(f"   Focal loss:    {df['train_focal_loss'].iloc[0]:.4f} -> {df['train_focal_loss'].iloc[-1]:.4f} (decreased)")
    print(f"   Contr loss:    {df['train_contrastive_loss'].iloc[0]:.4f} -> {df['train_contrastive_loss'].iloc[-1]:.4f} (decreased)")
    print(f"   Val F1:        {df['val_f1'].iloc[0]*100:.2f}% -> {df['val_f1'].iloc[-1]*100:.2f}% (increased)")
    
    return filename


def create_bad_training_logs(output_dir='training_logs'):
    """Create logs where contrastive loss is stuck (BAD)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    epochs = np.arange(1, 16)
    
    # Focal loss: decreases normally
    focal_loss = 0.65 * np.exp(-0.18 * (epochs - 1)) + 0.18 + np.random.normal(0, 0.01, len(epochs))
    
    # Contrastive loss: STUCK! Only slight noise
    contr_loss = 0.63 + np.random.normal(0, 0.02, len(epochs))
    
    # Total loss
    total_loss = 0.8 * focal_loss + 0.2 * contr_loss
    
    # Val F1: increases but not as much
    val_f1 = 0.88 + 0.06 * (1 - np.exp(-0.15 * (epochs - 1))) + np.random.normal(0, 0.005, len(epochs))
    val_f1 = np.clip(val_f1, 0.85, 0.95)  # Peaks at 95%, not 96%
    
    val_acc = val_f1 - np.random.uniform(0, 0.01, len(epochs))
    val_precision = val_f1 + np.random.uniform(0, 0.01, len(epochs))
    val_recall = val_f1 - np.random.uniform(0, 0.005, len(epochs))
    
    lr_start = 2e-5
    lr_end = 5e-7
    learning_rate = lr_end + (lr_start - lr_end) * 0.5 * (1 + np.cos(np.pi * (epochs - 1) / len(epochs)))
    
    df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': total_loss,
        'train_focal_loss': focal_loss,
        'train_contrastive_loss': contr_loss,
        'val_accuracy': val_acc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'learning_rate': learning_rate
    })
    
    filename = os.path.join(output_dir, 'epoch_losses_bad_training_dummy.csv')
    df.to_csv(filename, index=False)
    print(f"\nCreated: {filename}")
    print(f"\nPreview:")
    print(df.head(10))
    print(f"\nTraining summary:")
    print(f"   Focal loss:    {df['train_focal_loss'].iloc[0]:.4f} -> {df['train_focal_loss'].iloc[-1]:.4f} (decreased)")
    print(f"   Contr loss:    {df['train_contrastive_loss'].iloc[0]:.4f} -> {df['train_contrastive_loss'].iloc[-1]:.4f} (STUCK!)")
    print(f"   Val F1:        {df['val_f1'].iloc[0]*100:.2f}% -> {df['val_f1'].iloc[-1]*100:.2f}% (only to 95%)")
    
    return filename


def create_batch_logs(output_dir='training_logs', num_epochs=3, batches_per_epoch=100):
    """Create batch-level logs"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    batch_data = []
    
    for epoch in range(1, num_epochs + 1):
        # Base losses decrease per epoch
        focal_base = 0.7 - 0.15 * (epoch - 1)
        contr_base = 0.65 - 0.1 * (epoch - 1)
        
        for batch in range(batches_per_epoch):
            # Add within-epoch variation
            batch_progress = batch / batches_per_epoch
            focal_loss = focal_base * (1 - 0.2 * batch_progress) + np.random.normal(0, 0.05)
            contr_loss = contr_base * (1 - 0.15 * batch_progress) + np.random.normal(0, 0.04)
            total_loss = 0.8 * focal_loss + 0.2 * contr_loss
            
            batch_data.append({
                'epoch': epoch,
                'batch': batch,
                'total_loss': max(0.1, total_loss),
                'focal_loss': max(0.1, focal_loss),
                'contrastive_loss': max(0.1, contr_loss)
            })
    
    df = pd.DataFrame(batch_data)
    
    filename = os.path.join(output_dir, 'batch_losses_dummy.csv')
    df.to_csv(filename, index=False)
    print(f"\nCreated: {filename}")
    print(f"   Total batches: {len(df)}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batches per epoch: {batches_per_epoch}")
    
    return filename


if __name__ == '__main__':
    print("="*80)
    print("Creating Dummy Training Logs")
    print("="*80)
    
    # Create output directory
    output_dir = 'multi_label/training_logs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logs
    print("\n1. Creating GOOD training logs (both losses decrease)...")
    good_file = create_good_training_logs(output_dir)
    
    print("\n" + "-"*80)
    print("\n2. Creating BAD training logs (contrastive stuck)...")
    bad_file = create_bad_training_logs(output_dir)
    
    print("\n" + "-"*80)
    print("\n3. Creating batch logs...")
    batch_file = create_batch_logs(output_dir)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    
    print("\nCreated files:")
    print(f"   {good_file}")
    print(f"   {bad_file}")
    print(f"   {batch_file}")
    
    print("\nTo visualize:")
    print(f"\n   # Good training:")
    print(f"   python multi_label\\visualize_training_logs.py --epoch-log {good_file}")
    
    print(f"\n   # Bad training:")
    print(f"   python multi_label\\visualize_training_logs.py --epoch-log {bad_file}")
    
    print(f"\n   # With batch logs:")
    print(f"   python multi_label\\visualize_training_logs.py --epoch-log {good_file} --batch-log {batch_file}")
