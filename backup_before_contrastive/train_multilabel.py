"""
Training Script for Multi-Label ABSA
Train ViSoBERT to predict all 13 aspects simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
from datetime import datetime

from model_multilabel import MultiLabelViSoBERT
from dataset_multilabel import MultiLabelABSADataset

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def multilabel_loss(logits, labels, weights=None):
    """
    Multi-label loss: Cross-entropy per aspect
    
    Args:
        logits: [batch_size, num_aspects, num_sentiments]
        labels: [batch_size, num_aspects]
        weights: [num_aspects, num_sentiments] (optional)
    
    Returns:
        loss: scalar
    """
    batch_size, num_aspects, num_sentiments = logits.shape
    
    total_loss = 0
    for i in range(num_aspects):
        aspect_logits = logits[:, i, :]  # [batch_size, num_sentiments]
        aspect_labels = labels[:, i]      # [batch_size]
        
        if weights is not None:
            aspect_weights = weights[i]  # [num_sentiments]
            loss = F.cross_entropy(aspect_logits, aspect_labels, weight=aspect_weights)
        else:
            loss = F.cross_entropy(aspect_logits, aspect_labels)
        
        total_loss += loss
    
    return total_loss / num_aspects

def train_epoch(model, dataloader, optimizer, scheduler, device, weights=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(input_ids, attention_mask)
        
        # Loss
        loss = multilabel_loss(logits, labels, weights)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device, aspect_names):
    """Evaluate model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Predict
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)  # [batch_size, num_aspects]
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)  # [num_samples, num_aspects]
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics per aspect
    aspect_metrics = {}
    
    for i, aspect in enumerate(aspect_names):
        aspect_preds = all_preds[:, i].numpy()
        aspect_labels = all_labels[:, i].numpy()
        
        # Accuracy
        acc = accuracy_score(aspect_labels, aspect_preds)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='weighted', zero_division=0
        )
        
        aspect_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Overall metrics (average across aspects)
    overall_acc = (all_preds == all_labels).float().mean().item()
    
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    overall_precision = np.mean([m['precision'] for m in aspect_metrics.values()])
    overall_recall = np.mean([m['recall'] for m in aspect_metrics.values()])
    
    return {
        'overall_accuracy': overall_acc,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'per_aspect': aspect_metrics
    }

def print_metrics(metrics, epoch=None):
    """Pretty print metrics"""
    if epoch is not None:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Results")
        print(f"{'='*80}")
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:  {metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score:  {metrics['overall_f1']*100:.2f}%")
    print(f"   Precision: {metrics['overall_precision']*100:.2f}%")
    print(f"   Recall:    {metrics['overall_recall']*100:.2f}%")
    
    print(f"\n📊 Per-Aspect Metrics:")
    print(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for aspect, m in metrics['per_aspect'].items():
        print(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  {m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%")

def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """Save model checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"✓ Saved best model: {best_path}")
    
    return checkpoint_path

def main(args):
    print("=" * 80)
    print("Multi-Label ABSA Training")
    print("=" * 80)
    
    # Load config
    print(f"\n📖 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Set seed
    torch.manual_seed(config['general']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['general']['seed'])
    
    # Load tokenizer
    print(f"\n✓ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load datasets
    print(f"\n✓ Loading datasets...")
    train_dataset = MultiLabelABSADataset(
        config['paths']['train_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    val_dataset = MultiLabelABSADataset(
        config['paths']['validation_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    test_dataset = MultiLabelABSADataset(
        config['paths']['test_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    batch_size = config['training'].get('per_device_train_batch_size', 32)
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 64)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Calculate class weights for imbalanced data
    print(f"\n✓ Calculating class weights...")
    weights = train_dataset.get_label_weights().to(device)
    print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Create model
    print(f"\nCreating model...")
    model = MultiLabelViSoBERT(
        model_name=config['model']['name'],
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    learning_rate = config['training'].get('learning_rate', 2e-5)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler
    num_epochs = args.epochs
    total_steps = len(train_loader) * num_epochs
    warmup_ratio = config['training'].get('warmup_ratio', 0.06)
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n✓ Training setup:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Total steps: {total_steps}")
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}")
    
    best_f1 = 0.0
    aspect_names = train_dataset.aspects
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, weights)
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names)
        print_metrics(val_metrics)
        
        # Save checkpoint
        is_best = val_metrics['overall_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['overall_f1']
            print(f"\n🎉 New best F1: {best_f1*100:.2f}%")
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            args.output_dir, is_best=is_best
        )
    
    # Test with best model
    print(f"\n{'='*80}")
    print("Testing Best Model")
    print(f"{'='*80}")
    
    # Load best checkpoint
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, aspect_names)
    print_metrics(test_metrics)
    
    # Save final results
    results = {
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect']
    }
    
    import json
    results_file = os.path.join(args.output_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\n✅ Best Model Performance:")
    print(f"   Test Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   Test F1:       {test_metrics['overall_f1']*100:.2f}%")
    print(f"\n📁 Model saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Label ABSA Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='multilabel_model', help='Output directory')
    
    args = parser.parse_args()
    main(args)
