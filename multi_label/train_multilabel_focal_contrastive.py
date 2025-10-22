"""
Training Script for Multi-Label ABSA with Focal Loss + Contrastive Learning + LOGGING

Loss = Focal Loss + Contrastive Loss
- Focal Loss: Handle class imbalance
- Contrastive Loss: Learn better representations

NEW: Automatic logging to CSV files in training_logs/ subfolder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
from datetime import datetime

from model_multilabel_focal_contrastive import MultiLabelViSoBERTFocalContrastive
from model_multilabel_contrastive_v2 import ImprovedMultiLabelContrastiveLoss
from utils import FocalLoss
from dataset_multilabel import MultiLabelABSADataset


def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def combined_focal_contrastive_loss(logits, embeddings, labels, 
                                  focal_loss_fn, contrastive_loss_fn, 
                                  weights=None, 
                                  focal_weight=0.7, 
                                  contrastive_weight=0.3):
    """
    Combined loss: Focal Loss + Contrastive Loss
    
    Args:
        logits: [batch_size, num_aspects, num_sentiments]
        embeddings: [batch_size, projection_dim]
        labels: [batch_size, num_aspects]
        focal_loss_fn: Focal loss function
        contrastive_loss_fn: Contrastive loss function
        weights: [num_aspects, num_sentiments] class weights
        focal_weight: Weight for focal loss (0.7 = focus on classification)
        contrastive_weight: Weight for contrastive loss (0.3 = focus on representations)
    """
    batch_size, num_aspects, num_sentiments = logits.shape
    
    # Focal loss per aspect
    focal_loss = 0
    for i in range(num_aspects):
        aspect_logits = logits[:, i, :]
        aspect_labels = labels[:, i]
        
        if weights is not None:
            aspect_weights = weights[i]
            focal_loss_fn_i = FocalLoss(alpha=aspect_weights, gamma=2.0, reduction='mean')
        else:
            focal_loss_fn_i = focal_loss_fn
        
        loss = focal_loss_fn_i(aspect_logits, aspect_labels)
        focal_loss += loss
    
    focal_loss = focal_loss / num_aspects
    
    # Contrastive loss
    contr_loss = contrastive_loss_fn(embeddings, labels)
    
    # Combined loss
    total_loss = focal_weight * focal_loss + contrastive_weight * contr_loss
    
    return total_loss, focal_loss, contr_loss


def train_epoch(model, dataloader, optimizer, scheduler, device, 
                focal_loss_fn, contrastive_loss_fn, weights=None, 
                focal_weight=0.7, contrastive_weight=0.3):
    """Train for one epoch WITH LOGGING"""
    model.train()
    
    total_loss = 0
    total_focal_loss = 0
    total_contr_loss = 0
    batch_count = 0
    
    # Store batch-level losses
    batch_losses = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits, embeddings = model(input_ids, attention_mask, return_embeddings=True)
        
        # Combined loss
        loss, focal_loss, contr_loss = combined_focal_contrastive_loss(
            logits, embeddings, labels,
            focal_loss_fn, contrastive_loss_fn,
            weights=weights,
            focal_weight=focal_weight,
            contrastive_weight=contrastive_weight
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Accumulate
        total_loss += loss.item()
        total_focal_loss += focal_loss.item()
        total_contr_loss += contr_loss.item()
        batch_count += 1
        
        # Store batch loss
        batch_losses.append({
            'batch': batch_idx,
            'total_loss': loss.item(),
            'focal_loss': focal_loss.item(),
            'contrastive_loss': contr_loss.item()
        })
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'focal': f'{focal_loss.item():.4f}',
            'contr': f'{contr_loss.item():.4f}'
        })
    
    avg_loss = total_loss / batch_count
    avg_focal_loss = total_focal_loss / batch_count
    avg_contr_loss = total_contr_loss / batch_count
    
    return avg_loss, avg_focal_loss, avg_contr_loss, batch_losses


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
            
            logits = model(input_ids, attention_mask, return_embeddings=False)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate metrics per aspect
    aspect_metrics = {}
    
    for i, aspect in enumerate(aspect_names):
        aspect_preds = all_preds[:, i].numpy()
        aspect_labels = all_labels[:, i].numpy()
        
        acc = accuracy_score(aspect_labels, aspect_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='weighted', zero_division=0
        )
        
        aspect_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
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
    
    print(f"\nOverall Metrics:")
    print(f"   Accuracy:  {metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score:  {metrics['overall_f1']*100:.2f}%")
    print(f"   Precision: {metrics['overall_precision']*100:.2f}%")
    print(f"   Recall:    {metrics['overall_recall']*100:.2f}%")
    
    print(f"\nPer-Aspect Metrics:")
    print(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 60)
    
    for aspect, m in metrics['per_aspect'].items():
        print(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  {m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%")


def save_checkpoint(model, optimizer, epoch, metrics, output_dir, is_best=False):
    """Save checkpoint"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    return checkpoint_path


def main(args):
    print("=" * 80)
    print("Multi-Label ABSA Training with Focal + Contrastive + LOGGING")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Set seed
    torch.manual_seed(config['general']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['general']['seed'])
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load datasets
    print(f"\nLoading datasets...")
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # Calculate class weights
    print(f"\nCalculating class weights...")
    weights = train_dataset.get_label_weights().to(device)
    print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Create model
    print(f"\nCreating model with Focal + Contrastive...")
    model = MultiLabelViSoBERTFocalContrastive(
        model_name=config['model']['name'],
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        projection_dim=256,
        dropout=0.3
    )
    model = model.to(device)
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load loss settings
    multi_label_config = config.get('multi_label', {})
    focal_weight = args.focal_weight if args.focal_weight is not None else multi_label_config.get('focal_weight', 0.7)
    contrastive_weight = args.contrastive_weight if args.contrastive_weight is not None else multi_label_config.get('contrastive_weight', 0.3)
    temperature = args.temperature if args.temperature is not None else multi_label_config.get('contrastive_temperature', 0.1)
    focal_gamma = multi_label_config.get('focal_gamma', 2.0)
    contrastive_base_weight = multi_label_config.get('contrastive_base_weight', 0.1)
    
    # Loss functions
    focal_loss_fn = FocalLoss(alpha=None, gamma=focal_gamma, reduction='mean')
    contrastive_loss_fn = ImprovedMultiLabelContrastiveLoss(
        temperature=temperature,
        base_weight=contrastive_base_weight
    )
    
    print(f"\nLoss settings:")
    print(f"   Focal weight: {focal_weight}")
    print(f"   Contrastive weight: {contrastive_weight}")
    print(f"   Focal gamma: {focal_gamma}")
    print(f"   Contrastive temperature: {temperature}")
    print(f"   Contrastive base weight: {contrastive_base_weight}")
    
    # Optimizer & Scheduler
    learning_rate = config['training'].get('learning_rate', 2e-5)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    num_epochs = args.epochs if args.epochs is not None else config['training'].get('num_train_epochs', 8)
    output_dir = args.output_dir if args.output_dir is not None else config['paths'].get('output_dir', 'multilabel_focal_contrastive_model')
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_steps = len(train_loader) * num_epochs
    warmup_ratio = config['training'].get('warmup_ratio', 0.06)
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # ========== LOGGING SETUP ==========
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(output_dir, 'training_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    epoch_log_file = os.path.join(log_dir, f'epoch_losses_{timestamp}.csv')
    batch_log_file = os.path.join(log_dir, f'batch_losses_{timestamp}.csv')
    
    print(f"\nTraining setup:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Total steps: {total_steps}")
    print(f"   Output dir: {output_dir}")
    
    print(f"\nLogging setup:")
    print(f"   Epoch logs: {epoch_log_file}")
    print(f"   Batch logs: {batch_log_file}")
    
    # Training history
    training_history = []
    all_batch_losses = []
    
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
        train_loss, focal_loss, contr_loss, batch_losses = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            focal_loss_fn, contrastive_loss_fn, weights=weights,
            focal_weight=focal_weight,
            contrastive_weight=contrastive_weight
        )
        
        print(f"\nTrain Losses:")
        print(f"   Total: {train_loss:.4f}")
        print(f"   Focal: {focal_loss:.4f}")
        print(f"   Contrastive: {contr_loss:.4f}")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names)
        print_metrics(val_metrics)
        
        # ========== LOG EPOCH DATA ==========
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_focal_loss': focal_loss,
            'train_contrastive_loss': contr_loss,
            'val_accuracy': val_metrics['overall_accuracy'],
            'val_f1': val_metrics['overall_f1'],
            'val_precision': val_metrics['overall_precision'],
            'val_recall': val_metrics['overall_recall'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_data)
        
        # Log batch losses
        for batch_data in batch_losses:
            batch_data['epoch'] = epoch
            all_batch_losses.append(batch_data)
        
        # Save checkpoint
        is_best = val_metrics['overall_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['overall_f1']
            print(f"\nNew best F1: {best_f1*100:.2f}%")
        
        save_checkpoint(model, optimizer, epoch, val_metrics, output_dir, is_best=is_best)
        
        # ========== SAVE LOGS AFTER EACH EPOCH ==========
        pd.DataFrame(training_history).to_csv(epoch_log_file, index=False)
        print(f"Saved epoch log: {epoch_log_file}")
    
    # ========== SAVE BATCH LOGS ==========
    pd.DataFrame(all_batch_losses).to_csv(batch_log_file, index=False)
    print(f"\nSaved batch log: {batch_log_file}")
    
    # Test with best model
    print(f"\n{'='*80}")
    print("Testing Best Model")
    print(f"{'='*80}")
    
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, aspect_names)
    print_metrics(test_metrics)
    
    # Save results
    import json
    results = {
        'method': 'Focal Loss + Contrastive Learning',
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect'],
        'hyperparameters': {
            'focal_weight': focal_weight,
            'contrastive_weight': contrastive_weight,
            'focal_gamma': focal_gamma,
            'contrastive_temperature': temperature,
            'contrastive_base_weight': contrastive_base_weight
        },
        'training_log_files': {
            'epoch_losses': epoch_log_file,
            'batch_losses': batch_log_file
        }
    }
    
    results_file = os.path.join(output_dir, 'test_results_focal_contrastive.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Label ABSA with Focal + Contrastive + LOGGING')
    parser.add_argument('--config', type=str, default='multi_label/config_multi.yaml',
                        help='Path to config file (default: multi_label/config_multi.yaml)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config if provided)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config if provided)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Contrastive temperature (overrides config if provided)')
    parser.add_argument('--focal-weight', type=float, default=None,
                        help='Weight for focal loss (overrides config if provided)')
    parser.add_argument('--contrastive-weight', type=float, default=None,
                        help='Weight for contrastive loss (overrides config if provided)')
    
    args = parser.parse_args()
    main(args)
