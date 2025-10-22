"""
Training Script for Multi-Label ABSA with GHM-C Loss + Contrastive Learning + LOGGING

Loss = GHM-C Loss + Contrastive Loss
- GHM-C Loss: Dynamic gradient harmonizing for class imbalance (better than Focal Loss)
- Contrastive Loss: Learn better representations

GHM-C Loss advantages over Focal Loss:
1. Automatic adjustment based on gradient density
2. Better handling of easy, hard, and outlier samples
3. Less hyperparameter tuning
4. More stable training

Expected: 96.5-97% F1 (vs Focal: 95.99%)
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
import sys

# Add losses directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_multilabel_focal_contrastive import MultiLabelViSoBERTFocalContrastive
from model_multilabel_contrastive_v2 import ImprovedMultiLabelContrastiveLoss
from losses.ghm_loss import MultiLabelGHM_Loss
from utils import FocalLoss  # Fallback if needed
from dataset_multilabel import MultiLabelABSADataset


def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def combined_ghm_contrastive_loss(logits, embeddings, labels, 
                                  ghm_loss_fn, contrastive_loss_fn, 
                                  weights=None, 
                                  classification_weight=0.95, 
                                  contrastive_weight=0.05):
    """
    Combined loss: GHM-C Loss + Contrastive Loss
    
    Args:
        logits: [batch_size, num_aspects, num_sentiments]
        embeddings: [batch_size, projection_dim]
        labels: [batch_size, num_aspects]
        ghm_loss_fn: GHM-C loss function (handles all aspects at once)
        contrastive_loss_fn: Contrastive loss function
        weights: [num_aspects, num_sentiments] class weights (not used by GHM-C)
        classification_weight: Weight for GHM-C loss
        contrastive_weight: Weight for contrastive loss
    """
    # GHM-C Loss - handles all aspects automatically
    ghm_loss = ghm_loss_fn(logits, labels)
    
    # Contrastive loss
    contr_loss = contrastive_loss_fn(embeddings, labels)
    
    # Combined loss
    total_loss = classification_weight * ghm_loss + contrastive_weight * contr_loss
    
    return total_loss, ghm_loss, contr_loss


def train_epoch(model, dataloader, optimizer, scheduler, device, 
                ghm_loss_fn, contrastive_loss_fn, weights=None, 
                classification_weight=0.95, contrastive_weight=0.05):
    """Train for one epoch WITH LOGGING"""
    model.train()
    
    total_loss = 0
    total_classification_loss = 0
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
        loss, classification_loss, contr_loss = combined_ghm_contrastive_loss(
            logits, embeddings, labels,
            ghm_loss_fn, contrastive_loss_fn,
            weights=weights,
            classification_weight=classification_weight,
            contrastive_weight=contrastive_weight
        )
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Accumulate
        total_loss += loss.item()
        total_classification_loss += classification_loss.item()
        total_contr_loss += contr_loss.item()
        batch_count += 1
        
        # Store batch loss
        batch_losses.append({
            'batch': batch_idx,
            'total_loss': loss.item(),
            'ghm_loss': classification_loss.item(),
            'contrastive_loss': contr_loss.item()
        })
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ghm': f'{classification_loss.item():.4f}',
            'contr': f'{contr_loss.item():.4f}'
        })
    
    avg_loss = total_loss / batch_count
    avg_classification_loss = total_classification_loss / batch_count
    avg_contr_loss = total_contr_loss / batch_count
    
    return avg_loss, avg_classification_loss, avg_contr_loss, batch_losses


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
        print(f"✅ Saved best model: {best_path}")
    
    return checkpoint_path


def main(args):
    print("=" * 80)
    print("Multi-Label ABSA Training with GHM-C + Contrastive + LOGGING")
    print("=" * 80)
    print("\n🔥 Using GHM-C Loss (improved over Focal Loss)")
    print("   Expected: 96.5-97% F1 (vs Focal: 95.99%)")
    
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
    
    # Calculate class weights (for reference, not used by GHM-C)
    print(f"\nCalculating class weights (reference only)...")
    weights = train_dataset.get_label_weights().to(device)
    print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Create model
    print(f"\nCreating model with GHM-C + Contrastive...")
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
    loss_type = multi_label_config.get('loss_type', 'ghm')
    
    classification_weight = args.classification_weight if args.classification_weight is not None else multi_label_config.get('classification_weight', 0.95)
    contrastive_weight = args.contrastive_weight if args.contrastive_weight is not None else multi_label_config.get('contrastive_weight', 0.05)
    temperature = args.temperature if args.temperature is not None else multi_label_config.get('contrastive_temperature', 0.1)
    contrastive_base_weight = multi_label_config.get('contrastive_base_weight', 0.1)
    
    # GHM-C Loss settings
    ghm_bins = multi_label_config.get('ghm_bins', 10)
    ghm_momentum = multi_label_config.get('ghm_momentum', 0.75)
    
    # Create loss functions
    if loss_type == 'ghm':
        print(f"\n🔥 Using GHM-C Loss")
        classification_loss_fn = MultiLabelGHM_Loss(
            num_aspects=11,
            num_sentiments=3,
            bins=ghm_bins,
            momentum=ghm_momentum,
            loss_weight=1.0
        )
    else:
        print(f"\n⚠️  Fallback to Focal Loss")
        focal_gamma = multi_label_config.get('focal_gamma', 2.0)
        classification_loss_fn = FocalLoss(alpha=None, gamma=focal_gamma, reduction='mean')
    
    contrastive_loss_fn = ImprovedMultiLabelContrastiveLoss(
        temperature=temperature,
        base_weight=contrastive_base_weight
    )
    
    print(f"\nLoss settings:")
    print(f"   Loss type: {loss_type.upper()}")
    if loss_type == 'ghm':
        print(f"   GHM bins: {ghm_bins}")
        print(f"   GHM momentum: {ghm_momentum}")
    print(f"   Classification weight: {classification_weight}")
    print(f"   Contrastive weight: {contrastive_weight}")
    print(f"   Contrastive temperature: {temperature}")
    print(f"   Contrastive base weight: {contrastive_base_weight}")
    
    # Optimizer & Scheduler
    learning_rate = config['training'].get('learning_rate', 2e-5)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    num_epochs = args.epochs if args.epochs is not None else config['training'].get('num_train_epochs', 15)
    output_dir = args.output_dir if args.output_dir is not None else config['paths'].get('output_dir', 'multilabel_ghm_contrastive_model')
    
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
    print("Starting Training with GHM-C Loss")
    print(f"{'='*80}")
    
    best_f1 = 0.0
    aspect_names = train_dataset.aspects
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, classification_loss, contr_loss, batch_losses = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            classification_loss_fn, contrastive_loss_fn, weights=weights,
            classification_weight=classification_weight,
            contrastive_weight=contrastive_weight
        )
        
        print(f"\nTrain Losses:")
        print(f"   Total: {train_loss:.4f}")
        print(f"   GHM-C: {classification_loss:.4f}")
        print(f"   Contrastive: {contr_loss:.4f}")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names)
        print_metrics(val_metrics)
        
        # ========== LOG EPOCH DATA ==========
        epoch_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ghm_loss': classification_loss,
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
            print(f"\n🎉 New best F1: {best_f1*100:.2f}%")
        
        save_checkpoint(model, optimizer, epoch, val_metrics, output_dir, is_best=is_best)
        
        # ========== SAVE LOGS AFTER EACH EPOCH ==========
        pd.DataFrame(training_history).to_csv(epoch_log_file, index=False)
        print(f"📊 Saved epoch log: {epoch_log_file}")
    
    # ========== SAVE BATCH LOGS ==========
    pd.DataFrame(all_batch_losses).to_csv(batch_log_file, index=False)
    print(f"\n📊 Saved batch log: {batch_log_file}")
    
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
        'method': 'GHM-C Loss + Contrastive Learning',
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect'],
        'hyperparameters': {
            'loss_type': loss_type,
            'ghm_bins': ghm_bins,
            'ghm_momentum': ghm_momentum,
            'classification_weight': classification_weight,
            'contrastive_weight': contrastive_weight,
            'contrastive_temperature': temperature,
            'contrastive_base_weight': contrastive_base_weight
        },
        'training_log_files': {
            'epoch_losses': epoch_log_file,
            'batch_losses': batch_log_file
        }
    }
    
    results_file = os.path.join(output_dir, 'test_results_ghm_contrastive.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Final comparison message
    print(f"\n{'='*80}")
    print("🎊 Training Complete!")
    print(f"{'='*80}")
    print(f"\n📊 Final Results:")
    print(f"   Test F1: {test_metrics['overall_f1']*100:.2f}%")
    print(f"\n💡 Compare with Focal Loss baseline (95.99%):")
    print(f"   Improvement: {(test_metrics['overall_f1'] - 0.9599)*100:+.2f}%")
    print(f"\n✅ Check logs at: {log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Label ABSA with GHM-C + Contrastive + LOGGING')
    parser.add_argument('--config', type=str, default='multi_label/config_ghm.yaml',
                        help='Path to config file (default: multi_label/config_ghm.yaml)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config if provided)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config if provided)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Contrastive temperature (overrides config if provided)')
    parser.add_argument('--classification-weight', type=float, default=None,
                        help='Weight for GHM-C loss (overrides config if provided)')
    parser.add_argument('--contrastive-weight', type=float, default=None,
                        help='Weight for contrastive loss (overrides config if provided)')
    
    args = parser.parse_args()
    main(args)
