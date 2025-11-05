"""
Training Script for BiLSTM Aspect Detection

Single-task learning for binary multi-label aspect detection
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, hamming_loss, f1_score
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
import json
from datetime import datetime
from pathlib import Path

from model_bilstm_ad import BiLSTM_AspectDetection
from dataset_bilstm_ad import AspectDetectionDataset


def load_config(config_path='config_bilstm_ad.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device, aspect_names, threshold=0.5):
    """
    Evaluate model on validation/test set
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).float()
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # [num_samples, num_aspects]
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    
    # Convert to numpy
    y_true = all_labels.numpy()
    y_pred = all_predictions.numpy()
    y_prob = all_probabilities.numpy()
    
    # Calculate metrics
    metrics = {}
    
    # Overall metrics (micro/macro averaging)
    metrics['accuracy'] = accuracy_score(y_true.flatten(), y_pred.flatten())
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    
    # F1 scores
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Precision, Recall, F1 per aspect
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Per-aspect metrics
    aspect_metrics = []
    for i, aspect in enumerate(aspect_names):
        aspect_metrics.append({
            'aspect': aspect,
            'precision': precision[i],
            'recall': recall[i],
            'f1': f1[i],
            'support': int(support[i])
        })
    
    metrics['per_aspect'] = aspect_metrics
    
    # Print summary
    print(f"\n[Evaluation Results]")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    
    print(f"\n  Per-aspect metrics:")
    for am in aspect_metrics:
        print(f"    {am['aspect']:<15} P={am['precision']:.3f} R={am['recall']:.3f} F1={am['f1']:.3f} (n={am['support']})")
    
    return metrics


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)


def train(config):
    """Main training function"""
    
    # Set seed
    set_seed(config['reproducibility']['seed'])
    
    # Device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")
    
    # Get script directory for relative paths
    script_dir = Path(__file__).parent
    
    # Create output directories
    output_dir = script_dir / config['paths']['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = script_dir / Path(config['paths']['predictions_file']).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer (for text tokenization only, NOT for embeddings)
    print(f"\n[Loading Tokenizer] {config['model']['tokenizer_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name'])
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    
    # Update model vocab_size from tokenizer
    config['model']['vocab_size'] = len(tokenizer)
    
    # Load datasets (use absolute paths from script directory)
    print(f"\n[Loading Datasets]")
    train_dataset = AspectDetectionDataset(
        csv_file=str(script_dir / config['paths']['train_file']),
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    val_dataset = AspectDetectionDataset(
        csv_file=str(script_dir / config['paths']['validation_file']),
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    test_dataset = AspectDetectionDataset(
        csv_file=str(script_dir / config['paths']['test_file']),
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['per_device_train_batch_size'],
        shuffle=True,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['dataloader_pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['dataloader_pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        shuffle=False,
        num_workers=config['training']['dataloader_num_workers'],
        pin_memory=config['training']['dataloader_pin_memory']
    )
    
    # Create model
    print(f"\n[Creating Model]")
    model = BiLSTM_AspectDetection(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        num_aspects=config['model']['num_aspects'],
        lstm_hidden_size=config['model']['lstm_hidden_size'],
        lstm_num_layers=config['model']['lstm_num_layers'],
        lstm_dropout=config['model']['lstm_dropout'],
        spatial_dropout=config['model']['spatial_dropout'],
        conv_filters=config['model']['conv_filters'],
        conv_kernel_size=config['model']['conv_kernel_size'],
        dense_hidden_size=config['model']['dense_hidden_size'],
        dense_dropout=config['model']['dense_dropout'],
        padding_idx=config['model']['padding_idx']
    )
    model = model.to(device)
    
    # Loss function
    print(f"\n[Loss Function] BCEWithLogitsLoss")
    if config['loss']['use_pos_weight'] and config['loss']['pos_weight_auto']:
        pos_weight = train_dataset.get_pos_weight().to(device)
        print(f"  Using auto-calculated pos_weight:")
        for i, aspect in enumerate(config['valid_aspects']):
            print(f"    {aspect:<15} {pos_weight[i]:.3f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    num_epochs = config['training']['num_train_epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print(f"\n[Training]")
    print(f"  Epochs: {num_epochs}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    best_f1 = 0
    patience_counter = 0
    patience = config['training']['early_stopping_patience']
    
    training_history = {
        'train_loss': [],
        'val_f1_macro': [],
        'val_f1_micro': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, config['valid_aspects'], 
                              threshold=config['model']['threshold'])
        
        # Update history
        training_history['train_loss'].append(train_loss)
        training_history['val_f1_macro'].append(val_metrics['f1_macro'])
        training_history['val_f1_micro'].append(val_metrics['f1_micro'])
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            patience_counter = 0
            
            # Save model
            checkpoint_path = output_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Saved best model (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping triggered!")
            break
        
        # Step scheduler
        scheduler.step()
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Load best model for final evaluation
    print(f"\n[Loading Best Model]")
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print(f"Final Evaluation on Test Set")
    print(f"{'='*80}")
    
    test_metrics = evaluate(model, test_loader, device, config['valid_aspects'],
                           threshold=config['model']['threshold'])
    
    # Save test results
    test_results_path = output_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n[Training Complete]")
    print(f"  Best Val F1 (macro): {best_f1:.4f}")
    print(f"  Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Test F1 (micro): {test_metrics['f1_micro']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"\n  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train BiLSTM Aspect Detection Model')
    parser.add_argument('--config', type=str, default='config_bilstm_ad.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
