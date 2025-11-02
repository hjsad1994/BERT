"""
Train Multi-Label ABSA WITHOUT Oversampling
Use advanced techniques: Focal Loss + Better Class Weights + Label Smoothing
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

from model_multilabel import MultiLabelViSoBERT
from dataset_multilabel import MultiLabelABSADataset

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    
    Focus on hard examples (misclassified)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # [num_classes] weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_smooth_class_weights(dataset, beta=0.999):
    """
    Calculate smooth class weights using effective number of samples
    
    Formula: E_n = (1 - beta^n) / (1 - beta)
    Weight = 1 / E_n
    
    beta=0.999: Less extreme than inverse frequency
    """
    weights = []
    
    for aspect_idx in range(len(dataset.aspects)):
        aspect_counts = []
        for sentiment_idx in range(3):  # pos, neg, neu
            count = 0
            for i in range(len(dataset.df)):
                if dataset.df.iloc[i][dataset.aspects[aspect_idx]] == ['Positive', 'Negative', 'Neutral'][sentiment_idx]:
                    count += 1
            aspect_counts.append(count)
        
        # Effective number of samples
        effective_nums = [(1 - beta ** n) / (1 - beta) if n > 0 else 1 for n in aspect_counts]
        aspect_weights = [1.0 / en for en in effective_nums]
        
        # Normalize
        total = sum(aspect_weights)
        aspect_weights = [w / total * 3 for w in aspect_weights]
        
        weights.append(aspect_weights)
    
    return torch.tensor(weights, dtype=torch.float32)

def multilabel_focal_loss(logits, labels, alpha=None, gamma=2.0, label_smoothing=0.1):
    """
    Multi-label Focal Loss with Label Smoothing
    
    Args:
        logits: [batch_size, num_aspects, num_sentiments]
        labels: [batch_size, num_aspects]
        alpha: [num_aspects, num_sentiments] class weights
        gamma: Focal loss focusing parameter
        label_smoothing: Smooth hard labels (0.1 = 10% smoothing)
    """
    batch_size, num_aspects, num_sentiments = logits.shape
    
    total_loss = 0
    for i in range(num_aspects):
        aspect_logits = logits[:, i, :]  # [batch_size, num_sentiments]
        aspect_labels = labels[:, i]      # [batch_size]
        
        # Focal loss
        if alpha is not None:
            aspect_alpha = alpha[i]  # [num_sentiments]
        else:
            aspect_alpha = None
        
        # Cross entropy with label smoothing
        log_probs = F.log_softmax(aspect_logits, dim=-1)
        
        # One-hot encode labels
        one_hot = F.one_hot(aspect_labels, num_classes=num_sentiments).float()
        
        # Apply label smoothing
        if label_smoothing > 0:
            one_hot = one_hot * (1 - label_smoothing) + label_smoothing / num_sentiments
        
        # Focal loss calculation
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** gamma
        
        # Apply class weights
        if aspect_alpha is not None:
            aspect_alpha = aspect_alpha.to(aspect_logits.device)
            weighted_loss = -(one_hot * log_probs * focal_weight * aspect_alpha.unsqueeze(0)).sum(dim=-1)
        else:
            weighted_loss = -(one_hot * log_probs * focal_weight).sum(dim=-1)
        
        total_loss += weighted_loss.mean()
    
    return total_loss / num_aspects

def train_epoch(model, dataloader, optimizer, scheduler, device, weights=None, use_focal=True, gamma=2.0):
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
        if use_focal:
            loss = multilabel_focal_loss(logits, labels, alpha=weights, gamma=gamma, label_smoothing=0.1)
        else:
            # Standard cross-entropy
            batch_size, num_aspects, num_sentiments = logits.shape
            total_loss_ce = 0
            for i in range(num_aspects):
                aspect_logits = logits[:, i, :]
                aspect_labels = labels[:, i]
                if weights is not None:
                    aspect_weights = weights[i]
                    loss_ce = F.cross_entropy(aspect_logits, aspect_labels, weight=aspect_weights)
                else:
                    loss_ce = F.cross_entropy(aspect_logits, aspect_labels)
                total_loss_ce += loss_ce
            loss = total_loss_ce / num_aspects
        
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
            
            logits = model(input_ids, attention_mask)
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
    """Save model checkpoint"""
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
    print("Multi-Label ABSA Training (NO Oversampling)")
    print("Using: Focal Loss + Smooth Weights + Label Smoothing")
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
    
    # Load datasets (ORIGINAL, not balanced!)
    print(f"\nLoading datasets...")
    train_dataset = MultiLabelABSADataset(
        'data/train_multilabel.csv',  # Original unbalanced data
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
    
    print(f"   Train: {len(train_dataset)} samples (ORIGINAL, not oversampled)")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    batch_size = config['training'].get('per_device_train_batch_size', 32)
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 64)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # Calculate smooth class weights
    print(f"\nCalculating smooth class weights (beta=0.999)...")
    weights = get_smooth_class_weights(train_dataset, beta=0.999).to(device)
    print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"   Much smoother than inverse frequency!")
    
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
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    print(f"\nTraining setup:")
    print(f"   Strategy: Focal Loss (gamma=2.0) + Label Smoothing (0.1)")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   NO Oversampling! Using original data only")
    
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
        
        # Train with Focal Loss
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, 
                                weights=weights, use_focal=True, gamma=2.0)
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names)
        print_metrics(val_metrics)
        
        # Save checkpoint
        is_best = val_metrics['overall_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['overall_f1']
            print(f"\nNew best F1: {best_f1*100:.2f}%")
        
        save_checkpoint(model, optimizer, epoch, val_metrics, args.output_dir, is_best=is_best)
    
    # Test with best model
    print(f"\n{'='*80}")
    print("Testing Best Model")
    print(f"{'='*80}")
    
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, aspect_names)
    print_metrics(test_metrics)
    
    # Save results
    import json
    results = {
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect'],
        'method': 'Focal Loss + Smooth Weights + Label Smoothing (NO Oversampling)'
    }
    
    results_file = os.path.join(args.output_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {results_file}")
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\nBest Model Performance (NO Oversampling):")
    print(f"   Test Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   Test F1:       {test_metrics['overall_f1']*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Label ABSA WITHOUT Oversampling')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='multilabel_model_no_oversample')
    
    args = parser.parse_args()
    main(args)
