"""
Sequential Single-Task Learning for Vietnamese ABSA using BiLSTM
================================================================
Two-stage training approach:
    Stage 1: Aspect Detection (AD) - Binary classification for 11 aspects
    Stage 2: Sentiment Classification (SC) - 3-class for 11 aspects

Model: BiLSTM + CNN (trainable embeddings, NO pretrained models)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Stage 1: Aspect Detection
from model_bilstm_ad import BiLSTM_AspectDetection
from dataset_bilstm_ad import AspectDetectionDataset

# Stage 2: Sentiment Classification  
from model_bilstm_sc import BiLSTM_SentimentClassification
from dataset_bilstm_sc import SentimentClassificationDataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: str, stage_name: str) -> str:
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'{stage_name}_log_{timestamp}.txt')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


# =============================================================================
# STAGE 1: ASPECT DETECTION
# =============================================================================

def train_epoch_ad(model, dataloader, optimizer, scheduler, device, criterion, scaler):
    """Train one epoch for AD"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="[AD] Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
        
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_ad(model, dataloader, device, aspect_names):
    """Evaluate AD model"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[AD] Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Per-aspect metrics
    aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds[:, i], average='binary', zero_division=0
        )
        aspect_metrics[aspect] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    
    # Overall metrics (F1 Macro: average of per-aspect metrics)
    overall_acc = np.mean([m['accuracy'] for m in aspect_metrics.values()])
    overall_p = np.mean([m['precision'] for m in aspect_metrics.values()])
    overall_r = np.mean([m['recall'] for m in aspect_metrics.values()])
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    
    return {
        'overall_accuracy': overall_acc,
        'overall_precision': overall_p,
        'overall_recall': overall_r,
        'overall_f1': overall_f1,
        'per_aspect': aspect_metrics,
        'predictions': all_preds,
        'labels': all_labels
    }


def save_ad_confusion_matrix(metrics: dict, aspect_names: list, output_dir: str):
    """Save AD confusion matrix"""
    print("\n[AD] Generating confusion matrix...")
    
    preds = metrics['predictions']
    labels = metrics['labels']
    
    cm = confusion_matrix(labels.flatten(), preds.flatten())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Mentioned', 'Mentioned'],
               yticklabels=['Not Mentioned', 'Mentioned'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Aspect Detection - Overall Confusion Matrix (BiLSTM)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/confusion_matrix_overall.png")


def train_aspect_detection(config: dict, args: argparse.Namespace) -> str:
    """Train Stage 1: Aspect Detection"""
    print("\n" + "="*80)
    print("STAGE 1: ASPECT DETECTION (Binary Classification - BiLSTM)")
    print("="*80)
    
    output_dir = config['paths']['ad_output_dir']
    log_file = setup_logging(output_dir, 'aspect_detection')
    logging.info("Starting AD training (BiLSTM)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seed
    seed = config['reproducibility']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name'])
    
    # Update vocab size
    config['model']['vocab_size'] = tokenizer.vocab_size
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = AspectDetectionDataset(config['paths']['train_file'], tokenizer, config['model']['max_length'])
    val_dataset = AspectDetectionDataset(config['paths']['validation_file'], tokenizer, config['model']['max_length'])
    test_dataset = AspectDetectionDataset(config['paths']['test_file'], tokenizer, config['model']['max_length'])
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Dataloaders
    batch_size = config['training']['per_device_train_batch_size']
    eval_batch_size = config['training']['per_device_eval_batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # Model
    print("\nCreating BiLSTM AD model...")
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss function
    ad_config = config['two_stage']['aspect_detection']
    if ad_config.get('use_pos_weight') and ad_config.get('pos_weight_auto'):
        pos_weight = train_dataset.get_pos_weight().to(device)
        print(f"\n   Using auto pos_weight: {pos_weight[:3].tolist()}...")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # Optimizer & Scheduler
    num_epochs = ad_config.get('epochs', 30)
    learning_rate = config['training']['learning_rate']
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config['training']['weight_decay'])
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(config['training']['warmup_ratio'] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    print(f"\nTraining setup:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting AD Training")
    print("="*80)
    
    best_f1 = 0.0
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    patience_counter = 0
    early_stopping_patience = ad_config.get('early_stopping_patience', 5)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"[AD] Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch_ad(model, train_loader, optimizer, scheduler, device, criterion, scaler)
        print(f"\nTrain Loss: {train_loss:.4f}")
        logging.info(f"AD Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Validate
        print("\nValidating...")
        val_metrics = evaluate_ad(model, val_loader, device, train_dataset.aspects)
        print(f"   Accuracy: {val_metrics['overall_accuracy']*100:.2f}%")
        print(f"   F1 Score: {val_metrics['overall_f1']*100:.2f}%")
        
        logging.info(f"AD Val - Acc: {val_metrics['overall_accuracy']*100:.2f}%, F1: {val_metrics['overall_f1']*100:.2f}%")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_metrics['overall_accuracy'],
            'val_f1': val_metrics['overall_f1']
        })
        
        # Save best model
        if val_metrics['overall_f1'] > best_f1:
            best_f1 = val_metrics['overall_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"\nNew best F1: {best_f1*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False, encoding='utf-8-sig')
    
    # Test
    print("\n" + "="*80)
    print("[AD] Testing Best Model")
    print("="*80)
    
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate_ad(model, test_loader, device, train_dataset.aspects)
    print(f"\nTest Results:")
    print(f"   Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score: {test_metrics['overall_f1']*100:.2f}%")
    
    # Save results
    results = {
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect'],
        'training_completed': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save confusion matrix
    save_ad_confusion_matrix(test_metrics, train_dataset.aspects, output_dir)
    
    print(f"\n[AD] Training complete! Results saved to: {output_dir}")
    return output_dir


# =============================================================================
# STAGE 2: SENTIMENT CLASSIFICATION
# =============================================================================

def train_epoch_sc(model, dataloader, optimizer, scheduler, device, criterion, scaler):
    """Train one epoch for SC"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="[SC] Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss_mask = batch['loss_mask'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(input_ids, attention_mask)
            
            # Compute loss per aspect
            bsz, num_aspects, num_classes = logits.shape
            ce = F.cross_entropy(
                logits.view(bsz * num_aspects, num_classes),
                labels.view(bsz * num_aspects),
                reduction='none'
            )
            loss_per_aspect = ce.view(bsz, num_aspects)
            
            # Apply mask
            masked_loss = loss_per_aspect * loss_mask
            num_labeled = loss_mask.sum()
            loss = masked_loss.sum() / num_labeled if num_labeled > 0 else masked_loss.sum()
        
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_sc(model, dataloader, device, aspect_names):
    """Evaluate SC model"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[SC] Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Per-aspect metrics
    aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        aspect_preds = all_preds[:, i].numpy()
        aspect_labels = all_labels[:, i].numpy()
        
        acc = accuracy_score(aspect_labels, aspect_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='macro', zero_division=0
        )
        aspect_metrics[aspect] = {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}
    
    if aspect_metrics:
        overall_accuracy = np.mean([m['accuracy'] for m in aspect_metrics.values()])
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    overall_precision = np.mean([m['precision'] for m in aspect_metrics.values()])
    overall_recall = np.mean([m['recall'] for m in aspect_metrics.values()])
    else:
        overall_accuracy = 0.0
        overall_f1 = 0.0
        overall_precision = 0.0
        overall_recall = 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'per_aspect': aspect_metrics,
        'predictions': all_preds,
        'labels': all_labels
    }


def save_sc_confusion_matrices(metrics: dict, aspect_names: list, output_dir: str):
    """Save SC confusion matrices"""
    print("\n[SC] Generating confusion matrices...")
    
    preds = metrics['predictions'].numpy()
    labels = metrics['labels'].numpy()
    
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    
    # Overall confusion matrix
    cm_overall = confusion_matrix(labels.flatten(), preds.flatten(), labels=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues',
               xticklabels=sentiment_labels, yticklabels=sentiment_labels, ax=ax)
    ax.set_xlabel('Predicted Sentiment', fontsize=12)
    ax.set_ylabel('True Sentiment', fontsize=12)
    ax.set_title('Sentiment Classification - Overall (BiLSTM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/confusion_matrix_overall.png")
    
    # Per-aspect confusion matrices
    n_aspects = len(aspect_names)
    n_cols = 3
    n_rows = (n_aspects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_aspects > 1 else [axes]
    
    for i, aspect in enumerate(aspect_names):
        cm_aspect = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1, 2])
        sns.heatmap(cm_aspect, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sentiment_labels, yticklabels=sentiment_labels,
                   ax=axes[i], cbar=False)
        axes[i].set_title(f'{aspect}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    for i in range(n_aspects, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices_per_aspect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/confusion_matrices_per_aspect.png")


def save_sc_predictions(metrics: dict, aspect_names: list, output_dir: str):
    """Save SC predictions"""
    print("\n[SC] Saving predictions...")
    
    preds = metrics['predictions']
    labels = metrics['labels']
    
    pred_data = []
    for i in range(len(preds)):
        row = {'sample_id': i}
        for j, aspect in enumerate(aspect_names):
            row[f'{aspect}_pred'] = int(preds[i, j].item())
            row[f'{aspect}_true'] = int(labels[i, j].item())
            row[f'{aspect}_correct'] = int(preds[i, j].item() == labels[i, j].item())
        pred_data.append(row)
    
    df_preds = pd.DataFrame(pred_data)
    pred_file = os.path.join(output_dir, 'test_predictions_detailed.csv')
    df_preds.to_csv(pred_file, index=False, encoding='utf-8-sig')
    
    print(f"   Saved: {pred_file}")


def train_sentiment_classification(config: dict, args: argparse.Namespace) -> str:
    """Train Stage 2: Sentiment Classification"""
    print("\n" + "="*80)
    print("STAGE 2: SENTIMENT CLASSIFICATION (3-Class - BiLSTM)")
    print("="*80)
    
    output_dir = config['paths']['sc_output_dir']
    log_file = setup_logging(output_dir, 'sentiment_classification')
    logging.info("Starting SC training (BiLSTM)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seed
    seed = config['reproducibility']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer_name'])
    
    # Update vocab size
    config['model']['vocab_size'] = tokenizer.vocab_size
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = SentimentClassificationDataset(config['paths']['train_file'], tokenizer, config['model']['max_length'])
    val_dataset = SentimentClassificationDataset(config['paths']['validation_file'], tokenizer, config['model']['max_length'])
    test_dataset = SentimentClassificationDataset(config['paths']['test_file'], tokenizer, config['model']['max_length'])
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Dataloaders
    batch_size = config['training']['per_device_train_batch_size']
    eval_batch_size = config['training']['per_device_eval_batch_size']
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # Model
    print("\nCreating BiLSTM SC model...")
    model = BiLSTM_SentimentClassification(
        vocab_size=config['model']['vocab_size'],
        embedding_dim=config['model']['embedding_dim'],
        num_aspects=config['model']['num_aspects'],
        num_sentiments=config['model']['num_sentiments'],
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Loss function (standard CrossEntropy)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Optimizer & Scheduler
    sc_config = config['two_stage']['sentiment_classification']
    num_epochs = sc_config.get('epochs', 30)
    learning_rate = config['training']['learning_rate']
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config['training']['weight_decay'])
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(config['training']['warmup_ratio'] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    print(f"\nTraining setup:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting SC Training")
    print("="*80)
    
    best_acc = 0.0
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    patience_counter = 0
    early_stopping_patience = sc_config.get('early_stopping_patience', 7)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"[SC] Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train_epoch_sc(model, train_loader, optimizer, scheduler, device, criterion, scaler)
        print(f"\nTrain Loss: {train_loss:.4f}")
        logging.info(f"SC Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}")
        
        # Validate
        print("\nValidating...")
        val_metrics = evaluate_sc(model, val_loader, device, train_dataset.aspects)
        print(f"   Accuracy: {val_metrics['overall_accuracy']*100:.2f}%")
        print(f"   F1 Score: {val_metrics['overall_f1']*100:.2f}%")
        
        logging.info(f"SC Val - Acc: {val_metrics['overall_accuracy']*100:.2f}%, F1: {val_metrics['overall_f1']*100:.2f}%")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_metrics['overall_accuracy'],
            'val_f1': val_metrics['overall_f1']
        })
        
        # Save best model
        if val_metrics['overall_accuracy'] > best_acc:
            best_acc = val_metrics['overall_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"\nNew best Accuracy: {best_acc*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False, encoding='utf-8-sig')
    
    # Test
    print("\n" + "="*80)
    print("[SC] Testing Best Model")
    print("="*80)
    
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate_sc(model, test_loader, device, train_dataset.aspects)
    print(f"\nTest Results:")
    print(f"   Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score: {test_metrics['overall_f1']*100:.2f}%")
    
    # Save results
    results = {
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect'],
        'training_completed': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save predictions and confusion matrices
    save_sc_predictions(test_metrics, train_dataset.aspects, output_dir)
    save_sc_confusion_matrices(test_metrics, train_dataset.aspects, output_dir)
    
    print(f"\n[SC] Training complete! Results saved to: {output_dir}")
    return output_dir


# =============================================================================
# FINAL REPORT
# =============================================================================

def generate_final_report(ad_output_dir: str, sc_output_dir: str, final_results_dir: str, config: dict):
    """Generate consolidated final report"""
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)
    
    os.makedirs(final_results_dir, exist_ok=True)
    
    # Load results
    with open(os.path.join(ad_output_dir, 'test_results.json'), 'r', encoding='utf-8') as f:
        ad_results = json.load(f)
    
    with open(os.path.join(sc_output_dir, 'test_results.json'), 'r', encoding='utf-8') as f:
        sc_results = json.load(f)
    
    # Create report
    report_lines = [
        "="*80,
        "SEQUENTIAL SINGLE-TASK LEARNING FOR VIETNAMESE ABSA (BiLSTM + CNN)",
        "Two-Stage Training Results",
        "="*80,
        "",
        f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: BiLSTM + CNN (Trainable Embeddings)",
        "",
        "="*80,
        "STAGE 1: ASPECT DETECTION (Binary Classification)",
        "="*80,
        "",
        f"Test Accuracy:  {ad_results['test_accuracy']*100:.2f}%",
        f"Test F1 Score:  {ad_results['test_f1']*100:.2f}%",
        f"Test Precision: {ad_results['test_precision']*100:.2f}%",
        f"Test Recall:    {ad_results['test_recall']*100:.2f}%",
        "",
        "Per-Aspect Results (AD):",
        "-"*80
    ]
    
    for aspect, metrics in ad_results['per_aspect'].items():
        report_lines.append(
            f"{aspect:<15} Accuracy: {metrics['accuracy']*100:>6.2f}%  "
            f"F1 Score: {metrics['f1']*100:>6.2f}%  "
            f"Precision: {metrics['precision']*100:>6.2f}%  "
            f"Recall: {metrics['recall']*100:>6.2f}%"
        )
    
    report_lines.extend([
        "",
        "="*80,
        "STAGE 2: SENTIMENT CLASSIFICATION (3-Class per Aspect)",
        "="*80,
        "",
        f"Test Accuracy:  {sc_results['test_accuracy']*100:.2f}%",
        f"Test F1 Score:  {sc_results['test_f1']*100:.2f}%",
        f"Test Precision: {sc_results['test_precision']*100:.2f}%",
        f"Test Recall:    {sc_results['test_recall']*100:.2f}%",
        "",
        "Per-Aspect Results (SC):",
        "-"*80
    ])
    
    for aspect, metrics in sc_results['per_aspect'].items():
        report_lines.append(
            f"{aspect:<15} Accuracy: {metrics['accuracy']*100:>6.2f}%  "
            f"F1 Score: {metrics['f1']*100:>6.2f}%  "
            f"Precision: {metrics['precision']*100:>6.2f}%  "
            f"Recall: {metrics['recall']*100:>6.2f}%"
        )
    
    report_lines.extend([
        "",
        "="*80,
        "OUTPUT FILES",
        "="*80,
        "",
        "Aspect Detection:",
        f"  - Model: {ad_output_dir}/best_model.pt",
        f"  - Training history: {ad_output_dir}/training_history.csv",
        f"  - Test results: {ad_output_dir}/test_results.json",
        f"  - Confusion matrix: {ad_output_dir}/confusion_matrix_overall.png",
        "",
        "Sentiment Classification:",
        f"  - Model: {sc_output_dir}/best_model.pt",
        f"  - Training history: {sc_output_dir}/training_history.csv",
        f"  - Test results: {sc_output_dir}/test_results.json",
        f"  - Predictions: {sc_output_dir}/test_predictions_detailed.csv",
        f"  - Confusion matrix (overall): {sc_output_dir}/confusion_matrix_overall.png",
        f"  - Confusion matrices (per-aspect): {sc_output_dir}/confusion_matrices_per_aspect.png",
        "",
        "="*80,
        "TRAINING COMPLETE",
        "="*80
    ])
    
    report_text = '\n'.join(report_lines)
    
    # Save report
    report_file = os.path.join(final_results_dir, 'final_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nFinal report saved to: {report_file}")
    
    return report_file


# =============================================================================
# MAIN
# =============================================================================

def main(args: argparse.Namespace):
    """Main function for two-stage BiLSTM training"""
    print("="*80)
    print("SEQUENTIAL SINGLE-TASK LEARNING FOR VIETNAMESE ABSA (BiLSTM)")
    print("="*80)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Stage 1: Aspect Detection
    ad_output_dir = train_aspect_detection(config, args)
    
    # Stage 2: Sentiment Classification
    sc_output_dir = train_sentiment_classification(config, args)
    
    # Generate final report
    final_results_dir = config['paths']['final_results_dir']
    report_file = generate_final_report(ad_output_dir, sc_output_dir, final_results_dir, config)
    
    print("\n" + "="*80)
    print("TWO-STAGE BiLSTM TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal report: {report_file}")
    print(f"\nAll results saved to:")
    print(f"  - AD: {ad_output_dir}")
    print(f"  - SC: {sc_output_dir}")
    print(f"  - Final: {final_results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage BiLSTM ABSA Training')
    parser.add_argument('--config', type=str, default='BILSTM-STL/config_bilstm_stl.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    main(args)
