"""
Sequential Single-Task Learning for Vietnamese ABSA
====================================================
Two-stage training approach:
    Stage 1: Aspect Detection (AD) - Binary classification for 11 aspects (including Others)
    Stage 2: Sentiment Classification (SC) - 3-class for 10 aspects (excluding Others)

After training, automatically runs error analysis and generates reports.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
import json
import logging
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Stage 1: Aspect Detection
from model_visobert_ad import AspectDetectionModel
from dataset_visobert_ad import AspectDetectionDataset
from binary_focal_loss import BinaryFocalLoss, calculate_binary_alpha_auto

# Stage 2: Sentiment Classification
from model_visobert_sc import MultiLabelViSoBERT
from dataset_visobert_sc import MultiLabelABSADataset
from focal_loss_multilabel import MultilabelFocalLoss, calculate_global_alpha


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir: str, stage_name: str) -> str:
    """Setup logging to file and console"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'{stage_name}_log_{timestamp}.txt')
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file


# =============================================================================
# STAGE 1: ASPECT DETECTION
# =============================================================================

def train_epoch_ad(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    focal_loss_fn: BinaryFocalLoss,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> float:
    """Train one epoch for Aspect Detection"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="[AD] Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        use_amp = scaler is not None
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = focal_loss_fn(logits, labels)
        
        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate_ad(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    aspect_names: list,
    focal_loss_fn: Optional[BinaryFocalLoss] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate Aspect Detection model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[AD] Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            # Calculate loss if focal_loss_fn is provided
            if focal_loss_fn is not None:
                loss = focal_loss_fn(logits, labels)
                total_loss += loss.item()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader) if focal_loss_fn is not None else None
    
    # Per-aspect metrics (AD stage includes all 11 aspects, including "Others")
    aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        acc = accuracy_score(all_labels[:, i], all_preds[:, i])
        p, r, f1, _ = precision_recall_fscore_support(
            all_labels[:, i], all_preds[:, i], average='binary', zero_division=0
        )
        
        aspect_metrics[aspect] = {
            'accuracy': acc,
            'precision': p,
            'recall': r,
            'f1': f1
        }
    
    # Overall metrics (F1 Macro: average of per-aspect metrics, including "Others")
    overall_acc = np.mean([m['accuracy'] for m in aspect_metrics.values()])
    overall_p = np.mean([m['precision'] for m in aspect_metrics.values()])
    overall_r = np.mean([m['recall'] for m in aspect_metrics.values()])
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    
    result = {
        'overall_accuracy': overall_acc,
        'overall_precision': overall_p,
        'overall_recall': overall_r,
        'overall_f1': overall_f1,
        'per_aspect': aspect_metrics,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    if avg_loss is not None:
        result['val_loss'] = avg_loss
    
    return result


def train_aspect_detection(config: dict, args: argparse.Namespace) -> str:
    """Train Stage 1: Aspect Detection"""
    print("\n" + "="*80)
    print("STAGE 1: ASPECT DETECTION (Binary Classification)")
    print("="*80)
    
    # Setup
    output_dir = config['paths']['ad_output_dir']
    log_file = setup_logging(output_dir, 'aspect_detection')
    logging.info("Starting Aspect Detection training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seed
    seed = config['reproducibility']['training_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = AspectDetectionDataset(
        config['paths']['train_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    val_dataset = AspectDetectionDataset(
        config['paths']['validation_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    test_dataset = AspectDetectionDataset(
        config['paths']['test_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Dataloaders
    batch_size = config['training'].get('per_device_train_batch_size', 16)
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # Model
    print("\nCreating AD model...")
    model = AspectDetectionModel(
        model_name=config['model']['name'],
        num_aspects=11,  # AD stage includes "Others"
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Focal Loss
    print("\nSetting up Binary Focal Loss...")
    ad_config = config['two_stage']['aspect_detection']
    
    if ad_config.get('focal_alpha') == 'auto':
        alpha = calculate_binary_alpha_auto(
            config['paths']['train_file'],
            train_dataset.aspects,
            method='inverse_freq'
        )
    else:
        alpha = ad_config.get('focal_alpha', [1.0, 1.0])
    
    focal_loss = BinaryFocalLoss(
        alpha=alpha,
        gamma=ad_config.get('focal_gamma', 2.0),
        reduction='mean'
    )
    focal_loss = focal_loss.to(device)
    
    # Optimizer & Scheduler
    num_epochs = ad_config.get('epochs', 3)
    learning_rate = config['training'].get('learning_rate', 2e-5)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    warmup_ratio = config['training'].get('warmup_ratio', 0.06)
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTraining setup:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Total steps: {total_steps}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting AD Training")
    print("="*80)
    
    best_f1 = 0.0
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"[AD] Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        logging.info(f"AD Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch_ad(model, train_loader, optimizer, scheduler, device, focal_loss, scaler)
        print(f"\nTrain Loss: {train_loss:.4f}")
        logging.info(f"AD Train Loss: {train_loss:.4f}")
        
        # Validate
        print("\nValidating...")
        val_metrics = evaluate_ad(model, val_loader, device, train_dataset.aspects, focal_loss)
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_metrics.get('val_loss', 'N/A'):.4f}" if val_metrics.get('val_loss') is not None else "   Val Loss: N/A")
        print(f"   Accuracy: {val_metrics['overall_accuracy']*100:.2f}%")
        print(f"   F1 Score: {val_metrics['overall_f1']*100:.2f}%")
        print(f"   Precision: {val_metrics['overall_precision']*100:.2f}%")
        print(f"   Recall: {val_metrics['overall_recall']*100:.2f}%")
        
        val_loss = val_metrics.get('val_loss', None)
        logging.info(f"AD Val - Loss: {val_loss:.4f}, " if val_loss is not None else "AD Val - Loss: N/A, " +
                    f"Acc: {val_metrics['overall_accuracy']*100:.2f}%, "
                    f"F1: {val_metrics['overall_f1']*100:.2f}%")
        
        # Save history
        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_metrics['overall_accuracy'],
            'val_f1': val_metrics['overall_f1'],
            'val_precision': val_metrics['overall_precision'],
            'val_recall': val_metrics['overall_recall']
        }
        if val_loss is not None:
            history_entry['val_loss'] = val_loss
        history.append(history_entry)
        
        # Save best model
        if val_metrics['overall_f1'] > best_f1:
            best_f1 = val_metrics['overall_f1']
            best_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, best_path)
            print(f"\nNew best F1: {best_f1*100:.2f}%")
            logging.info(f"New best AD F1: {best_f1*100:.2f}%")
    
    # Save history
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False, encoding='utf-8-sig')
    
    # Test with best model
    print("\n" + "="*80)
    print("[AD] Testing Best Model")
    print("="*80)
    
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate_ad(model, test_loader, device, train_dataset.aspects, focal_loss)
    print(f"\nTest Results:")
    print(f"   Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score: {test_metrics['overall_f1']*100:.2f}%")
    print(f"   Precision: {test_metrics['overall_precision']*100:.2f}%")
    print(f"   Recall: {test_metrics['overall_recall']*100:.2f}%")
    
    logging.info(f"AD Test - Acc: {test_metrics['overall_accuracy']*100:.2f}%, "
                f"F1: {test_metrics['overall_f1']*100:.2f}%")
    
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
    
    # Save predictions
    pred_df = pd.DataFrame(test_metrics['predictions'], columns=train_dataset.aspects)
    pred_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False, encoding='utf-8-sig')
    
    # Generate confusion matrix
    save_ad_confusion_matrix(test_metrics, train_dataset.aspects, output_dir)
    
    print(f"\n[AD] Training complete! Results saved to: {output_dir}")
    logging.info("[AD] Training completed successfully")
    
    return output_dir


def save_ad_confusion_matrix(metrics: dict, aspect_names: list, output_dir: str):
    """Save confusion matrices for AD"""
    print("\n[AD] Generating confusion matrices...")
    
    preds = metrics['predictions']
    labels = metrics['labels']
    
    # Overall confusion matrix
    cm_overall = confusion_matrix(labels.flatten(), preds.flatten())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Mentioned', 'Mentioned'],
               yticklabels=['Not Mentioned', 'Mentioned'],
               ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Aspect Detection - Overall Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/confusion_matrix_overall.png")


# =============================================================================
# STAGE 2: SENTIMENT CLASSIFICATION
# =============================================================================

def train_epoch_sc(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    focal_loss_fn: Optional[MultilabelFocalLoss],
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> float:
    """Train one epoch for Sentiment Classification"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="[SC] Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss_mask = batch['loss_mask'].to(device)
        
        optimizer.zero_grad()
        use_amp = scaler is not None
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            
            if focal_loss_fn is not None:
                loss_per_aspect = focal_loss_fn(logits, labels)
            else:
                bsz, num_aspects, num_classes = logits.shape
                ce = F.cross_entropy(
                    logits.view(bsz * num_aspects, num_classes),
                    labels.view(bsz * num_aspects),
                    reduction='none'
                )
                loss_per_aspect = ce.view(bsz, num_aspects)
            
            masked_loss = loss_per_aspect * loss_mask
            num_labeled = loss_mask.sum()
            
            if num_labeled > 0:
                loss = masked_loss.sum() / num_labeled
            else:
                loss = masked_loss.sum()
        
        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate_sc(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    aspect_names: list,
    focal_loss_fn: Optional[MultilabelFocalLoss] = None,
    raw_data_file: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate Sentiment Classification model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[SC] Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            # Calculate loss if focal_loss_fn is provided or if we need to track loss
            if focal_loss_fn is not None:
                loss_per_aspect = focal_loss_fn(logits, labels)
            else:
                bsz, num_aspects, num_classes = logits.shape
                ce = F.cross_entropy(
                    logits.view(bsz * num_aspects, num_classes),
                    labels.view(bsz * num_aspects),
                    reduction='none'
                )
                loss_per_aspect = ce.view(bsz, num_aspects)
            
            masked_loss = loss_per_aspect * loss_mask
            num_labeled = loss_mask.sum()
            
            if num_labeled > 0:
                loss = masked_loss.sum() / num_labeled
            else:
                loss = masked_loss.sum()
            
            total_loss += loss.item()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    
    # Load raw data for NaN masking
    labeled_mask = None
    if raw_data_file and os.path.exists(raw_data_file):
        try:
            raw_df = pd.read_csv(raw_data_file, encoding='utf-8-sig')
            labeled_mask = torch.zeros_like(all_labels, dtype=torch.bool)
            for i, aspect in enumerate(aspect_names):
                if aspect in raw_df.columns:
                    labeled_mask[:, i] = torch.tensor(raw_df[aspect].notna().values)
        except:
            labeled_mask = None
    
    # Per-aspect metrics (excluding "Others")
    aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        # Skip "Others" aspect - excluded from training and evaluation
        if aspect == 'Others':
            continue
            
        if labeled_mask is not None:
            mask = labeled_mask[:, i]
            if mask.sum() == 0:
                continue
            aspect_preds = all_preds[:, i][mask].numpy()
            aspect_labels = all_labels[:, i][mask].numpy()
        else:
            aspect_preds = all_preds[:, i].numpy()
            aspect_labels = all_labels[:, i].numpy()
        
        acc = accuracy_score(aspect_labels, aspect_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='macro', zero_division=0
        )
        
        aspect_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate overall metrics (excluding "Others" aspect)
    if aspect_metrics:
        overall_accuracy = np.mean([m['accuracy'] for m in aspect_metrics.values()])
        overall_precision = np.mean([m['precision'] for m in aspect_metrics.values()])
        overall_recall = np.mean([m['recall'] for m in aspect_metrics.values()])
        overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    else:
        overall_accuracy = 0.0
        overall_precision = 0.0
        overall_recall = 0.0
        overall_f1 = 0.0
    
    return {
        'overall_accuracy': overall_accuracy,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'val_loss': avg_loss,
        'per_aspect': aspect_metrics,
        'predictions': all_preds,
        'labels': all_labels
    }


def train_sentiment_classification(config: dict, args: argparse.Namespace) -> str:
    """Train Stage 2: Sentiment Classification"""
    print("\n" + "="*80)
    print("STAGE 2: SENTIMENT CLASSIFICATION (3-Class per Aspect)")
    print("="*80)
    
    # Setup
    output_dir = config['paths']['sc_output_dir']
    log_file = setup_logging(output_dir, 'sentiment_classification')
    logging.info("Starting Sentiment Classification training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seed
    seed = config['reproducibility']['training_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Datasets
    print("\nLoading datasets...")
    # Use oversampled data for SC stage to improve minority sentiment performance
    train_file_sc = config['paths'].get('train_file_sc', config['paths']['train_file'])
    print(f"   Training file: {train_file_sc}")
    if train_file_sc != config['paths']['train_file']:
        print(f"   [INFO] Using oversampled data for SC stage to balance sentiments")
    
    train_dataset = MultiLabelABSADataset(
        train_file_sc,
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
    
    # Dataloaders
    batch_size = config['training'].get('per_device_train_batch_size', 16)
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 32)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # Model
    print("\nCreating SC model...")
    model = MultiLabelViSoBERT(
        model_name=config['model']['name'],
        num_aspects=10,
        num_sentiments=3,
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Focal Loss
    print("\nSetting up Focal Loss...")
    sc_config = config['two_stage']['sentiment_classification']
    sentiment_to_idx = config['sentiment_labels']
    
    if sc_config.get('focal_alpha') == 'auto':
        # IMPORTANT: Calculate alpha from the SAME data used for training
        # If using balanced data for training, use it for alpha calculation too!
        alpha = calculate_global_alpha(
            train_file_sc,  # Use the actual training file (balanced or original)
            train_dataset.aspects,
            sentiment_to_idx
        )
    else:
        alpha = sc_config.get('focal_alpha', [1.0, 1.0, 1.0])
    
    focal_loss = MultilabelFocalLoss(
        alpha=alpha,
        gamma=sc_config.get('focal_gamma', 2.0),
        num_aspects=10,
        reduction='none'
    )
    focal_loss = focal_loss.to(device)
    
    # Optimizer & Scheduler
    num_epochs = sc_config.get('epochs', 3)
    learning_rate = config['training'].get('learning_rate', 2e-5)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    warmup_ratio = config['training'].get('warmup_ratio', 0.06)
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nTraining setup:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Total steps: {total_steps}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting SC Training")
    print("="*80)
    
    best_f1 = 0.0
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"[SC] Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        logging.info(f"SC Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch_sc(model, train_loader, optimizer, scheduler, device, focal_loss, scaler)
        print(f"\nTrain Loss: {train_loss:.4f}")
        logging.info(f"SC Train Loss: {train_loss:.4f}")
        
        # Validate
        print("\nValidating...")
        val_metrics = evaluate_sc(model, val_loader, device, train_dataset.aspects,
                                 focal_loss,
                                 raw_data_file=config['paths']['validation_file'])
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"   Accuracy: {val_metrics['overall_accuracy']*100:.2f}%")
        print(f"   F1 Score: {val_metrics['overall_f1']*100:.2f}%")
        print(f"   Precision: {val_metrics['overall_precision']*100:.2f}%")
        print(f"   Recall: {val_metrics['overall_recall']*100:.2f}%")
        
        logging.info(f"SC Val - Loss: {val_metrics['val_loss']:.4f}, "
                    f"Acc: {val_metrics['overall_accuracy']*100:.2f}%, "
                    f"F1: {val_metrics['overall_f1']*100:.2f}%")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_accuracy': val_metrics['overall_accuracy'],
            'val_f1': val_metrics['overall_f1'],
            'val_precision': val_metrics['overall_precision'],
            'val_recall': val_metrics['overall_recall']
        })
        
        # Save best model
        if val_metrics['overall_f1'] > best_f1:
            best_f1 = val_metrics['overall_f1']
            best_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, best_path)
            print(f"\nNew best F1: {best_f1*100:.2f}%")
            logging.info(f"New best SC F1: {best_f1*100:.2f}%")
    
    # Save history
    df_history = pd.DataFrame(history)
    df_history.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False, encoding='utf-8-sig')
    
    # Test with best model
    print("\n" + "="*80)
    print("[SC] Testing Best Model")
    print("="*80)
    
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate_sc(
        model,
        test_loader,
        device,
        train_dataset.aspects,
        focal_loss,
        raw_data_file=config['paths']['test_file']
    )
    print(f"\nTest Results:")
    print(f"   Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score: {test_metrics['overall_f1']*100:.2f}%")
    print(f"   Precision: {test_metrics['overall_precision']*100:.2f}%")
    print(f"   Recall: {test_metrics['overall_recall']*100:.2f}%")
    
    logging.info(f"SC Test - Acc: {test_metrics['overall_accuracy']*100:.2f}%, "
                f"F1: {test_metrics['overall_f1']*100:.2f}%")
    
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
    
    # Save detailed predictions
    save_sc_predictions(test_metrics, train_dataset.aspects, test_dataset, output_dir)
    
    # Save confusion matrices
    save_sc_confusion_matrix(test_metrics, train_dataset.aspects, output_dir)

    # Also generate confusion matrices on validation and training sets for analysis
    val_metrics = evaluate_sc(
        model,
        val_loader,
        device,
        train_dataset.aspects,
        focal_loss,
        raw_data_file=config['paths']['validation_file']
    )
    save_sc_confusion_matrix(val_metrics, train_dataset.aspects, output_dir, prefix='validation')

    train_metrics = evaluate_sc(
        model,
        train_loader,
        device,
        train_dataset.aspects,
        focal_loss,
        raw_data_file=config['paths']['train_file']
    )
    save_sc_confusion_matrix(train_metrics, train_dataset.aspects, output_dir, prefix='train')
    
    print(f"\n[SC] Training complete! Results saved to: {output_dir}")
    logging.info("[SC] Training completed successfully")
    
    return output_dir


def save_sc_predictions(metrics: dict, aspect_names: list, dataset, output_dir: str):
    """Save detailed SC predictions
    
    IMPORTANT: Save true labels from RAW data (with NaN for unlabeled),
    NOT from tensor (which has placeholder 0 for unlabeled aspects)
    """
    print("\n[SC] Saving predictions...")
    
    preds = metrics['predictions']
    labels = metrics['labels']
    
    # Load raw test data to get TRUE labels (with NaN for unlabeled)
    raw_df = dataset.df  # Access raw DataFrame from dataset
    
    # Sentiment mapping (inverse)
    sentiment_map_inv = {0: 'Positive', 1: 'Negative', 2: 'Neutral'}
    
    # Save predictions
    pred_data = []
    for i in range(len(preds)):
        row = {'sample_id': i}
        
        # Get raw row from DataFrame
        raw_row = raw_df.iloc[i]
        
        for j, aspect in enumerate(aspect_names):
            # Prediction: Always save as int (0/1/2)
            row[f'{aspect}_pred'] = int(preds[i, j].item())
            
            # True label: Get from RAW data (may be NaN for unlabeled aspects)
            true_sentiment = raw_row[aspect]
            
            if pd.isna(true_sentiment):
                # Unlabeled aspect: Save as NaN (NOT 0!)
                row[f'{aspect}_true'] = np.nan
                row[f'{aspect}_correct'] = np.nan  # Can't evaluate unlabeled
            else:
                # Labeled aspect: Convert sentiment string to ID
                sentiment_str = str(true_sentiment).strip()
                if sentiment_str == 'Positive':
                    row[f'{aspect}_true'] = 0
                elif sentiment_str == 'Negative':
                    row[f'{aspect}_true'] = 1
                elif sentiment_str == 'Neutral':
                    row[f'{aspect}_true'] = 2
                else:
                    # Invalid label: Treat as unlabeled
                    row[f'{aspect}_true'] = np.nan
                    row[f'{aspect}_correct'] = np.nan
                    continue
                
                # Check if correct
                row[f'{aspect}_correct'] = int(preds[i, j].item() == row[f'{aspect}_true'])
        
        pred_data.append(row)
    
    df_preds = pd.DataFrame(pred_data)
    pred_file = os.path.join(output_dir, 'test_predictions_detailed.csv')
    df_preds.to_csv(pred_file, index=False, encoding='utf-8-sig')
    
    print(f"   Saved: {pred_file}")
    print(f"   Note: Unlabeled aspects saved as NaN in '{aspect}_true' columns")


def save_sc_confusion_matrix(metrics: dict, aspect_names: list, output_dir: str, prefix: str = ''):
    """Save confusion matrices for SC (Sentiment Classification)
    
    Note: 'Others' aspect is excluded as it does not have sentiment labels
    """
    if prefix:
        print(f"\n[SC] Generating confusion matrices ({prefix})...")
    else:
        print("\n[SC] Generating confusion matrices...")
    
    preds = metrics['predictions'].numpy()
    labels = metrics['labels'].numpy()
    
    # Filter out "Others" aspect for confusion matrix
    # (Others has no sentiment labels, always NaN)
    aspect_names_filtered = [a for a in aspect_names if a != 'Others']
    others_idx = aspect_names.index('Others') if 'Others' in aspect_names else None
    
    if others_idx is not None:
        # Remove "Others" column from predictions and labels
        mask = np.ones(len(aspect_names), dtype=bool)
        mask[others_idx] = False
        preds = preds[:, mask]
        labels = labels[:, mask]
        aspect_names = aspect_names_filtered
    
    # Overall confusion matrix (all aspects combined)
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    cm_overall = confusion_matrix(labels.flatten(), preds.flatten(), labels=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_overall, annot=True, fmt='d', cmap='Blues',
               xticklabels=sentiment_labels,
               yticklabels=sentiment_labels,
               ax=ax)
    ax.set_xlabel('Predicted Sentiment', fontsize=12)
    ax.set_ylabel('True Sentiment', fontsize=12)
    title = 'Sentiment Classification - Overall Confusion Matrix'
    if prefix:
        title += f' ({prefix.capitalize()})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = 'confusion_matrix_overall.png'
    if prefix:
        filename = f'confusion_matrix_overall_{prefix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/{filename}")
    
    # Per-aspect confusion matrices (grid view)
    n_aspects = len(aspect_names)
    n_cols = 3
    n_rows = (n_aspects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_aspects > 1 else [axes]
    
    for i, aspect in enumerate(aspect_names):
        cm_aspect = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1, 2])
        
        sns.heatmap(cm_aspect, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sentiment_labels,
                   yticklabels=sentiment_labels,
                   ax=axes[i], cbar=False)
        axes[i].set_title(f'{aspect}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    # Hide unused subplots
    for i in range(n_aspects, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    filename = 'confusion_matrices_per_aspect.png'
    if prefix:
        filename = f'confusion_matrices_per_aspect_{prefix}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/{filename}")


# =============================================================================
# MAIN: TWO-STAGE TRAINING
# =============================================================================

def run_error_analysis(sc_output_dir: str, config: dict):
    """Run error analysis automatically"""
    print("\n" + "="*80)
    print("RUNNING ERROR ANALYSIS")
    print("="*80)
    
    try:
        # Import error analysis
        import sys
        sys.path.append('multi_label/test')
        from error_analysis import ErrorAnalyzer
        
        # Run analysis
        analyzer = ErrorAnalyzer(
            test_file=config['paths']['test_file'],
            predictions_file=os.path.join(sc_output_dir, 'test_predictions_detailed.csv')
        )
        analyzer.run_full_analysis()
        
        print("\n[OK] Error analysis completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Error analysis failed: {e}")
        logging.error(f"Error analysis failed: {e}")


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
        "SEQUENTIAL SINGLE-TASK LEARNING FOR VIETNAMESE ABSA",
        "Two-Stage Training Results",
        "="*80,
        "",
        f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {config['model']['name']}",
        "",
        "="*80,
        "STAGE 1: ASPECT DETECTION (Binary Classification)",
        "="*80,
        "",
        "Note: AD stage includes all 11 aspects (including 'Others')",
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
        "Note: SC stage excludes 'Others' aspect (10 aspects only, no sentiment labels for Others)",
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
        "Error Analysis:",
        f"  - Report: multi_label/error_analysis_results/error_analysis_report.txt",
        f"  - Confusion matrix: multi_label/error_analysis_results/confusion_matrix.png",
        f"  - All errors: multi_label/error_analysis_results/all_errors_detailed.csv",
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


def main(args: argparse.Namespace):
    """Main function for two-stage training"""
    print("="*80)
    print("SEQUENTIAL SINGLE-TASK LEARNING FOR VIETNAMESE ABSA")
    print("="*80)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Stage 1: Aspect Detection
    ad_output_dir = train_aspect_detection(config, args)
    
    # Stage 2: Sentiment Classification
    sc_output_dir = train_sentiment_classification(config, args)
    
    # Run error analysis
    if config['two_stage'].get('run_error_analysis', True):
        run_error_analysis(sc_output_dir, config)
    
    # Generate final report
    final_results_dir = config['paths']['final_results_dir']
    report_file = generate_final_report(ad_output_dir, sc_output_dir, final_results_dir, config)
    
    print("\n" + "="*80)
    print("TWO-STAGE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal report: {report_file}")
    print(f"\nAll results saved to:")
    print(f"  - AD: {ad_output_dir}")
    print(f"  - SC: {sc_output_dir}")
    print(f"  - Final: {final_results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Stage ABSA Training')
    parser.add_argument('--config', type=str, default='multi_label/config_multi.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    main(args)
