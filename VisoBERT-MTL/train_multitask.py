"""
Training Script for Dual Task Learning (Aspect Detection + Sentiment Classification)
Train ViSoBERT for aspect detection (binary) and sentiment classification (3-class)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from model_multitask import DualTaskViSoBERT
from dataset_multitask import MultiLabelABSADataset
from focal_loss_multitask import DualTaskFocalLoss, calculate_global_alpha, calculate_aspect_detection_alpha

def get_script_directory():
    """Get the directory where this script is located (dual-task-learning/)"""
    return Path(__file__).parent.absolute()


def load_config(config_path='config.yaml'):
    """
    Load configuration and resolve paths relative to script directory
    
    Args:
        config_path: Path to config file (relative or absolute)
    
    Returns:
        Config dict with resolved paths
    """
    script_dir = get_script_directory()
    
    # Resolve config path
    if not os.path.isabs(config_path):
        config_path = str(script_dir / config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Resolve all paths relative to script directory
    if 'paths' in config:
        paths = config['paths']
        for key in ['train_file', 'validation_file', 'test_file', 'output_dir', 
                    'evaluation_report', 'predictions_file', 'data_dir']:
            if key in paths:
                path_value = paths[key]
                if not os.path.isabs(path_value):
                    # Resolve relative to script directory
                    paths[key] = str((script_dir / path_value).resolve())
    
    return config

def train_epoch(model, dataloader, optimizer, scheduler, device, dual_task_loss, max_grad_norm=1.0):
    """
    Train for one epoch with dual task learning
    
    Args:
        model: DualTaskViSoBERT model
        dataloader: DataLoader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (CPU/GPU)
        dual_task_loss: DualTaskFocalLoss combining both tasks
        max_grad_norm: Maximum gradient norm for clipping
    """
    model.train()
    total_loss = 0
    total_loss_ad = 0
    total_loss_sc = 0
    total_labeled_aspects = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        aspect_labels = batch['aspect_detection_labels'].to(device)  # Binary: [batch_size, num_aspects]
        sentiment_labels = batch['sentiment_labels'].to(device)  # 3-class: [batch_size, num_aspects]
        aspect_mask = batch['aspect_mask'].to(device)  # [batch_size, num_aspects]
        
        optimizer.zero_grad()
        
        # Forward pass: get both outputs
        aspect_logits, sentiment_logits = model(input_ids, attention_mask)
        
        # Compute combined dual-task loss (0.3 * AD + 0.7 * SC)
        total_batch_loss = dual_task_loss(
            aspect_logits, sentiment_logits,
            aspect_labels, sentiment_labels, aspect_mask
        )
        
        # Get individual loss components for monitoring
        loss_components = dual_task_loss.get_loss_components(
            aspect_logits, sentiment_logits,
            aspect_labels, sentiment_labels, aspect_mask
        )
        
        # Backward
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += total_batch_loss.item()
        total_loss_ad += loss_components['ad'].item()
        total_loss_sc += loss_components['sc'].item()
        total_labeled_aspects += aspect_mask.sum().item()
        
        progress_bar.set_postfix({
            'loss': f'{total_batch_loss.item():.4f}',
            'AD': f'{loss_components["ad"].item():.4f}',
            'SC': f'{loss_components["sc"].item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_loss_ad = total_loss_ad / len(dataloader)
    avg_loss_sc = total_loss_sc / len(dataloader)
    avg_labeled_per_batch = total_labeled_aspects / len(dataloader)
    
    print(f"\n   Avg labeled aspects per batch: {avg_labeled_per_batch:.1f} / 11")
    print(f"   Avg AD loss: {avg_loss_ad:.4f}, Avg SC loss: {avg_loss_sc:.4f}")
    
    return avg_loss, avg_loss_ad, avg_loss_sc

def evaluate(model, dataloader, device, aspect_names, raw_data_file=None, return_predictions=False):
    """
    Evaluate dual task model (Aspect Detection + Sentiment Classification)
    
    Args:
        model: DualTaskViSoBERT model to evaluate
        dataloader: DataLoader with test data
        device: Device (CPU/GPU)
        aspect_names: List of aspect names
        raw_data_file: Path to raw CSV to check for NaN labels (optional)
        return_predictions: If True, return predictions along with metrics
    
    Returns:
        Dictionary with metrics for both tasks
    """
    model.eval()
    
    all_aspect_preds = []
    all_aspect_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_aspect_masks = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            aspect_labels = batch['aspect_detection_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)
            
            # Predict both tasks
            aspect_logits, sentiment_logits = model(input_ids, attention_mask)
            
            # Aspect Detection: binary prediction (sigmoid + threshold)
            aspect_probs = torch.sigmoid(aspect_logits)
            aspect_preds = (aspect_probs > 0.5).long()
            
            # Sentiment Classification: multi-class prediction (argmax)
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)
            
            all_aspect_preds.append(aspect_preds.cpu())
            all_aspect_labels.append(aspect_labels.cpu())
            all_sentiment_preds.append(sentiment_preds.cpu())
            all_sentiment_labels.append(sentiment_labels.cpu())
            all_aspect_masks.append(aspect_mask.cpu())
            
            # Store texts if needed
            if return_predictions and 'text' in batch:
                all_texts.extend(batch['text'])
    
    # Concatenate all predictions
    all_aspect_preds = torch.cat(all_aspect_preds, dim=0)  # [num_samples, num_aspects]
    all_aspect_labels = torch.cat(all_aspect_labels, dim=0)
    all_sentiment_preds = torch.cat(all_sentiment_preds, dim=0)
    all_sentiment_labels = torch.cat(all_sentiment_labels, dim=0)
    all_aspect_masks = torch.cat(all_aspect_masks, dim=0)
    
    # Convert masks to boolean
    valid_mask = all_aspect_masks.bool()
    
    n_labeled = valid_mask.sum().item()
    n_total = valid_mask.numel()
    print(f"\nEvaluation Coverage:")
    print(f"   Labeled aspects: {n_labeled:,} ({n_labeled/n_total*100:.1f}%)")
    print(f"   Unlabeled (NaN): {n_total-n_labeled:,} ({(n_total-n_labeled)/n_total*100:.1f}%)")
    
    # Task 1: Aspect Detection Metrics (Binary Classification)
    # CRITICAL FIX: Evaluate on ALL aspects, not just labeled ones
    # - Labeled aspects (valid_mask=True) → label = 1 (present)
    # - Unlabeled aspects (valid_mask=False) → label = 0 (absent)
    # This gives us true binary classification metrics
    aspect_detection_metrics = {}
    
    # Overall accuracy: compare ALL predictions vs ALL labels (all aspects)
    overall_ad_acc = (all_aspect_preds == all_aspect_labels).float().mean().item()
    
    for i, aspect in enumerate(aspect_names):
        # Evaluate on ALL samples for this aspect (both labeled and unlabeled)
        aspect_ad_preds = all_aspect_preds[:, i].numpy()
        aspect_ad_labels = all_aspect_labels[:, i].numpy()
        n_samples = len(aspect_ad_preds)
        
        # Count labeled vs unlabeled for this aspect
        labeled_samples = valid_mask[:, i].sum().item()
        unlabeled_samples = n_samples - labeled_samples
        
        # Binary classification metrics on ALL samples
        ad_acc = accuracy_score(aspect_ad_labels, aspect_ad_preds)
        ad_precision, ad_recall, ad_f1, _ = precision_recall_fscore_support(
            aspect_ad_labels, aspect_ad_preds, average='binary', zero_division=0
        )
        
        aspect_detection_metrics[aspect] = {
            'accuracy': ad_acc,
            'precision': ad_precision,
            'recall': ad_recall,
            'f1': ad_f1,
            'n_samples': n_samples,
            'n_labeled': labeled_samples,
            'n_unlabeled': unlabeled_samples
        }
    
    # Task 2: Sentiment Classification Metrics (Independent of predicted AD)
    # Evaluate sentiment ONLY on aspects that are labeled as present in ground truth
    sentiment_metrics = {}
    valid_sentiment_mask = valid_mask & (all_aspect_labels == 1)
    
    overall_sc_acc = (all_sentiment_preds[valid_sentiment_mask] == all_sentiment_labels[valid_sentiment_mask]).float().mean().item()
    
    for i, aspect in enumerate(aspect_names):
        mask = valid_sentiment_mask[:, i]
        if mask.sum() == 0:
            continue
        
        sentiment_preds = all_sentiment_preds[:, i][mask].numpy()
        sentiment_labels = all_sentiment_labels[:, i][mask].numpy()
        n_samples = mask.sum().item()
        
        # Multi-class classification metrics
        sc_acc = accuracy_score(sentiment_labels, sentiment_preds)
        sc_precision, sc_recall, sc_f1, _ = precision_recall_fscore_support(
            sentiment_labels, sentiment_preds, average='weighted', zero_division=0
        )
        
        sentiment_metrics[aspect] = {
            'accuracy': sc_acc,
            'precision': sc_precision,
            'recall': sc_recall,
            'f1': sc_f1,
            'n_samples': n_samples
        }
    
    # Overall metrics
    overall_ad_f1 = np.mean([m['f1'] for m in aspect_detection_metrics.values()]) if aspect_detection_metrics else 0.0
    overall_sc_f1 = np.mean([m['f1'] for m in sentiment_metrics.values()]) if sentiment_metrics else 0.0
    
    results = {
        'aspect_detection': {
            'overall_accuracy': overall_ad_acc,
            'overall_f1': overall_ad_f1,
            'per_aspect': aspect_detection_metrics
        },
        'sentiment_classification': {
            'overall_accuracy': overall_sc_acc,
            'overall_f1': overall_sc_f1,
            'per_aspect': sentiment_metrics
        },
        'n_labeled': n_labeled,
        'n_total': n_total
    }
    
    # Add predictions if requested
    if return_predictions:
        results['aspect_predictions'] = all_aspect_preds
        results['aspect_labels'] = all_aspect_labels
        results['sentiment_predictions'] = all_sentiment_preds
        results['sentiment_labels'] = all_sentiment_labels
        if all_texts:
            results['texts'] = all_texts
    
    return results

def print_metrics(metrics, epoch=None):
    """Pretty print dual task metrics"""
    if epoch is not None:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Results - Dual Task Learning")
        print(f"{'='*80}")
    
    # Show labeled vs total if available
    if metrics.get('n_labeled') is not None and metrics.get('n_total') is not None:
        n_labeled = metrics['n_labeled']
        n_total = metrics['n_total']
        print(f"\nCoverage: {n_labeled:,}/{n_total:,} labeled aspects ({n_labeled/n_total*100:.1f}%)")
    
    # Task 1: Aspect Detection
    print(f"\n{'='*80}")
    print("Task 1: Aspect Detection (Binary Classification)")
    print(f"{'='*80}")
    ad_metrics = metrics['aspect_detection']
    print(f"\nOverall Aspect Detection Metrics:")
    print(f"   Accuracy:  {ad_metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score:  {ad_metrics['overall_f1']*100:.2f}%")
    
    print(f"\nPer-Aspect Detection Metrics:")
    print(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Samples':<10}")
    print("-" * 75)
    for aspect, m in ad_metrics['per_aspect'].items():
        n_samples_str = f"({m.get('n_samples', 'N/A')})" if 'n_samples' in m else ""
        print(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  {m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%  {n_samples_str:<10}")
    
    # Task 2: Sentiment Classification
    print(f"\n{'='*80}")
    print("Task 2: Sentiment Classification (3-Class)")
    print(f"{'='*80}")
    sc_metrics = metrics['sentiment_classification']
    print(f"\nOverall Sentiment Classification Metrics:")
    print(f"   Accuracy:  {sc_metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score:  {sc_metrics['overall_f1']*100:.2f}%")
    
    print(f"\nPer-Aspect Sentiment Classification Metrics:")
    print(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Samples':<10}")
    print("-" * 75)
    for aspect, m in sc_metrics['per_aspect'].items():
        n_samples_str = f"({m.get('n_samples', 'N/A')})" if 'n_samples' in m else ""
        print(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  {m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%  {n_samples_str:<10}")

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
        print(f"Saved best model: {best_path}")
    
    return checkpoint_path

def setup_logging(output_dir):
    """Setup logging to file and console"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'training_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def save_training_history(history, output_dir):
    """Save training history to CSV and JSON"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(history)
    csv_path = os.path.join(output_dir, 'training_history.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # Save as JSON with full details
    json_path = os.path.join(output_dir, 'training_history.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    
    logging.info(f"Training history saved to: {csv_path} and {json_path}")
    return csv_path, json_path

def save_evaluation_results(metrics, all_preds, all_labels, aspect_names, output_dir, split='test'):
    """Save detailed evaluation results including predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions vs labels
    pred_data = []
    for i in range(len(all_preds)):
        row = {'sample_id': i}
        for j, aspect in enumerate(aspect_names):
            row[f'{aspect}_pred'] = int(all_preds[i, j].item())
            row[f'{aspect}_true'] = int(all_labels[i, j].item())
            row[f'{aspect}_correct'] = int(all_preds[i, j].item() == all_labels[i, j].item())
        pred_data.append(row)
    
    df_preds = pd.DataFrame(pred_data)
    pred_file = os.path.join(output_dir, f'{split}_predictions_detailed.csv')
    df_preds.to_csv(pred_file, index=False, encoding='utf-8-sig')
    
    # Save summary metrics
    summary_file = os.path.join(output_dir, f'{split}_evaluation_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results - {split.upper()} Set\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {metrics['overall_accuracy']*100:.2f}%\n")
        f.write(f"  F1 Score:  {metrics['overall_f1']*100:.2f}%\n")
        f.write(f"  Precision: {metrics['overall_precision']*100:.2f}%\n")
        f.write(f"  Recall:    {metrics['overall_recall']*100:.2f}%\n\n")
        
        if metrics.get('n_labeled') and metrics.get('n_total'):
            f.write(f"Coverage:\n")
            f.write(f"  Labeled: {metrics['n_labeled']:,}/{metrics['n_total']:,} ")
            f.write(f"({metrics['n_labeled']/metrics['n_total']*100:.1f}%)\n\n")
        
        f.write(f"Per-Aspect Metrics:\n")
        f.write(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}\n")
        f.write("-" * 65 + "\n")
        
        for aspect, m in metrics['per_aspect'].items():
            f.write(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  ")
            f.write(f"{m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%\n")
    
    logging.info(f"Evaluation results saved to: {pred_file} and {summary_file}")
    return pred_file, summary_file

def main(args):
    print("=" * 80)
    print("Dual Task Learning Training (Aspect Detection + Sentiment Classification)")
    print("=" * 80)
    
    # Get script directory
    script_dir = get_script_directory()
    print(f"\nScript directory: {script_dir}")
    
    # Load config (paths will be resolved relative to script directory)
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Display resolved paths
    print(f"\nResolved paths:")
    print(f"   Train file:      {config['paths']['train_file']}")
    print(f"   Validation file: {config['paths']['validation_file']}")
    print(f"   Test file:       {config['paths']['test_file']}")
    print(f"   Output dir:      {config['paths']['output_dir']}")
    
    # Verify data files exist
    for key in ['train_file', 'validation_file', 'test_file']:
        file_path = config['paths'][key]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        print(f"   ✓ {key}: {file_path}")
    
    # Setup logging
    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
    if not os.path.isabs(output_dir):
        output_dir = str((script_dir / output_dir).resolve())
    log_file = setup_logging(output_dir)
    logging.info(f"Training started at {datetime.now()}")
    logging.info(f"Log file: {log_file}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed from reproducibility config
    seed = config['reproducibility']['training_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load tokenizer
    model_name = config['model']['name']
    print(f"\nLoading tokenizer from: {model_name}")
    print(f"Model reference: https://huggingface.co/{model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load datasets
    print(f"\nLoading datasets...")
    train_file = config['paths']['train_file']
    val_file = config['paths']['validation_file']
    test_file = config['paths']['test_file']
    
    print(f"   Loading train from: {train_file}")
    train_dataset = MultiLabelABSADataset(
        train_file,
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    print(f"   Loading validation from: {val_file}")
    val_dataset = MultiLabelABSADataset(
        val_file,
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    print(f"   Loading test from: {test_file}")
    test_dataset = MultiLabelABSADataset(
        test_file,
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
    
    # Load model hyperparameters from config (needed early for loss function)
    model_config = config.get('model', {})
    num_aspects = model_config.get('num_aspects', 11)
    num_sentiments = model_config.get('num_sentiments', 3)
    hidden_size = model_config.get('hidden_size', 512)
    dropout = model_config.get('dropout', 0.3)
    
    # =====================================================================
    # SETUP DUAL TASK FOCAL LOSS
    # =====================================================================
    print(f"\n{'='*80}")
    print("Setting up Dual Task Focal Loss...")
    print(f"{'='*80}")
    
    # Read dual task config
    dual_task_config = config.get('dual_task', {})
    loss_weight_ad = dual_task_config.get('loss_weight_ad', 0.3)
    loss_weight_sc = dual_task_config.get('loss_weight_sc', 0.7)
    ad_alpha_config = dual_task_config.get('ad_alpha', [1.0, 1.0])
    sc_alpha_config = dual_task_config.get('sc_alpha', 'auto')
    
    # Read multi-task config (renamed from multi_label)
    multitask_config = config.get('multi_task', {})
    if not multitask_config:
        # Fallback to old multi_label config name for compatibility
        multitask_config = config.get('multi_label', {})
    
    use_focal_loss = multitask_config.get('use_focal_loss', True)
    focal_gamma = multitask_config.get('focal_gamma', 2.0)
    
    print(f"\nDual Task Loss Weights:")
    print(f"   Aspect Detection (AD): {loss_weight_ad}")
    print(f"   Sentiment Classification (SC): {loss_weight_sc}")
    
    if not use_focal_loss:
        print(f"\nWARNING: Focal Loss is DISABLED in config!")
        raise ValueError("Focal Loss is required for dual task learning")
    
    # Determine alpha weights from config
    sentiment_to_idx = config['sentiment_labels']
    train_file_path = config['paths']['train_file']
    
    # Aspect Detection alpha weights
    if ad_alpha_config == 'auto':
        print(f"\nCalculating aspect detection alpha weights from training data...")
        alpha_ad = calculate_aspect_detection_alpha(
            train_file_path,
            train_dataset.aspects
        )
    elif isinstance(ad_alpha_config, list) and len(ad_alpha_config) == 2:
        alpha_ad = ad_alpha_config
        print(f"\nUsing aspect detection alpha weights from config: {alpha_ad}")
    elif ad_alpha_config is None:
        print(f"\nUsing equal aspect detection alpha weights...")
        alpha_ad = [1.0, 1.0]
    else:
        print(f"\nWARNING: Invalid ad_alpha config, using AUTO")
        alpha_ad = calculate_aspect_detection_alpha(
            train_file_path,
            train_dataset.aspects
        )
    
    # Sentiment Classification alpha weights
    if sc_alpha_config == 'auto':
        print(f"\nCalculating sentiment alpha weights from training data...")
        alpha_sc = calculate_global_alpha(
            train_file_path,
            train_dataset.aspects,
            sentiment_to_idx
        )
    elif isinstance(sc_alpha_config, list) and len(sc_alpha_config) == 3:
        print(f"\nUsing user-defined sentiment alpha weights from config...")
        alpha_sc = sc_alpha_config
    elif sc_alpha_config is None:
        print(f"\nUsing equal sentiment alpha weights...")
        alpha_sc = [1.0, 1.0, 1.0]
    else:
        print(f"\nWARNING: Invalid sc_alpha config, using AUTO")
        alpha_sc = calculate_global_alpha(
            train_file_path,
            train_dataset.aspects,
            sentiment_to_idx
        )
    
    # Create combined dual-task loss function
    dual_task_loss = DualTaskFocalLoss(
        alpha_ad=alpha_ad,
        alpha_sc=alpha_sc,
        gamma=focal_gamma,
        weight_ad=loss_weight_ad,
        weight_sc=loss_weight_sc,
        num_aspects=num_aspects
    )
    dual_task_loss = dual_task_loss.to(device)
    
    print(f"\nTRAINING WILL SKIP NaN ASPECTS (masking enabled)")
    
    # Create model
    model_name = config['model']['name']
    print(f"\nCreating Dual Task Model...")
    print(f"   Using base model: {model_name}")
    print(f"   Model reference: https://huggingface.co/{model_name}")
    
    model = DualTaskViSoBERT(
        model_name=model_name,
        num_aspects=num_aspects,
        num_sentiments=num_sentiments,
        hidden_size=hidden_size,
        dropout=dropout
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Optimizer - load all hyperparameters from config
    training_config = config.get('training', {})
    learning_rate = training_config.get('learning_rate', 2e-5)
    weight_decay = training_config.get('weight_decay', 0.01)
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    # Use epochs from config unless explicitly overridden
    if args.epochs is not None:
        num_epochs = args.epochs
        print(f"\nWARNING: Using epochs from command line: {num_epochs} (overrides config: {config['training'].get('num_train_epochs')})")
    else:
        num_epochs = config['training'].get('num_train_epochs', 5)
        print(f"\nUsing epochs from config: {num_epochs}")
    
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
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Total steps: {total_steps}")
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}")
    logging.info("Starting training loop")
    
    best_f1 = 0.0
    aspect_names = train_dataset.aspects
    training_history = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        logging.info(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_loss_ad, train_loss_sc = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            dual_task_loss, max_grad_norm=max_grad_norm
        )
        print(f"\nTrain Loss: {train_loss:.4f} (AD: {train_loss_ad:.4f}, SC: {train_loss_sc:.4f})")
        logging.info(f"Train Loss: {train_loss:.4f} (AD: {train_loss_ad:.4f}, SC: {train_loss_sc:.4f})")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names, 
                               raw_data_file=config['paths']['validation_file'])
        print_metrics(val_metrics, epoch=epoch)
        
        # Log validation metrics
        ad_metrics = val_metrics['aspect_detection']
        sc_metrics = val_metrics['sentiment_classification']
        logging.info(f"Validation - AD F1: {ad_metrics['overall_f1']*100:.2f}%, "
                    f"SC F1: {sc_metrics['overall_f1']*100:.2f}%")
        
        # Record history
        epoch_history = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_loss_ad': train_loss_ad,
            'train_loss_sc': train_loss_sc,
            'val_ad_accuracy': ad_metrics['overall_accuracy'],
            'val_ad_f1': ad_metrics['overall_f1'],
            'val_sc_accuracy': sc_metrics['overall_accuracy'],
            'val_sc_f1': sc_metrics['overall_f1'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add per-aspect metrics to history
        for aspect, m in ad_metrics['per_aspect'].items():
            epoch_history[f'{aspect}_ad_f1'] = m['f1']
        for aspect, m in sc_metrics['per_aspect'].items():
            epoch_history[f'{aspect}_sc_f1'] = m['f1']
        
        training_history.append(epoch_history)
        
        # Save checkpoint (use combined F1 as best model metric)
        combined_f1 = (loss_weight_ad * ad_metrics['overall_f1'] + 
                       loss_weight_sc * sc_metrics['overall_f1'])
        is_best = combined_f1 > best_f1
        if is_best:
            best_f1 = combined_f1
            print(f"\nNew best combined F1: {best_f1*100:.2f}% (AD: {ad_metrics['overall_f1']*100:.2f}%, SC: {sc_metrics['overall_f1']*100:.2f}%)")
            logging.info(f"New best combined F1: {best_f1*100:.2f}%")
        
        # Use output_dir from config if not specified
        output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
        if not os.path.isabs(output_dir):
            output_dir = str((script_dir / output_dir).resolve())
        
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            output_dir, is_best=is_best
        )
        
        # Save training history after each epoch
        save_training_history(training_history, output_dir)
    
    # Test with best model
    print(f"\n{'='*80}")
    print("Testing Best Model")
    print(f"{'='*80}")
    
    # Use output_dir from config if not specified
    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
    if not os.path.isabs(output_dir):
        output_dir = str((script_dir / output_dir).resolve())
    
    # Load best checkpoint
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    
    # Load with strict=False to handle old checkpoints with pooler
    missing_keys, unexpected_keys = model.load_state_dict(best_checkpoint['model_state_dict'], strict=False)
    if unexpected_keys:
        print(f"WARNING: Ignored unexpected keys from old checkpoint: {len(unexpected_keys)} keys")
    if missing_keys:
        print(f"WARNING: Missing keys: {missing_keys}")
    
    test_metrics = evaluate(model, test_loader, device, aspect_names,
                           raw_data_file=config['paths']['test_file'],
                           return_predictions=True)
    print_metrics(test_metrics)
    
    # Log test metrics
    ad_metrics = test_metrics['aspect_detection']
    sc_metrics = test_metrics['sentiment_classification']
    logging.info(f"Test Results - AD F1: {ad_metrics['overall_f1']*100:.2f}%, "
                f"SC F1: {sc_metrics['overall_f1']*100:.2f}%")
    
    # Save final results JSON
    results = {
        'aspect_detection': {
            'test_accuracy': ad_metrics['overall_accuracy'],
            'test_f1': ad_metrics['overall_f1'],
            'per_aspect': ad_metrics['per_aspect']
        },
        'sentiment_classification': {
            'test_accuracy': sc_metrics['overall_accuracy'],
            'test_f1': sc_metrics['overall_f1'],
            'per_aspect': sc_metrics['per_aspect']
        },
        'training_completed': datetime.now().isoformat(),
        'config': {
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_name': config['model']['name'],
            'loss_weight_ad': loss_weight_ad,
            'loss_weight_sc': loss_weight_sc
        }
    }
    
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {results_file}")
    logging.info(f"Results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\nBest Model Performance:")
    print(f"   Aspect Detection - Test Accuracy: {ad_metrics['overall_accuracy']*100:.2f}%, F1: {ad_metrics['overall_f1']*100:.2f}%")
    print(f"   Sentiment Classification - Test Accuracy: {sc_metrics['overall_accuracy']*100:.2f}%, F1: {sc_metrics['overall_f1']*100:.2f}%")
    print(f"\nModel saved to: {output_dir}")
    print(f"\nTraining logs and results:")
    print(f"   - Training history: {output_dir}/training_history.csv")
    print(f"   - Test predictions: {output_dir}/test_predictions_detailed.csv")
    print(f"   - Evaluation summary: {output_dir}/test_evaluation_summary.txt")
    print(f"   - Training log: {log_file}")
    
    logging.info("Training completed successfully")
    ad_metrics = test_metrics['aspect_detection']
    sc_metrics = test_metrics['sentiment_classification']
    logging.info(f"Final Test Results - AD Accuracy: {ad_metrics['overall_accuracy']*100:.2f}%, AD F1: {ad_metrics['overall_f1']*100:.2f}%")
    logging.info(f"Final Test Results - SC Accuracy: {sc_metrics['overall_accuracy']*100:.2f}%, SC F1: {sc_metrics['overall_f1']*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dual Task Learning Model (Aspect Detection + Sentiment Classification)')
    parser.add_argument('--config', type=str, default='config_multi.yaml', help='Path to config file (default: config_multi.yaml)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config if specified)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config if specified)')
    
    args = parser.parse_args()
    main(args)
