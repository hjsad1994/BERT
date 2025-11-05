"""
Training Script for Multi-Label ABSA
Train ViSoBERT to predict all 13 aspects simultaneously
"""

import torch
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
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from model_multilabel import MultiLabelViSoBERT
from dataset_multilabel import MultiLabelABSADataset
from focal_loss_multilabel import MultilabelFocalLoss, calculate_global_alpha

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    focal_loss_fn: Optional[MultilabelFocalLoss],
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> float:
    """Train for one epoch - ONLY on labeled aspects (masked)"""
    model.train()
    total_loss = 0
    total_masked_aspects = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        loss_mask = batch['loss_mask'].to(device)  # NEW: Get mask
        
        optimizer.zero_grad()
        use_amp = scaler is not None
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Forward
            logits = model(input_ids, attention_mask)

            # Compute per-aspect loss: focal if provided, else CrossEntropy per aspect
            if focal_loss_fn is not None:
                # returns [batch_size, num_aspects]
                loss_per_aspect = focal_loss_fn(logits, labels)
            else:
                # logits: [B, A, C], labels: [B, A]
                bsz, num_aspects, num_classes = logits.shape
                ce = F.cross_entropy(
                    logits.view(bsz * num_aspects, num_classes),
                    labels.view(bsz * num_aspects),
                    reduction='none'
                )
                loss_per_aspect = ce.view(bsz, num_aspects)

        # Apply mask: ONLY train on labeled aspects (mask=1.0)
        # NaN aspects have mask=0.0, so their loss is zeroed out
        masked_loss = loss_per_aspect * loss_mask
        
        # Average over labeled aspects only
        num_labeled = loss_mask.sum()
        if num_labeled > 0:
            loss = masked_loss.sum() / num_labeled
        else:
            loss = masked_loss.sum()  # Fallback (shouldn't happen)
        
        # Backward
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
        total_masked_aspects += num_labeled.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    avg_labeled_per_batch = total_masked_aspects / len(dataloader)
    
    print(f"\n   Avg labeled aspects per batch: {avg_labeled_per_batch:.1f} / 11")
    
    return avg_loss

def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    aspect_names: list,
    raw_data_file: Optional[str] = None,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model - ONLY on labeled aspects (skip NaN)
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device (CPU/GPU)
        aspect_names: List of aspect names
        raw_data_file: Path to raw CSV to check for NaN labels (optional)
        return_predictions: If True, return predictions along with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_texts = []
    
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
            
            # Store texts if needed for detailed results
            if return_predictions and 'text' in batch:
                all_texts.extend(batch['text'])
    
    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)  # [num_samples, num_aspects]
    all_labels = torch.cat(all_labels, dim=0)
    
    # Load raw data to identify NaN labels (unlabeled aspects)
    labeled_mask = None
    if raw_data_file and os.path.exists(raw_data_file):
        try:
            raw_df = pd.read_csv(raw_data_file, encoding='utf-8-sig')
            # Create mask: True where label exists, False where NaN
            labeled_mask = torch.zeros_like(all_labels, dtype=torch.bool)
            for i, aspect in enumerate(aspect_names):
                if aspect in raw_df.columns:
                    # True for non-NaN (labeled) aspects
                    labeled_mask[:, i] = torch.tensor(raw_df[aspect].notna().values)
            
            n_labeled = labeled_mask.sum().item()
            n_total = labeled_mask.numel()
            print(f"\nEvaluation Coverage:")
            print(f"   Labeled aspects: {n_labeled:,} ({n_labeled/n_total*100:.1f}%)")
            print(f"   Unlabeled (NaN): {n_total-n_labeled:,} ({(n_total-n_labeled)/n_total*100:.1f}%)")
            print(f"   WARNING: Metrics calculated ONLY on labeled aspects")
        except Exception as e:
            print(f"WARNING: Could not load raw data for NaN masking: {e}")
            print(f"   Calculating metrics on ALL aspects (may be inflated)")
            labeled_mask = None
    
    # Calculate metrics per aspect
    aspect_metrics = {}
    
    for i, aspect in enumerate(aspect_names):
        if labeled_mask is not None:
            # Only evaluate on labeled samples for this aspect
            mask = labeled_mask[:, i]
            if mask.sum() == 0:
                # No labeled data for this aspect, skip
                print(f"   WARNING: {aspect}: No labeled data, skipping")
                continue
            
            aspect_preds = all_preds[:, i][mask].numpy()
            aspect_labels = all_labels[:, i][mask].numpy()
            n_samples = mask.sum().item()
        else:
            # Evaluate on all samples (old behavior)
            aspect_preds = all_preds[:, i].numpy()
            aspect_labels = all_labels[:, i].numpy()
            n_samples = len(aspect_preds)
        
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
            'f1': f1,
            'n_samples': n_samples
        }
    
    # Overall metrics (average across aspects)
    if labeled_mask is not None:
        # Calculate overall accuracy ONLY on labeled aspects
        overall_acc = (all_preds[labeled_mask] == all_labels[labeled_mask]).float().mean().item()
    else:
        # Old behavior: all aspects
        overall_acc = (all_preds == all_labels).float().mean().item()
    
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    overall_precision = np.mean([m['precision'] for m in aspect_metrics.values()])
    overall_recall = np.mean([m['recall'] for m in aspect_metrics.values()])
    
    results = {
        'overall_accuracy': overall_acc,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'per_aspect': aspect_metrics,
        'n_labeled': labeled_mask.sum().item() if labeled_mask is not None else None,
        'n_total': labeled_mask.numel() if labeled_mask is not None else None
    }
    
    # Add predictions if requested
    if return_predictions:
        results['predictions'] = all_preds
        results['labels'] = all_labels
        if all_texts:
            results['texts'] = all_texts
    
    return results

def print_metrics(metrics: Dict[str, Any], epoch: Optional[int] = None) -> None:
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
    
    # Show labeled vs total if available
    if metrics.get('n_labeled') is not None and metrics.get('n_total') is not None:
        n_labeled = metrics['n_labeled']
        n_total = metrics['n_total']
        print(f"\n   Coverage: {n_labeled:,}/{n_total:,} labeled aspects ({n_labeled/n_total*100:.1f}%)")
        print(f"   WARNING: Metrics above are calculated ONLY on labeled aspects")
    
    print(f"\nPer-Aspect Metrics:")
    print(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Samples':<10}")
    print("-" * 75)
    
    for aspect, m in metrics['per_aspect'].items():
        n_samples_str = f"({m.get('n_samples', 'N/A')})" if 'n_samples' in m else ""
        print(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  {m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%  {n_samples_str:<10}")

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    output_dir: str,
    is_best: bool = False,
) -> str:
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

def setup_logging(output_dir: str) -> str:
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

def save_training_history(history: list, output_dir: str) -> Tuple[str, str]:
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

def save_evaluation_results(
    metrics: Dict[str, Any],
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    aspect_names: list,
    output_dir: str,
    split: str = 'test',
) -> Tuple[str, str]:
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
        
        if metrics.get('n_labeled') is not None and metrics.get('n_total') is not None:
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

def main(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("Multi-Label ABSA Training")
    print("=" * 80)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    # Setup logging
    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
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
    
    # =====================================================================
    # SETUP FOCAL LOSS
    # =====================================================================
    print(f"\n{'='*80}")
    print("Setting up Focal Loss...")
    print(f"{'='*80}")
    
    # Read focal loss config
    focal_config = config.get('multi_label', {})
    use_focal_loss = focal_config.get('use_focal_loss', True)
    focal_gamma = focal_config.get('focal_gamma', 2.0)
    focal_alpha_config = focal_config.get('focal_alpha', 'auto')
    
    if not use_focal_loss:
        print(f"\nWARNING: Focal Loss is DISABLED in config!")
        print(f"   Using standard CrossEntropyLoss")
        # Fallback to cross entropy (not recommended for imbalanced data)
        focal_loss_fn = None  # Will handle this in train_epoch
    else:
        sentiment_to_idx = config['sentiment_labels']
        
        # Determine alpha weights
        if focal_alpha_config == 'auto':
            print(f"\nAlpha mode: AUTO (global inverse frequency)")
            alpha = calculate_global_alpha(
                config['paths']['train_file'],
                train_dataset.aspects,
                sentiment_to_idx
            )
        
        elif isinstance(focal_alpha_config, list) and len(focal_alpha_config) == 3:
            print(f"\nAlpha mode: USER-DEFINED (global)")
            alpha = focal_alpha_config
            print(f"   Using custom alpha: {alpha}")
        
        elif focal_alpha_config is None:
            print(f"\nAlpha mode: EQUAL (no class weighting)")
            alpha = [1.0, 1.0, 1.0]
            print(f"   Using equal weights: {alpha}")
        
        else:
            print(f"\nWARNING: Invalid focal_alpha config: {focal_alpha_config}")
            print(f"   Falling back to AUTO mode")
            alpha = calculate_global_alpha(
                config['paths']['train_file'],
                train_dataset.aspects,
                sentiment_to_idx
            )
        
        # Create Focal Loss with reduction='none' for per-aspect masking
        focal_loss_fn = MultilabelFocalLoss(
            alpha=alpha,
            gamma=focal_gamma,
            num_aspects=11,
            reduction='none'  # Return loss per aspect for masking
        )
        focal_loss_fn = focal_loss_fn.to(device)
        
        print(f"\nFocal Loss ready:")
        print(f"   Gamma: {focal_gamma}")
        print(f"   Alpha: {alpha}")
        print(f"   Reduction: 'none' (for per-aspect masking)")
        print(f"   Mode: Global (same alpha for all 11 aspects)")
        print(f"\nTRAINING WILL SKIP NaN ASPECTS (masking enabled)")
    
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
    
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        logging.info(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            focal_loss_fn,
            scaler,
        )
        print(f"\nTrain Loss: {train_loss:.4f}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names, 
                               raw_data_file=config['paths']['validation_file'])
        print_metrics(val_metrics)
        
        # Log validation metrics
        logging.info(f"Validation - Acc: {val_metrics['overall_accuracy']*100:.2f}%, "
                    f"F1: {val_metrics['overall_f1']*100:.2f}%, "
                    f"Precision: {val_metrics['overall_precision']*100:.2f}%, "
                    f"Recall: {val_metrics['overall_recall']*100:.2f}%")
        
        # Record history
        epoch_history = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_accuracy': val_metrics['overall_accuracy'],
            'val_f1': val_metrics['overall_f1'],
            'val_precision': val_metrics['overall_precision'],
            'val_recall': val_metrics['overall_recall'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add per-aspect metrics to history
        for aspect, metrics in val_metrics['per_aspect'].items():
            epoch_history[f'{aspect}_f1'] = metrics['f1']
            epoch_history[f'{aspect}_acc'] = metrics['accuracy']
        
        training_history.append(epoch_history)
        
        # Save checkpoint
        is_best = val_metrics['overall_f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['overall_f1']
            print(f"\nNew best F1: {best_f1*100:.2f}%")
            logging.info(f"New best F1: {best_f1*100:.2f}%")
        
        # Use output_dir from config if not specified
        output_dir = args.output_dir if args.output_dir else config['paths']['output_dir']
        
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
    
    # Load best checkpoint (robust to absence/incompatibility)
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        try:
            best_checkpoint = torch.load(best_model_path, map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(
                best_checkpoint['model_state_dict'], strict=False
            )
            if unexpected_keys:
                print(f"WARNING: Ignored unexpected keys from old checkpoint: {len(unexpected_keys)} keys")
            if missing_keys:
                print(f"WARNING: Missing keys: {missing_keys}")
        except Exception as e:
            print(f"WARNING: Could not load best model ('{best_model_path}'): {e}")
            print("   Proceeding with current model weights for testing.")
    else:
        print(f"WARNING: Best model not found at '{best_model_path}'. Testing current model.")
    
    test_metrics = evaluate(model, test_loader, device, aspect_names,
                           raw_data_file=config['paths']['test_file'],
                           return_predictions=True)
    print_metrics(test_metrics)
    
    # Log test metrics
    logging.info(f"Test Results - Acc: {test_metrics['overall_accuracy']*100:.2f}%, "
                f"F1: {test_metrics['overall_f1']*100:.2f}%")
    
    # Save detailed evaluation results
    save_evaluation_results(
        test_metrics,
        test_metrics['predictions'],
        test_metrics['labels'],
        aspect_names,
        output_dir,
        split='test'
    )
    
    # Save final results JSON
    results = {
        'test_accuracy': test_metrics['overall_accuracy'],
        'test_f1': test_metrics['overall_f1'],
        'test_precision': test_metrics['overall_precision'],
        'test_recall': test_metrics['overall_recall'],
        'per_aspect': test_metrics['per_aspect'],
        'training_completed': datetime.now().isoformat(),
        'config': {
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_name': config['model']['name']
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
    print(f"   Test Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    print(f"   Test F1:       {test_metrics['overall_f1']*100:.2f}%")
    print(f"\nModel saved to: {output_dir}")
    print(f"\nTraining logs and results:")
    print(f"   - Training history: {output_dir}/training_history.csv")
    print(f"   - Test predictions: {output_dir}/test_predictions_detailed.csv")
    print(f"   - Evaluation summary: {output_dir}/test_evaluation_summary.txt")
    print(f"   - Training log: {log_file}")
    
    logging.info("Training completed successfully")
    logging.info(f"Final Test Accuracy: {test_metrics['overall_accuracy']*100:.2f}%")
    logging.info(f"Final Test F1: {test_metrics['overall_f1']*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multi-Label ABSA Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config if specified)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config if specified)')
    
    args = parser.parse_args()
    main(args)
