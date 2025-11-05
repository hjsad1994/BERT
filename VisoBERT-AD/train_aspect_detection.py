"""
Training Script for Aspect Detection
Train ViSoBERT to detect which aspects are mentioned in the text (binary classification)
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
import json
import logging
from datetime import datetime

from model_aspect_detection import AspectDetectionModel
from dataset_aspect_detection import AspectDetectionDataset

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn, pos_weight=None):
    """Train for one epoch - Binary Cross-Entropy Loss for aspect detection"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)  # [batch_size, num_aspects] - binary labels (0 or 1)
        loss_mask = batch['loss_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(input_ids, attention_mask)  # [batch_size, num_aspects]
        
        # Binary Cross-Entropy Loss with masking
        # BCEWithLogitsLoss includes sigmoid internally, so pass logits directly
        loss_per_aspect = loss_fn(logits, labels)  # [batch_size, num_aspects]
        masked_loss = loss_per_aspect * loss_mask
        
        # Average over all aspects (all aspects are labeled in aspect detection)
        num_aspects = loss_mask.sum()
        if num_aspects > 0:
            loss = masked_loss.sum() / num_aspects
        else:
            loss = masked_loss.sum()  # Fallback (shouldn't happen)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_samples += len(batch['input_ids'])
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss

def evaluate(model, dataloader, device, aspect_names, threshold=0.5, per_aspect_thresholds=None, return_predictions=False):
    """
    Evaluate model - Binary classification for aspect detection
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device (CPU/GPU)
        aspect_names: List of aspect names
        threshold: Default threshold for binary prediction (default: 0.5)
        per_aspect_thresholds: Dict mapping aspect names to thresholds (overrides default threshold)
        return_predictions: If True, return predictions along with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [batch_size, num_aspects] - binary labels
            
            # Predict
            logits = model(input_ids, attention_mask)  # [batch_size, num_aspects]
            probs = torch.sigmoid(logits)  # [batch_size, num_aspects]
            
            # Apply per-aspect thresholds if provided
            if per_aspect_thresholds:
                preds = torch.zeros_like(probs, dtype=torch.long)
                for i, aspect in enumerate(aspect_names):
                    aspect_threshold = per_aspect_thresholds.get(aspect, threshold)
                    preds[:, i] = (probs[:, i] >= aspect_threshold).long()
            else:
                preds = (probs >= threshold).long()  # Binary: 0 or 1
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            
            # Store texts if needed for detailed results
            if return_predictions and 'text' in batch:
                all_texts.extend(batch['text'])
    
    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)  # [num_samples, num_aspects]
    all_labels = torch.cat(all_labels, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    # Calculate metrics per aspect (binary classification)
    aspect_metrics = {}
    
    for i, aspect in enumerate(aspect_names):
        aspect_preds = all_preds[:, i].numpy()
        aspect_labels = all_labels[:, i].numpy()
        aspect_probs = all_probs[:, i].numpy()
        
        n_samples = len(aspect_preds)
        
        # Binary classification metrics
        # Accuracy
        acc = accuracy_score(aspect_labels, aspect_preds)
        
        # Binary precision, recall, F1 (average='binary' for binary classification)
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='binary', zero_division=0
        )
        
        # Additional stats
        tp = ((aspect_preds == 1) & (aspect_labels == 1)).sum()
        fp = ((aspect_preds == 1) & (aspect_labels == 0)).sum()
        tn = ((aspect_preds == 0) & (aspect_labels == 0)).sum()
        fn = ((aspect_preds == 0) & (aspect_labels == 1)).sum()
        
        aspect_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': n_samples,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'n_positive': int(aspect_labels.sum()),
            'n_negative': int(n_samples - aspect_labels.sum())
        }
    
    # Overall metrics (average across aspects)
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
        'n_total': all_preds.numel()
    }
    
    # Add predictions if requested
    if return_predictions:
        results['predictions'] = all_preds
        results['labels'] = all_labels
        results['probabilities'] = all_probs
        if all_texts:
            results['texts'] = all_texts
    
    return results

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
    
    if metrics.get('n_total'):
        n_total = metrics['n_total']
        print(f"\n   Total predictions: {n_total:,}")
    
    print(f"\nPer-Aspect Binary Classification Metrics:")
    print(f"{'Aspect':<15} {'Acc':<7} {'F1':<7} {'Prec':<7} {'Rec':<7} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5}")
    print("-" * 80)
    
    for aspect, m in metrics['per_aspect'].items():
        tp = m.get('tp', 0)
        fp = m.get('fp', 0)
        fn = m.get('fn', 0)
        tn = m.get('tn', 0)
        print(f"{aspect:<15} {m['accuracy']*100:>6.2f}% {m['f1']*100:>6.2f}% {m['precision']*100:>6.2f}% "
              f"{m['recall']*100:>6.2f}% {tp:>4} {fp:>4} {fn:>4} {tn:>4}")

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

def save_evaluation_results(metrics, all_preds, all_labels, all_probs, aspect_names, output_dir, split='test'):
    """Save detailed evaluation results including predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions vs labels
    pred_data = []
    for i in range(len(all_preds)):
        row = {'sample_id': i}
        for j, aspect in enumerate(aspect_names):
            row[f'{aspect}_pred'] = int(all_preds[i, j].item())
            row[f'{aspect}_true'] = int(all_labels[i, j].item())
            row[f'{aspect}_prob'] = float(all_probs[i, j].item())
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
        
        if metrics.get('n_total'):
            f.write(f"Total Predictions: {metrics['n_total']:,}\n\n")
        
        f.write(f"Per-Aspect Binary Classification Metrics:\n")
        f.write(f"{'Aspect':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}\n")
        f.write("-" * 85 + "\n")
        
        for aspect, m in metrics['per_aspect'].items():
            tp = m.get('tp', 0)
            fp = m.get('fp', 0)
            fn = m.get('fn', 0)
            tn = m.get('tn', 0)
            f.write(f"{aspect:<15} {m['accuracy']*100:>8.2f}%  {m['f1']*100:>8.2f}%  ")
            f.write(f"{m['precision']*100:>8.2f}%  {m['recall']*100:>8.2f}%  ")
            f.write(f"{tp:>4}  {fp:>4}  {fn:>4}  {tn:>4}\n")
    
    logging.info(f"Evaluation results saved to: {pred_file} and {summary_file}")
    return pred_file, summary_file

def main(args):
    print("=" * 80)
    print("Aspect Detection Training")
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
    # SETUP BCE LOSS
    # =====================================================================
    print(f"\n{'='*80}")
    print("Setting up Binary Cross-Entropy Loss...")
    print(f"{'='*80}")
    
    # Read aspect detection config
    aspect_config = config.get('aspect_detection', {})
    use_class_weights = aspect_config.get('use_class_weights', True)
    prediction_threshold = aspect_config.get('prediction_threshold', 0.5)
    
    # Calculate positive weights for imbalanced binary data
    pos_weight = None
    max_weight = aspect_config.get('max_class_weight', None)
    
    if use_class_weights:
        print(f"\nCalculating class weights for binary classification...")
        weights = train_dataset.get_label_weights()  # [num_aspects, 2]
        
        # Extract positive weights (for mentioned class)
        pos_weights = weights[:, 1].to(device)  # [num_aspects]
        
        # Cap weights if max_weight is specified (to prevent overfitting)
        if max_weight is not None:
            print(f"   Capping class weights at {max_weight:.2f} to prevent overfitting...")
            pos_weights = torch.clamp(pos_weights, max=max_weight)
        
        print(f"   Positive weights (per aspect):")
        for i, aspect in enumerate(train_dataset.aspects):
            weight_val = pos_weights[i].item()
            original_weight = weights[i, 1].item()
            if max_weight is not None and original_weight > max_weight:
                print(f"      {aspect:<15} {weight_val:.3f} (capped from {original_weight:.3f})")
            else:
                print(f"      {aspect:<15} {weight_val:.3f}")
        
        pos_weight = pos_weights  # Use per-aspect positive weights
    else:
        print(f"\nWARNING: Class weights DISABLED")
        print(f"   Using equal weights for all classes")
    
    # Create BCE Loss with reduction='none' for per-aspect masking
    # Note: BCE with logits includes sigmoid, so we pass logits directly
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    bce_loss_fn = bce_loss_fn.to(device)
    
    print(f"\nBCE Loss ready:")
    print(f"   Reduction: 'none' (for per-aspect masking)")
    print(f"   Class weights: {'Enabled' if use_class_weights else 'Disabled'}")
    print(f"   Prediction threshold: {prediction_threshold}")
    
    # Create model
    print(f"\nCreating model...")
    dropout = config['model'].get('dropout', 0.3)
    model = AspectDetectionModel(
        model_name=config['model']['name'],
        num_aspects=11,
        hidden_size=512,
        dropout=dropout
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
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        logging.info(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, bce_loss_fn, pos_weight)
        print(f"\nTrain Loss: {train_loss:.4f}")
        logging.info(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        print(f"\nValidating...")
        
        # Try to load optimal thresholds for validation if available
        per_aspect_thresholds_val = None
        thresholds_file = os.path.join(output_dir, 'optimal_thresholds.json')
        if os.path.exists(thresholds_file) and config.get('aspect_detection', {}).get('use_per_aspect_thresholds', False):
            import json
            with open(thresholds_file, 'r', encoding='utf-8') as f:
                per_aspect_thresholds_val = json.load(f)
        
        val_metrics = evaluate(model, val_loader, device, aspect_names, 
                               threshold=prediction_threshold,
                               per_aspect_thresholds=per_aspect_thresholds_val)
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
    
    # Load best checkpoint
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), weights_only=False)
    
    # Load with strict=False to handle old checkpoints with pooler
    missing_keys, unexpected_keys = model.load_state_dict(best_checkpoint['model_state_dict'], strict=False)
    if unexpected_keys:
        print(f"WARNING: Ignored unexpected keys from old checkpoint: {len(unexpected_keys)} keys")
    if missing_keys:
        print(f"WARNING: Missing keys: {missing_keys}")
    
    # Try to load optimal thresholds if available
    per_aspect_thresholds = None
    thresholds_file = os.path.join(output_dir, 'optimal_thresholds.json')
    if os.path.exists(thresholds_file):
        import json
        with open(thresholds_file, 'r', encoding='utf-8') as f:
            per_aspect_thresholds = json.load(f)
        print(f"\nUsing optimal per-aspect thresholds from: {thresholds_file}")
        print(f"   Thresholds: {per_aspect_thresholds}")
    else:
        print(f"\nUsing default threshold: {prediction_threshold}")
        print(f"   (Run find_optimal_thresholds.py to find optimal thresholds)")
    
    test_metrics = evaluate(model, test_loader, device, aspect_names,
                           threshold=prediction_threshold,
                           per_aspect_thresholds=per_aspect_thresholds,
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
        test_metrics['probabilities'],
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
    parser = argparse.ArgumentParser(description='Train Aspect Detection Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config if specified)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (overrides config if specified)')
    
    args = parser.parse_args()
    main(args)


