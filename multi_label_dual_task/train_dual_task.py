"""
Training Script for Dual-Task ABSA (Focal Loss ONLY)
====================================================
- Task 1: Aspect Detection (binary focal loss)
- Task 2: Sentiment Classification (3-way focal loss)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml
import argparse
import json
import logging
from datetime import datetime

from model_dual_task import DualTaskViSoBERT
from dataset_dual_task import DualTaskABSADataset
# Import only focal loss! No other loss here
from focal_loss_dual_task import (
    DualTaskFocalLoss,
    DualTaskWeightedBCELoss,
    calculate_detection_alpha,
    calculate_sentiment_alpha,
    calculate_detection_pos_weight
)

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    """Train for one epoch with dual-task loss"""
    model.train()
    total_loss = 0
    total_det_loss = 0
    total_sent_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        detection_labels = batch['detection_labels'].to(device)
        sentiment_labels = batch['sentiment_labels'].to(device)
        detection_mask = batch['detection_mask'].to(device)
        sentiment_mask = batch['sentiment_mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        detection_logits, sentiment_logits = model(input_ids, attention_mask)
        
        # Compute dual-task loss
        loss_per_aspect, det_loss, sent_loss = loss_fn(
            detection_logits,
            sentiment_logits,
            detection_labels,
            sentiment_labels,
            detection_mask,
            sentiment_mask
        )
        
        # Average over aspects
        loss = loss_per_aspect.mean()
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_det_loss += det_loss.item()
        total_sent_loss += sent_loss.item()
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'det': f'{det_loss.item():.4f}',
            'sent': f'{sent_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_det_loss = total_det_loss / len(dataloader)
    avg_sent_loss = total_sent_loss / len(dataloader)
    
    return avg_loss, avg_det_loss, avg_sent_loss

def evaluate(model, dataloader, device, aspect_names, detection_thresholds=None):
    """
    Evaluate dual-task model
    Returns metrics for detection and sentiment.
    detection_thresholds: list[float] length=len(aspect_names); default 0.5 each
    """
    model.eval()
    if detection_thresholds is None:
        detection_thresholds = [0.5] * len(aspect_names)

    all_det_preds = []
    all_det_labels = []
    all_sent_preds = []
    all_sent_labels = []
    all_sent_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            detection_labels = batch['detection_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            sentiment_mask = batch['sentiment_mask'].to(device)

            detection_logits, sentiment_logits = model(input_ids, attention_mask)

            # Detection probs via sigmoid on single-logit per aspect
            det_probs = torch.sigmoid(detection_logits)  # [batch, aspects]
            thresholds_tensor = torch.tensor(detection_thresholds, device=det_probs.device).unsqueeze(0)
            det_pred = (det_probs >= thresholds_tensor).long()

            sent_pred = torch.argmax(torch.softmax(sentiment_logits, dim=-1), dim=-1)

            all_det_preds.append(det_pred.cpu())
            all_det_labels.append(detection_labels.cpu())
            all_sent_preds.append(sent_pred.cpu())
            all_sent_labels.append(sentiment_labels.cpu())
            all_sent_masks.append(sentiment_mask.cpu())

    # Concatenate
    all_det_preds = torch.cat(all_det_preds, dim=0)
    all_det_labels = torch.cat(all_det_labels, dim=0)
    all_sent_preds = torch.cat(all_sent_preds, dim=0)
    all_sent_labels = torch.cat(all_sent_labels, dim=0)
    all_sent_masks = torch.cat(all_sent_masks, dim=0)

    # Metrics (same as before)...
    detection_metrics = {}
    for i, aspect in enumerate(aspect_names):
        det_pred_np = all_det_preds[:, i].numpy()
        det_label_np = all_det_labels[:, i].numpy()
        acc = accuracy_score(det_label_np, det_pred_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            det_label_np, det_pred_np, average='binary', pos_label=1, zero_division=0
        )
        detection_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'threshold': detection_thresholds[i]
        }

    det_pred_flat = all_det_preds.numpy().flatten()
    det_label_flat = all_det_labels.numpy().flatten()
    overall_det_acc = accuracy_score(det_label_flat, det_pred_flat)
    overall_det_p, overall_det_r, overall_det_f1, _ = precision_recall_fscore_support(
        det_label_flat, det_pred_flat, average='binary', pos_label=1, zero_division=0
    )

    # Sentiment metrics (only on detected)
    sentiment_metrics = {}
    for i, aspect in enumerate(aspect_names):
        mask = all_sent_masks[:, i].bool()
        if mask.sum() == 0:
            sentiment_metrics[aspect] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'n_samples': 0
            }
            continue
        sent_pred_np = all_sent_preds[:, i][mask].numpy()
        sent_label_np = all_sent_labels[:, i][mask].numpy()
        acc = accuracy_score(sent_label_np, sent_pred_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            sent_label_np, sent_pred_np, average='weighted', zero_division=0
        )
        sentiment_metrics[aspect] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': mask.sum().item()
        }

    sent_mask_flat = all_sent_masks.bool().flatten()
    if sent_mask_flat.sum() > 0:
        sent_pred_masked = all_sent_preds.numpy().flatten()[sent_mask_flat.numpy()]
        sent_label_masked = all_sent_labels.numpy().flatten()[sent_mask_flat.numpy()]
        overall_sent_acc = accuracy_score(sent_label_masked, sent_pred_masked)
        overall_sent_p, overall_sent_r, overall_sent_f1, _ = precision_recall_fscore_support(
            sent_label_masked, sent_pred_masked, average='weighted', zero_division=0
        )
    else:
        overall_sent_acc = 0.0
        overall_sent_p = 0.0
        overall_sent_r = 0.0
        overall_sent_f1 = 0.0

    return {
        'detection': {
            'overall_accuracy': overall_det_acc,
            'overall_precision': overall_det_p,
            'overall_recall': overall_det_r,
            'overall_f1': overall_det_f1,
            'per_aspect': detection_metrics
        },
        'sentiment': {
            'overall_accuracy': overall_sent_acc,
            'overall_precision': overall_sent_p,
            'overall_recall': overall_sent_r,
            'overall_f1': overall_sent_f1,
            'per_aspect': sentiment_metrics
        }
    }


def find_optimal_detection_thresholds(model, dataloader, device, aspect_names):
    """Grid-search simple per-aspect thresholds on validation set to maximize F1."""
    print("\nOptimizing detection thresholds on validation set...")
    candidate_thresholds = [round(x, 2) for x in [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80]]
    best_thresholds = [0.5] * len(aspect_names)
    best_f1s = [0.0] * len(aspect_names)

    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting val probs"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            detection_labels = batch['detection_labels'].to(device)
            det_logits, _ = model(input_ids, attention_mask)
            det_probs = torch.sigmoid(det_logits)  # [batch, aspects]
            all_probs.append(det_probs.cpu())
            all_labels.append(detection_labels.cpu())
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    for i, aspect in enumerate(aspect_names):
        y_true = all_labels[:, i]
        best_f1 = 0.0
        best_t = 0.5
        for t in candidate_thresholds:
            y_pred = (all_probs[:, i] >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[i] = best_t
        best_f1s[i] = best_f1
    print("✓ Optimal thresholds:")
    for i, aspect in enumerate(aspect_names):
        print(f"   {aspect:<15} t*={best_thresholds[i]:.2f}  (val F1={best_f1s[i]*100:.2f}%)")
    return best_thresholds

def print_metrics(metrics, epoch=None):
    """Pretty print dual-task metrics"""
    if epoch is not None:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} Results")
        print(f"{'='*80}")
    
    # Detection metrics
    print(f"\nDETECTION METRICS (Aspect Present/Not Present):")
    det = metrics['detection']
    print(f"   Accuracy:  {det['overall_accuracy']*100:.2f}%")
    print(f"   Precision: {det['overall_precision']*100:.2f}%")
    print(f"   Recall:    {det['overall_recall']*100:.2f}%")
    print(f"   F1 Score:  {det['overall_f1']*100:.2f}%")
    
    # Sentiment metrics
    print(f"\nSENTIMENT METRICS (When Aspect Detected):")
    sent = metrics['sentiment']
    print(f"   Accuracy:  {sent['overall_accuracy']*100:.2f}%")
    print(f"   Precision: {sent['overall_precision']*100:.2f}%")
    print(f"   Recall:    {sent['overall_recall']*100:.2f}%")
    print(f"   F1 Score:  {sent['overall_f1']*100:.2f}%")
    
    # Per-aspect breakdown
    print(f"\nPer-Aspect Metrics:")
    print(f"{'Aspect':<15} {'Detection F1':<15} {'Sentiment F1':<15} {'Sent. Samples':<15}")
    print("-" * 65)
    
    for aspect in det['per_aspect'].keys():
        det_f1 = det['per_aspect'][aspect]['f1']
        sent_f1 = sent['per_aspect'][aspect]['f1']
        sent_n = sent['per_aspect'][aspect]['n_samples']
        
        print(f"{aspect:<15} {det_f1*100:>13.2f}%  {sent_f1*100:>13.2f}%  {sent_n:>13}")

def setup_logging(output_dir):
    """Setup logging"""
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
        print(f"✓ Saved best model: {best_path}")
    
    return checkpoint_path

def main(args):
    print("=" * 80)
    print("Dual-Task ABSA Training (Focal Loss Only)")
    print("=" * 80)
    print("\n⚡ Using FOCAL LOSS for BOTH DETECTION and SENTIMENT tasks ONLY!")
    
    # Load config
    print(f"\n📖 Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Setup logging
    output_dir = args.output_dir if args.output_dir else config['paths']['output_dir'] + '_dual_task'
    log_file = setup_logging(output_dir)
    logging.info(f"Training started at {datetime.now()}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    logging.info(f"Device: {device}")
    
    # Set seed
    seed = config['reproducibility']['training_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Load tokenizer
    print(f"\n✓ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load datasets
    print(f"\n✓ Loading datasets...")
    train_dataset = DualTaskABSADataset(
        config['paths']['train_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    val_dataset = DualTaskABSADataset(
        config['paths']['validation_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    test_dataset = DualTaskABSADataset(
        config['paths']['test_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    print(f"\n   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Create dataloaders
    batch_size = config['training'].get('per_device_train_batch_size', 32)
    eval_batch_size = config['training'].get('per_device_eval_batch_size', 64)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
    
    # === CRITERION BLOCK ===
    print(f"\n{'='*80}")
    print("Setting up Dual-Task criterion ...")
    print(f"{'='*80}")
    focal_config = config.get('multi_label', {})
    detection_weight = focal_config.get('detection_weight', 0.3)
    sentiment_weight = focal_config.get('sentiment_weight', 0.7)
    detection_loss_type = focal_config.get('detection_loss_type', 'focal')  # 'focal' or 'bce'
    print(f"Detection loss type: {detection_loss_type}")

    if detection_loss_type.lower() == 'bce':
        # Weighted BCE for detection using per-aspect pos_weight
        pos_weight = calculate_detection_pos_weight(
            config['paths']['train_file'],
            train_dataset.aspects
        )
        sentiment_alpha = calculate_sentiment_alpha(
            config['paths']['train_file'],
            train_dataset.aspects,
            train_dataset.sentiment_map
        )
        loss_fn = DualTaskWeightedBCELoss(
            detection_pos_weight=pos_weight.to(device),
            sentiment_alpha=sentiment_alpha,
            sentiment_gamma=focal_config.get('sentiment_gamma', 2.0),
            detection_weight=detection_weight,
            sentiment_weight=sentiment_weight,
            reduction='none'
        ).to(device)
    else:
        # Default: Focal for both tasks
        focal_gamma = focal_config.get('focal_gamma', 2.0)
        detection_gamma = focal_config.get('detection_gamma', None)
        sentiment_gamma = focal_config.get('sentiment_gamma', None)
        detection_alpha = calculate_detection_alpha(
            config['paths']['train_file'],
            train_dataset.aspects
        )
        sentiment_alpha = calculate_sentiment_alpha(
            config['paths']['train_file'],
            train_dataset.aspects,
            train_dataset.sentiment_map
        )
        loss_fn = DualTaskFocalLoss(
            detection_alpha=detection_alpha,
            sentiment_alpha=sentiment_alpha,
            gamma=focal_gamma,
            detection_weight=detection_weight,
            sentiment_weight=sentiment_weight,
            reduction='none',
            detection_gamma=detection_gamma,
            sentiment_gamma=sentiment_gamma
        ).to(device)
    # === END CRITERION BLOCK ===
    
    # Create model
    print(f"\n{'='*80}")
    print(f"Creating Dual-Task Model")
    print(f"{'='*80}")
    model_name = config['model']['name']
    print(f"Model: {model_name}")
    logging.info(f"Model: {model_name}")
    
    model = DualTaskViSoBERT(
        model_name=model_name,
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3,
        use_separate_encoders=False
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"   Base model: {model_name}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Detection head: Binary classification (11 aspects)")
    print(f"   Sentiment head: 3-class classification (11 aspects)")
    
    logging.info(f"Model: {model_name}")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer & Scheduler
    learning_rate = config['training'].get('learning_rate', 2e-5)
    num_epochs = args.epochs if args.epochs else config['training'].get('num_train_epochs', 5)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    warmup_ratio = config['training'].get('warmup_ratio', 0.06)
    warmup_steps = int(warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Total steps: {total_steps}")
    print(f"   Optimizer: AdamW (weight_decay=0.01)")
    print(f"   Scheduler: Cosine with warmup")
    
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
        train_loss, train_det_loss, train_sent_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn
        )

        # Lấy trọng số từ focal_config (nếu chưa có, lấy mặc định)
        focal_config = config.get('multi_label', {})
        detection_weight = focal_config.get('detection_weight', 0.3)
        sentiment_weight = focal_config.get('sentiment_weight', 0.7)
        total_loss_weighted = detection_weight * train_det_loss + sentiment_weight * train_sent_loss

        print(f"\n[Epoch {epoch} Train Loss]:")
        print(f"   Detection loss:  {train_det_loss:.4f}")
        print(f"   Sentiment loss:  {train_sent_loss:.4f}")
        print(f"   Total loss (weighted): {total_loss_weighted:.4f}  [= {detection_weight:.2f}*det + {sentiment_weight:.2f}*sent]")
        logging.info(f"Epoch {epoch} Detection Loss: {train_det_loss:.4f}")
        logging.info(f"Epoch {epoch} Sentiment Loss: {train_sent_loss:.4f}")
        logging.info(f"Epoch {epoch} Total Weighted Loss: {total_loss_weighted:.4f} (det*{detection_weight:.2f} + sent*{sentiment_weight:.2f})")
        
        # Validate
        print(f"\nValidating...")
        val_metrics = evaluate(model, val_loader, device, aspect_names, detection_thresholds=None)
        print_metrics(val_metrics)

        # Find and log optimal detection thresholds on validation set
        optimal_thresholds = find_optimal_detection_thresholds(model, val_loader, device, aspect_names)
        logging.info(f"Optimal detection thresholds: {optimal_thresholds}")

        # Track best model (based on sentiment F1, since that's the main task)
        val_sent_f1 = val_metrics['sentiment']['overall_f1']
        is_best = val_sent_f1 > best_f1
        
        if is_best:
            best_f1 = val_sent_f1
            print(f"\nNew best Sentiment F1: {best_f1*100:.2f}%")
            logging.info(f"New best Sentiment F1: {best_f1*100:.2f}%")
        
        save_checkpoint(model, optimizer, epoch, val_metrics, output_dir, is_best=is_best)
    
    # Test with best model
    print(f"\n{'='*80}")
    print("Testing Best Model")
    print(f"{'='*80}")

    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # Recompute optimal thresholds on validation with best model (for fairness)
    optimal_thresholds = find_optimal_detection_thresholds(model, val_loader, device, aspect_names)
    print("\nUsing optimal detection thresholds for TEST:")
    print(f"{optimal_thresholds}")

    test_metrics = evaluate(model, test_loader, device, aspect_names, detection_thresholds=optimal_thresholds)
    print_metrics(test_metrics)
    
    # Save results
    results = {
        'detection': test_metrics['detection'],
        'sentiment': test_metrics['sentiment'],
        'training_completed': datetime.now().isoformat()
    }
    
    results_file = os.path.join(output_dir, 'test_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"\nBest Model Performance:")
    print(f"   Detection F1:  {test_metrics['detection']['overall_f1']*100:.2f}%")
    print(f"   Sentiment F1:  {test_metrics['sentiment']['overall_f1']*100:.2f}%")
    print(f"\n📁 Model saved to: {output_dir}")
    
    logging.info("Training completed successfully")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Dual-Task ABSA Model')
    parser.add_argument('--config', type=str, default='multi_label/config_multi.yaml', help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    main(args)
