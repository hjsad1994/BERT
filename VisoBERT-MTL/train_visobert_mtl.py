"""
ViSoBERT Multi-Task Learning for Vietnamese ABSA
================================================
Train both AD and SC simultaneously with shared ViSoBERT backbone

Combined Focal Loss = α * BinaryFocal_AD + β * Focal_SC
Both with auto alpha calculation
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
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from VisoBERT-STL for focal losses
import sys
sys.path.append('VisoBERT-STL')
from binary_focal_loss import BinaryFocalLoss, calculate_binary_alpha_auto
from focal_loss_multilabel import MultilabelFocalLoss, calculate_global_alpha

# MTL components
from model_visobert_mtl import ViSoBERT_MTL
from dataset_visobert_mtl import MTLDataset


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: str) -> str:
    """Setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f'visobert_mtl_log_{timestamp}.txt')
    
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


def train_epoch_mtl(model, dataloader, optimizer, scheduler, device, 
                    ad_focal_loss, sc_focal_loss, loss_weight_ad, loss_weight_sc, scaler):
    """Train one epoch with multi-task learning and focal losses"""
    model.train()
    total_loss = 0
    total_ad_loss = 0
    total_sc_loss = 0
    
    for batch in tqdm(dataloader, desc="[ViSoBERT-MTL] Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        ad_labels = batch['ad_labels'].to(device)
        sc_labels = batch['sc_labels'].to(device)
        sc_mask = batch['sc_loss_mask'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            # Forward pass (both heads)
            ad_logits, sc_logits = model(input_ids, attention_mask)
            
            # AD loss (Binary Focal Loss)
            ad_loss = ad_focal_loss(ad_logits, ad_labels)
            
            # SC loss (Multi-label Focal Loss with masking)
            sc_loss_per_aspect = sc_focal_loss(sc_logits, sc_labels)  # [batch, aspects]
            sc_masked_loss = sc_loss_per_aspect * sc_mask
            num_labeled = sc_mask.sum()
            sc_loss = sc_masked_loss.sum() / num_labeled if num_labeled > 0 else sc_masked_loss.sum()
            
            # Combined focal loss
            loss = loss_weight_ad * ad_loss + loss_weight_sc * sc_loss
        
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
        total_ad_loss += ad_loss.item()
        total_sc_loss += sc_loss.item()
    
    return total_loss / len(dataloader), total_ad_loss / len(dataloader), total_sc_loss / len(dataloader)


def evaluate_mtl(model, dataloader, device, aspect_names):
    """Evaluate both AD and SC tasks"""
    model.eval()
    
    ad_preds_all, ad_labels_all = [], []
    sc_preds_all, sc_labels_all = [], []
    sc_masks_all = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[ViSoBERT-MTL] Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ad_labels = batch['ad_labels'].to(device)
            sc_labels = batch['sc_labels'].to(device)
            sc_mask = batch['sc_loss_mask'].to(device)
            
            ad_logits, sc_logits = model(input_ids, attention_mask)
            
            # AD predictions
            ad_probs = torch.sigmoid(ad_logits)
            ad_preds = (ad_probs >= 0.5).float()
            
            # SC predictions
            sc_preds = torch.argmax(sc_logits, dim=-1)
            
            ad_preds_all.append(ad_preds.cpu())
            ad_labels_all.append(ad_labels.cpu())
            sc_preds_all.append(sc_preds.cpu())
            sc_labels_all.append(sc_labels.cpu())
            sc_masks_all.append(sc_mask.cpu())
    
    ad_preds_all = torch.cat(ad_preds_all, dim=0).numpy()
    ad_labels_all = torch.cat(ad_labels_all, dim=0).numpy()
    sc_preds_all = torch.cat(sc_preds_all, dim=0)
    sc_labels_all = torch.cat(sc_labels_all, dim=0)
    sc_masks_all = torch.cat(sc_masks_all, dim=0)
    
    # AD per-aspect metrics
    ad_aspect_metrics = {}
    for i, aspect in enumerate(aspect_names):
        acc = accuracy_score(ad_labels_all[:, i], ad_preds_all[:, i])
        p, r, f1, _ = precision_recall_fscore_support(
            ad_labels_all[:, i], ad_preds_all[:, i], average='binary', zero_division=0
        )
        ad_aspect_metrics[aspect] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    
    # AD overall metrics (F1 Macro: average of per-aspect metrics)
    ad_acc = np.mean([m['accuracy'] for m in ad_aspect_metrics.values()])
    ad_p = np.mean([m['precision'] for m in ad_aspect_metrics.values()])
    ad_r = np.mean([m['recall'] for m in ad_aspect_metrics.values()])
    ad_f1 = np.mean([m['f1'] for m in ad_aspect_metrics.values()])
    
    # SC metrics (ONLY on labeled aspects where ad_labels==1)
    # CRITICAL FIX: Evaluate sentiment ONLY where aspect is actually present
    valid_sentiment_mask = (sc_masks_all > 0) & (torch.tensor(ad_labels_all) == 1)
    
    if valid_sentiment_mask.sum() == 0:
        print("WARNING: No valid sentiment labels found!")
        sc_acc = 0.0
        sc_f1 = 0.0
        sc_p = 0.0
        sc_r = 0.0
        sc_aspect_metrics = {aspect: {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0} for aspect in aspect_names}
    else:
        # Per-aspect SC metrics (only on valid aspects)
        sc_aspect_metrics = {}
        for i, aspect in enumerate(aspect_names):
            aspect_mask = valid_sentiment_mask[:, i]
            if aspect_mask.sum() == 0:
                sc_aspect_metrics[aspect] = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'n_samples': 0}
                continue
            
            aspect_preds = sc_preds_all[:, i][aspect_mask].numpy()
            aspect_labels = sc_labels_all[:, i][aspect_mask].numpy()
            
            acc = accuracy_score(aspect_labels, aspect_preds)
            p, r, f1, _ = precision_recall_fscore_support(
                aspect_labels, aspect_preds, average='macro', zero_division=0
            )
            sc_aspect_metrics[aspect] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'n_samples': aspect_mask.sum().item()}
        
        # Overall SC metrics (F1 Macro: average of per-aspect metrics)
        valid_aspects = [m for m in sc_aspect_metrics.values() if m['n_samples'] > 0]
        if len(valid_aspects) > 0:
            sc_acc = np.mean([m['accuracy'] for m in valid_aspects])
            sc_p = np.mean([m['precision'] for m in valid_aspects])
            sc_r = np.mean([m['recall'] for m in valid_aspects])
            sc_f1 = np.mean([m['f1'] for m in valid_aspects])
        else:
            sc_acc = sc_p = sc_r = sc_f1 = 0.0
    
    return {
        'ad': {
            'overall_accuracy': ad_acc,
            'overall_precision': ad_p,
            'overall_recall': ad_r,
            'overall_f1': ad_f1,
            'per_aspect': ad_aspect_metrics,
            'predictions': ad_preds_all,
            'labels': ad_labels_all
        },
        'sc': {
            'overall_accuracy': sc_acc,
            'overall_precision': sc_p,
            'overall_recall': sc_r,
            'overall_f1': sc_f1,
            'per_aspect': sc_aspect_metrics,
            'predictions': sc_preds_all,
            'labels': sc_labels_all
        }
    }


def save_confusion_matrices(metrics: dict, aspect_names: list, output_dir: str):
    """Save confusion matrices for both AD and SC"""
    print("\n[ViSoBERT-MTL] Generating confusion matrices...")
    
    # AD confusion matrix
    ad_preds = metrics['ad']['predictions']
    ad_labels = metrics['ad']['labels']
    cm_ad = confusion_matrix(ad_labels.flatten(), ad_preds.flatten())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_ad, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Mentioned', 'Mentioned'],
               yticklabels=['Not Mentioned', 'Mentioned'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Aspect Detection - Confusion Matrix (ViSoBERT-MTL)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_ad.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SC overall confusion matrix
    sc_preds = metrics['sc']['predictions'].numpy()
    sc_labels = metrics['sc']['labels'].numpy()
    cm_sc = confusion_matrix(sc_labels.flatten(), sc_preds.flatten(), labels=[0, 1, 2])
    
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_sc, annot=True, fmt='d', cmap='Blues',
               xticklabels=sentiment_labels, yticklabels=sentiment_labels, ax=ax)
    ax.set_xlabel('Predicted Sentiment', fontsize=12)
    ax.set_ylabel('True Sentiment', fontsize=12)
    ax.set_title('Sentiment Classification - Overall (ViSoBERT-MTL)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_sc_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # SC per-aspect confusion matrices
    n_aspects = len(aspect_names)
    n_cols = 3
    n_rows = (n_aspects + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_aspects > 1 else [axes]
    
    for i, aspect in enumerate(aspect_names):
        cm_aspect = confusion_matrix(sc_labels[:, i], sc_preds[:, i], labels=[0, 1, 2])
        sns.heatmap(cm_aspect, annot=True, fmt='d', cmap='Blues',
                   xticklabels=sentiment_labels, yticklabels=sentiment_labels,
                   ax=axes[i], cbar=False)
        axes[i].set_title(f'{aspect}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    for i in range(n_aspects, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_sc_per_aspect.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: {output_dir}/confusion_matrix_ad.png")
    print(f"   Saved: {output_dir}/confusion_matrix_sc_overall.png")
    print(f"   Saved: {output_dir}/confusion_matrix_sc_per_aspect.png")


def generate_final_report(metrics: dict, output_dir: str, config: dict):
    """Generate final report"""
    print("\n[ViSoBERT-MTL] Generating final report...")
    
    report_lines = [
        "="*80,
        "VISOBERT MULTI-TASK LEARNING FOR VIETNAMESE ABSA",
        "Joint Training Results (AD + SC)",
        "="*80,
        "",
        f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: ViSoBERT (Pretrained BERT) with 2 Task Heads",
        f"Loss: Combined Focal Loss (AD=Binary Focal, SC=Multi-label Focal)",
        f"Loss Weights: AD={config['multi_task']['loss_weight_ad']}, SC={config['multi_task']['loss_weight_sc']}",
        "",
        "="*80,
        "TASK 1: ASPECT DETECTION (Binary Classification)",
        "="*80,
        "",
        f"Test Accuracy:  {metrics['ad']['overall_accuracy']*100:.2f}%",
        f"Test F1 Score:  {metrics['ad']['overall_f1']*100:.2f}%",
        f"Test Precision: {metrics['ad']['overall_precision']*100:.2f}%",
        f"Test Recall:    {metrics['ad']['overall_recall']*100:.2f}%",
        "",
        "Per-Aspect Results (AD):",
        "-"*80
    ]
    
    for aspect, m in metrics['ad']['per_aspect'].items():
        report_lines.append(
            f"{aspect:<15} Accuracy: {m['accuracy']*100:>6.2f}%  "
            f"F1 Score: {m['f1']*100:>6.2f}%  "
            f"Precision: {m['precision']*100:>6.2f}%  "
            f"Recall: {m['recall']*100:>6.2f}%"
        )
    
    report_lines.extend([
        "",
        "="*80,
        "TASK 2: SENTIMENT CLASSIFICATION (3-Class per Aspect)",
        "="*80,
        "",
        f"Test Accuracy:  {metrics['sc']['overall_accuracy']*100:.2f}%",
        f"Test F1 Score:  {metrics['sc']['overall_f1']*100:.2f}%",
        f"Test Precision: {metrics['sc']['overall_precision']*100:.2f}%",
        f"Test Recall:    {metrics['sc']['overall_recall']*100:.2f}%",
        "",
        "Per-Aspect Results (SC):",
        "-"*80
    ])
    
    for aspect, m in metrics['sc']['per_aspect'].items():
        report_lines.append(
            f"{aspect:<15} Accuracy: {m['accuracy']*100:>6.2f}%  "
            f"F1 Score: {m['f1']*100:>6.2f}%  "
            f"Precision: {m['precision']*100:>6.2f}%  "
            f"Recall: {m['recall']*100:>6.2f}%"
        )
    
    report_lines.extend([
        "",
        "="*80,
        "OUTPUT FILES",
        "="*80,
        "",
        "Model:",
        f"  - Best model: {output_dir}/best_model.pt",
        f"  - Training history: {output_dir}/training_history.csv",
        "",
        "Results:",
        f"  - Test results: {output_dir}/test_results.json",
        f"  - Predictions: {output_dir}/test_predictions_detailed.csv",
        "",
        "Confusion Matrices:",
        f"  - AD confusion matrix: {output_dir}/confusion_matrix_ad.png",
        f"  - SC overall confusion matrix: {output_dir}/confusion_matrix_sc_overall.png",
        f"  - SC per-aspect confusion matrices: {output_dir}/confusion_matrix_sc_per_aspect.png",
        "",
        "="*80,
        "TRAINING COMPLETE",
        "="*80
    ])
    
    report_text = '\n'.join(report_lines)
    
    report_file = os.path.join(output_dir, 'final_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nFinal report saved to: {report_file}")


def main(args: argparse.Namespace):
    """Main training function"""
    print("="*80)
    print("VISOBERT MULTI-TASK LEARNING FOR VIETNAMESE ABSA")
    print("="*80)
    
    # Load config
    print(f"\nLoading config from: {args.config}")
    config = load_config(args.config)
    
    output_dir = config['paths']['output_dir']
    log_file = setup_logging(output_dir)
    logging.info("Starting ViSoBERT-MTL training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Set seed
    seed = config['reproducibility']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = MTLDataset(config['paths']['train_file'], tokenizer, config['model']['max_length'])
    val_dataset = MTLDataset(config['paths']['validation_file'], tokenizer, config['model']['max_length'])
    test_dataset = MTLDataset(config['paths']['test_file'], tokenizer, config['model']['max_length'])
    
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
    print("\nCreating ViSoBERT MTL model...")
    model = ViSoBERT_MTL(
        model_name=config['model']['name'],
        num_aspects=config['model']['num_aspects'],
        num_sentiments=config['model']['num_sentiments'],
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Setup Focal Losses
    print("\n" + "="*80)
    print("Setting up Focal Losses")
    print("="*80)
    
    mtl_config = config['multi_task']
    ad_config = mtl_config['aspect_detection']
    sc_config = mtl_config['sentiment_classification']
    
    # AD: Binary Focal Loss
    print("\n[AD] Binary Focal Loss:")
    if ad_config.get('focal_alpha') == 'auto':
        ad_alpha = calculate_binary_alpha_auto(
            config['paths']['train_file'],
            train_dataset.aspects,
            method='inverse_freq'
        )
    else:
        ad_alpha = ad_config.get('focal_alpha', [1.0, 1.0])
    
    ad_focal_loss = BinaryFocalLoss(
        alpha=ad_alpha,
        gamma=ad_config.get('focal_gamma', 2.0),
        reduction='mean'
    )
    ad_focal_loss = ad_focal_loss.to(device)
    
    # SC: Multi-label Focal Loss
    print("\n[SC] Multi-label Focal Loss:")
    sentiment_to_idx = config['sentiment_labels']
    
    if sc_config.get('focal_alpha') == 'auto':
        sc_alpha = calculate_global_alpha(
            config['paths']['train_file'],
            train_dataset.aspects,
            sentiment_to_idx
        )
    else:
        sc_alpha = sc_config.get('focal_alpha', [1.0, 1.0, 1.0])
    
    sc_focal_loss = MultilabelFocalLoss(
        alpha=sc_alpha,
        gamma=sc_config.get('focal_gamma', 2.0),
        num_aspects=11,
        reduction='none'  # For masking
    )
    sc_focal_loss = sc_focal_loss.to(device)
    
    # Loss weights
    loss_weight_ad = mtl_config['loss_weight_ad']
    loss_weight_sc = mtl_config['loss_weight_sc']
    
    print(f"\n[Combined Loss] Weights: AD={loss_weight_ad}, SC={loss_weight_sc}")
    print(f"   Formula: Total = {loss_weight_ad} × Focal_AD + {loss_weight_sc} × Focal_SC")
    
    # Optimizer & Scheduler
    num_epochs = config['training']['num_train_epochs']
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
    print("Starting ViSoBERT-MTL Training")
    print("="*80)
    
    selection_metric_name = config['multi_task'].get('best_model_metric', 'combined_f1')
    best_selection_metric = -float('inf')
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    patience_counter = 0
    early_stopping_patience = config['training']['early_stopping_patience']
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"[ViSoBERT-MTL] Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_ad_loss, train_sc_loss = train_epoch_mtl(
            model, train_loader, optimizer, scheduler, device,
            ad_focal_loss, sc_focal_loss, loss_weight_ad, loss_weight_sc, scaler
        )
        print(f"\nTrain Loss: {train_loss:.4f} (AD: {train_ad_loss:.4f}, SC: {train_sc_loss:.4f})")
        logging.info(f"ViSoBERT-MTL Epoch {epoch}/{num_epochs} - Loss: {train_loss:.4f}")
        
        # Validate
        print("\nValidating...")
        val_metrics = evaluate_mtl(model, val_loader, device, train_dataset.aspects)
        ad_f1 = val_metrics['ad']['overall_f1']
        sc_f1 = val_metrics['sc']['overall_f1']
        ad_acc = val_metrics['ad']['overall_accuracy']
        sc_acc = val_metrics['sc']['overall_accuracy']

        print(f"   AD - Accuracy: {ad_acc*100:.2f}%, F1: {ad_f1*100:.2f}%")
        print(f"   SC - Accuracy: {sc_acc*100:.2f}%, F1: {sc_f1*100:.2f}%")

        if selection_metric_name == 'ad_f1':
            current_selection_metric = ad_f1
        elif selection_metric_name == 'sc_f1':
            current_selection_metric = sc_f1
        else:
            current_selection_metric = (ad_f1 + sc_f1) / 2

        print(f"   Selection metric ({selection_metric_name}): {current_selection_metric*100:.2f}%")
        
        logging.info(
            "ViSoBERT-MTL Val - AD F1: %.2f%%, SC F1: %.2f%%" % (ad_f1*100, sc_f1*100)
        )
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ad_loss': train_ad_loss,
            'train_sc_loss': train_sc_loss,
            'val_ad_accuracy': ad_acc,
            'val_ad_f1': ad_f1,
            'val_sc_accuracy': sc_acc,
            'val_sc_f1': sc_f1,
            'val_selection_metric': current_selection_metric
        })
        
        # Save best model
        if current_selection_metric > best_selection_metric:
            best_selection_metric = current_selection_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, os.path.join(output_dir, 'best_model.pt'))
            print(f"\nNew best {selection_metric_name}: {best_selection_metric*100:.2f}%")
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
    print("[ViSoBERT-MTL] Testing Best Model")
    print("="*80)
    
    best_checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = evaluate_mtl(model, test_loader, device, train_dataset.aspects)
    print(f"\nTest Results:")
    print(f"   AD - Accuracy: {test_metrics['ad']['overall_accuracy']*100:.2f}%, F1: {test_metrics['ad']['overall_f1']*100:.2f}%")
    print(f"   SC - Accuracy: {test_metrics['sc']['overall_accuracy']*100:.2f}%, F1: {test_metrics['sc']['overall_f1']*100:.2f}%")
    
    # Save results
    results = {
        'ad': {
            'test_accuracy': test_metrics['ad']['overall_accuracy'],
            'test_f1': test_metrics['ad']['overall_f1'],
            'test_precision': test_metrics['ad']['overall_precision'],
            'test_recall': test_metrics['ad']['overall_recall'],
            'per_aspect': test_metrics['ad']['per_aspect']
        },
        'sc': {
            'test_accuracy': test_metrics['sc']['overall_accuracy'],
            'test_f1': test_metrics['sc']['overall_f1'],
            'test_precision': test_metrics['sc']['overall_precision'],
            'test_recall': test_metrics['sc']['overall_recall'],
            'per_aspect': test_metrics['sc']['per_aspect']
        },
        'training_completed': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save confusion matrices
    save_confusion_matrices(test_metrics, train_dataset.aspects, output_dir)
    
    # Generate final report
    generate_final_report(test_metrics, output_dir, config)
    
    print("\n" + "="*80)
    print("VISOBERT-MTL TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ViSoBERT Multi-Task Learning for ABSA')
    parser.add_argument('--config', type=str, default='VisoBERT-MTL/config_visobert_mtl.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    main(args)
