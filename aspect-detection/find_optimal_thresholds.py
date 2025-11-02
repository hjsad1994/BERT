"""
Find Optimal Thresholds for Each Aspect
========================================
This script finds the optimal prediction threshold for each aspect
by maximizing F1 score on validation set.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import yaml
import json
import argparse
import os

from model_multilabel import AspectDetectionModel
from dataset_multilabel import AspectDetectionDataset


def find_optimal_thresholds(model, dataloader, device, aspect_names, threshold_range=(0.1, 0.9), step=0.01):
    """
    Find optimal threshold for each aspect by maximizing F1 score.
    
    Args:
        model: Trained model
        dataloader: DataLoader with validation/test data
        device: Device (CPU/GPU)
        aspect_names: List of aspect names
        threshold_range: Tuple of (min_threshold, max_threshold)
        step: Step size for threshold search
    
    Returns:
        Dictionary mapping aspect names to optimal thresholds
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    print(f"\nCollecting predictions from {len(dataloader)} batches...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs, dim=0).numpy()  # [num_samples, num_aspects]
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    print(f"\nSearching optimal thresholds in range [{threshold_range[0]}, {threshold_range[1]}] with step {step}...")
    
    optimal_thresholds = {}
    optimal_scores = {}
    
    for i, aspect in enumerate(tqdm(aspect_names, desc="Finding thresholds")):
        aspect_probs = all_probs[:, i]
        aspect_labels = all_labels[:, i]
        
        best_threshold = 0.5
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        
        thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
        
        for threshold in thresholds:
            preds = (aspect_probs >= threshold).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                aspect_labels, preds, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        
        optimal_thresholds[aspect] = float(best_threshold)
        optimal_scores[aspect] = {
            'f1': float(best_f1),
            'precision': float(best_precision),
            'recall': float(best_recall),
            'threshold': float(best_threshold)
        }
        
        print(f"   {aspect:<15} threshold={best_threshold:.3f}, F1={best_f1:.4f}, "
              f"Precision={best_precision:.4f}, Recall={best_recall:.4f}")
    
    return optimal_thresholds, optimal_scores


def evaluate_with_thresholds(model, dataloader, device, aspect_names, thresholds_dict, return_predictions=False):
    """
    Evaluate model with per-aspect thresholds.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with test data
        device: Device (CPU/GPU)
        aspect_names: List of aspect names
        thresholds_dict: Dictionary mapping aspect names to thresholds
        return_predictions: If True, return predictions along with metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs, dim=0)  # [num_samples, num_aspects]
    all_labels = torch.cat(all_labels, dim=0)
    
    # Apply per-aspect thresholds
    all_preds = torch.zeros_like(all_probs, dtype=torch.long)
    for i, aspect in enumerate(aspect_names):
        threshold = thresholds_dict.get(aspect, 0.5)
        all_preds[:, i] = (all_probs[:, i] >= threshold).long()
    
    all_probs = all_probs.numpy()
    all_preds = all_preds.numpy()
    all_labels = all_labels.numpy()
    
    # Calculate metrics per aspect
    aspect_metrics = {}
    
    for i, aspect in enumerate(aspect_names):
        aspect_preds = all_preds[:, i]
        aspect_labels = all_labels[:, i]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            aspect_labels, aspect_preds, average='binary', zero_division=0
        )
        
        tp = ((aspect_preds == 1) & (aspect_labels == 1)).sum()
        fp = ((aspect_preds == 1) & (aspect_labels == 0)).sum()
        tn = ((aspect_preds == 0) & (aspect_labels == 0)).sum()
        fn = ((aspect_preds == 0) & (aspect_labels == 1)).sum()
        
        aspect_metrics[aspect] = {
            'accuracy': float(np.mean(aspect_preds == aspect_labels)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'threshold': float(thresholds_dict.get(aspect, 0.5))
        }
    
    # Overall metrics
    overall_acc = np.mean(all_preds == all_labels)
    overall_f1 = np.mean([m['f1'] for m in aspect_metrics.values()])
    overall_precision = np.mean([m['precision'] for m in aspect_metrics.values()])
    overall_recall = np.mean([m['recall'] for m in aspect_metrics.values()])
    
    results = {
        'overall_accuracy': float(overall_acc),
        'overall_f1': float(overall_f1),
        'overall_precision': float(overall_precision),
        'overall_recall': float(overall_recall),
        'per_aspect': aspect_metrics,
        'n_total': int(all_preds.size)
    }
    
    if return_predictions:
        results['predictions'] = all_preds
        results['labels'] = all_labels
        results['probabilities'] = all_probs
    
    return results


def load_config(config_path):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    print("=" * 80)
    print("Find Optimal Thresholds for Each Aspect")
    print("=" * 80)
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load validation dataset (use validation to find optimal thresholds)
    print(f"\nLoading validation dataset...")
    val_dataset = AspectDetectionDataset(
        config['paths']['validation_file'],
        tokenizer,
        max_length=config['model']['max_length']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('per_device_eval_batch_size', 32),
        shuffle=False,
        num_workers=2
    )
    
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Load model
    print(f"\nLoading best model...")
    model = AspectDetectionModel(
        model_name=config['model']['name'],
        num_aspects=11,
        hidden_size=config['model'].get('hidden_size', 512),
        dropout=config['model'].get('dropout', 0.3)
    )
    
    checkpoint_path = os.path.join(config['paths']['output_dir'], 'best_model.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   Loaded checkpoint from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    aspect_names = val_dataset.aspects
    
    # Find optimal thresholds
    print(f"\n{'='*80}")
    print("Step 1: Finding Optimal Thresholds on Validation Set")
    print(f"{'='*80}")
    
    optimal_thresholds, optimal_scores = find_optimal_thresholds(
        model, val_loader, device, aspect_names,
        threshold_range=(args.min_threshold, args.max_threshold),
        step=args.step
    )
    
    # Save thresholds
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds_file = os.path.join(output_dir, 'optimal_thresholds.json')
    with open(thresholds_file, 'w', encoding='utf-8') as f:
        json.dump(optimal_thresholds, f, indent=2, ensure_ascii=False)
    print(f"\nOptimal thresholds saved to: {thresholds_file}")
    
    scores_file = os.path.join(output_dir, 'optimal_threshold_scores.json')
    with open(scores_file, 'w', encoding='utf-8') as f:
        json.dump(optimal_scores, f, indent=2, ensure_ascii=False)
    
    # Evaluate on validation set with optimal thresholds
    print(f"\n{'='*80}")
    print("Step 2: Evaluating with Optimal Thresholds (Validation Set)")
    print(f"{'='*80}")
    
    val_results = evaluate_with_thresholds(
        model, val_loader, device, aspect_names, optimal_thresholds
    )
    
    print(f"\nValidation Metrics with Optimal Thresholds:")
    print(f"   Accuracy:  {val_results['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score:  {val_results['overall_f1']*100:.2f}%")
    print(f"   Precision: {val_results['overall_precision']*100:.2f}%")
    print(f"   Recall:    {val_results['overall_recall']*100:.2f}%")
    
    print(f"\nPer-Aspect Metrics:")
    print(f"{'Aspect':<15} {'Threshold':<10} {'F1':<8} {'Prec':<8} {'Rec':<8}")
    print("-" * 60)
    for aspect, metrics in val_results['per_aspect'].items():
        print(f"{aspect:<15} {metrics['threshold']:>9.3f} {metrics['f1']*100:>7.2f}% "
              f"{metrics['precision']*100:>7.2f}% {metrics['recall']*100:>7.2f}%")
    
    # Evaluate on test set if available
    if os.path.exists(config['paths']['test_file']):
        print(f"\n{'='*80}")
        print("Step 3: Evaluating on Test Set with Optimal Thresholds")
        print(f"{'='*80}")
        
        test_dataset = AspectDetectionDataset(
            config['paths']['test_file'],
            tokenizer,
            max_length=config['model']['max_length']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training'].get('per_device_eval_batch_size', 32),
            shuffle=False,
            num_workers=2
        )
        
        print(f"   Test samples: {len(test_dataset)}")
        
        test_results = evaluate_with_thresholds(
            model, test_loader, device, aspect_names, optimal_thresholds,
            return_predictions=True
        )
        
        print(f"\nTest Metrics with Optimal Thresholds:")
        print(f"   Accuracy:  {test_results['overall_accuracy']*100:.2f}%")
        print(f"   F1 Score:  {test_results['overall_f1']*100:.2f}%")
        print(f"   Precision: {test_results['overall_precision']*100:.2f}%")
        print(f"   Recall:    {test_results['overall_recall']*100:.2f}%")
        
        print(f"\nPer-Aspect Test Metrics:")
        print(f"{'Aspect':<15} {'Threshold':<10} {'F1':<8} {'Prec':<8} {'Rec':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}")
        print("-" * 80)
        for aspect, metrics in test_results['per_aspect'].items():
            print(f"{aspect:<15} {metrics['threshold']:>9.3f} {metrics['f1']*100:>7.2f}% "
                  f"{metrics['precision']*100:>7.2f}% {metrics['recall']*100:>7.2f}% "
                  f"{metrics['tp']:>4} {metrics['fp']:>4} {metrics['fn']:>4} {metrics['tn']:>4}")
        
        # Save test results
        test_results_file = os.path.join(output_dir, 'test_results_optimal_thresholds.json')
        with open(test_results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nTest results saved to: {test_results_file}")
    
    print(f"\n{'='*80}")
    print("Complete!")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"   Optimal thresholds saved to: {thresholds_file}")
    print(f"   Validation F1 improvement: {val_results['overall_f1']*100:.2f}%")
    if os.path.exists(config['paths']['test_file']):
        print(f"   Test F1 with optimal thresholds: {test_results['overall_f1']*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find optimal thresholds for each aspect')
    parser.add_argument('--config', type=str, default='aspect-detection/config_multi.yaml',
                      help='Path to config file')
    parser.add_argument('--min-threshold', type=float, default=0.1,
                      help='Minimum threshold to search')
    parser.add_argument('--max-threshold', type=float, default=0.9,
                      help='Maximum threshold to search')
    parser.add_argument('--step', type=float, default=0.01,
                      help='Step size for threshold search')
    
    args = parser.parse_args()
    main(args)

