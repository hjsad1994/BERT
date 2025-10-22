"""
Ensemble Multiple Multi-Label Models
Average predictions from 3+ models for better performance
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
import argparse
import os

from model_multilabel import MultiLabelViSoBERT
from dataset_multilabel import MultiLabelABSADataset

def load_model(checkpoint_path, device):
    """Load a trained model"""
    model = MultiLabelViSoBERT(
        model_name="5CD-AI/Vietnamese-Sentiment-visobert",
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def predict_single_model(model, dataloader, device):
    """Get predictions from a single model"""
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)  # [batch, 11, 3]
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)  # [num_samples, 11, 3]
    all_labels = torch.cat(all_labels, dim=0)  # [num_samples, 11]
    
    return all_logits, all_labels

def ensemble_predict(model_paths, dataloader, device, weights=None):
    """
    Ensemble predictions from multiple models
    
    Args:
        model_paths: List of checkpoint paths
        dataloader: Test dataloader
        device: cuda or cpu
        weights: Optional weights for each model [w1, w2, w3]
                 If None, use equal weights
    """
    if weights is None:
        weights = [1.0 / len(model_paths)] * len(model_paths)
    
    print(f"\nEnsembling {len(model_paths)} models:")
    print(f"Weights: {weights}")
    
    # Get predictions from all models
    all_model_logits = []
    labels = None
    
    for i, model_path in enumerate(model_paths):
        print(f"\nModel {i+1}/{len(model_paths)}: {model_path}")
        model = load_model(model_path, device)
        
        logits, labels = predict_single_model(model, dataloader, device)
        all_model_logits.append(logits)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Weighted average of logits
    print(f"\nAveraging logits...")
    ensemble_logits = torch.zeros_like(all_model_logits[0])
    
    for i, logits in enumerate(all_model_logits):
        ensemble_logits += weights[i] * logits
    
    # Get predictions
    ensemble_preds = torch.argmax(ensemble_logits, dim=-1)  # [num_samples, 11]
    
    return ensemble_preds, labels

def evaluate(predictions, labels, aspect_names):
    """Evaluate ensemble predictions"""
    # Calculate metrics per aspect
    aspect_metrics = {}
    
    for i, aspect in enumerate(aspect_names):
        aspect_preds = predictions[:, i].numpy()
        aspect_labels = labels[:, i].numpy()
        
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
    
    # Overall metrics
    overall_acc = (predictions == labels).float().mean().item()
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

def print_metrics(metrics):
    """Pretty print metrics"""
    print(f"\n{'='*80}")
    print("Ensemble Results")
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

def main(args):
    print("=" * 80)
    print("Multi-Label Model Ensemble")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert")
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = MultiLabelABSADataset(
        args.test_file,
        tokenizer,
        max_length=256
    )
    
    print(f"   Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    
    # Model paths
    model_paths = args.model_paths
    
    if not all(os.path.exists(p) for p in model_paths):
        print("\nERROR: Some model checkpoints not found!")
        for p in model_paths:
            exists = "OK" if os.path.exists(p) else "NOT FOUND"
            print(f"   {p}: {exists}")
        return
    
    # Ensemble predict
    predictions, labels = ensemble_predict(
        model_paths,
        test_loader,
        device,
        weights=args.weights
    )
    
    # Evaluate
    metrics = evaluate(predictions, labels, test_dataset.aspects)
    print_metrics(metrics)
    
    # Save results
    import json
    results = {
        'ensemble_accuracy': metrics['overall_accuracy'],
        'ensemble_f1': metrics['overall_f1'],
        'ensemble_precision': metrics['overall_precision'],
        'ensemble_recall': metrics['overall_recall'],
        'per_aspect': metrics['per_aspect'],
        'num_models': len(model_paths),
        'weights': args.weights if args.weights else 'equal',
        'model_paths': model_paths
    }
    
    output_file = args.output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*80}")
    print("Ensemble Complete!")
    print(f"{'='*80}")
    print(f"\nFinal Ensemble Performance:")
    print(f"   Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"   F1 Score: {metrics['overall_f1']*100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble Multi-Label Models')
    parser.add_argument('--model-paths', nargs='+', required=True,
                        help='Paths to model checkpoints (e.g., model1/best_model.pt model2/best_model.pt)')
    parser.add_argument('--test-file', type=str, default='data/test_multilabel.csv',
                        help='Test data file')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Optional weights for each model (e.g., 0.3 0.35 0.35)')
    parser.add_argument('--output-file', type=str, default='ensemble_results.json',
                        help='Output results file')
    
    args = parser.parse_args()
    main(args)
