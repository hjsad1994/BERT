"""
Generate Test Predictions CSV for Analysis
Loads trained model and generates predictions in long format for analyze_results.py
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse
import yaml

from model_multilabel import MultiLabelViSoBERT
from dataset_multilabel import MultiLabelABSADataset


def load_config(config_path='multi_label/config_multi.yaml'):
    """Load configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_predictions(model, dataloader, device, aspect_names, sentiment_names, test_df):
    """
    Generate predictions and return in wide format (multi-label format)
    
    Args:
        test_df: DataFrame with test data (to get text)
    
    Returns:
        DataFrame with columns: data, Battery, Camera, ..., Others (predicted sentiments)
        DataFrame with columns: data, Battery, Camera, ..., Others (true sentiments)
    """
    model.eval()
    
    all_predictions = []
    all_true_labels = []
    all_texts = []
    sample_idx = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # [batch_size, num_aspects]
            
            batch_size = input_ids.size(0)
            
            # Predict
            logits = model(input_ids, attention_mask)  # [batch_size, num_aspects, num_sentiments]
            preds = torch.argmax(logits, dim=-1)  # [batch_size, num_aspects]
            
            # Convert to CPU numpy
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Store in wide format
            for i in range(batch_size):
                text = test_df.iloc[sample_idx]['data']
                all_texts.append(text)
                
                # Predicted sentiments for all aspects
                pred_row = {}
                true_row = {}
                true_source_row = test_df.iloc[sample_idx]
                for j, aspect in enumerate(aspect_names):
                    pred_label = preds_np[i, j]
                    pred_row[aspect] = sentiment_names[pred_label].capitalize()

                    true_value = true_source_row[aspect]
                    if pd.isna(true_value):
                        true_row[aspect] = np.nan
                    else:
                        true_str = str(true_value).strip()
                        true_row[aspect] = true_str.capitalize() if true_str else np.nan
                
                all_predictions.append(pred_row)
                all_true_labels.append(true_row)
                
                sample_idx += 1
    
    # Create DataFrames in wide format
    pred_df = pd.DataFrame(all_predictions)
    pred_df.insert(0, 'data', all_texts)
    
    true_df = pd.DataFrame(all_true_labels)
    true_df.insert(0, 'data', all_texts)
    
    return pred_df, true_df


def main(args):
    """Main function"""
    print("\n" + "="*80)
    print("GENERATING TEST PREDICTIONS FOR ANALYSIS")
    print("="*80)
    
    # Load config
    config = load_config(args.config)
    
    # Paths
    if args.model_dir:
        model_dir = args.model_dir
    else:
        # Try to find available models
        default_dir = config['paths']['output_dir']
        possible_dirs = [
            default_dir,
            'multi_label/models/single_task_focal',
            'multi_label/models/dual_task_focal',
            'multi_label/models/multilabel_focal_contrastive'
        ]
        
        model_dir = None
        for dir_path in possible_dirs:
            checkpoint_path = os.path.join(dir_path, 'best_model.pt')
            if os.path.exists(checkpoint_path):
                model_dir = dir_path
                break
        
        if model_dir is None:
            print("\nERROR: ERROR: No trained model found!")
            print("\nSearched in:")
            for dir_path in possible_dirs:
                print(f"   - {dir_path}")
            print("\nPlease train a model first using one of:")
            print("   python multi_label/train_multilabel.py --config multi_label/config_single_task.yaml")
            print("   python multi_label/train_dual_task.py --config multi_label/config_dual_task.yaml")
            sys.exit(1)
    
    test_file = config['paths']['test_file']
    output_file = args.output if args.output else 'multi_label/results/test_predictions_multi.csv'
    
    print(f"\nModel directory: {model_dir}")
    print(f"Test file: {test_file}")
    print(f"Output file: {output_file}")
    
    # Create results directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load test dataset
    print(f"\nLoading test dataset...")
    test_dataset = MultiLabelABSADataset(
        csv_file=test_file,
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training'].get('per_device_eval_batch_size', 64),
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    print(f"\nLoading model...")
    model = MultiLabelViSoBERT(
        model_name=config['model']['name'],
        num_aspects=len(test_dataset.aspects),
        num_sentiments=3,
        hidden_size=config['model']['hidden_size'],
        dropout=config['model']['dropout']
    )
    
    # Load best checkpoint
    checkpoint_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load with strict=False to ignore pooler keys (old model has pooler, new doesn't)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if unexpected_keys:
        print(f"WARNING: Ignored unexpected keys (old pooler): {unexpected_keys}")
    if missing_keys:
        print(f"WARNING: Missing keys: {missing_keys}")
    
    model.to(device)
    
    # Get metrics
    epoch = checkpoint.get('epoch', 'unknown')
    metrics = checkpoint.get('metrics', {})
    f1_score = metrics.get('overall_f1', 0) * 100 if 'overall_f1' in metrics else 0
    
    print(f"Model loaded (Epoch {epoch}, F1: {f1_score:.2f}%)")
    
    # Aspect and sentiment names
    aspect_names = test_dataset.aspects
    sentiment_names = ['positive', 'negative', 'neutral']
    
    # Load test DataFrame (for text)
    test_df = pd.read_csv(test_file, encoding='utf-8-sig')
    
    # Generate predictions
    print(f"\nGenerating predictions...")
    pred_df, true_df = generate_predictions(model, test_loader, device, aspect_names, sentiment_names, test_df)
    
    # Save predictions in wide format
    print(f"\nSaving predictions in MULTI-LABEL WIDE FORMAT...")

    # Primary output (for analysis scripts expecting the original path)
    pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"   Predictions saved: {output_file}")

    # Optional suffix files for clarity/backward compatibility
    pred_output = output_file.replace('.csv', '_pred.csv')
    if pred_output != output_file:
        pred_df.to_csv(pred_output, index=False, encoding='utf-8-sig')
        print(f"   Predictions saved (alias): {pred_output}")
    
    # Save true sentiments (for reference)
    true_output = output_file.replace('.csv', '_true.csv')
    true_df.to_csv(true_output, index=False, encoding='utf-8-sig')
    print(f"   True labels saved: {true_output}")
    
    print(f"\nPREDICTIONS SAVED IN MULTI-LABEL FORMAT!")
    print(f"   Format: Wide format (one row per sentence, all aspects as columns)")
    print(f"   Total sentences: {len(pred_df)}")
    print(f"   Aspects per sentence: {len(aspect_names)}")
    
    # Calculate accuracy for each aspect
    print(f"\nAccuracy by Aspect:")
    total_correct = 0
    total_predictions = 0
    
    for aspect in aspect_names:
        mask = true_df[aspect].notna()
        total = mask.sum()
        if total == 0:
            print(f"   {aspect:15s}:   N/A   (no ground-truth labels)")
            continue
        correct = (pred_df.loc[mask, aspect] == true_df.loc[mask, aspect]).sum()
        accuracy = correct / total * 100
        print(f"   {aspect:15s}: {accuracy:5.2f}% ({correct}/{total})")
        total_correct += correct
        total_predictions += total
    
    # Overall accuracy
    if total_predictions > 0:
        overall_accuracy = total_correct / total_predictions * 100
        print(f"\n   {'Overall':15s}: {overall_accuracy:5.2f}% ({total_correct}/{total_predictions})")
    else:
        print(f"\n   {'Overall':15s}:   N/A   (no ground-truth labels)")
    
    # Sentiment distribution
    print(f"\nSentiment Distribution:")
    print(f"\n   Predicted:")
    for sentiment in ['Positive', 'Negative', 'Neutral']:
        count = (pred_df[aspect_names] == sentiment).sum().sum()
        pct = count / (len(pred_df) * len(aspect_names)) * 100
        print(f"      {sentiment:8s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n   True:")
    total_true_labels = true_df[aspect_names].notna().sum().sum()
    if total_true_labels == 0:
        print(f"      (no ground-truth labels)")
    else:
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            count = ((true_df[aspect_names] == sentiment) & true_df[aspect_names].notna()).sum().sum()
            pct = count / total_true_labels * 100 if total_true_labels else 0
            print(f"      {sentiment:8s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"DONE! Multi-label predictions saved in wide format.")
    print(f"   You can now use these files for multi-label evaluation.")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Test Predictions CSV')
    parser.add_argument('--config', type=str, default='multi_label/config_multi.yaml',
                        help='Path to config file')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model directory (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: multi_label/results/test_predictions_multi.csv)')
    
    args = parser.parse_args()
    main(args)
