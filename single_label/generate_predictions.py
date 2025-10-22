"""
Generate Predictions cho bất kỳ dataset nào
Để dùng cho Error Analysis
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os


def generate_predictions(input_file, output_file, model_path, batch_size=32):
    """
    Generate predictions cho dataset
    
    Args:
        input_file: Path to CSV file (train.csv, validation.csv, test.csv)
        output_file: Path to save predictions
        model_path: Path to trained model
        batch_size: Batch size for inference
    """
    print(f"\n{'='*70}")
    print(f"📊 GENERATING PREDICTIONS")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n📁 Loading data from: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    print(f"✓ Loaded {len(df)} samples")
    
    # Load model and tokenizer
    print(f"\n🤖 Loading model from: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Label mapping
    id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    # Prepare predictions
    predictions = []
    
    print(f"\n🔮 Generating predictions...")
    print(f"   Batch size: {batch_size}")
    print(f"   Total batches: {len(df) // batch_size + 1}")
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(df), batch_size), desc="Predicting"):
            batch_df = df.iloc[i:i+batch_size]
            
            # Prepare inputs
            sentences = batch_df['sentence'].tolist()
            aspects = batch_df['aspect'].tolist()
            
            # Tokenize
            inputs = tokenizer(
                sentences,
                aspects,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_sentiments = [id2label[id] for id in pred_ids]
            
            predictions.extend(pred_sentiments)
    
    # Create predictions dataframe
    pred_df = pd.DataFrame({
        'sentence': df['sentence'],
        'aspect': df['aspect'],
        'true_sentiment': df['sentiment'],
        'predicted_sentiment': predictions
    })
    
    # Calculate accuracy
    accuracy = (pred_df['true_sentiment'] == pred_df['predicted_sentiment']).mean()
    
    # Save
    pred_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✓ Predictions saved to: {output_file}")
    print(f"✓ Accuracy: {accuracy:.2%}")
    print(f"✓ Total samples: {len(pred_df)}")
    
    # Error count
    errors = (pred_df['true_sentiment'] != pred_df['predicted_sentiment']).sum()
    print(f"✓ Errors: {errors} / {len(pred_df)}")
    
    return pred_df


def main():
    parser = argparse.ArgumentParser(description='Generate predictions for Error Analysis')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (e.g., data/train.csv)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output predictions CSV file')
    parser.add_argument('--model', type=str, default='finetuned_visobert_absa_model',
                       help='Path to trained model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Check model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    # Generate predictions
    generate_predictions(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model,
        batch_size=args.batch_size
    )
    
    print(f"\n{'='*70}")
    print(f"✅ DONE!")
    print(f"{'='*70}")
    print(f"\n💡 Now you can run error analysis:")
    print(f"   python error_analysis.py")


if __name__ == '__main__':
    main()
