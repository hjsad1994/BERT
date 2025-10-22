"""
Generate Test Predictions
=========================
Generate predictions on test set from trained model

Usage:
    python generate_test_predictions.py
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def main():
    print("="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    
    # Config
    model_dir = "finetuned_visobert_absa_model"
    test_file = "data/test.csv"
    output_file = "test_predictions.csv"  # Will overwrite with new predictions
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    print("OK Model loaded")
    
    # Load test data
    print(f"\nLoading test data: {test_file}")
    df = pd.read_csv(test_file)
    print(f"OK {len(df)} samples")
    
    # Label map
    id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    # Generate predictions
    print(f"\nGenerating predictions...")
    predictions = []
    confidences = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Tokenize
        inputs = tokenizer(
            row['sentence'],
            row['aspect'],
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_id = probs.argmax().item()
            confidence = probs.max().item()
        
        predictions.append(id2label[pred_id])
        confidences.append(confidence)
    
    # Add to dataframe
    df['predicted_sentiment'] = predictions
    df['confidence'] = confidences
    
    # Save
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nOK Saved to: {output_file}")
    
    # Quick stats
    print(f"\nQuick Stats:")
    print(f"  Total samples: {len(df)}")
    print(f"  Avg confidence: {sum(confidences)/len(confidences):.4f}")
    
    # Accuracy
    if 'sentiment' in df.columns:
        correct = (df['sentiment'] == df['predicted_sentiment']).sum()
        accuracy = correct / len(df)
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(df)})")
    
    print(f"\nDone! You can now run:")
    print(f"  python error_analysis.py")
    print(f"  python analyze_results.py")

if __name__ == '__main__':
    main()
