"""
Multi-Label Aspect Detection Dataset for Vietnamese
Detects which aspects are mentioned in the text (binary classification)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AspectDetectionDataset(Dataset):
    """
    Multi-Label Aspect Detection Dataset
    
    Returns binary labels for all 11 aspects: 0 = not mentioned, 1 = mentioned
    """
    
    def __init__(self, csv_file, tokenizer, max_length=256):
        """
        Args:
            csv_file: Path to CSV file (train_aspect_detection.csv, etc.)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Aspect columns in order
        self.aspects = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        print(f"Loaded {len(self.df)} samples from {csv_file}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            labels: [num_aspects] (binary: 0 = not mentioned, 1 = mentioned)
            loss_mask: [num_aspects] (1.0 for labeled, 0.0 for NaN)
        """
        row = self.df.iloc[idx]
        
        # Get text
        text = str(row['data'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get binary labels for all aspects
        # 0 = aspect not mentioned (NaN or empty)
        # 1 = aspect mentioned (any value: Positive, Negative, Neutral)
        labels = []
        masks = []
        
        for aspect in self.aspects:
            aspect_value = row[aspect]
            
            # Check if aspect is mentioned (has any value, including Positive/Negative/Neutral)
            if pd.isna(aspect_value) or str(aspect_value).strip() == '':
                # Aspect not mentioned
                label = 0  # Not mentioned
                mask = 1.0  # Still train on this (negative example)
            else:
                # Aspect mentioned (regardless of sentiment)
                label = 1  # Mentioned
                mask = 1.0  # Train on this (positive example)
            
            labels.append(label)
            masks.append(mask)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float),  # Binary: float for BCE loss
            'loss_mask': torch.tensor(masks, dtype=torch.float)
        }
    
    def get_aspect_counts(self):
        """Get binary counts per aspect (mentioned vs not mentioned)"""
        counts = {}
        
        for aspect in self.aspects:
            # Count: 0 = not mentioned (NaN), 1 = mentioned (any value)
            n_mentioned = self.df[aspect].notna().sum()
            n_not_mentioned = self.df[aspect].isna().sum()
            
            counts[aspect] = {
                'mentioned': n_mentioned,
                'not_mentioned': n_not_mentioned,
                'total': len(self.df)
            }
        
        return counts
    
    def get_label_weights(self):
        """
        Calculate class weights for imbalanced binary data
        
        Returns:
            weights: [num_aspects, 2] (weights for [not_mentioned, mentioned])
        """
        weights = []
        
        for aspect in self.aspects:
            # Count positive (mentioned) and negative (not mentioned)
            n_mentioned = self.df[aspect].notna().sum()
            n_not_mentioned = self.df[aspect].isna().sum()
            total = len(self.df)
            
            # Calculate weights (inverse frequency)
            if n_mentioned > 0 and n_not_mentioned > 0:
                weight_mentioned = total / (2 * n_mentioned)
                weight_not_mentioned = total / (2 * n_not_mentioned)
            else:
                weight_mentioned = 1.0
                weight_not_mentioned = 1.0
            
            weights.append([weight_not_mentioned, weight_mentioned])
        
        return torch.tensor(weights, dtype=torch.float32)


def test_dataset():
    """Test dataset loading"""
    print("=" * 80)
    print("Testing Aspect Detection Dataset")
    print("=" * 80)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert")
    
    # Create dataset
    print("\n2. Creating dataset...")
    try:
        dataset = AspectDetectionDataset(
            csv_file='aspect-detection/data/train_aspect_detection.csv',
            tokenizer=tokenizer,
            max_length=256
        )
        
        print(f"   Dataset size: {len(dataset)}")
        
        # Test loading
        print("\n3. Testing sample loading...")
        sample = dataset[0]
        
        print(f"   Input IDs shape: {sample['input_ids'].shape}")
        print(f"   Attention mask shape: {sample['attention_mask'].shape}")
        print(f"   Labels shape: {sample['labels'].shape}")
        print(f"   Labels (binary): {sample['labels']}")
        
        # Decode text
        text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"   Text: {text[:100]}...")
        
        # Get label distribution
        print("\n4. Analyzing label distribution...")
        counts = dataset.get_aspect_counts()
        
        print(f"\n   Per-aspect binary distribution:")
        for aspect, aspect_counts in counts.items():
            total = aspect_counts['total']
            mentioned = aspect_counts['mentioned']
            not_mentioned = aspect_counts['not_mentioned']
            print(f"   {aspect:<15} Total: {total}")
            print(f"      {'Mentioned':<10} {mentioned:4d} ({mentioned/total*100:5.1f}%)")
            print(f"      {'Not Mentioned':<10} {not_mentioned:4d} ({not_mentioned/total*100:5.1f}%)")
        
        # Calculate weights
        print("\n5. Calculating class weights...")
        weights = dataset.get_label_weights()
        print(f"   Weight shape: {weights.shape}")
        print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        print("\nDataset tests passed!")
        
    except FileNotFoundError as e:
        print(f"   WARNING: File not found: {e}")
        print("   Please prepare data first")
    
    print("=" * 80)


if __name__ == '__main__':
    test_dataset()
