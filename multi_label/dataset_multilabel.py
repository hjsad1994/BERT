"""
Multi-Label Dataset for Vietnamese ABSA
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MultiLabelABSADataset(Dataset):
    """
    Multi-Label ABSA Dataset
    
    Returns all 13 aspect sentiments for each review
    """
    
    def __init__(self, csv_file, tokenizer, max_length=256):
        """
        Args:
            csv_file: Path to CSV file (train_multilabel.csv, etc.)
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
        
        # Sentiment to ID mapping
        self.sentiment_map = {
            'Positive': 0,
            'Negative': 1,
            'Neutral': 2
        }
        
        print(f"Loaded {len(self.df)} samples from {csv_file}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            labels: [num_aspects] (sentiment IDs)
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
        
        # Get labels for all aspects
        labels = []
        for aspect in self.aspects:
            sentiment = row[aspect]
            
            # Handle missing or NaN values
            if pd.isna(sentiment):
                label_id = 2  # Default to Neutral
            else:
                sentiment = str(sentiment).strip()
                label_id = self.sentiment_map.get(sentiment, 2)
            
            labels.append(label_id)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def get_aspect_counts(self):
        """Get sentiment counts per aspect"""
        counts = {}
        
        for aspect in self.aspects:
            aspect_sentiments = self.df[aspect].fillna('Neutral')
            counts[aspect] = aspect_sentiments.value_counts().to_dict()
        
        return counts
    
    def get_label_weights(self):
        """
        Calculate class weights for imbalanced data
        
        Returns:
            weights: [num_aspects, num_sentiments]
        """
        weights = []
        
        for aspect in self.aspects:
            # Count sentiments for this aspect
            sentiments = self.df[aspect].fillna('Neutral')
            sentiment_counts = sentiments.value_counts()
            
            # Calculate weights (inverse frequency)
            total = len(sentiments)
            aspect_weights = []
            
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                count = sentiment_counts.get(sentiment, 1)  # At least 1
                weight = total / (count * 3)  # 3 classes
                aspect_weights.append(weight)
            
            weights.append(aspect_weights)
        
        return torch.tensor(weights, dtype=torch.float32)


def test_dataset():
    """Test dataset loading"""
    print("=" * 80)
    print("Testing Multi-Label Dataset")
    print("=" * 80)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert")
    
    # Create dataset
    print("\n2. Creating dataset...")
    try:
        dataset = MultiLabelABSADataset(
            csv_file='data/train_multilabel.csv',
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
        print(f"   Labels: {sample['labels']}")
        
        # Decode text
        text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"   Text: {text[:100]}...")
        
        # Get label distribution
        print("\n4. Analyzing label distribution...")
        counts = dataset.get_aspect_counts()
        
        print(f"\n   Per-aspect sentiment distribution:")
        for aspect, aspect_counts in counts.items():
            total = sum(aspect_counts.values())
            print(f"   {aspect:<15} Total: {total}")
            for sentiment, count in aspect_counts.items():
                print(f"      {sentiment:<10} {count:4d} ({count/total*100:5.1f}%)")
        
        # Calculate weights
        print("\n5. Calculating class weights...")
        weights = dataset.get_label_weights()
        print(f"   Weight shape: {weights.shape}")
        print(f"   Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        print("\n✅ Dataset tests passed!")
        
    except FileNotFoundError:
        print("   ⚠️  File not found: data/train_multilabel.csv")
        print("   Please run: python prepare_data_multilabel.py first")
    
    print("=" * 80)


if __name__ == '__main__':
    test_dataset()
