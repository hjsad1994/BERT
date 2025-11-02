"""
Dual-Task Dataset for Vietnamese ABSA
Provides labels for both:
  1. Aspect Detection (binary: present/not present)
  2. Sentiment Classification (3-way: pos/neg/neu when present)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DualTaskABSADataset(Dataset):
    """
    Dual-Task ABSA Dataset
    
    Returns:
        - Detection labels: 1 if aspect mentioned, 0 if not (NaN)
        - Sentiment labels: 0/1/2 for pos/neg/neu (only valid when detected=1)
        - Masks: For both detection and sentiment tasks
    """
    
    def __init__(self, csv_file, tokenizer, max_length=256):
        """
        Args:
            csv_file: Path to CSV (train/val/test)
            tokenizer: BERT tokenizer
            max_length: Max sequence length
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Aspect columns
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
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics for detection and sentiment"""
        total_possible = len(self.df) * len(self.aspects)
        total_detected = 0
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        for aspect in self.aspects:
            aspect_data = self.df[aspect]
            detected = aspect_data.notna().sum()
            total_detected += detected
            
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                sentiment_counts[sentiment] += (aspect_data == sentiment).sum()
        
        total_not_detected = total_possible - total_detected
        
        print(f"   Detection: {total_detected:,} present, {total_not_detected:,} not present " +
              f"({total_detected/total_possible*100:.1f}% positive rate)")
        print(f"   Sentiment: Pos={sentiment_counts['Positive']:,}, " +
              f"Neg={sentiment_counts['Negative']:,}, Neu={sentiment_counts['Neutral']:,}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            detection_labels: [num_aspects] - binary 0/1
            sentiment_labels: [num_aspects] - class 0/1/2 (only valid when detected=1)
            detection_mask: [num_aspects] - always 1.0 (train on all detection)
            sentiment_mask: [num_aspects] - 1.0 only when aspect detected, 0.0 otherwise
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
        
        # Prepare labels for both tasks
        detection_labels = []
        sentiment_labels = []
        detection_masks = []
        sentiment_masks = []
        
        for aspect in self.aspects:
            sentiment = row[aspect]
            
            if pd.isna(sentiment):
                # Aspect NOT present
                detection_labels.append(0)  # Not detected
                sentiment_labels.append(2)  # Placeholder (will be masked)
                detection_masks.append(1.0)  # Train detection on this
                sentiment_masks.append(0.0)  # DON'T train sentiment (no aspect)
            else:
                # Aspect IS present
                sentiment = str(sentiment).strip()
                sentiment_id = self.sentiment_map.get(sentiment, 2)
                
                detection_labels.append(1)  # Detected
                sentiment_labels.append(sentiment_id)  # Actual sentiment
                detection_masks.append(1.0)  # Train detection on this
                sentiment_masks.append(1.0)  # Train sentiment on this
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'detection_labels': torch.tensor(detection_labels, dtype=torch.long),
            'sentiment_labels': torch.tensor(sentiment_labels, dtype=torch.long),
            'detection_mask': torch.tensor(detection_masks, dtype=torch.float),
            'sentiment_mask': torch.tensor(sentiment_masks, dtype=torch.float)
        }
    
    def get_detection_stats(self):
        """Get detection statistics per aspect"""
        stats = {}
        
        for aspect in self.aspects:
            detected = self.df[aspect].notna().sum()
            not_detected = self.df[aspect].isna().sum()
            
            stats[aspect] = {
                'detected': int(detected),
                'not_detected': int(not_detected),
                'detection_rate': float(detected / len(self.df))
            }
        
        return stats
    
    def get_sentiment_stats(self):
        """Get sentiment statistics per aspect (only for detected)"""
        stats = {}
        
        for aspect in self.aspects:
            aspect_data = self.df[aspect].dropna()
            
            if len(aspect_data) == 0:
                stats[aspect] = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
                continue
            
            stats[aspect] = {
                'Positive': int((aspect_data == 'Positive').sum()),
                'Negative': int((aspect_data == 'Negative').sum()),
                'Neutral': int((aspect_data == 'Neutral').sum())
            }
        
        return stats


def test_dataset():
    """Test dual-task dataset"""
    print("=" * 80)
    print("Testing Dual-Task ABSA Dataset")
    print("=" * 80)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/visobert-14gb-corpus")
    
    # Test with a dataset file
    csv_file = "D:/BERT/multi_label/data/train_multilabel.csv"
    
    try:
        dataset = DualTaskABSADataset(csv_file, tokenizer, max_length=128)
        
        print(f"\n✓ Dataset loaded: {len(dataset)} samples")
        
        # Test __getitem__
        print("\nTesting sample retrieval...")
        sample = dataset[0]
        
        print(f"   input_ids shape: {sample['input_ids'].shape}")
        print(f"   attention_mask shape: {sample['attention_mask'].shape}")
        print(f"   detection_labels shape: {sample['detection_labels'].shape}")
        print(f"   sentiment_labels shape: {sample['sentiment_labels'].shape}")
        print(f"   detection_mask shape: {sample['detection_mask'].shape}")
        print(f"   sentiment_mask shape: {sample['sentiment_mask'].shape}")
        
        print(f"\n   Detection labels: {sample['detection_labels'].tolist()}")
        print(f"   Sentiment labels: {sample['sentiment_labels'].tolist()}")
        print(f"   Detection mask: {sample['detection_mask'].tolist()}")
        print(f"   Sentiment mask: {sample['sentiment_mask'].tolist()}")
        
        # Show statistics
        print("\n" + "=" * 80)
        print("Detection Statistics per Aspect:")
        print("=" * 80)
        
        det_stats = dataset.get_detection_stats()
        print(f"{'Aspect':<15} {'Detected':<12} {'Not Detected':<15} {'Rate':<10}")
        print("-" * 60)
        
        for aspect, stats in det_stats.items():
            print(f"{aspect:<15} {stats['detected']:<12} {stats['not_detected']:<15} {stats['detection_rate']*100:>6.2f}%")
        
        print("\n" + "=" * 80)
        print("Sentiment Statistics per Aspect (when detected):")
        print("=" * 80)
        
        sent_stats = dataset.get_sentiment_stats()
        print(f"{'Aspect':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10}")
        print("-" * 50)
        
        for aspect, stats in sent_stats.items():
            print(f"{aspect:<15} {stats['Positive']:<10} {stats['Negative']:<10} {stats['Neutral']:<10}")
        
        print("\n✅ All tests passed!")
        
    except FileNotFoundError:
        print(f"\n⚠️  Dataset file not found: {csv_file}")
        print("   Please run with correct path to test")
    
    print("=" * 80)


if __name__ == '__main__':
    test_dataset()
