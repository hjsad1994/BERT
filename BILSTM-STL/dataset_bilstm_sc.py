"""
Dataset Loader for BiLSTM Multi-Label Sentiment Classification

Multi-label: Predicts sentiment (Positive/Negative/Neutral) for each of 11 aspects
Uses loss masking to skip unlabeled/NaN aspects
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SentimentClassificationDataset(Dataset):
    """
    Multi-Label Sentiment Classification Dataset
    
    Returns sentiment labels for 11 aspects:
        0 = Positive
        1 = Negative
        2 = Neutral
    
    Uses loss_mask to indicate labeled (1.0) vs unlabeled/NaN (0.0) aspects
    """
    
    def __init__(self, csv_file, tokenizer, max_length=256):
        """
        Args:
            csv_file: Path to CSV file (train_multilabel.csv, etc.)
            tokenizer: Tokenizer (e.g., ViSoBERT tokenizer)
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Aspect columns in order (11 aspects)
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
        
        # Reverse mapping for display
        self.sentiment_labels = ['Positive', 'Negative', 'Neutral']
        
        print(f"[Multi-Label Sentiment Dataset] Loaded {len(self.df)} samples from {csv_file}")
        
        # Print dataset statistics
        self._print_statistics()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            labels: [num_aspects] - sentiment IDs (0, 1, 2) for each aspect
            loss_mask: [num_aspects] - 1.0 for labeled, 0.0 for NaN/unlabeled
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
        
        # Get labels and masks for all aspects
        labels = []
        masks = []
        
        for aspect in self.aspects:
            sentiment = row[aspect]
            
            # Handle missing or NaN values (UNLABELED - NOT Neutral!)
            if pd.isna(sentiment) or str(sentiment).strip() == '':
                # IMPORTANT: NaN/empty != Neutral!
                # Use label_id = 0 as placeholder (value doesn't matter since mask=0.0)
                # The loss_mask=0.0 ensures this aspect is NOT trained on
                label_id = 0  # Placeholder (unused when mask=0.0)
                mask = 0.0    # MASK = 0 (skip training on unlabeled aspects)
            else:
                sentiment = str(sentiment).strip()
                # Only map valid sentiments: Positive, Negative, Neutral
                label_id = self.sentiment_map.get(sentiment, None)
                if label_id is None:
                    # Invalid sentiment value -> treat as unlabeled
                    label_id = 0  # Placeholder (unused when mask=0.0)
                    mask = 0.0    # Don't train on invalid labels
                else:
                    # Valid labeled aspect (Positive/Negative/Neutral)
                    mask = 1.0    # MASK = 1 (train on this labeled aspect)
            
            labels.append(label_id)
            masks.append(mask)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long),  # [11] - Long for CrossEntropyLoss
            'loss_mask': torch.tensor(masks, dtype=torch.float)  # [11] - Float for masking
        }
    
    def _print_statistics(self):
        """Print dataset statistics (ONLY labeled aspects)"""
        print(f"\n[Dataset Statistics]")
        print(f"  Total samples: {len(self.df)}")
        
        # Count sentiment distribution per aspect
        total_labeled = 0
        total_unlabeled = 0
        
        for aspect in self.aspects:
            # IMPORTANT: Do NOT fillna('Neutral') - NaN is UNLABELED, not Neutral!
            labeled_sentiments = self.df[aspect].dropna()  # Drop NaN (unlabeled)
            n_labeled = len(labeled_sentiments)
            n_unlabeled = len(self.df) - n_labeled
            
            total_labeled += n_labeled
            total_unlabeled += n_unlabeled
        
        print(f"  Total aspect slots: {len(self.aspects) * len(self.df):,}")
        print(f"  Labeled aspects: {total_labeled:,} ({total_labeled/(len(self.aspects)*len(self.df))*100:.1f}%)")
        print(f"  Unlabeled aspects (NaN): {total_unlabeled:,} ({total_unlabeled/(len(self.aspects)*len(self.df))*100:.1f}%)")
        
        print(f"\n  Per-aspect sentiment distribution (labeled only):")
        for aspect in self.aspects:
            labeled_sentiments = self.df[aspect].dropna()
            n_labeled = len(labeled_sentiments)
            
            if n_labeled > 0:
                sentiment_counts = labeled_sentiments.value_counts()
                n_pos = sentiment_counts.get('Positive', 0)
                n_neg = sentiment_counts.get('Negative', 0)
                n_neu = sentiment_counts.get('Neutral', 0)
                
                print(f"    {aspect:<15} Labeled: {n_labeled:>4} | Pos: {n_pos:>3}, Neg: {n_neg:>3}, Neu: {n_neu:>3}")
            else:
                print(f"    {aspect:<15} [No labeled data]")
    
    def get_label_weights(self):
        """
        Calculate class weights for imbalanced data (ONLY labeled aspects)
        
        Returns:
            weights: [num_aspects, num_sentiments] - weights for each aspect-sentiment pair
        """
        weights = []
        
        for aspect in self.aspects:
            # IMPORTANT: Do NOT fillna('Neutral') - NaN is UNLABELED, not Neutral!
            # Only count labeled sentiments (drop NaN/unlabeled)
            sentiments = self.df[aspect].dropna()  # Drop NaN (unlabeled)
            sentiment_counts = sentiments.value_counts()
            
            # Calculate weights (inverse frequency) - only on labeled data
            total = len(sentiments)  # Only labeled count (excluding NaN)
            aspect_weights = []
            
            for sentiment in ['Positive', 'Negative', 'Neutral']:
                count = sentiment_counts.get(sentiment, 1)  # At least 1
                weight = total / (count * 3) if total > 0 else 1.0  # 3 classes
                aspect_weights.append(weight)
            
            weights.append(aspect_weights)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_aspect_counts(self):
        """Get sentiment counts per aspect (ONLY labeled, excludes NaN/unlabeled)"""
        counts = {}
        
        for aspect in self.aspects:
            # IMPORTANT: Do NOT fillna('Neutral') - NaN is UNLABELED, not Neutral!
            # Only count labeled aspects (Positive, Negative, Neutral)
            aspect_sentiments = self.df[aspect].dropna()  # Drop NaN (unlabeled)
            counts[aspect] = aspect_sentiments.value_counts().to_dict()
            
            # Add count for unlabeled
            n_unlabeled = self.df[aspect].isna().sum()
            if n_unlabeled > 0:
                counts[aspect]['Unlabeled'] = n_unlabeled
        
        return counts


def test_dataset():
    """Test dataset loading"""
    print("=" * 80)
    print("Testing Multi-Label Sentiment Classification Dataset")
    print("=" * 80)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/visobert-14gb-corpus")
    
    # Create dataset
    print("\n2. Creating dataset...")
    try:
        dataset = SentimentClassificationDataset(
            csv_file='data/train_multilabel.csv',
            tokenizer=tokenizer,
            max_length=256
        )
        
        print(f"\n3. Dataset size: {len(dataset)}")
        
        # Test loading
        print("\n4. Testing sample loading...")
        sample = dataset[0]
        
        print(f"   input_ids shape: {sample['input_ids'].shape}")
        print(f"   attention_mask shape: {sample['attention_mask'].shape}")
        print(f"   labels shape: {sample['labels'].shape}")  # [11]
        print(f"   loss_mask shape: {sample['loss_mask'].shape}")  # [11]
        print(f"   labels: {sample['labels']}")
        print(f"   loss_mask: {sample['loss_mask']}")
        print(f"   Labeled aspects: {sample['loss_mask'].sum().item()} / 11")
        
        # Decode text
        text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"   Text length: {len(text)} chars")
        
        # Show labeled aspects with sentiments
        print(f"\n   Labeled aspects in this sample:")
        for i, aspect in enumerate(dataset.aspects):
            if sample['loss_mask'][i] == 1.0:
                sentiment = dataset.sentiment_labels[sample['labels'][i]]
                print(f"     {aspect}: {sentiment}")
        
        # Get class weights
        print("\n5. Calculating class weights...")
        class_weights = dataset.get_label_weights()
        print(f"   Class weights shape: {class_weights.shape}")  # [11, 3]
        print(f"   Class weights (first 3 aspects):")
        for i in range(min(3, len(dataset.aspects))):
            aspect = dataset.aspects[i]
            print(f"     {aspect}: Pos={class_weights[i, 0]:.3f}, Neg={class_weights[i, 1]:.3f}, Neu={class_weights[i, 2]:.3f}")
        
        print("\n[OK] Dataset tests passed!")
        
    except FileNotFoundError as e:
        print(f"   WARNING: File not found: {e}")
        print("   Please prepare multi-label data first")
    
    print("=" * 80)


if __name__ == '__main__':
    test_dataset()
