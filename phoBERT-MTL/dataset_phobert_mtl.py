"""
Dataset Loader for Multi-Task Learning

Returns both AD labels (binary) and SC labels (multi-class) for joint training
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class MTLDataset(Dataset):
    """
    Multi-Task Learning Dataset
    
    Returns both:
        - AD labels: Binary (0=absent, 1=present) for 11 aspects
        - SC labels: Multi-class (0=pos, 1=neg, 2=neu) for 11 aspects
        - SC loss mask: Mask for unlabeled aspects
    """
    
    def __init__(self, csv_file, tokenizer, max_length=256):
        """
        Args:
            csv_file: Path to CSV file
            tokenizer: Tokenizer
            max_length: Maximum sequence length
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
        
        # Sentiment mapping
        self.sentiment_map = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }
        
        print(f"[MTLDataset] Loaded {len(self.df)} samples from {csv_file}")
        self._print_statistics()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            ad_labels: [num_aspects] - binary (0=absent, 1=present)
            sc_labels: [num_aspects] - multi-class (0/1/2 or default 2 for neutral)
            sc_loss_mask: [num_aspects] - mask for labeled aspects (1.0=train, 0.0=skip)
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
        ad_labels = []
        sc_labels = []
        sc_masks = []
        
        for aspect in self.aspects:
            sentiment = row[aspect]
            
            # AD label (binary): 1 if present, 0 if absent
            if pd.isna(sentiment) or str(sentiment).strip() == '':
                ad_label = 0  # Absent
                sc_label = 2  # Default to neutral (not used in loss)
                sc_mask = 0.0  # Don't train SC on this aspect
            else:
                ad_label = 1  # Present
                
                # SC label (multi-class): map sentiment to class
                sentiment_str = str(sentiment).strip().lower()
                sc_label = self.sentiment_map.get(sentiment_str, 2)  # Default neutral
                sc_mask = 1.0  # Train SC on this aspect
            
            ad_labels.append(ad_label)
            sc_labels.append(sc_label)
            sc_masks.append(sc_mask)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ad_labels': torch.tensor(ad_labels, dtype=torch.float32),  # Float for BCE
            'sc_labels': torch.tensor(sc_labels, dtype=torch.long),     # Long for CE
            'sc_loss_mask': torch.tensor(sc_masks, dtype=torch.float32)
        }
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\n[Dataset Statistics]")
        
        total_aspects = len(self.df) * len(self.aspects)
        present_count = 0
        absent_count = 0
        
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            for aspect in self.aspects:
                sentiment = row[aspect]
                
                if pd.isna(sentiment) or str(sentiment).strip() == '':
                    absent_count += 1
                else:
                    present_count += 1
                    sentiment_str = str(sentiment).strip().lower()
                    if sentiment_str in sentiment_counts:
                        sentiment_counts[sentiment_str] += 1
        
        print(f"  Total samples: {len(self.df)}")
        print(f"  Total aspect slots: {total_aspects:,}")
        print(f"\n  AD task (binary):")
        print(f"    Present: {present_count:,} ({present_count/total_aspects*100:.1f}%)")
        print(f"    Absent:  {absent_count:,} ({absent_count/total_aspects*100:.1f}%)")
        print(f"\n  SC task (multi-class, only on present aspects):")
        print(f"    Positive: {sentiment_counts['positive']:,} ({sentiment_counts['positive']/present_count*100:.1f}%)")
        print(f"    Negative: {sentiment_counts['negative']:,} ({sentiment_counts['negative']/present_count*100:.1f}%)")
        print(f"    Neutral:  {sentiment_counts['neutral']:,} ({sentiment_counts['neutral']/present_count*100:.1f}%)")
    
    def get_ad_pos_weight(self):
        """
        Calculate pos_weight for AD task (BCE loss)
        
        Returns:
            pos_weight: [num_aspects] - weight for positive class
        """
        pos_weights = []
        
        for aspect in self.aspects:
            n_present = self.df[aspect].notna().sum()
            n_absent = self.df[aspect].isna().sum()
            
            # pos_weight = n_negative / n_positive
            if n_present > 0:
                weight = n_absent / n_present
            else:
                weight = 1.0
            
            pos_weights.append(weight)
        
        return torch.tensor(pos_weights, dtype=torch.float32)
    
    def get_sc_class_weights(self):
        """
        Calculate class weights for SC task (CE loss)
        
        Returns:
            class_weights: [3] - weights for [positive, negative, neutral]
        """
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            for aspect in self.aspects:
                sentiment = row[aspect]
                if not pd.isna(sentiment) and str(sentiment).strip() != '':
                    sentiment_str = str(sentiment).strip().lower()
                    if sentiment_str in sentiment_counts:
                        sentiment_counts[sentiment_str] += 1
        
        total = sum(sentiment_counts.values())
        
        if total > 0:
            weights = [
                total / (3 * sentiment_counts['positive']) if sentiment_counts['positive'] > 0 else 1.0,
                total / (3 * sentiment_counts['negative']) if sentiment_counts['negative'] > 0 else 1.0,
                total / (3 * sentiment_counts['neutral']) if sentiment_counts['neutral'] > 0 else 1.0
            ]
        else:
            weights = [1.0, 1.0, 1.0]
        
        return torch.tensor(weights, dtype=torch.float32)


def test_dataset():
    """Test MTL dataset"""
    print("="*80)
    print("Testing MTL Dataset")
    print("="*80)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Create dataset
    print("\n2. Creating dataset...")
    try:
        dataset = MTLDataset(
            csv_file='data/train_multilabel.csv',
            tokenizer=tokenizer,
            max_length=256
        )
        
        print(f"   Dataset size: {len(dataset)}")
        
        # Test sample
        print("\n3. Testing sample loading...")
        sample = dataset[0]
        
        print(f"   Input IDs shape: {sample['input_ids'].shape}")
        print(f"   Attention mask shape: {sample['attention_mask'].shape}")
        print(f"   AD labels shape: {sample['ad_labels'].shape}")
        print(f"   SC labels shape: {sample['sc_labels'].shape}")
        print(f"   SC mask shape: {sample['sc_loss_mask'].shape}")
        
        print(f"\n   AD labels (binary): {sample['ad_labels']}")
        print(f"   SC labels (0/1/2): {sample['sc_labels']}")
        print(f"   SC mask (train/skip): {sample['sc_loss_mask']}")
        
        # Get weights
        print("\n4. Calculating class weights...")
        ad_pos_weight = dataset.get_ad_pos_weight()
        sc_class_weight = dataset.get_sc_class_weights()
        
        print(f"   AD pos_weight: {ad_pos_weight[:3].tolist()}...")
        print(f"   SC class_weight: {sc_class_weight.tolist()}")
        
        print("\nDataset tests passed!")
        
    except FileNotFoundError as e:
        print(f"   WARNING: {e}")
        print("   Please ensure data files exist")
    
    print("="*80)


if __name__ == '__main__':
    test_dataset()

