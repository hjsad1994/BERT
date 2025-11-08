"""
Multi-Label Aspect Detection Dataset for Vietnamese ABSA
Binary classification dataset: Detects which aspects are mentioned
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class AspectDetectionDataset(Dataset):
    """
    Multi-Label Aspect Detection Dataset
    
    Returns binary labels for 11 aspects (including "Others"):
        0 = aspect not mentioned (NaN/empty)
        1 = aspect mentioned (has any sentiment: positive/negative/neutral)
    """
    
    def __init__(self, csv_file, tokenizer, max_length=256):
        """
        Args:
            csv_file: Path to CSV file (train/val/test_multilabel.csv)
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.df = pd.read_csv(csv_file, encoding='utf-8-sig')
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Aspect columns in order (including "Others" for AD stage)
        self.aspects = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        print(f"Loaded {len(self.df)} samples from {csv_file}")
        self._print_statistics()
    
    def _print_statistics(self):
        """Print dataset statistics"""
        total_aspects = len(self.df) * len(self.aspects)
        mentioned_count = 0
        not_mentioned_count = 0
        
        for aspect in self.aspects:
            mentioned = self.df[aspect].notna().sum()
            not_mentioned = self.df[aspect].isna().sum()
            mentioned_count += mentioned
            not_mentioned_count += not_mentioned
        
        print(f"   Binary label distribution:")
        print(f"      Mentioned (1):     {mentioned_count:,} ({mentioned_count/total_aspects*100:.1f}%)")
        print(f"      Not mentioned (0): {not_mentioned_count:,} ({not_mentioned_count/total_aspects*100:.1f}%)")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            labels: [num_aspects] (binary: 0 = not mentioned, 1 = mentioned)
            loss_mask: [num_aspects] (1.0 for all - we train on both positive and negative examples)
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
        labels = []
        masks = []
        
        for aspect in self.aspects:
            aspect_value = row[aspect]
            
            # Binary label: 1 if aspect mentioned (has any value), 0 if not mentioned (NaN)
            if pd.isna(aspect_value) or str(aspect_value).strip() == '':
                label = 0  # Not mentioned
            else:
                label = 1  # Mentioned (regardless of sentiment)
            
            labels.append(label)
            masks.append(1.0)  # Train on all aspects (both positive and negative examples)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float),  # Float for BCE loss
            'loss_mask': torch.tensor(masks, dtype=torch.float)
        }
    
    def get_label_distribution(self):
        """Get per-aspect label distribution"""
        stats = []
        
        for aspect in self.aspects:
            mentioned = self.df[aspect].notna().sum()
            not_mentioned = self.df[aspect].isna().sum()
            total = len(self.df)
            
            stats.append({
                'aspect': aspect,
                'mentioned': mentioned,
                'not_mentioned': not_mentioned,
                'mentioned_ratio': mentioned / total,
                'imbalance_ratio': not_mentioned / mentioned if mentioned > 0 else float('inf')
            })
        
        return pd.DataFrame(stats)
    
    def get_pos_weights(self):
        """
        Calculate positive class weights for imbalanced binary data
        Used for BCE loss pos_weight parameter
        
        Returns:
            pos_weights: [num_aspects] (weight for positive class per aspect)
        """
        pos_weights = []
        
        for aspect in self.aspects:
            n_mentioned = self.df[aspect].notna().sum()
            n_not_mentioned = self.df[aspect].isna().sum()
            
            # Weight = n_negative / n_positive (higher weight for rare positive class)
            if n_mentioned > 0:
                weight = n_not_mentioned / n_mentioned
            else:
                weight = 1.0
            
            pos_weights.append(weight)
        
        return torch.tensor(pos_weights, dtype=torch.float32)
