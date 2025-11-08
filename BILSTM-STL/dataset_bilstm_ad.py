"""
Dataset Loader for BiLSTM Aspect Detection

Single-task learning: Only Aspect Detection (binary multi-label)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class AspectDetectionDataset(Dataset):
    """
    Dataset for Aspect Detection Task
    
    Returns binary labels for 11 fixed aspect categories:
        1 = aspect is present (has sentiment label)
        0 = aspect is absent (NaN/empty)
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
        
        # Aspect columns in order (11 aspects)
        self.aspects = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        print(f"[AspectDetectionDataset] Loaded {len(self.df)} samples from {csv_file}")
        
        # Print dataset statistics
        self._print_statistics()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_ids: [max_length]
            attention_mask: [max_length]
            labels: [num_aspects] - binary labels (1=present, 0=absent)
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
        
        # Binary aspect detection labels
        labels = []
        
        for aspect in self.aspects:
            sentiment = row[aspect]
            
            # Check if aspect is present (has any sentiment label)
            if pd.isna(sentiment) or str(sentiment).strip() == '':
                labels.append(0)  # Absent
            else:
                labels.append(1)  # Present
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float32)  # Float for BCEWithLogitsLoss
        }
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print(f"\n[Dataset Statistics]")
        
        total_aspects = 0
        present_aspects = 0
        
        aspect_counts = {aspect: 0 for aspect in self.aspects}
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            for aspect in self.aspects:
                total_aspects += 1
                sentiment = row[aspect]
                if not pd.isna(sentiment) and str(sentiment).strip() != '':
                    present_aspects += 1
                    aspect_counts[aspect] += 1
        
        print(f"  Total samples: {len(self.df)}")
        print(f"  Total aspect slots: {total_aspects:,}")
        print(f"  Present aspects: {present_aspects:,} ({present_aspects/total_aspects*100:.1f}%)")
        print(f"  Absent aspects: {total_aspects-present_aspects:,} ({(total_aspects-present_aspects)/total_aspects*100:.1f}%)")
        
        print(f"\n  Per-aspect distribution:")
        for aspect in self.aspects:
            count = aspect_counts[aspect]
            percentage = count / len(self.df) * 100
            print(f"    {aspect:<15} {count:>5} samples ({percentage:>5.1f}%)")
    
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced data
        
        Returns:
            weights: [num_aspects, 2] - weights for [absent, present] for each aspect
        """
        weights = []
        
        for aspect in self.aspects:
            # Count present and absent
            sentiments = self.df[aspect]
            n_total = len(sentiments)
            n_present = sentiments.notna().sum()
            n_absent = n_total - n_present
            
            # Calculate weights (inverse frequency)
            if n_absent > 0 and n_present > 0:
                weight_absent = n_total / (2 * n_absent)
                weight_present = n_total / (2 * n_present)
            else:
                weight_absent = 1.0
                weight_present = 1.0
            
            weights.append([weight_absent, weight_present])
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_pos_weight(self):
        """
        Calculate pos_weight for BCEWithLogitsLoss
        
        pos_weight: [num_aspects] - ratio of negative to positive samples for each aspect
        """
        pos_weights = []
        
        for aspect in self.aspects:
            sentiments = self.df[aspect]
            n_total = len(sentiments)
            n_present = sentiments.notna().sum()
            n_absent = n_total - n_present
            
            # pos_weight = #negative / #positive
            if n_present > 0:
                pos_weight = n_absent / n_present
            else:
                pos_weight = 1.0
            
            pos_weights.append(pos_weight)
        
        return torch.tensor(pos_weights, dtype=torch.float32)


def test_dataset():
    """Test dataset loading"""
    print("=" * 80)
    print("Testing Aspect Detection Dataset")
    print("=" * 80)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/visobert-14gb-corpus")
    
    # Create dataset
    print("\n2. Creating dataset...")
    try:
        dataset = AspectDetectionDataset(
            csv_file='../dual-task-learning/data/train_multilabel.csv',
            tokenizer=tokenizer,
            max_length=256
        )
        
        print(f"\n3. Dataset size: {len(dataset)}")
        
        # Test loading
        print("\n4. Testing sample loading...")
        sample = dataset[0]
        
        print(f"   input_ids shape: {sample['input_ids'].shape}")
        print(f"   attention_mask shape: {sample['attention_mask'].shape}")
        print(f"   labels shape: {sample['labels'].shape}")
        print(f"   labels: {sample['labels']}")
        
        # Decode text
        text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        print(f"   Text: {text[:100]}...")
        
        # Get class weights
        print("\n5. Calculating class weights...")
        class_weights = dataset.get_class_weights()
        print(f"   Class weights shape: {class_weights.shape}")
        print(f"   Class weights (first 3 aspects):")
        for i, aspect in enumerate(dataset.aspects[:3]):
            print(f"     {aspect}: Absent={class_weights[i][0]:.3f}, Present={class_weights[i][1]:.3f}")
        
        # Get pos_weight
        print("\n6. Calculating pos_weight...")
        pos_weight = dataset.get_pos_weight()
        print(f"   pos_weight shape: {pos_weight.shape}")
        print(f"   pos_weight (first 5 aspects): {pos_weight[:5]}")
        
        print("\n✓ Dataset tests passed!")
        
    except FileNotFoundError:
        print("   WARNING: File not found: ../dual-task-learning/data/train_multilabel.csv")
        print("   Please run: python prepare_data_multilabel.py first")
    
    print("=" * 80)


if __name__ == '__main__':
    test_dataset()
