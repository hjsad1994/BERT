"""
Focal Loss for Multi-Label ABSA
================================
Handles class imbalance by focusing on hard examples

Usage:
    focal_loss = MultilabelFocalLoss(alpha=[0.8, 1.0, 2.5], gamma=2.0)
    loss = focal_loss(logits, labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label ABSA
    
    Applies Focal Loss to each aspect independently with GLOBAL alpha weights.
    
    Args:
        alpha: List of 3 weights [w_positive, w_negative, w_neutral]
               or "auto" to compute from data
        gamma: Focusing parameter (default: 2.0)
        num_aspects: Number of aspects (default: 11)
    
    Example:
        >>> focal_loss = MultilabelFocalLoss(alpha=[0.8, 1.0, 2.5], gamma=2.0)
        >>> logits = torch.randn(16, 11, 3)  # [batch, aspects, sentiments]
        >>> labels = torch.randint(0, 3, (16, 11))  # [batch, aspects]
        >>> loss = focal_loss(logits, labels)
    """
    
    def __init__(self, alpha=[1.0, 1.0, 1.0], gamma=2.0, num_aspects=11):
        super().__init__()
        
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.num_aspects = num_aspects
        
        print(f"✓ MultilabelFocalLoss initialized:")
        print(f"   Alpha: {alpha}")
        print(f"   Gamma: {gamma}")
        print(f"   Num aspects: {num_aspects}")
    
    def forward(self, logits, labels):
        """
        Compute Focal Loss for multi-label ABSA
        
        Args:
            logits: [batch_size, num_aspects, num_sentiments]
            labels: [batch_size, num_aspects]
        
        Returns:
            loss: scalar
        """
        batch_size = logits.size(0)
        num_aspects = logits.size(1)
        
        # Move alpha to same device as logits
        if not isinstance(self.alpha, torch.Tensor):
            self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        
        alpha = self.alpha.to(logits.device)
        
        total_loss = 0.0
        
        # Loop over aspects (apply same alpha to all aspects)
        for aspect_idx in range(num_aspects):
            aspect_logits = logits[:, aspect_idx, :]  # [batch, 3]
            aspect_labels = labels[:, aspect_idx]      # [batch]
            
            # Softmax probabilities
            probs = F.softmax(aspect_logits, dim=1)  # [batch, 3]
            
            # Get probability of correct class
            class_probs = probs[range(batch_size), aspect_labels]  # [batch]
            
            # Focal weight: (1 - p_t)^gamma
            focal_weight = (1.0 - class_probs) ** self.gamma
            
            # Alpha weight for each sample
            alpha_weight = alpha[aspect_labels]  # [batch]
            
            # Cross entropy loss (without reduction)
            ce_loss = F.cross_entropy(aspect_logits, aspect_labels, reduction='none')
            
            # Focal loss = alpha * (1-p)^gamma * CE
            focal_loss = alpha_weight * focal_weight * ce_loss
            
            # Average over batch
            total_loss += focal_loss.mean()
        
        # Average over aspects
        return total_loss / num_aspects


def calculate_global_alpha(train_file_path, aspect_cols, sentiment_to_idx):
    """
    Calculate global alpha weights from training data
    
    Args:
        train_file_path: Path to training CSV file
        aspect_cols: List of aspect column names
        sentiment_to_idx: {'positive': 0, 'negative': 1, 'neutral': 2}
    
    Returns:
        alpha: [alpha_pos, alpha_neg, alpha_neu]
    """
    print(f"\n📊 Calculating global alpha weights...")
    
    # Load training data
    import pandas as pd
    df = pd.read_csv(train_file_path, encoding='utf-8-sig')
    
    # Collect all sentiments from all aspects
    from collections import Counter
    all_sentiments = []
    
    for aspect in aspect_cols:
        if aspect in df.columns:
            sentiments = df[aspect].dropna()
            # Convert to lowercase and strip
            sentiments = sentiments.astype(str).str.strip().str.lower()
            all_sentiments.extend(sentiments.tolist())
    
    # Count sentiments
    counts = Counter(all_sentiments)
    total = sum(counts.values())
    
    print(f"\n   Total aspect-sentiment pairs: {total:,}")
    print(f"\n   Sentiment distribution:")
    for sentiment in ['positive', 'negative', 'neutral']:
        count = counts.get(sentiment, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"     {sentiment:10}: {count:6,} ({pct:5.2f}%)")
    
    # Calculate inverse frequency alpha
    alpha = []
    num_classes = len(sentiment_to_idx)
    
    for sentiment in ['positive', 'negative', 'neutral']:
        count = counts.get(sentiment, 1)  # Avoid division by zero
        # Inverse frequency: total / (num_classes * count)
        weight = total / (num_classes * count)
        alpha.append(weight)
    
    print(f"\n   Calculated alpha (inverse frequency):")
    for sentiment, weight in zip(['positive', 'negative', 'neutral'], alpha):
        print(f"     {sentiment:10}: {weight:.4f}")
    
    return alpha


# Test
if __name__ == '__main__':
    print("Testing MultilabelFocalLoss...")
    
    # Create dummy data
    batch_size = 4
    num_aspects = 11
    num_sentiments = 3
    
    logits = torch.randn(batch_size, num_aspects, num_sentiments)
    labels = torch.randint(0, num_sentiments, (batch_size, num_aspects))
    
    print(f"\nInput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Labels: {labels.shape}")
    
    # Test with different alpha
    print(f"\n1. Equal weights (no class weighting):")
    focal_loss = MultilabelFocalLoss(alpha=[1.0, 1.0, 1.0], gamma=2.0)
    loss = focal_loss(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    print(f"\n2. Weighted (boost neutral):")
    focal_loss = MultilabelFocalLoss(alpha=[0.8, 1.0, 2.5], gamma=2.0)
    loss = focal_loss(logits, labels)
    print(f"   Loss: {loss.item():.4f}")
    
    print(f"\n✓ Test passed!")
