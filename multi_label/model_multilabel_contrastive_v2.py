"""
Multi-Label ABSA Model with Improved Contrastive Learning

Key changes:
- Use SOFT similarity weighting (no hard threshold)
- Focus on NON-NEUTRAL aspects
- Works better with diverse batches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from model_multilabel_contrastive import MultiLabelViSoBERTContrastive

class ImprovedMultiLabelContrastiveLoss(nn.Module):
    """
    Improved Contrastive Loss with Soft Similarity
    
    Changes from original:
    1. No hard threshold - use soft similarity weighting
    2. Focus on non-neutral aspects
    3. Works better with diverse batches
    """
    
    def __init__(self, temperature=0.07, base_weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.base_weight = base_weight  # Minimum weight for all pairs
    
    def forward(self, embeddings, labels):
        """
        Soft contrastive loss
        
        Args:
            embeddings: [batch_size, projection_dim] (L2 normalized)
            labels: [batch_size, num_aspects] (aspect sentiments)
        
        Returns:
            loss: scalar
        """
        batch_size = embeddings.shape[0]
        num_aspects = labels.shape[1]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create SOFT label similarity weights
        # Focus on non-neutral aspects (value != 2)
        label_weights = torch.zeros(batch_size, batch_size, device=labels.device)
        
        for i in range(num_aspects):
            aspect_labels = labels[:, i]  # [batch_size]
            
            # Create match matrix for this aspect
            # aspect_match[i,j] = 1 if same sentiment, 0 otherwise
            aspect_match = (aspect_labels.unsqueeze(1) == aspect_labels.unsqueeze(0)).float()
            
            # Weight by importance: non-neutral aspects are more important
            # If both samples have non-neutral sentiment for this aspect → higher weight
            is_non_neutral = (aspect_labels != 2).float()  # [batch_size]
            importance = is_non_neutral.unsqueeze(1) * is_non_neutral.unsqueeze(0)  # [batch_size, batch_size]
            
            # Add weighted match
            # If both non-neutral and match: weight = 2.0
            # If one neutral and match: weight = 0.5
            # If both neutral and match: weight = 0.1
            weighted_match = aspect_match * (1.0 + importance)
            label_weights += weighted_match
        
        # Normalize by number of aspects
        label_weights = label_weights / (num_aspects * 2.0)  # Divide by 2 because max weight per aspect is 2
        
        # Add base weight to all pairs (avoid zero weights)
        label_weights = label_weights + self.base_weight
        
        # Set diagonal to 0 (exclude self)
        label_weights.fill_diagonal_(0)
        
        # Soft contrastive loss with weighted pairs
        exp_sim = torch.exp(similarity_matrix)
        
        loss = 0.0
        for i in range(batch_size):
            # Weighted positive similarities
            weighted_positive_sim = (exp_sim[i] * label_weights[i]).sum()
            
            # All similarities (exclude self)
            all_sim = exp_sim[i].sum() - exp_sim[i, i]
            
            if weighted_positive_sim > 0 and all_sim > 0:
                loss += -torch.log(weighted_positive_sim / (all_sim + 1e-8))
        
        # Average over all samples
        loss = loss / batch_size
        
        return loss


class VeryAggressiveContrastiveLoss(nn.Module):
    """
    Very Aggressive: Use ANY matching aspect as positive pair
    No threshold at all!
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """
        Contrastive loss: ANY aspect match = positive pair
        """
        batch_size = embeddings.shape[0]
        num_aspects = labels.shape[1]
        
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive mask: any aspect matches
        positive_mask = torch.zeros(batch_size, batch_size, device=labels.device)
        
        for i in range(num_aspects):
            aspect_match = (labels[:, i].unsqueeze(1) == labels[:, i].unsqueeze(0)).float()
            positive_mask = torch.max(positive_mask, aspect_match)  # Any match = positive
        
        positive_mask.fill_diagonal_(0)  # Exclude self
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        
        loss = 0.0
        num_positives = 0
        
        for i in range(batch_size):
            positive_sim = (exp_sim[i] * positive_mask[i]).sum()
            all_sim = exp_sim[i].sum() - exp_sim[i, i]
            
            if positive_sim > 0:
                loss += -torch.log(positive_sim / (all_sim + 1e-8))
                num_positives += 1
        
        if num_positives > 0:
            loss = loss / num_positives
        else:
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss


def test_losses():
    """Test different loss functions"""
    print("=" * 80)
    print("Testing Different Contrastive Loss Functions")
    print("=" * 80)
    
    # Create fake data
    batch_size = 8
    embeddings = F.normalize(torch.randn(batch_size, 256), dim=1)
    
    # Labels: most aspects are neutral (2), only 1-2 aspects differ
    labels = torch.tensor([
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Battery=pos
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Battery=pos (same as above)
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Battery=neg
        [2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Camera=pos
        [2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Camera=neg
        [2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2],  # Performance=pos
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Battery=pos, Camera=neg
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # All neutral
    ])
    
    # Test original loss with threshold 0.3
    print("\n1. Original Loss (threshold=0.3):")
    from model_multilabel_contrastive import MultiLabelContrastiveLoss
    loss_fn1 = MultiLabelContrastiveLoss(temperature=0.1, similarity_threshold=0.3)
    loss1 = loss_fn1(embeddings, labels)
    print(f"   Loss: {loss1.item():.4f}")
    
    # Test improved soft loss
    print("\n2. Improved Soft Loss:")
    loss_fn2 = ImprovedMultiLabelContrastiveLoss(temperature=0.1, base_weight=0.1)
    loss2 = loss_fn2(embeddings, labels)
    print(f"   Loss: {loss2.item():.4f}")
    
    # Test very aggressive loss
    print("\n3. Very Aggressive Loss (any match):")
    loss_fn3 = VeryAggressiveContrastiveLoss(temperature=0.1)
    loss3 = loss_fn3(embeddings, labels)
    print(f"   Loss: {loss3.item():.4f}")
    
    print("\n" + "=" * 80)
    print("Loss Comparison:")
    print(f"   Original (threshold):  {loss1.item():.4f}")
    print(f"   Improved (soft):       {loss2.item():.4f}")
    print(f"   Very Aggressive (any): {loss3.item():.4f}")
    print("=" * 80)
    print("\nRecommendation: Use 'Improved Soft Loss' or 'Very Aggressive Loss'")
    print("Both should give loss > 0.1 (much better than 0.0007!)")


if __name__ == '__main__':
    test_losses()
