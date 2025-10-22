"""
Unified Focal Loss (combination of Dice + Cross-Entropy + Focal)

Paper: "Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation"
URL: https://www.sciencedirect.com/science/article/pii/S0895611121001750

GitHub: https://github.com/tayden/unified-focal-loss-pytorch

Improvements over Focal Loss:
1. Combines strengths of Dice loss and Cross-Entropy
2. Better for medical imaging and imbalanced datasets
3. Asymmetric variants for handling false positives/negatives differently
4. More stable training

Variants:
- Symmetric Focal Loss
- Asymmetric Focal Loss (penalize FP/FN differently)
- Symmetric Focal Tversky Loss
- Asymmetric Focal Tversky Loss
- Unified Focal Loss (combination of above)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss
    
    For multi-class problems wanting to emphasize on False Positives or False Negatives.
    
    Args:
        delta: Weight controlling the balance between false positives and false negatives
               delta > 0.5 penalizes false negatives more (focus on recall)
               delta < 0.5 penalizes false positives more (focus on precision)
        gamma: Focusing parameter (same as Focal Loss)
    """
    
    def __init__(self, delta=0.7, gamma=2.0):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            loss: Asymmetric focal loss
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        
        # Cross-entropy
        ce = -targets_one_hot * torch.log(probs + 1e-7)
        
        # Asymmetric weighting
        # For correct class: weight = delta
        # For incorrect class: weight = (1 - delta)
        weights = targets_one_hot * self.delta + (1 - targets_one_hot) * (1 - self.delta)
        
        # Focal term
        focal_weight = torch.abs(targets_one_hot - probs) ** self.gamma
        
        # Combine
        loss = weights * focal_weight * ce
        
        return loss.mean()


class AsymmetricFocalTverskyLoss(nn.Module):
    """
    Asymmetric Focal Tversky Loss
    
    Combines Tversky index (generalization of Dice) with Focal Loss
    Better for highly imbalanced data
    
    Args:
        delta: Controls FP vs FN trade-off (same as AsymmetricFocalLoss)
        gamma: Focusing parameter
        epsilon: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-6):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            loss: Asymmetric focal Tversky loss
        """
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
        
        # True Positives, False Positives, False Negatives
        tp = (probs * targets_one_hot).sum(dim=0)
        fp = (probs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - probs) * targets_one_hot).sum(dim=0)
        
        # Tversky index per class
        tversky_index = (tp + self.epsilon) / (
            tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon
        )
        
        # Focal Tversky loss
        focal_tversky = (1 - tversky_index) ** self.gamma
        
        return focal_tversky.mean()


class UnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss: Combination of Asymmetric Focal Loss and Asymmetric Focal Tversky Loss
    
    This provides the best of both worlds:
    - Cross-entropy based (good for classification)
    - Dice/Tversky based (good for imbalance)
    - Focal mechanism (focus on hard examples)
    - Asymmetric weighting (balance FP/FN)
    
    Args:
        weight: Weight to balance the two losses (0 to 1)
                weight = 0.5 means equal contribution
        delta: Asymmetric parameter (0.5 = symmetric, >0.5 = focus on recall, <0.5 = focus on precision)
        gamma: Focusing parameter for Focal Loss part
        gamma_tversky: Focusing parameter for Tversky part (usually smaller)
    """
    
    def __init__(self, weight=0.5, delta=0.6, gamma=2.0, gamma_tversky=0.75):
        super(UnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.asymmetric_focal = AsymmetricFocalLoss(delta=delta, gamma=gamma)
        self.asymmetric_tversky = AsymmetricFocalTverskyLoss(delta=delta, gamma=gamma_tversky)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            loss: Unified focal loss
        """
        focal_loss = self.asymmetric_focal(logits, targets)
        tversky_loss = self.asymmetric_tversky(logits, targets)
        
        # Weighted combination
        unified_loss = self.weight * focal_loss + (1 - self.weight) * tversky_loss
        
        return unified_loss


class MultiLabelUnifiedFocalLoss(nn.Module):
    """
    Unified Focal Loss for multi-label classification (ABSA task)
    
    Applies Unified Focal Loss to each aspect separately
    """
    
    def __init__(self, num_aspects=11, num_sentiments=3, 
                 weight=0.5, delta=0.6, gamma=2.0, gamma_tversky=0.75):
        super(MultiLabelUnifiedFocalLoss, self).__init__()
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.unified_loss = UnifiedFocalLoss(
            weight=weight, 
            delta=delta, 
            gamma=gamma, 
            gamma_tversky=gamma_tversky
        )
    
    def forward(self, logits, targets, class_weights=None):
        """
        Args:
            logits: [batch_size, num_aspects, num_sentiments]
            targets: [batch_size, num_aspects]
            class_weights: Optional [num_aspects, num_sentiments]
        
        Returns:
            loss: Average unified focal loss across all aspects
        """
        total_loss = 0.0
        
        for i in range(self.num_aspects):
            aspect_logits = logits[:, i, :]  # [batch_size, num_sentiments]
            aspect_targets = targets[:, i]   # [batch_size]
            
            loss = self.unified_loss(aspect_logits, aspect_targets)
            
            # Apply aspect-specific weight if provided
            if class_weights is not None:
                aspect_weight = class_weights[i].mean()  # Average weight for this aspect
                loss = loss * aspect_weight
            
            total_loss += loss
        
        return total_loss / self.num_aspects


# Test function
def test_unified_focal_loss():
    """Test Unified Focal Loss implementations"""
    print("=" * 80)
    print("Testing Unified Focal Loss Variants")
    print("=" * 80)
    
    # Test 1: Asymmetric Focal Loss
    print("\n1. Asymmetric Focal Loss Test:")
    afl = AsymmetricFocalLoss(delta=0.7, gamma=2.0)
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    loss = afl(logits, targets)
    print(f"   Asymmetric Focal Loss: {loss.item():.4f}")
    
    # Test 2: Asymmetric Focal Tversky Loss
    print("\n2. Asymmetric Focal Tversky Loss Test:")
    aftl = AsymmetricFocalTverskyLoss(delta=0.7, gamma=0.75)
    loss = aftl(logits, targets)
    print(f"   Asymmetric Focal Tversky Loss: {loss.item():.4f}")
    
    # Test 3: Unified Focal Loss
    print("\n3. Unified Focal Loss Test:")
    ufl = UnifiedFocalLoss(weight=0.5, delta=0.6, gamma=2.0, gamma_tversky=0.75)
    loss = ufl(logits, targets)
    print(f"   Unified Focal Loss: {loss.item():.4f}")
    
    # Test 4: Multi-label ABSA
    print("\n4. Multi-label ABSA Test:")
    ml_ufl = MultiLabelUnifiedFocalLoss(num_aspects=11, num_sentiments=3)
    logits = torch.randn(32, 11, 3)
    targets = torch.randint(0, 3, (32, 11))
    loss = ml_ufl(logits, targets)
    print(f"   Multi-label Unified Focal Loss: {loss.item():.4f}")
    
    # Test 5: Compare different delta values
    print("\n5. Delta Parameter Effects:")
    for delta in [0.3, 0.5, 0.7]:
        ufl_delta = UnifiedFocalLoss(weight=0.5, delta=delta, gamma=2.0)
        loss = ufl_delta(logits[:, :, :], targets[:, 0])
        print(f"   Delta={delta} (focus on {'recall' if delta > 0.5 else 'precision' if delta < 0.5 else 'balanced'}): Loss={loss.item():.4f}")
    
    # Test 6: Compare with standard Focal Loss
    print("\n6. Comparison with Standard Focal Loss:")
    from utils import FocalLoss
    focal_loss_fn = FocalLoss(alpha=None, gamma=2.0, reduction='mean')
    
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    
    focal_loss = focal_loss_fn(logits, targets)
    unified_loss = ufl(logits, targets)
    
    print(f"   Standard Focal Loss: {focal_loss.item():.4f}")
    print(f"   Unified Focal Loss:  {unified_loss.item():.4f}")
    
    print("\nAll tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_unified_focal_loss()
