"""
Improved Focal Loss with Auxiliary Losses for Better Detection F1
Target: >92% Detection F1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

class ImprovedDualTaskFocalLoss(nn.Module):
    """
    Enhanced loss with:
    1. Higher gamma for detection
    2. Dice loss for imbalance
    3. Asymmetric loss for false negatives
    """
    
    def __init__(
        self,
        detection_alpha: List[float] = [0.2, 5.0],  # Aggressive weighting
        sentiment_alpha: List[float] = [1.0, 1.2, 0.8],
        detection_gamma: float = 3.5,  # Higher for extreme imbalance
        sentiment_gamma: float = 2.0,
        detection_weight: float = 0.65,  # Prioritize detection
        sentiment_weight: float = 0.35,
        use_dice: bool = True,
        dice_weight: float = 0.1,
        use_asymmetric: bool = True,
        asymmetric_weight: float = 0.1,
        asymmetric_pos_weight: float = 5.0,
        reduction: str = 'none'
    ):
        super().__init__()
        
        self.detection_gamma = detection_gamma
        self.sentiment_gamma = sentiment_gamma
        self.detection_weight = detection_weight
        self.sentiment_weight = sentiment_weight
        self.use_dice = use_dice
        self.dice_weight = dice_weight
        self.use_asymmetric = use_asymmetric
        self.asymmetric_weight = asymmetric_weight
        self.asymmetric_pos_weight = asymmetric_pos_weight
        self.reduction = reduction
        
        # Convert alpha to tensors
        self.detection_alpha = torch.tensor(detection_alpha, dtype=torch.float32)
        self.sentiment_alpha = torch.tensor(sentiment_alpha, dtype=torch.float32)
        
        print("\n" + "="*60)
        print("IMPROVED DUAL-TASK FOCAL LOSS")
        print("="*60)
        print(f"Detection:")
        print(f"  - Gamma: {detection_gamma} (high for imbalance)")
        print(f"  - Alpha: {detection_alpha} (25x weight for 'present')")
        print(f"  - Weight: {detection_weight}")
        print(f"Sentiment:")
        print(f"  - Gamma: {sentiment_gamma}")
        print(f"  - Alpha: {sentiment_alpha}")
        print(f"  - Weight: {sentiment_weight}")
        print(f"Auxiliary:")
        print(f"  - Dice Loss: {use_dice} (weight={dice_weight})")
        print(f"  - Asymmetric Loss: {use_asymmetric} (weight={asymmetric_weight})")
        print("="*60)
    
    def focal_loss_sigmoid(self, logits, labels, alpha=None, gamma=2.0):
        """BCE Focal Loss for detection"""
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        
        p = torch.sigmoid(logits)
        pt = p * labels + (1 - p) * (1 - labels)
        focal_term = (1 - pt) ** gamma
        loss = focal_term * bce
        
        if alpha is not None:
            if alpha.device != logits.device:
                alpha = alpha.to(logits.device)
            alpha_t = alpha[1] * labels + alpha[0] * (1 - labels)
            loss = alpha_t * loss
        
        return loss
    
    def dice_loss(self, logits, labels, smooth=1e-5):
        """
        Dice Loss - good for imbalanced binary classification
        Directly optimizes F1 score
        """
        probs = torch.sigmoid(logits)
        
        # Flatten for calculation
        probs_flat = probs.view(-1)
        labels_flat = labels.view(-1)
        
        # Calculate intersection and union
        intersection = (probs_flat * labels_flat).sum()
        union = probs_flat.sum() + labels_flat.sum()
        
        # Dice coefficient
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Return loss (1 - dice)
        return 1 - dice
    
    def asymmetric_loss(self, logits, labels, gamma_pos=0, gamma_neg=4, pos_weight=5.0):
        """
        Asymmetric Loss - penalizes false negatives more than false positives
        Good for imbalanced detection where missing positives is worse
        """
        probs = torch.sigmoid(logits)
        
        # Asymmetric focusing
        pos_term = labels * torch.pow(1 - probs, gamma_pos)
        neg_term = (1 - labels) * torch.pow(probs, gamma_neg)
        
        # BCE with asymmetric weighting
        pos_loss = -pos_term * torch.log(probs + 1e-7) * pos_weight
        neg_loss = -neg_term * torch.log(1 - probs + 1e-7)
        
        loss = pos_loss + neg_loss
        return loss
    
    def focal_loss_multiclass(self, logits, labels, alpha=None, gamma=2.0):
        """Standard focal loss for sentiment"""
        ce_loss = F.cross_entropy(logits.view(-1, 3), labels.view(-1), reduction='none')
        ce_loss = ce_loss.view(logits.size(0), -1)
        
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        focal_term = (1 - pt) ** gamma
        
        if alpha is not None:
            if alpha.device != logits.device:
                alpha = alpha.to(logits.device)
            alpha_t = alpha[labels]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        return focal_loss
    
    def forward(self, detection_logits, sentiment_logits, detection_labels, 
                sentiment_labels, detection_mask=None, sentiment_mask=None):
        """
        Combined loss with focal + auxiliary losses
        """
        
        # 1. Main Focal Losses
        detection_focal = self.focal_loss_sigmoid(
            detection_logits, detection_labels, 
            self.detection_alpha, self.detection_gamma
        )
        
        sentiment_focal = self.focal_loss_multiclass(
            sentiment_logits, sentiment_labels,
            self.sentiment_alpha, self.sentiment_gamma  
        )
        
        # Apply masks
        if detection_mask is not None:
            detection_focal = detection_focal * detection_mask
        if sentiment_mask is not None:
            sentiment_focal = sentiment_focal * sentiment_mask
        
        # 2. Auxiliary Detection Losses
        auxiliary_det_loss = 0
        
        if self.use_dice:
            dice = self.dice_loss(detection_logits, detection_labels)
            auxiliary_det_loss += self.dice_weight * dice
        
        if self.use_asymmetric:
            asym = self.asymmetric_loss(
                detection_logits, detection_labels,
                pos_weight=self.asymmetric_pos_weight
            ).mean()
            auxiliary_det_loss += self.asymmetric_weight * asym
        
        # 3. Combine all losses
        if self.reduction == 'none':
            # Per-aspect loss
            total_loss = (
                self.detection_weight * detection_focal +
                self.sentiment_weight * sentiment_focal
            )
            
            # Add auxiliary (broadcast to match shape)
            if auxiliary_det_loss > 0:
                total_loss = total_loss + auxiliary_det_loss
            
            # Calculate averages for logging
            det_loss_avg = detection_focal.mean() + auxiliary_det_loss
            sent_loss_avg = sentiment_focal.mean()
            
        else:  # mean reduction
            det_loss_avg = detection_focal.mean() + auxiliary_det_loss
            sent_loss_avg = sentiment_focal.mean()
            
            total_loss = (
                self.detection_weight * det_loss_avg +
                self.sentiment_weight * sent_loss_avg
            )
        
        return total_loss, det_loss_avg, sent_loss_avg


def test_improved_loss():
    """Test the improved loss function"""
    
    batch_size = 4
    num_aspects = 11
    
    # Create dummy data
    detection_logits = torch.randn(batch_size, num_aspects)
    sentiment_logits = torch.randn(batch_size, num_aspects, 3)
    
    # Imbalanced labels (more 0s than 1s)
    detection_labels = torch.tensor([
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    ])
    
    sentiment_labels = torch.randint(0, 3, (batch_size, num_aspects))
    
    # Create loss function
    loss_fn = ImprovedDualTaskFocalLoss()
    
    # Calculate loss
    total_loss, det_loss, sent_loss = loss_fn(
        detection_logits, sentiment_logits,
        detection_labels, sentiment_labels
    )
    
    print(f"\nTest Results:")
    print(f"  Detection loss: {det_loss.item():.4f}")
    print(f"  Sentiment loss: {sent_loss.item():.4f}")
    print(f"  Total loss mean: {total_loss.mean().item():.4f}")
    
    # Check gradients
    total_loss.mean().backward()
    print(f"\n✓ Backward pass successful!")
    print("="*60)


if __name__ == '__main__':
    test_improved_loss()
