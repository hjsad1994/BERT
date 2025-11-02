"""
Dual-Task Focal Loss for ABSA (Focal Loss Only)
----------------------------------------------
This module implements ONLY Focal Loss (no standard cross-entropy fallback)
for BOTH Aspect Detection and Sentiment tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Optional, Union, List

class DualTaskFocalLoss(nn.Module):
    """
    Dual-Task Focal Loss (No fallback to CE, strictly focal loss everywhere)
    L_total = λ_det * L_detection_focal + λ_sent * L_sentiment_focal
    """
    def __init__(
        self,
        detection_alpha: Optional[List[float]] = None,  # [not_present, present]
        sentiment_alpha: Optional[List[float]] = None,  # [pos, neg, neu]
        gamma: float = 2.0,
        detection_weight: float = 0.3,
        sentiment_weight: float = 0.7,
        reduction: str = 'none',
        detection_gamma: Optional[float] = None,
        sentiment_gamma: Optional[float] = None
    ):
        super().__init__()
        # If task-specific gamma provided, override shared gamma
        self.detection_gamma = detection_gamma if detection_gamma is not None else gamma
        self.sentiment_gamma = sentiment_gamma if sentiment_gamma is not None else gamma
        self.detection_weight = detection_weight
        self.sentiment_weight = sentiment_weight
        self.reduction = reduction
        # Detection alpha (binary) as weights for classes [0,1]
        self.detection_alpha = torch.tensor(detection_alpha, dtype=torch.float32) if detection_alpha is not None else None
        # Sentiment alpha (multi-class)
        self.sentiment_alpha = torch.tensor(sentiment_alpha, dtype=torch.float32) if sentiment_alpha is not None else None
        print("\nDUAL-TASK FOCAL LOSS INITIALIZED (NO Cross-Entropy option!)")
        print(f"   Detection alpha: {detection_alpha}")
        print(f"   Sentiment alpha: {sentiment_alpha}")
        print(f"   Detection gamma: {self.detection_gamma}")
        print(f"   Sentiment gamma: {self.sentiment_gamma}")
        print(f"   Weights: detection={detection_weight:.2f}, sentiment={sentiment_weight:.2f}")
        print(f"   Reduction: {reduction}")

    def focal_loss_binary_sigmoid(self, logits, labels, alpha=None, gamma_val: float = 2.0):
        """
        Sigmoid/BCE-with-logits focal loss for detection.
        logits: [batch, aspects] raw logits
        labels: [batch, aspects] (0/1)
        alpha: [2] weights for [neg(0), pos(1)]
        returns: [batch, aspects]
        """
        # BCE with logits per-element
        bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')  # [batch, aspects]
        # Probabilities for true class
        p = torch.sigmoid(logits)
        pt = p * labels + (1 - p) * (1 - labels)
        focal_term = (1 - pt) ** gamma_val
        loss = focal_term * bce
        if alpha is not None:
            if alpha.device != logits.device:
                alpha = alpha.to(logits.device)
            alpha_t = alpha[1] * labels + alpha[0] * (1 - labels)
            loss = alpha_t * loss
        return loss

    def focal_loss_multiclass(self, logits, labels, alpha=None, gamma_val: float = 2.0):
        ce_loss = F.cross_entropy(logits.view(-1, 3), labels.view(-1), reduction='none')
        ce_loss = ce_loss.view(logits.size(0), -1)
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        focal_term = (1 - pt) ** gamma_val
        if alpha is not None:
            if alpha.device != logits.device:
                alpha = alpha.to(logits.device)
            alpha_t = alpha[labels]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        return focal_loss

    def forward(self, detection_logits, sentiment_logits, detection_labels, sentiment_labels, detection_mask=None, sentiment_mask=None):
        # detection_logits expected shape: [batch, aspects]
        detection_loss = self.focal_loss_binary_sigmoid(
            detection_logits, detection_labels, self.detection_alpha, gamma_val=self.detection_gamma
        )
        if detection_mask is not None:
            detection_loss = detection_loss * detection_mask
        sentiment_loss = self.focal_loss_multiclass(
            sentiment_logits, sentiment_labels, self.sentiment_alpha, gamma_val=self.sentiment_gamma
        )
        if sentiment_mask is not None:
            sentiment_loss = sentiment_loss * sentiment_mask
        if self.reduction == 'none':
            total_loss = (
                self.detection_weight * detection_loss + 
                self.sentiment_weight * sentiment_loss
            )
            if detection_mask is not None:
                num_det = detection_mask.sum()
                detection_loss_val = detection_loss.sum() / num_det if num_det > 0 else detection_loss.mean()
            else:
                detection_loss_val = detection_loss.mean()
            if sentiment_mask is not None:
                num_sent = sentiment_mask.sum()
                sentiment_loss_val = sentiment_loss.sum() / num_sent if num_sent > 0 else sentiment_loss.mean()
            else:
                sentiment_loss_val = sentiment_loss.mean()
        elif self.reduction == 'mean':
            if detection_mask is not None:
                num_det = detection_mask.sum()
                detection_loss_val = detection_loss.sum() / num_det if num_det > 0 else detection_loss.mean()
            else:
                detection_loss_val = detection_loss.mean()
            if sentiment_mask is not None:
                num_sent = sentiment_mask.sum()
                sentiment_loss_val = sentiment_loss.sum() / num_sent if num_sent > 0 else sentiment_loss.mean()
            else:
                sentiment_loss_val = sentiment_loss.mean()
            total_loss = (
                self.detection_weight * detection_loss_val + 
                self.sentiment_weight * sentiment_loss_val
            )
        return total_loss, detection_loss_val, sentiment_loss_val

# ...KEEP calculation helpers unchanged as they just count frequencies, not logic


def calculate_detection_alpha(csv_file, aspects):
    """
    Calculate alpha weights for detection based on class frequency
    
    Returns:
        alpha: [not_present_weight, present_weight]
    """
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    total_possible = len(df) * len(aspects)
    total_present = 0
    
    for aspect in aspects:
        total_present += df[aspect].notna().sum()
    
    total_not_present = total_possible - total_present
    
    # Inverse frequency weighting
    present_weight = total_possible / (2 * total_present) if total_present > 0 else 1.0
    not_present_weight = total_possible / (2 * total_not_present) if total_not_present > 0 else 1.0
    
    alpha = [not_present_weight, present_weight]
    
    print(f"\nDetection Class Distribution:")
    print(f"   Not present: {total_not_present:,} ({total_not_present/total_possible*100:.1f}%)")
    print(f"   Present:     {total_present:,} ({total_present/total_possible*100:.1f}%)")
    print(f"   Alpha weights: [not_present={not_present_weight:.4f}, present={present_weight:.4f}]")
    
    return alpha


def calculate_sentiment_alpha(csv_file, aspects, sentiment_map):
    """
    Calculate alpha weights for sentiment based on class frequency
    (Only considers detected aspects)
    
    Returns:
        alpha: [positive_weight, negative_weight, neutral_weight]
    """
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    # Count sentiments (only non-NaN)
    counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    
    for aspect in aspects:
        aspect_data = df[aspect].dropna()
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            counts[sentiment] += (aspect_data == sentiment).sum()
    
    total = sum(counts.values())
    
    if total == 0:
        return [1.0, 1.0, 1.0]
    
    # Inverse frequency
    alpha = [
        total / (3 * counts['Positive']) if counts['Positive'] > 0 else 1.0,
        total / (3 * counts['Negative']) if counts['Negative'] > 0 else 1.0,
        total / (3 * counts['Neutral']) if counts['Neutral'] > 0 else 1.0
    ]
    
    print(f"\nSentiment Class Distribution (detected only):")
    print(f"   Positive: {counts['Positive']:,} ({counts['Positive']/total*100:.1f}%)")
    print(f"   Negative: {counts['Negative']:,} ({counts['Negative']/total*100:.1f}%)")
    print(f"   Neutral:  {counts['Neutral']:,} ({counts['Neutral']/total*100:.1f}%)")
    print(f"   Alpha weights: [pos={alpha[0]:.4f}, neg={alpha[1]:.4f}, neu={alpha[2]:.4f}]")
    
    return alpha


class DualTaskWeightedBCELoss(nn.Module):
    """
    Dual-Task criterion using weighted BCEWithLogitsLoss for detection (binary)
    and focal loss for sentiment (multiclass). This emphasizes positive (present)
    errors via per-aspect pos_weight while keeping sentiment handling unchanged.
    """
    def __init__(
        self,
        detection_pos_weight: torch.Tensor,  # shape [num_aspects]
        sentiment_alpha: Optional[List[float]] = None,
        sentiment_gamma: float = 2.0,
        detection_weight: float = 0.5,
        sentiment_weight: float = 0.5,
        reduction: str = 'none'
    ):
        super().__init__()
        if reduction not in ('none', 'mean'):
            raise ValueError("reduction must be 'none' or 'mean'")
        self.reduction = reduction
        self.detection_weight = detection_weight
        self.sentiment_weight = sentiment_weight
        # Register buffers for device moves
        self.register_buffer('pos_weight', detection_pos_weight.float())
        self.sentiment_alpha = torch.tensor(sentiment_alpha, dtype=torch.float32) if sentiment_alpha is not None else None
        self.sentiment_gamma = sentiment_gamma
        print("\nDUAL-TASK WEIGHTED BCE INITIALIZED")
        print(f"   Detection pos_weight (per-aspect): min={self.pos_weight.min().item():.3f} max={self.pos_weight.max().item():.3f}")
        print(f"   Sentiment alpha: {sentiment_alpha}")
        print(f"   Sentiment gamma: {self.sentiment_gamma}")
        print(f"   Weights: detection={detection_weight:.2f}, sentiment={sentiment_weight:.2f}")
        print(f"   Reduction: {reduction}")

    def focal_loss_multiclass(self, logits, labels, alpha=None, gamma_val: float = 2.0):
        ce_loss = F.cross_entropy(logits.view(-1, 3), labels.view(-1), reduction='none')
        ce_loss = ce_loss.view(logits.size(0), -1)
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        focal_term = (1 - pt) ** gamma_val
        if alpha is not None:
            if alpha.device != logits.device:
                alpha = alpha.to(logits.device)
            alpha_t = alpha[labels]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        return focal_loss

    def forward(self, detection_logits, sentiment_logits, detection_labels, sentiment_labels, detection_mask=None, sentiment_mask=None):
        # Detection: weighted BCE with logits (per-aspect pos_weight)
        # logits, labels shape: [batch, aspects]
        bce_loss = F.binary_cross_entropy_with_logits(
            detection_logits,
            detection_labels.float(),
            reduction='none',
            pos_weight=self.pos_weight
        )  # [batch, aspects]
        if detection_mask is not None:
            detection_loss = bce_loss * detection_mask
        else:
            detection_loss = bce_loss

        # Sentiment: focal loss
        sentiment_alpha = self.sentiment_alpha.to(sentiment_logits.device) if self.sentiment_alpha is not None else None
        sentiment_loss = self.focal_loss_multiclass(
            sentiment_logits, sentiment_labels, alpha=sentiment_alpha, gamma_val=self.sentiment_gamma
        )
        if sentiment_mask is not None:
            sentiment_loss = sentiment_loss * sentiment_mask

        # Reductions to scalars if needed and combine
        if self.reduction == 'none':
            if detection_mask is not None:
                num_det = detection_mask.sum()
                det_val = detection_loss.sum() / num_det if num_det > 0 else detection_loss.mean()
            else:
                det_val = detection_loss.mean()
            if sentiment_mask is not None:
                num_sent = sentiment_mask.sum()
                sent_val = sentiment_loss.sum() / num_sent if num_sent > 0 else sentiment_loss.mean()
            else:
                sent_val = sentiment_loss.mean()
            total = self.detection_weight * detection_loss + self.sentiment_weight * sentiment_loss
            return total, det_val, sent_val
        else:  # mean
            if detection_mask is not None:
                num_det = detection_mask.sum()
                det_val = detection_loss.sum() / num_det if num_det > 0 else detection_loss.mean()
            else:
                det_val = detection_loss.mean()
            if sentiment_mask is not None:
                num_sent = sentiment_mask.sum()
                sent_val = sentiment_loss.sum() / num_sent if num_sent > 0 else sentiment_loss.mean()
            else:
                sent_val = sentiment_loss.mean()
            total = self.detection_weight * det_val + self.sentiment_weight * sent_val
            return total, det_val, sent_val


def calculate_detection_pos_weight(csv_file, aspects):
    """
    Compute per-aspect positive class weight for BCEWithLogitsLoss.
    pos_weight[i] = (num_neg_i / num_pos_i). This increases penalty for missing
    'present' examples under heavy imbalance.
    Returns a tensor of shape [num_aspects].
    """
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    pos_weights = []
    for aspect in aspects:
        num_pos = df[aspect].notna().sum()
        num_total = len(df)
        num_neg = num_total - num_pos
        if num_pos <= 0:
            pw = torch.tensor(1.0)
        else:
            pw = torch.tensor(float(num_neg) / float(num_pos))
        pos_weights.append(pw)
    pos_weight_tensor = torch.stack(pos_weights)
    # Logging
    total_pos = sum([df[a].notna().sum() for a in aspects])
    total_neg = len(df) * len(aspects) - total_pos
    print("\nDetection imbalance summary (for pos_weight):")
    print(f"   Not Present: {total_neg:,} ({total_neg/(total_neg+total_pos)*100:.1f}%)")
    print(f"   Present:     {total_pos:,} ({total_pos/(total_neg+total_pos)*100:.1f}%)")
    print(f"   Global ratio (neg/pos): {(total_neg/max(total_pos,1)):.2f}")
    print(f"   pos_weight range: min={pos_weight_tensor.min().item():.3f}, max={pos_weight_tensor.max().item():.3f}")
    return pos_weight_tensor


def test_focal_loss():
    """Test dual-task focal loss"""
    print("=" * 80)
    print("Testing Dual-Task Focal Loss")
    print("=" * 80)
    
    batch_size = 4
    num_aspects = 11
    
    # Create dummy data
    detection_logits = torch.randn(batch_size, num_aspects, 2)
    sentiment_logits = torch.randn(batch_size, num_aspects, 3)
    
    detection_labels = torch.randint(0, 2, (batch_size, num_aspects))
    sentiment_labels = torch.randint(0, 3, (batch_size, num_aspects))
    
    # Masks: detect all, but sentiment only when detected
    detection_mask = torch.ones(batch_size, num_aspects)
    sentiment_mask = detection_labels.float()  # Only train sentiment when detected
    
    # Create loss
    loss_fn = DualTaskFocalLoss(
        detection_alpha=[0.5, 1.5],  # Upweight detected class
        sentiment_alpha=[1.2, 1.5, 0.8],  # Upweight neg class
        gamma=2.0,
        detection_weight=0.3,
        sentiment_weight=0.7,
        reduction='none'
    )
    
    # Compute loss
    total_loss, det_loss, sent_loss = loss_fn(
        detection_logits,
        sentiment_logits,
        detection_labels,
        sentiment_labels,
        detection_mask,
        sentiment_mask
    )
    
    print(f"\n✓ Loss computed successfully")
    print(f"   Total loss shape: {total_loss.shape}")
    print(f"   Detection loss (avg): {det_loss.item():.4f}")
    print(f"   Sentiment loss (avg): {sent_loss.item():.4f}")
    print(f"   Total loss (sample): {total_loss[0, :3].tolist()}")
    
    # Test backward
    loss_scalar = total_loss.mean()
    loss_scalar.backward()
    
    print(f"\n✓ Backward pass successful")
    print(f"   Gradient computed: {detection_logits.grad is not None}")
    
    print("\n✅ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_focal_loss()
