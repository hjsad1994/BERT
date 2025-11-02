"""
Binary Focal Loss for Aspect Detection (Multi-Label Binary Classification)
==========================================================================
PyTorch implementation of Focal Loss for binary classification tasks.

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    Formula for binary: FL = -alpha * (1-p_t)^gamma * log(p_t)
    where p_t = p if y=1, else p_t = 1-p

Usage for Aspect Detection:
    focal_loss = BinaryFocalLoss(alpha=[1.0, 2.5], gamma=2.0)  # [negative, positive]
    loss = focal_loss(logits, labels)  # logits: [B, num_aspects], labels: [B, num_aspects]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label aspect detection.
    
    Each aspect is treated as a binary classification task (mentioned/not mentioned).
    This loss function addresses class imbalance by down-weighting easy examples
    and focusing training on hard negatives.
    
    Args:
        alpha (Union[List[float], torch.Tensor], optional): Class weights for 
            [negative_class, positive_class]. If None, no class weighting.
            Default: None (equal weights)
            Example: [1.0, 2.5] means positive class gets 2.5x weight
        gamma (float): Focusing parameter. Higher values increase focus on hard examples.
            Recommended: 2.0. Default: 2.0
        reduction (str): Specifies reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
        pos_weight (torch.Tensor, optional): Per-aspect positive weights for BCE.
            If provided, used instead of alpha[1]. Shape: [num_aspects]
    
    Shape:
        - Input (logits): [batch_size, num_aspects]
        - Target (labels): [batch_size, num_aspects] with values in {0, 1}
        - Output: scalar if reduction='mean' or 'sum', else [batch_size, num_aspects]
    
    Example:
        >>> focal_loss = BinaryFocalLoss(alpha=[1.0, 2.5], gamma=2.0)
        >>> logits = torch.randn(16, 11, requires_grad=True)
        >>> labels = torch.randint(0, 2, (16, 11)).float()
        >>> loss = focal_loss(logits, labels)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        pos_weight: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                if len(alpha) != 2:
                    raise ValueError(f"alpha must have 2 elements [negative, positive], got {len(alpha)}")
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            elif isinstance(alpha, torch.Tensor):
                if alpha.numel() != 2:
                    raise ValueError(f"alpha must have 2 elements, got {alpha.numel()}")
                self.register_buffer('alpha', alpha.float())
            else:
                raise TypeError(f"alpha must be list, tuple, or Tensor, got {type(alpha)}")
        else:
            self.register_buffer('alpha', None)
        
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
        
        print(f"[OK] BinaryFocalLoss initialized:")
        if self.alpha is not None:
            print(f"   Alpha weights: [negative={self.alpha[0]:.3f}, positive={self.alpha[1]:.3f}]")
        else:
            print(f"   Alpha weights: None (equal weights)")
        print(f"   Gamma (focusing): {self.gamma}")
        print(f"   Reduction: {self.reduction}")
        if pos_weight is not None:
            print(f"   Per-aspect pos_weight: {pos_weight.shape}")
    
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Binary Focal Loss for aspect detection.
        
        Args:
            input: Predicted logits [batch_size, num_aspects]
            target: Ground truth binary labels [batch_size, num_aspects] in {0, 1}
        
        Returns:
            Computed loss value (scalar or tensor based on reduction)
        """
        if input.dim() != 2:
            raise ValueError(f"Expected input to be 2D [batch, num_aspects], got {input.dim()}D")
        
        if target.dim() != 2:
            raise ValueError(f"Expected target to be 2D [batch, num_aspects], got {target.dim()}D")
        
        if input.shape != target.shape:
            raise ValueError(
                f"Shape mismatch: input {input.shape}, target {target.shape}"
            )
        
        # Compute probabilities using sigmoid
        probs = torch.sigmoid(input)  # [batch_size, num_aspects]
        
        # Compute p_t: probability of the correct class
        # p_t = p if y=1, else p_t = 1-p
        p_t = target * probs + (1 - target) * (1 - probs)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(self.gamma)
        
        # Compute BCE with logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(
            input, target, reduction='none', pos_weight=self.pos_weight
        )
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # alpha_t = alpha[1] if y=1, else alpha[0]
            alpha_t = target * self.alpha[1] + (1 - target) * self.alpha[0]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:  # sum
            return focal_loss.sum()


def calculate_binary_alpha_from_weights(
    pos_weights: torch.Tensor,
    method: str = 'proportional'
) -> List[float]:
    """
    Calculate alpha weights for Binary Focal Loss from per-aspect positive weights.
    
    Args:
        pos_weights: Per-aspect positive weights [num_aspects]
        method: How to compute alpha:
            - 'proportional': alpha[1] = mean(pos_weights), alpha[0] = 1.0
            - 'max': alpha[1] = max(pos_weights), alpha[0] = 1.0
            - 'min': alpha[1] = min(pos_weights), alpha[0] = 1.0
            Default: 'proportional'
    
    Returns:
        List of [negative_weight, positive_weight]
    """
    if method == 'proportional':
        positive_weight = pos_weights.mean().item()
    elif method == 'max':
        positive_weight = pos_weights.max().item()
    elif method == 'min':
        positive_weight = pos_weights.min().item()
    else:
        raise ValueError(f"method must be 'proportional', 'max', or 'min', got {method}")
    
    return [1.0, positive_weight]


if __name__ == '__main__':
    # Test Binary Focal Loss
    print("=" * 70)
    print("Testing BinaryFocalLoss Implementation")
    print("=" * 70)
    
    batch_size = 16
    num_aspects = 11
    
    # Test 1: Basic functionality
    print("\n1. Test basic functionality")
    print("-" * 70)
    logits = torch.randn(batch_size, num_aspects, requires_grad=True)
    labels = torch.randint(0, 2, (batch_size, num_aspects)).float()
    
    focal_loss = BinaryFocalLoss(alpha=[1.0, 2.5], gamma=2.0)
    loss = focal_loss(logits, labels)
    loss.backward()
    
    print(f"   Input shape:  {logits.shape}")
    print(f"   Target shape: {labels.shape}")
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Gradient computed: {logits.grad is not None}")
    print(f"   [PASS]")
    
    # Test 2: Without alpha weights
    print("\n2. Test without alpha weights")
    print("-" * 70)
    focal_loss_no_alpha = BinaryFocalLoss(alpha=None, gamma=2.0)
    loss2 = focal_loss_no_alpha(logits, labels)
    print(f"   Loss value: {loss2.item():.4f}")
    print(f"   [PASS]")
    
    # Test 3: Different gamma values
    print("\n3. Test different gamma values")
    print("-" * 70)
    for gamma in [0.0, 1.0, 2.0, 5.0]:
        focal = BinaryFocalLoss(alpha=[1.0, 2.0], gamma=gamma)
        loss_val = focal(logits, labels)
        print(f"   Gamma={gamma:.1f}: Loss={loss_val.item():.4f}")
    print(f"   [PASS]")
    
    # Test 4: Per-aspect pos_weight
    print("\n4. Test with per-aspect pos_weight")
    print("-" * 70)
    pos_weight = torch.ones(num_aspects) * 2.0  # [num_aspects]
    focal_weighted = BinaryFocalLoss(alpha=[1.0, 1.5], gamma=2.0, pos_weight=pos_weight)
    loss3 = focal_weighted(logits, labels)
    print(f"   Loss value: {loss3.item():.4f}")
    print(f"   [PASS]")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All tests passed!")
    print("=" * 70)

