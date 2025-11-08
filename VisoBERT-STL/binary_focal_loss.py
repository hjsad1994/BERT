"""
Binary Focal Loss for Aspect Detection (Multi-Label Binary Classification)
PyTorch implementation for addressing class imbalance in binary detection tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Optional, Union, List

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label aspect detection.
    
    Addresses class imbalance by down-weighting easy examples and focusing on hard cases.
    
    Args:
        alpha: Class weights [negative_weight, positive_weight]. Default: [1.0, 1.0]
        gamma: Focusing parameter (0-5). Higher = more focus on hard examples. Default: 2.0
        reduction: 'none' | 'mean' | 'sum'. Default: 'mean'
    
    Shape:
        - Input: [batch_size, num_aspects] (logits)
        - Target: [batch_size, num_aspects] (binary labels in {0, 1})
        - Output: scalar if reduction='mean' or 'sum', else [batch_size, num_aspects]
    """
    
    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
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
                raise TypeError(f"alpha must be list or Tensor, got {type(alpha)}")
        else:
            self.register_buffer('alpha', torch.tensor([1.0, 1.0], dtype=torch.float32))
        
        self.gamma = gamma
        self.reduction = reduction
        
        print(f"   BinaryFocalLoss initialized:")
        print(f"      Alpha: [neg={self.alpha[0]:.3f}, pos={self.alpha[1]:.3f}]")
        print(f"      Gamma: {self.gamma}")
        print(f"      Reduction: {self.reduction}")
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Focal Loss
        
        Args:
            input: Predicted logits [batch_size, num_aspects]
            target: Ground truth binary labels [batch_size, num_aspects]
        
        Returns:
            Loss value (scalar or tensor based on reduction)
        """
        if input.shape != target.shape:
            raise ValueError(f"Shape mismatch: input {input.shape}, target {target.shape}")
        
        # Sigmoid probabilities
        probs = torch.sigmoid(input)
        
        # p_t = p if y=1, else (1-p)
        p_t = target * probs + (1 - target) * (1 - probs)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t).pow(self.gamma)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * bce
        
        # Apply alpha weighting
        alpha_t = target * self.alpha[1] + (1 - target) * self.alpha[0]
        focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        else:  # sum
            return focal_loss.sum()


def calculate_binary_alpha_auto(
    csv_file: str,
    aspects: list,
    method: str = 'inverse_freq'
) -> List[float]:
    """
    Auto-calculate alpha weights from training data
    
    Args:
        csv_file: Path to training CSV
        aspects: List of aspect names
        method: 'inverse_freq' | 'balanced'
    
    Returns:
        [negative_weight, positive_weight]
    """
    print(f"\n   Calculating auto alpha weights from: {csv_file}")
    
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
    
    total_mentioned = 0
    total_not_mentioned = 0
    
    for aspect in aspects:
        mentioned = df[aspect].notna().sum()
        not_mentioned = df[aspect].isna().sum()
        total_mentioned += mentioned
        total_not_mentioned += not_mentioned
    
    total = total_mentioned + total_not_mentioned
    
    if method == 'inverse_freq':
        # Weight inversely proportional to frequency
        # More weight to minority class (usually positive/mentioned)
        neg_weight = 1.0
        pos_weight = total_not_mentioned / total_mentioned if total_mentioned > 0 else 1.0
    
    elif method == 'balanced':
        # Balanced weights
        neg_weight = total / (2 * total_not_mentioned) if total_not_mentioned > 0 else 1.0
        pos_weight = total / (2 * total_mentioned) if total_mentioned > 0 else 1.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    alpha = [neg_weight, pos_weight]
    
    print(f"      Total samples: {len(df):,}")
    print(f"      Mentioned (positive):     {total_mentioned:,} ({total_mentioned/total*100:.1f}%)")
    print(f"      Not mentioned (negative): {total_not_mentioned:,} ({total_not_mentioned/total*100:.1f}%)")
    print(f"      Imbalance ratio: {total_not_mentioned/total_mentioned:.2f}:1")
    print(f"      Calculated alpha: {alpha}")
    
    return alpha
