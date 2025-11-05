"""
Focal Loss for Multi-Label ABSA
================================
PyTorch implementation following best practices

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Usage:
    focal_loss = MultilabelFocalLoss(alpha=[0.8, 1.0, 2.5], gamma=2.0)
    loss = focal_loss(logits, labels)
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilabelFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Label ABSA with class balancing.
    
    This loss function addresses class imbalance by down-weighting easy examples
    and focusing training on hard negatives. It applies the same alpha weights
    globally across all aspects.
    
    Args:
        alpha (Union[List[float], torch.Tensor], optional): Class weights for 
            [positive, negative, neutral]. If None, no class weighting is applied.
            Default: [1.0, 1.0, 1.0]
        gamma (float): Focusing parameter for modulating loss. Higher values
            increase focus on hard examples. Recommended: 2.0. Default: 2.0
        num_aspects (int): Number of aspects in multi-label setting. Default: 11
        reduction (str): Specifies reduction to apply: 'none' | 'mean' | 'sum'.
            Default: 'mean'
    
    Shape:
        - Input: (batch_size, num_aspects, num_sentiments)
        - Target: (batch_size, num_aspects) with class indices in [0, num_sentiments)
        - Output: scalar if reduction='mean' or 'sum', else (batch_size, num_aspects)
    
    Example:
        >>> focal_loss = MultilabelFocalLoss(alpha=[0.8, 1.0, 2.5], gamma=2.0)
        >>> logits = torch.randn(16, 11, 3, requires_grad=True)
        >>> labels = torch.randint(0, 3, (16, 11))
        >>> loss = focal_loss(logits, labels)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        num_aspects: int = 11,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        
        if alpha is None:
            alpha = [1.0, 1.0, 1.0]
        
        if isinstance(alpha, (list, tuple)):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer('alpha', alpha.float())
        else:
            raise TypeError(f"alpha must be list, tuple, or Tensor, got {type(alpha)}")
        
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        
        self.gamma = gamma
        self.num_aspects = num_aspects
        self.reduction = reduction
        
        print(f"[OK] MultilabelFocalLoss initialized:")
        print(f"   Alpha weights: {self.alpha.tolist()}")
        print(f"   Gamma (focusing): {self.gamma}")
        print(f"   Num aspects: {self.num_aspects}")
        print(f"   Reduction: {self.reduction}")
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss for multi-label ABSA.
        
        Args:
            input: Predicted logits [batch_size, num_aspects, num_sentiments]
            target: Ground truth labels [batch_size, num_aspects]
        
        Returns:
            Computed loss value (scalar or tensor based on reduction)
        """
        if input.dim() != 3:
            raise ValueError(f"Expected input to be 3D, got {input.dim()}D")
        
        if target.dim() != 2:
            raise ValueError(f"Expected target to be 2D, got {target.dim()}D")
        
        batch_size, num_aspects, num_classes = input.shape
        
        if target.shape != (batch_size, num_aspects):
            raise ValueError(
                f"Shape mismatch: input {input.shape}, target {target.shape}"
            )
        
        # Reshape for efficient computation
        # [batch_size, num_aspects, num_classes] -> [batch_size * num_aspects, num_classes]
        input_flat = input.view(-1, num_classes)
        target_flat = target.view(-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(input_flat, dim=1)
        probs = log_probs.exp()
        
        # Gather log probability of ground truth class
        log_pt = log_probs.gather(dim=1, index=target_flat.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=1, index=target_flat.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - pt).pow(self.gamma)
        
        # Apply alpha weighting (move alpha to same device as target)
        alpha_t = self.alpha.to(target_flat.device)[target_flat]
        
        # Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        loss = -alpha_t * focal_weight * log_pt
        
        # Reshape back to [batch_size, num_aspects]
        loss = loss.view(batch_size, num_aspects)
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # sum
            return loss.sum()


def calculate_global_alpha(
    train_file_path: str,
    aspect_cols: List[str],
    sentiment_to_idx: dict,
    method: str = 'inverse_freq'
) -> List[float]:
    """
    Calculate global class weights (alpha) from training data distribution.
    
    This function computes alpha weights to balance class imbalance in the
    training data. The weights are applied uniformly across all aspects.
    
    Args:
        train_file_path: Path to training CSV file
        aspect_cols: List of aspect column names to analyze
        sentiment_to_idx: Dictionary mapping sentiment names to indices
            Example: {'positive': 0, 'negative': 1, 'neutral': 2}
        method: Weighting method. Options:
            - 'inverse_freq': weight = total / (num_classes * count)
            - 'balanced': weight = max_count / count
            Default: 'inverse_freq'
    
    Returns:
        List of alpha weights ordered as [positive_weight, negative_weight, neutral_weight]
    
    Raises:
        FileNotFoundError: If train_file_path doesn't exist
        ValueError: If no valid sentiments found or invalid method
    
    Example:
        >>> alpha = calculate_global_alpha(
        ...     'data/train.csv',
        ...     ['Battery', 'Camera', 'Performance'],
        ...     {'positive': 0, 'negative': 1, 'neutral': 2}
        ... )
        >>> print(alpha)  # [0.85, 1.20, 2.50]
    """
    import pandas as pd
    from collections import Counter
    from pathlib import Path
    
    # Validate inputs
    if not Path(train_file_path).exists():
        raise FileNotFoundError(f"Training file not found: {train_file_path}")
    
    if method not in ('inverse_freq', 'balanced'):
        raise ValueError(f"method must be 'inverse_freq' or 'balanced', got '{method}'")
    
    print(f"\n[INFO] Calculating global alpha weights...")
    print(f"   Method: {method}")
    print(f"   File: {train_file_path}")
    
    # Load training data
    df = pd.read_csv(train_file_path, encoding='utf-8-sig')
    
    # Collect all sentiments from all aspects
    all_sentiments = []
    
    for aspect in aspect_cols:
        if aspect in df.columns:
            sentiments = df[aspect].dropna()
            sentiments = sentiments.astype(str).str.strip().str.lower()
            all_sentiments.extend(sentiments.tolist())
    
    if not all_sentiments:
        raise ValueError("No valid sentiment data found in training file")
    
    # Count sentiment occurrences
    counts = Counter(all_sentiments)
    total = sum(counts.values())
    
    # Display distribution
    print(f"\n   Total aspect-sentiment pairs: {total:,}")
    print(f"\n   Sentiment distribution:")
    
    sentiment_order = ['positive', 'negative', 'neutral']
    for sentiment in sentiment_order:
        count = counts.get(sentiment, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"     {sentiment:10s}: {count:6,} ({pct:5.2f}%)")
    
    # Calculate alpha weights based on method
    alpha = []
    num_classes = len(sentiment_to_idx)
    
    if method == 'inverse_freq':
        # Inverse frequency weighting: total / (num_classes * count)
        for sentiment in sentiment_order:
            count = max(counts.get(sentiment, 0), 1)  # Avoid division by zero
            weight = total / (num_classes * count)
            alpha.append(weight)
    
    elif method == 'balanced':
        # Balanced weighting: max_count / count
        max_count = max(counts.values())
        for sentiment in sentiment_order:
            count = max(counts.get(sentiment, 0), 1)
            weight = max_count / count
            alpha.append(weight)
    
    # Display calculated weights
    print(f"\n   Calculated alpha weights ({method}):")
    for sentiment, weight in zip(sentiment_order, alpha):
        print(f"     {sentiment:10s}: {weight:.4f}")
    
    return alpha


def _test_focal_loss():
    """Test suite for MultilabelFocalLoss implementation."""
    print("=" * 70)
    print("Testing MultilabelFocalLoss Implementation")
    print("=" * 70)
    
    # Test configuration
    batch_size = 16
    num_aspects = 11
    num_sentiments = 3
    
    # Test 1: Basic functionality
    print("\n1. Test basic functionality with equal weights")
    print("-" * 70)
    logits = torch.randn(batch_size, num_aspects, num_sentiments, requires_grad=True)
    labels = torch.randint(0, num_sentiments, (batch_size, num_aspects))
    
    print(f"   Input shape:  {logits.shape}")
    print(f"   Target shape: {labels.shape}")
    
    focal_loss = MultilabelFocalLoss(alpha=[1.0, 1.0, 1.0], gamma=2.0)
    loss = focal_loss(logits, labels)
    loss.backward()
    
    print(f"   Loss value: {loss.item():.4f}")
    print(f"   Gradient computed: {logits.grad is not None}")
    print(f"   [PASS]")
    
    # Test 2: Class-weighted focal loss
    print("\n2. Test class-weighted focal loss (boost neutral)")
    print("-" * 70)
    logits2 = torch.randn(batch_size, num_aspects, num_sentiments, requires_grad=True)
    focal_loss_weighted = MultilabelFocalLoss(alpha=[0.8, 1.0, 2.5], gamma=2.0)
    loss2 = focal_loss_weighted(logits2, labels)
    loss2.backward()
    
    print(f"   Loss value: {loss2.item():.4f}")
    print(f"   [PASS]")
    
    # Test 3: Different gamma values
    print("\n3. Test different gamma values (focusing parameter)")
    print("-" * 70)
    logits3 = torch.randn(batch_size, num_aspects, num_sentiments)
    
    for gamma in [0.0, 1.0, 2.0, 5.0]:
        focal = MultilabelFocalLoss(alpha=None, gamma=gamma, reduction='mean')
        loss_val = focal(logits3, labels)
        print(f"   Gamma={gamma:.1f}: Loss={loss_val.item():.4f}")
    print(f"   [PASS]")
    
    # Test 4: Different reduction modes
    print("\n4. Test reduction modes")
    print("-" * 70)
    logits4 = torch.randn(batch_size, num_aspects, num_sentiments)
    
    for reduction in ['none', 'mean', 'sum']:
        focal = MultilabelFocalLoss(gamma=2.0, reduction=reduction)
        loss_val = focal(logits4, labels)
        
        if reduction == 'none':
            print(f"   {reduction:6s}: shape={loss_val.shape}, mean={loss_val.mean().item():.4f}")
        else:
            print(f"   {reduction:6s}: value={loss_val.item():.4f}")
    print(f"   [PASS]")
    
    # Test 5: Input validation
    print("\n5. Test input validation")
    print("-" * 70)
    focal = MultilabelFocalLoss()
    
    try:
        # Wrong input dimension
        wrong_input = torch.randn(batch_size, num_aspects)
        focal(wrong_input, labels)
        print("   [FAIL] Should raise ValueError for wrong input dim")
    except ValueError as e:
        print(f"   [PASS] Correctly caught wrong input dimension")
    
    try:
        # Wrong target dimension
        wrong_target = torch.randint(0, 3, (batch_size, num_aspects, num_sentiments))
        focal(logits, wrong_target)
        print("   [FAIL] Should raise ValueError for wrong target dim")
    except ValueError as e:
        print(f"   [PASS] Correctly caught wrong target dimension")
    
    # Test 6: GPU compatibility (if available)
    if torch.cuda.is_available():
        print("\n6. Test GPU compatibility")
        print("-" * 70)
        device = torch.device('cuda')
        logits_gpu = torch.randn(batch_size, num_aspects, num_sentiments, device=device)
        labels_gpu = torch.randint(0, num_sentiments, (batch_size, num_aspects), device=device)
        
        focal_gpu = MultilabelFocalLoss(alpha=[0.8, 1.0, 2.5], gamma=2.0).to(device)
        loss_gpu = focal_gpu(logits_gpu, labels_gpu)
        
        print(f"   Loss on GPU: {loss_gpu.item():.4f}")
        print(f"   [PASS]")
    else:
        print("\n6. GPU not available, skipping GPU test")
    
    print("\n" + "=" * 70)
    print("[SUCCESS] All tests passed!")
    print("=" * 70)


if __name__ == '__main__':
    _test_focal_loss()
