"""
Focal Loss for Dual Task Learning (Aspect Detection + Sentiment Classification)
===============================================================================
PyTorch implementation optimized for dual-task ABSA training.

Dual Task Learning:
    Task 1: Aspect Detection (Binary) - Predict if aspect is present/absent
    Task 2: Sentiment Classification (3-class) - Predict sentiment for detected aspects

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

Key Features:
    - BinaryFocalLoss: For aspect detection (present/absent classification)
    - MultilabelFocalLoss: For sentiment classification (positive/negative/neutral)
    - Handles class imbalance through alpha weighting
    - Focuses on hard examples through gamma parameter
    - Numerical stability optimizations

Usage:
    # Task 1: Aspect Detection (Binary Focal Loss)
    focal_loss_ad = BinaryFocalLoss(alpha=[1.0, 5.0], gamma=2.0, reduction='none')
    
    # Task 2: Sentiment Classification (Multi-class Focal Loss)
    focal_loss_sc = MultilabelFocalLoss(alpha=[1.06, 0.92, 1.04], gamma=2.0, reduction='none')
    
    # In training loop:
    loss_ad = focal_loss_ad(aspect_logits, aspect_labels).mean()  # On ALL aspects
    loss_sc = (focal_loss_sc(sentiment_logits, sentiment_labels) * aspect_mask).sum() / aspect_mask.sum()
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for Aspect Detection Task (Task 1).
    
    Predicts whether each aspect is present (1) or absent (0) in the text.
    This loss is computed on ALL aspects (both labeled and unlabeled) to ensure
    the model learns both positive and negative predictions.
    
    Args:
        alpha: Class weights [weight_for_absent, weight_for_present].
            Higher weight for present class helps with class imbalance.
            Default: [1.0, 1.0]
        gamma: Focusing parameter. Higher values (>1) down-weight easy examples.
            Recommended: 2.0. Default: 2.0
        reduction: Loss reduction mode. 'none' returns per-aspect losses,
            'mean'/'sum' returns aggregated loss. Default: 'mean'
    
    Shape:
        - Input: (batch_size, num_aspects) - raw logits
        - Target: (batch_size, num_aspects) - binary labels (0 or 1)
        - Output: (batch_size, num_aspects) if reduction='none', else scalar
    
    Example:
        >>> loss_fn = BinaryFocalLoss(alpha=[0.2, 1.0], gamma=2.0)
        >>> logits = torch.randn(16, 11)
        >>> labels = torch.randint(0, 2, (16, 11)).float()
        >>> loss = loss_fn(logits, labels)  # Computes loss on all 11 aspects
    """
    
    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        
        # Set default alpha if not provided
        if alpha is None:
            alpha = [1.0, 1.0]
        
        # Convert alpha to tensor and register as buffer
        if isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.float()
        else:
            raise TypeError(f"alpha must be list, tuple, or Tensor, got {type(alpha)}")
        
        if len(alpha_tensor) != 2:
            raise ValueError(f"alpha must have 2 elements [absent, present], got {len(alpha_tensor)}")
        
        self.register_buffer('alpha', alpha_tensor)
        
        # Validate gamma
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.gamma = gamma
        
        # Validate reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        self.reduction = reduction
        
        # Print initialization info (only once during module creation)
        if not hasattr(self, '_initialized'):
            print(f"[OK] BinaryFocalLoss initialized:")
            print(f"   Alpha weights: {self.alpha.tolist()} (absent, present)")
            print(f"   Gamma (focusing): {self.gamma}")
            print(f"   Reduction: {self.reduction}")
            self._initialized = True
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Focal Loss.
        
        Args:
            input: Predicted logits [batch_size, num_aspects]
            target: Ground truth binary labels [batch_size, num_aspects] (0 or 1)
        
        Returns:
            Loss tensor with shape based on reduction mode
        """
        # Input validation
        if input.dim() != 2:
            raise ValueError(f"Expected input to be 2D [batch_size, num_aspects], got {input.dim()}D")
        if target.dim() != 2:
            raise ValueError(f"Expected target to be 2D [batch_size, num_aspects], got {target.dim()}D")
        
        batch_size, num_aspects = input.shape
        if target.shape != (batch_size, num_aspects):
            raise ValueError(f"Shape mismatch: input {input.shape}, target {target.shape}")
        
        # Compute probabilities using sigmoid (numerically stable)
        probs = torch.sigmoid(input)  # [batch_size, num_aspects]
        
        # Compute probability of true class: p_t = p when target=1, (1-p) when target=0
        # This gives high probability when prediction matches ground truth
        p_t = probs * target + (1.0 - probs) * (1.0 - target)
        
        # Compute log probability with numerical stability
        # Use log(sigmoid) trick for better numerical stability
        log_p_t = F.logsigmoid(input) * target + F.logsigmoid(-input) * (1.0 - target)
        
        # Focal weight: down-weight easy examples, focus on hard examples
        # (1 - p_t)^gamma: smaller when p_t is high (easy example), larger when p_t is low (hard example)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        
        # Apply class weighting: alpha[1] for present, alpha[0] for absent
        alpha_t = self.alpha[1] * target + self.alpha[0] * (1.0 - target)
        
        # Binary focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
        loss = -alpha_t * focal_weight * log_p_t
        
        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        else:  # sum
            return loss.sum()


class MultilabelFocalLoss(nn.Module):
    """
    Multi-class Focal Loss for Sentiment Classification Task (Task 2).
    
    Predicts sentiment (positive/negative/neutral) for each detected aspect.
    This loss is computed ONLY on labeled aspects (where aspect is present).
    Unlabeled aspects should be masked out using aspect_mask in training.
    
    Args:
        alpha: Class weights [weight_for_positive, weight_for_negative, weight_for_neutral].
            Used to balance class imbalance. Can be computed using calculate_global_alpha().
            Default: [1.0, 1.0, 1.0]
        gamma: Focusing parameter. Higher values (>1) down-weight easy examples.
            Recommended: 2.0. Default: 2.0
        num_aspects: Number of aspects (for documentation/logging). Default: 11
        reduction: Loss reduction mode. 'none' returns per-aspect losses,
            'mean'/'sum' returns aggregated loss. Default: 'mean'
    
    Shape:
        - Input: (batch_size, num_aspects, num_sentiments) - raw logits
        - Target: (batch_size, num_aspects) - class indices (0=positive, 1=negative, 2=neutral)
        - Output: (batch_size, num_aspects) if reduction='none', else scalar
    
    Example:
        >>> loss_fn = MultilabelFocalLoss(alpha=[1.06, 0.92, 1.04], gamma=2.0)
        >>> logits = torch.randn(16, 11, 3)
        >>> labels = torch.randint(0, 3, (16, 11))
        >>> mask = torch.randint(0, 2, (16, 11)).float()
        >>> loss = (loss_fn(logits, labels) * mask).sum() / mask.sum()
    """
    
    def __init__(
        self,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        num_aspects: int = 11,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        
        # Set default alpha if not provided
        if alpha is None:
            alpha = [1.0, 1.0, 1.0]
        
        # Convert alpha to tensor and register as buffer
        if isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        elif isinstance(alpha, torch.Tensor):
            alpha_tensor = alpha.float()
        else:
            raise TypeError(f"alpha must be list, tuple, or Tensor, got {type(alpha)}")
        
        if len(alpha_tensor) != 3:
            raise ValueError(f"alpha must have 3 elements [positive, negative, neutral], got {len(alpha_tensor)}")
        
        self.register_buffer('alpha', alpha_tensor)
        
        # Validate gamma
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.gamma = gamma
        
        # Validate reduction
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        self.reduction = reduction
        self.num_aspects = num_aspects
        
        # Print initialization info (only once during module creation)
        if not hasattr(self, '_initialized'):
            print(f"[OK] MultilabelFocalLoss initialized:")
            print(f"   Alpha weights: {self.alpha.tolist()} (positive, negative, neutral)")
            print(f"   Gamma (focusing): {self.gamma}")
            print(f"   Num aspects: {self.num_aspects}")
            print(f"   Reduction: {self.reduction}")
            self._initialized = True
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Multi-class Focal Loss for sentiment classification.
        
        Args:
            input: Predicted logits [batch_size, num_aspects, num_sentiments]
            target: Ground truth class indices [batch_size, num_aspects]
                   (0=positive, 1=negative, 2=neutral)
        
        Returns:
            Loss tensor with shape based on reduction mode
        """
        # Input validation
        if input.dim() != 3:
            raise ValueError(f"Expected input to be 3D [batch_size, num_aspects, num_sentiments], got {input.dim()}D")
        if target.dim() != 2:
            raise ValueError(f"Expected target to be 2D [batch_size, num_aspects], got {target.dim()}D")
        
        batch_size, num_aspects, num_classes = input.shape
        if target.shape != (batch_size, num_aspects):
            raise ValueError(f"Shape mismatch: input {input.shape}, target {target.shape}")
        
        # Flatten for efficient computation: [batch_size * num_aspects, num_classes]
        input_flat = input.view(-1, num_classes)
        target_flat = target.view(-1)
        
        # Compute log probabilities using log_softmax (numerically stable)
        log_probs = F.log_softmax(input_flat, dim=1)
        probs = log_probs.exp()
        
        # Gather log probability and probability of ground truth class
        # This selects the predicted probability for the correct class
        log_pt = log_probs.gather(dim=1, index=target_flat.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=1, index=target_flat.unsqueeze(1)).squeeze(1)
        
        # Focal weight: down-weight easy examples, focus on hard examples
        # (1 - p_t)^gamma: smaller when p_t is high (easy example), larger when p_t is low (hard example)
        focal_weight = (1.0 - pt).pow(self.gamma)
        
        # Apply class weighting: select alpha weight for each ground truth class
        alpha_t = self.alpha[target_flat]
        
        # Multi-class focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)
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


class DualTaskFocalLoss(nn.Module):
    """
    Combined Focal Loss for Dual Task Learning.
    
    Combines Aspect Detection (Task 1) and Sentiment Classification (Task 2)
    into a single loss function with configurable task weights.
    
    Args:
        alpha_ad: Class weights for aspect detection [weight_for_absent, weight_for_present].
            Default: [1.0, 1.0]
        alpha_sc: Class weights for sentiment classification [positive, negative, neutral].
            Default: [1.0, 1.0, 1.0]
        gamma: Focusing parameter for both tasks. Default: 2.0
        weight_ad: Weight for aspect detection loss in combined loss. Default: 0.3
        weight_sc: Weight for sentiment classification loss in combined loss. Default: 0.7
        num_aspects: Number of aspects. Default: 11
    
    Shape:
        - aspect_logits: (batch_size, num_aspects) - raw logits for aspect detection
        - sentiment_logits: (batch_size, num_aspects, num_sentiments) - raw logits for sentiment
        - aspect_labels: (batch_size, num_aspects) - binary labels (0 or 1)
        - sentiment_labels: (batch_size, num_aspects) - class indices (0, 1, or 2)
        - aspect_mask: (batch_size, num_aspects) - 1.0 for labeled, 0.0 for unlabeled
        - Output: scalar loss value
    
    Example:
        >>> loss_fn = DualTaskFocalLoss(weight_ad=0.3, weight_sc=0.7)
        >>> aspect_logits = torch.randn(16, 11)
        >>> sentiment_logits = torch.randn(16, 11, 3)
        >>> aspect_labels = torch.randint(0, 2, (16, 11)).float()
        >>> sentiment_labels = torch.randint(0, 3, (16, 11))
        >>> aspect_mask = torch.randint(0, 2, (16, 11)).float()
        >>> loss = loss_fn(aspect_logits, sentiment_logits, aspect_labels, 
        ...                sentiment_labels, aspect_mask)
    """
    
    def __init__(
        self,
        alpha_ad: Optional[Union[List[float], torch.Tensor]] = None,
        alpha_sc: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        weight_ad: float = 0.3,
        weight_sc: float = 0.7,
        num_aspects: int = 11
    ) -> None:
        super().__init__()
        
        # Validate task weights
        if abs(weight_ad + weight_sc - 1.0) > 1e-6:
            print(f"[WARNING] Task weights sum to {weight_ad + weight_sc:.3f}, expected 1.0")
            # Normalize weights
            total = weight_ad + weight_sc
            weight_ad = weight_ad / total
            weight_sc = weight_sc / total
            print(f"   Normalized to: AD={weight_ad:.3f}, SC={weight_sc:.3f}")
        
        self.weight_ad = weight_ad
        self.weight_sc = weight_sc
        
        # Initialize individual loss functions
        self.loss_ad = BinaryFocalLoss(alpha=alpha_ad, gamma=gamma, reduction='mean')
        self.loss_sc = MultilabelFocalLoss(alpha=alpha_sc, gamma=gamma, 
                                           num_aspects=num_aspects, reduction='none')
        
        # Store config for printing
        self.gamma = gamma
        self.num_aspects = num_aspects
        
        # Print initialization info
        if not hasattr(self, '_initialized'):
            print(f"[OK] DualTaskFocalLoss initialized:")
            print(f"   Aspect Detection (AD):")
            print(f"      Alpha: {self.loss_ad.alpha.tolist()} (absent, present)")
            print(f"      Weight: {self.weight_ad:.3f}")
            print(f"   Sentiment Classification (SC):")
            print(f"      Alpha: {self.loss_sc.alpha.tolist()} (positive, negative, neutral)")
            print(f"      Weight: {self.weight_sc:.3f}")
            print(f"   Gamma (focusing): {self.gamma}")
            print(f"   Total weight: {self.weight_ad + self.weight_sc:.3f}")
            self._initialized = True
    
    def forward(
        self,
        aspect_logits: torch.Tensor,
        sentiment_logits: torch.Tensor,
        aspect_labels: torch.Tensor,
        sentiment_labels: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined dual-task focal loss.
        
        Args:
            aspect_logits: Predicted logits for aspect detection [batch_size, num_aspects]
            sentiment_logits: Predicted logits for sentiment [batch_size, num_aspects, num_sentiments]
            aspect_labels: Ground truth aspect labels [batch_size, num_aspects] (0 or 1)
            sentiment_labels: Ground truth sentiment labels [batch_size, num_aspects] (0, 1, or 2)
            aspect_mask: Mask indicating labeled aspects [batch_size, num_aspects] (1.0 for labeled, 0.0 for unlabeled)
        
        Returns:
            Combined loss scalar: weight_ad * loss_ad + weight_sc * loss_sc
        """
        # Validate input shapes
        batch_size = aspect_logits.shape[0]
        
        if aspect_logits.dim() != 2:
            raise ValueError(f"Expected aspect_logits to be 2D, got {aspect_logits.dim()}D")
        if sentiment_logits.dim() != 3:
            raise ValueError(f"Expected sentiment_logits to be 3D, got {sentiment_logits.dim()}D")
        if aspect_labels.shape != aspect_logits.shape:
            raise ValueError(f"Aspect label shape mismatch: {aspect_labels.shape} vs {aspect_logits.shape}")
        if sentiment_labels.shape != (batch_size, self.num_aspects):
            raise ValueError(f"Sentiment label shape mismatch: {sentiment_labels.shape}")
        if aspect_mask.shape != aspect_logits.shape:
            raise ValueError(f"Aspect mask shape mismatch: {aspect_mask.shape}")
        
        # Task 1: Aspect Detection Loss
        # CRITICAL: Compute on ALL aspects (both labeled=1 and unlabeled=0)
        # This ensures model learns both positive and negative predictions
        loss_ad = self.loss_ad(aspect_logits, aspect_labels.float())
        
        # Task 2: Sentiment Classification Loss
        # Compute per-aspect loss first (reduction='none' returns [batch_size, num_aspects])
        loss_sc_per_aspect = self.loss_sc(sentiment_logits, sentiment_labels)
        
        # Apply mask: only compute loss on labeled aspects (where mask=1.0)
        # Unlabeled aspects (mask=0.0) should not contribute to sentiment loss
        masked_loss_sc = loss_sc_per_aspect * aspect_mask
        
        # Average over labeled aspects only
        num_labeled = aspect_mask.sum()
        if num_labeled > 0:
            loss_sc = masked_loss_sc.sum() / num_labeled
        else:
            # Fallback if no labeled aspects (shouldn't happen in normal training)
            loss_sc = masked_loss_sc.sum()
        
        # Combined loss: weighted sum of both tasks
        total_loss = self.weight_ad * loss_ad + self.weight_sc * loss_sc
        
        return total_loss
    
    def get_loss_components(
        self,
        aspect_logits: torch.Tensor,
        sentiment_logits: torch.Tensor,
        aspect_labels: torch.Tensor,
        sentiment_labels: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> dict:
        """
        Compute loss and return individual components for monitoring.
        
        Returns:
            Dictionary with keys: 'total', 'ad', 'sc', 'ad_weighted', 'sc_weighted'
        """
        # Compute both losses
        loss_ad = self.loss_ad(aspect_logits, aspect_labels.float())
        
        loss_sc_per_aspect = self.loss_sc(sentiment_logits, sentiment_labels)
        masked_loss_sc = loss_sc_per_aspect * aspect_mask
        num_labeled = aspect_mask.sum()
        loss_sc = masked_loss_sc.sum() / num_labeled if num_labeled > 0 else masked_loss_sc.sum()
        
        # Weighted components
        loss_ad_weighted = self.weight_ad * loss_ad
        loss_sc_weighted = self.weight_sc * loss_sc
        
        return {
            'total': loss_ad_weighted + loss_sc_weighted,
            'ad': loss_ad,
            'sc': loss_sc,
            'ad_weighted': loss_ad_weighted,
            'sc_weighted': loss_sc_weighted
        }


def calculate_global_alpha(
    train_file_path: str,
    aspect_cols: List[str],
    sentiment_to_idx: dict,
    method: str = 'inverse_freq'
) -> List[float]:
    """
    Calculate global class weights (alpha) from training data distribution.
    
    Computes alpha weights to balance class imbalance in sentiment classification.
    The weights are applied uniformly across all aspects.
    
    IMPORTANT: Only counts labeled aspects (non-NaN). NaN aspects are excluded
    as they represent unlabeled data, not neutral sentiment.
    
    Args:
        train_file_path: Path to training CSV file
        aspect_cols: List of aspect column names (e.g., ['Battery', 'Camera', ...])
        sentiment_to_idx: Dictionary mapping sentiment names to indices.
            Example: {'positive': 0, 'negative': 1, 'neutral': 2}
        method: Weighting method:
            - 'inverse_freq': weight = total / (num_classes * count)
                Balances classes by inverse frequency
            - 'balanced': weight = max_count / count
                Makes all classes have equal effective weight
            Default: 'inverse_freq'
    
    Returns:
        List of alpha weights [positive_weight, negative_weight, neutral_weight]
        ordered according to sentiment_to_idx
    
    Raises:
        FileNotFoundError: If train_file_path doesn't exist
        ValueError: If no valid sentiments found or invalid method
    
    Example:
        >>> alpha = calculate_global_alpha(
        ...     'data/train_multilabel_balanced.csv',
        ...     ['Battery', 'Camera', 'Performance'],
        ...     {'positive': 0, 'negative': 1, 'neutral': 2}
        ... )
        >>> print(alpha)  # [1.0612, 0.9160, 1.0353]
    """
    import pandas as pd
    from collections import Counter
    from pathlib import Path
    
    # Validate inputs
    train_path = Path(train_file_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_file_path}")
    
    if method not in ('inverse_freq', 'balanced'):
        raise ValueError(f"method must be 'inverse_freq' or 'balanced', got '{method}'")
    
    print(f"\n[INFO] Calculating global alpha weights...")
    print(f"   Method: {method}")
    print(f"   File: {train_file_path}")
    
    # Load training data
    df = pd.read_csv(train_file_path, encoding='utf-8-sig')
    
    # Collect all sentiments from all aspects (ONLY labeled, drop NaN)
    all_sentiments = []
    for aspect in aspect_cols:
        if aspect in df.columns:
            # IMPORTANT: dropna() - NaN means unlabeled, not neutral!
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
    print(f"\n   Sentiment distribution (labeled only):")
    
    sentiment_order = ['positive', 'negative', 'neutral']
    for sentiment in sentiment_order:
        count = counts.get(sentiment, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"     {sentiment:10s}: {count:6,} ({pct:5.2f}%)")
    
    # Calculate alpha weights based on method
    alpha = []
    num_classes = len(sentiment_to_idx)
    
    if method == 'inverse_freq':
        # Inverse frequency weighting: total / (num_classes * count)
        # Rare classes get higher weight
        for sentiment in sentiment_order:
            count = max(counts.get(sentiment, 0), 1)  # Avoid division by zero
            weight = total / (num_classes * count)
            alpha.append(weight)
    
    elif method == 'balanced':
        # Balanced weighting: max_count / count
        # All classes get equal effective weight
        max_count = max(counts.values()) if counts else 1
        for sentiment in sentiment_order:
            count = max(counts.get(sentiment, 0), 1)
            weight = max_count / count
            alpha.append(weight)
    
    # Display calculated weights
    print(f"\n   Calculated alpha weights ({method}):")
    for sentiment, weight in zip(sentiment_order, alpha):
        print(f"     {sentiment:10s}: {weight:.4f}")
    
    return alpha


def calculate_aspect_detection_alpha(
    train_file_path: str,
    aspect_cols: List[str],
    method: str = 'inverse_freq'
) -> List[float]:
    """
    Calculate alpha weights for aspect detection (binary classification: absent/present).
    
    Computes alpha weights to balance class imbalance in aspect detection.
    The weights are calculated based on the distribution of absent vs present aspects.
    
    Args:
        train_file_path: Path to training CSV file
        aspect_cols: List of aspect column names (e.g., ['Battery', 'Camera', ...])
        method: Weighting method:
            - 'inverse_freq': weight = total / (num_classes * count)
                Balances classes by inverse frequency
            - 'balanced': weight = max_count / count
                Makes all classes have equal effective weight
            Default: 'inverse_freq'
    
    Returns:
        List of alpha weights [absent_weight, present_weight]
        for binary classification (0=absent, 1=present)
    
    Raises:
        FileNotFoundError: If train_file_path doesn't exist
        ValueError: If no valid data found or invalid method
    
    Example:
        >>> alpha = calculate_aspect_detection_alpha(
        ...     'data/train_multilabel.csv',
        ...     ['Battery', 'Camera', 'Performance']
        ... )
        >>> print(alpha)  # [0.5, 2.0]
    """
    import pandas as pd
    from collections import Counter
    from pathlib import Path
    
    # Validate inputs
    train_path = Path(train_file_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_file_path}")
    
    if method not in ('inverse_freq', 'balanced'):
        raise ValueError(f"method must be 'inverse_freq' or 'balanced', got '{method}'")
    
    print(f"\n[INFO] Calculating aspect detection alpha weights...")
    print(f"   Method: {method}")
    print(f"   File: {train_file_path}")
    
    # Load training data
    df = pd.read_csv(train_file_path, encoding='utf-8-sig')
    
    # Collect all aspect detection labels (absent=0, present=1)
    all_labels = []
    for aspect in aspect_cols:
        if aspect in df.columns:
            # Check if aspect is present (non-NaN) or absent (NaN)
            for value in df[aspect]:
                if pd.isna(value) or str(value).strip() == '':
                    all_labels.append(0)  # Absent
                else:
                    # Check if it's a valid sentiment
                    sentiment = str(value).strip().lower()
                    if sentiment in ['positive', 'negative', 'neutral']:
                        all_labels.append(1)  # Present
                    else:
                        all_labels.append(0)  # Absent (invalid)
    
    if not all_labels:
        raise ValueError("No valid aspect detection data found in training file")
    
    # Count absent (0) vs present (1) occurrences
    counts = Counter(all_labels)
    total = sum(counts.values())
    absent_count = counts.get(0, 0)
    present_count = counts.get(1, 0)
    
    # Display distribution
    print(f"\n   Total aspect-label pairs: {total:,}")
    print(f"\n   Aspect detection distribution:")
    absent_pct = (absent_count / total * 100) if total > 0 else 0.0
    present_pct = (present_count / total * 100) if total > 0 else 0.0
    print(f"     absent     : {absent_count:6,} ({absent_pct:5.2f}%)")
    print(f"     present    : {present_count:6,} ({present_pct:5.2f}%)")
    
    # Calculate alpha weights based on method
    num_classes = 2  # Binary classification
    
    if method == 'inverse_freq':
        # Inverse frequency weighting: total / (num_classes * count)
        # Rare classes get higher weight
        absent_weight = total / (num_classes * max(absent_count, 1))
        present_weight = total / (num_classes * max(present_count, 1))
        alpha = [absent_weight, present_weight]
    
    elif method == 'balanced':
        # Balanced weighting: max_count / count
        # All classes get equal effective weight
        max_count = max(absent_count, present_count) if total > 0 else 1
        absent_weight = max_count / max(absent_count, 1)
        present_weight = max_count / max(present_count, 1)
        alpha = [absent_weight, present_weight]
    
    # Display calculated weights
    print(f"\n   Calculated alpha weights ({method}):")
    print(f"     absent     : {alpha[0]:.4f}")
    print(f"     present    : {alpha[1]:.4f}")
    
    return alpha


def _test_focal_loss():
    """Test suite for Focal Loss implementations (both Binary and Multi-class)."""
    print("=" * 80)
    print("Testing Focal Loss Implementation for Dual Task Learning")
    print("=" * 80)
    
    batch_size = 16
    num_aspects = 11
    num_sentiments = 3
    
    # Test 1: BinaryFocalLoss basic functionality
    print("\n[Test 1] BinaryFocalLoss - Basic functionality")
    print("-" * 80)
    logits_ad = torch.randn(batch_size, num_aspects, requires_grad=True)
    labels_ad = torch.randint(0, 2, (batch_size, num_aspects)).float()
    
    focal_ad = BinaryFocalLoss(alpha=[1.0, 5.0], gamma=2.0)
    loss_ad = focal_ad(logits_ad, labels_ad)
    loss_ad.backward()
    
    print(f"   Input shape:  {logits_ad.shape}")
    print(f"   Target shape: {labels_ad.shape}")
    print(f"   Loss value:   {loss_ad.item():.4f}")
    print(f"   Gradient:     {logits_ad.grad is not None}")
    print(f"   [PASS]")
    
    # Test 2: MultilabelFocalLoss basic functionality
    print("\n[Test 2] MultilabelFocalLoss - Basic functionality")
    print("-" * 80)
    logits_sc = torch.randn(batch_size, num_aspects, num_sentiments, requires_grad=True)
    labels_sc = torch.randint(0, num_sentiments, (batch_size, num_aspects))
    
    focal_sc = MultilabelFocalLoss(alpha=[1.0, 1.0, 1.0], gamma=2.0)
    loss_sc = focal_sc(logits_sc, labels_sc)
    loss_sc.backward()
    
    print(f"   Input shape:  {logits_sc.shape}")
    print(f"   Target shape: {labels_sc.shape}")
    print(f"   Loss value:   {loss_sc.item():.4f}")
    print(f"   Gradient:     {logits_sc.grad is not None}")
    print(f"   [PASS]")
    
    # Test 3: Reduction modes
    print("\n[Test 3] Reduction modes")
    print("-" * 80)
    for reduction in ['none', 'mean', 'sum']:
        focal = BinaryFocalLoss(gamma=2.0, reduction=reduction)
        loss = focal(logits_ad, labels_ad)
        
        if reduction == 'none':
            print(f"   {reduction:6s}: shape={loss.shape}, mean={loss.mean().item():.4f}")
        else:
            print(f"   {reduction:6s}: value={loss.item():.4f}")
    print(f"   [PASS]")
    
    # Test 4: GPU compatibility
    if torch.cuda.is_available():
        print("\n[Test 4] GPU compatibility")
        print("-" * 80)
        device = torch.device('cuda')
        
        logits_ad_gpu = logits_ad.to(device)
        labels_ad_gpu = labels_ad.to(device)
        focal_ad_gpu = focal_ad.to(device)
        loss_ad_gpu = focal_ad_gpu(logits_ad_gpu, labels_ad_gpu)
        
        print(f"   Loss on GPU: {loss_ad_gpu.item():.4f}")
        print(f"   [PASS]")
    else:
        print("\n[Test 4] GPU not available, skipping")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    _test_focal_loss()
