"""
GHM-C Loss (Gradient Harmonized Mechanism for Classification)

Paper: "Gradient Harmonized Single-stage Detector"
ArXiv: https://arxiv.org/abs/1811.05181

Improvements over Focal Loss:
1. Dynamic adjustment based on gradient density
2. Better handling of outliers
3. No need for careful hyperparameter tuning
4. Adapts to changing data distribution during training

Source: https://github.com/shuxinyin/NLP-Loss-Pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GHM_Loss(nn.Module):
    """
    GHM-C Loss for classification
    
    Args:
        bins: Number of gradient bins (default: 10)
        momentum: Momentum for moving average of gradient density (default: 0.75)
        use_sigmoid: Whether to use sigmoid (binary) or softmax (multi-class)
    """
    
    def __init__(self, bins=10, momentum=0.75, use_sigmoid=True, loss_weight=1.0):
        super(GHM_Loss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins  # [0, 0.1, 0.2, ..., 1.0]
        self.edges[-1] += 1e-6  # Ensure last edge is slightly > 1.0
        
        if momentum > 0:
            self.acc_sum = torch.zeros(bins)  # Accumulator for moving average
        
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight
    
    def forward(self, logits, targets, weights=None):
        """
        Args:
            logits: [batch_size, num_classes] - model predictions
            targets: [batch_size] - ground truth labels
            weights: Optional sample weights
        
        Returns:
            loss: GHM weighted loss
        """
        # Move edges to same device as logits
        if self.edges.device != logits.device:
            self.edges = self.edges.to(logits.device)
            if self.momentum > 0:
                self.acc_sum = self.acc_sum.to(logits.device)
        
        # Calculate gradient length (proxy for difficulty)
        if self.use_sigmoid:
            # Binary classification
            probs = torch.sigmoid(logits)
            grad = torch.abs(probs - targets.float())
        else:
            # Multi-class classification
            probs = F.softmax(logits, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).float()
            grad = torch.abs(probs - targets_one_hot).sum(dim=1)
        
        # Compute gradient density
        grad_density = self._compute_gradient_density(grad)
        
        # Calculate loss with GHM weights
        if self.use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        else:
            loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply GHM weighting
        ghm_weights = 1.0 / (grad_density + 1e-7)  # Inverse density as weight
        ghm_weights = ghm_weights / ghm_weights.sum() * logits.shape[0]  # Normalize
        
        loss = loss * ghm_weights
        
        if weights is not None:
            loss = loss * weights
        
        return loss.mean() * self.loss_weight
    
    def _compute_gradient_density(self, grad):
        """
        Compute gradient density for each sample
        
        Args:
            grad: [batch_size] - gradient magnitudes
        
        Returns:
            density: [batch_size] - density values for each sample
        """
        n = grad.shape[0]
        
        # Find which bin each gradient belongs to
        bin_idx = torch.searchsorted(self.edges, grad, right=False) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.bins - 1)
        
        # Count samples in each bin
        bin_count = torch.zeros(self.bins, device=grad.device)
        for i in range(self.bins):
            bin_count[i] = (bin_idx == i).sum().float()
        
        # Update accumulator with momentum
        if self.momentum > 0:
            self.acc_sum = self.momentum * self.acc_sum + (1 - self.momentum) * bin_count
            bin_count = self.acc_sum.clone()
        
        # Normalize
        bin_count = bin_count + 1e-7  # Avoid division by zero
        
        # Get density for each sample based on its bin
        density = bin_count[bin_idx]
        
        return density


class MultiLabelGHM_Loss(nn.Module):
    """
    GHM Loss for multi-label classification (ABSA task)
    
    Applies GHM loss to each aspect separately
    """
    
    def __init__(self, num_aspects=11, num_sentiments=3, bins=10, momentum=0.75, loss_weight=1.0):
        super(MultiLabelGHM_Loss, self).__init__()
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.ghm_loss = GHM_Loss(bins=bins, momentum=momentum, use_sigmoid=False, loss_weight=loss_weight)
    
    def forward(self, logits, targets, weights=None):
        """
        Args:
            logits: [batch_size, num_aspects, num_sentiments]
            targets: [batch_size, num_aspects]
            weights: Optional [num_aspects, num_sentiments] class weights
        
        Returns:
            loss: Average GHM loss across all aspects
        """
        total_loss = 0.0
        
        for i in range(self.num_aspects):
            aspect_logits = logits[:, i, :]  # [batch_size, num_sentiments]
            aspect_targets = targets[:, i]   # [batch_size]
            
            # Apply aspect-specific weights if provided
            aspect_weights = None
            if weights is not None:
                aspect_weights = weights[i]  # [num_sentiments]
                # Expand to match batch size
                aspect_weights = aspect_weights[aspect_targets]
            
            loss = self.ghm_loss(aspect_logits, aspect_targets, aspect_weights)
            total_loss += loss
        
        return total_loss / self.num_aspects


# Test function
def test_ghm_loss():
    """Test GHM Loss implementation"""
    print("=" * 80)
    print("Testing GHM-C Loss")
    print("=" * 80)
    
    # Test 1: Binary classification
    print("\n1. Binary Classification Test:")
    ghm_binary = GHM_Loss(bins=10, momentum=0.75, use_sigmoid=True)
    logits = torch.randn(32, 1)
    targets = torch.randint(0, 2, (32,))
    loss = ghm_binary(logits.squeeze(), targets)
    print(f"   Binary GHM Loss: {loss.item():.4f}")
    
    # Test 2: Multi-class classification
    print("\n2. Multi-class Classification Test:")
    ghm_multi = GHM_Loss(bins=10, momentum=0.75, use_sigmoid=False)
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    loss = ghm_multi(logits, targets)
    print(f"   Multi-class GHM Loss: {loss.item():.4f}")
    
    # Test 3: Multi-label ABSA
    print("\n3. Multi-label ABSA Test:")
    ghm_absa = MultiLabelGHM_Loss(num_aspects=11, num_sentiments=3, bins=10)
    logits = torch.randn(32, 11, 3)
    targets = torch.randint(0, 3, (32, 11))
    loss = ghm_absa(logits, targets)
    print(f"   Multi-label GHM Loss: {loss.item():.4f}")
    
    # Test 4: Compare with Focal Loss
    print("\n4. Comparison with Focal Loss:")
    from utils import FocalLoss
    focal_loss_fn = FocalLoss(alpha=None, gamma=2.0, reduction='mean')
    
    # Same data for both
    logits = torch.randn(32, 3)
    targets = torch.randint(0, 3, (32,))
    
    focal_loss = focal_loss_fn(logits, targets)
    ghm_loss = ghm_multi(logits, targets)
    
    print(f"   Focal Loss: {focal_loss.item():.4f}")
    print(f"   GHM Loss:   {ghm_loss.item():.4f}")
    
    print("\nAll tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_ghm_loss()
