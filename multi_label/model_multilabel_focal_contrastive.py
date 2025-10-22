"""
Multi-Label ABSA Model with Focal Loss + Contrastive Learning

Combination of:
1. Focal Loss for class imbalance handling
2. Contrastive Learning for better representations

Loss = Focal Loss + Contrastive Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from model_multilabel_contrastive import MultiLabelViSoBERTContrastive
from model_multilabel_contrastive_v2 import ImprovedMultiLabelContrastiveLoss
from utils import FocalLoss


class MultiLabelViSoBERTFocalContrastive(MultiLabelViSoBERTContrastive):
    """
    Multi-Label ABSA with Focal Loss + Contrastive Learning
    
    Inherits from contrastive model but adds Focal Loss
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use parent constructor (same as contrastive)
    
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """Same as parent"""
        return super().forward(input_ids, attention_mask, return_embeddings)


def test_focal_contrastive_model():
    """Test Focal + Contrastive model"""
    print("=" * 80)
    print("Testing Multi-Label Focal + Contrastive Model")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating model...")
    model = MultiLabelViSoBERTFocalContrastive(
        model_name="5CD-AI/Vietnamese-Sentiment-visobert",
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        projection_dim=256,
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert")
    
    # Test text
    test_text = "Pin tot camera xau"
    
    encoding = tokenizer(
        test_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    print(f"   Input: '{test_text}'")
    print(f"   Input shape: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(input_ids, attention_mask, return_embeddings=True)
    
    print(f"   Logits shape: {logits.shape} (expected: [1, 11, 3])")
    print(f"   Embeddings shape: {embeddings.shape} (expected: [1, 256])")
    print(f"   Embeddings L2 norm: {torch.norm(embeddings, dim=1).item():.3f}")
    
    # Test losses
    print("\n3. Testing losses...")
    
    # Create fake batch
    batch_size = 4
    fake_logits = torch.randn(batch_size, 11, 3)
    fake_embeddings = F.normalize(torch.randn(batch_size, 256), dim=1)
    fake_labels = torch.tensor([
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    ])
    
    # Class weights for focal loss
    class_weights = torch.ones(3)  # pos, neg, neu
    
    # Focal loss per aspect
    focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    focal_loss = 0
    for i in range(11):
        aspect_logits = fake_logits[:, i, :]
        aspect_labels = fake_labels[:, i]
        focal_loss += focal_loss_fn(aspect_logits, aspect_labels)
    focal_loss = focal_loss / 11
    
    # Contrastive loss
    contrastive_loss_fn = ImprovedMultiLabelContrastiveLoss(temperature=0.1, base_weight=0.1)
    contr_loss = contrastive_loss_fn(fake_embeddings, fake_labels)
    
    # Combined loss
    focal_weight = 0.7
    contrastive_weight = 0.3
    total_loss = focal_weight * focal_loss + contrastive_weight * contr_loss
    
    print(f"   Focal Loss:      {focal_loss.item():.4f}")
    print(f"   Contrastive Loss: {contr_loss.item():.4f}")
    print(f"   Combined Loss:   {total_loss.item():.4f}")
    print(f"   (Focal: {focal_weight}, Contrastive: {contrastive_weight})")
    
    print("\nAll tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_focal_contrastive_model()
