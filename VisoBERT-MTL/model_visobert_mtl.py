"""
ViSoBERT Multi-Task Learning Model for Vietnamese ABSA
======================================================
One shared ViSoBERT backbone with two task-specific heads:
    - Head 1: Aspect Detection (AD) - Binary classification (11 aspects)
    - Head 2: Sentiment Classification (SC) - 3-class per aspect (11 × 3)

Architecture:
    Input → ViSoBERT → [CLS] Token
                ↓
    ┌───────────┴───────────┐
    │                       │
  AD Head                SC Head
  Dense → Sigmoid    Dense → Reshape [11, 3]
  11 binary          11 × 3 multi-class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class ViSoBERT_MTL(nn.Module):
    """
    ViSoBERT Multi-Task Learning Model
    
    Shared ViSoBERT backbone + 2 task-specific heads:
        - AD Head: Binary classification (11 aspects)
        - SC Head: Multi-class classification (11 × 3)
    """
    
    def __init__(
        self,
        model_name="5CD-AI/visobert-14gb-corpus",
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3
    ):
        """
        Args:
            model_name: Pretrained ViSoBERT model name
            num_aspects: Number of aspects (11)
            num_sentiments: Number of sentiments (3)
            hidden_size: Hidden size for task heads
            dropout: Dropout rate
        """
        super(ViSoBERT_MTL, self).__init__()
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        
        # Aspect names (for reference)
        self.aspect_names = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        # =====================================================================
        # SHARED BACKBONE: ViSoBERT
        # =====================================================================
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        bert_hidden_size = self.bert.config.hidden_size  # 768
        
        # Shared dense layer (from [CLS] token)
        self.shared_dense = nn.Linear(bert_hidden_size, hidden_size)
        self.shared_activation = nn.ReLU()
        self.shared_dropout = nn.Dropout(dropout)
        
        # =====================================================================
        # TASK-SPECIFIC HEADS
        # =====================================================================
        
        # Head 1: Aspect Detection (Binary classification)
        self.ad_dense = nn.Linear(hidden_size, hidden_size // 2)
        self.ad_activation = nn.ReLU()
        self.ad_dropout = nn.Dropout(dropout)
        self.ad_output = nn.Linear(hidden_size // 2, num_aspects)  # Binary logits
        
        # Head 2: Sentiment Classification (Multi-class)
        self.sc_dense = nn.Linear(hidden_size, hidden_size // 2)
        self.sc_activation = nn.ReLU()
        self.sc_dropout = nn.Dropout(dropout)
        self.sc_output = nn.Linear(hidden_size // 2, num_aspects * num_sentiments)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            ad_logits: [batch_size, num_aspects] (binary logits)
            sc_logits: [batch_size, num_aspects, num_sentiments] (multi-class logits)
        """
        batch_size = input_ids.size(0)
        
        # =====================================================================
        # SHARED BACKBONE
        # =====================================================================
        
        # ViSoBERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Shared dense layer
        shared = self.shared_dense(cls_output)
        shared = self.shared_activation(shared)
        shared = self.shared_dropout(shared)  # [batch_size, hidden_size]
        
        # =====================================================================
        # TASK-SPECIFIC HEADS (PARALLEL, NO DEPENDENCY)
        # =====================================================================
        
        # Head 1: Aspect Detection (from shared, NOT from SC)
        ad_hidden = self.ad_dense(shared)
        ad_hidden = self.ad_activation(ad_hidden)
        ad_hidden = self.ad_dropout(ad_hidden)
        ad_logits = self.ad_output(ad_hidden)  # [batch_size, num_aspects]
        
        # Head 2: Sentiment Classification (from shared, NOT from AD)
        sc_hidden = self.sc_dense(shared)
        sc_hidden = self.sc_activation(sc_hidden)
        sc_hidden = self.sc_dropout(sc_hidden)
        sc_flat = self.sc_output(sc_hidden)  # [batch_size, num_aspects * num_sentiments]
        sc_logits = sc_flat.view(batch_size, self.num_aspects, self.num_sentiments)
        
        return ad_logits, sc_logits
    
    def predict_ad(self, input_ids, attention_mask, threshold=0.5):
        """Predict aspect detection only"""
        with torch.no_grad():
            ad_logits, _ = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(ad_logits)
            preds = (probs >= threshold).long()
        return preds, probs
    
    def predict_sc(self, input_ids, attention_mask):
        """Predict sentiment classification only"""
        with torch.no_grad():
            _, sc_logits = self.forward(input_ids, attention_mask)
            probs = F.softmax(sc_logits, dim=-1)
            preds = torch.argmax(sc_logits, dim=-1)
        return preds, probs
    
    def predict_both(self, input_ids, attention_mask, ad_threshold=0.5):
        """Predict both tasks"""
        with torch.no_grad():
            ad_logits, sc_logits = self.forward(input_ids, attention_mask)
            
            # AD predictions
            ad_probs = torch.sigmoid(ad_logits)
            ad_preds = (ad_probs >= ad_threshold).long()
            
            # SC predictions
            sc_probs = F.softmax(sc_logits, dim=-1)
            sc_preds = torch.argmax(sc_logits, dim=-1)
        
        return ad_preds, ad_probs, sc_preds, sc_probs


def test_model():
    """Test ViSoBERT MTL model"""
    print("="*80)
    print("Testing ViSoBERT Multi-Task Learning Model")
    print("="*80)
    
    # Create model
    print("\n1. Creating model...")
    model = ViSoBERT_MTL(
        model_name="5CD-AI/visobert-14gb-corpus",
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/visobert-14gb-corpus")
    test_text = "Pin tốt camera xấu"
    
    encoding = tokenizer(test_text, max_length=128, padding='max_length', 
                        truncation=True, return_tensors='pt')
    
    model.eval()
    with torch.no_grad():
        ad_logits, sc_logits = model(encoding['input_ids'], encoding['attention_mask'])
    
    print(f"   Input: '{test_text}'")
    print(f"   AD logits shape: {ad_logits.shape}  (expected: [1, 11])")
    print(f"   SC logits shape: {sc_logits.shape}  (expected: [1, 11, 3])")
    
    # Test predictions
    print("\n3. Testing predictions...")
    ad_preds, ad_probs, sc_preds, sc_probs = model.predict_both(
        encoding['input_ids'], encoding['attention_mask']
    )
    
    print(f"   AD predictions (binary): {ad_preds[0]}")
    print(f"   SC predictions (classes): {sc_preds[0]}")
    
    print("\nAll tests passed!")
    print("="*80)


if __name__ == '__main__':
    test_model()
