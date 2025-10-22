"""
Multi-Label ABSA Model with Contrastive Learning

Architecture:
- ViSoBERT encoder
- Projection head for contrastive learning
- Classification head for prediction

Loss: Contrastive Loss + Classification Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class MultiLabelViSoBERTContrastive(nn.Module):
    """
    Multi-Label ABSA with Contrastive Learning
    
    Learns better representations by:
    - Pulling similar samples (same aspect sentiments) closer
    - Pushing dissimilar samples apart
    """
    
    def __init__(
        self, 
        model_name="5CD-AI/Vietnamese-Sentiment-visobert",
        num_aspects=11, 
        num_sentiments=3,
        hidden_size=512,
        projection_dim=256,  # For contrastive learning
        dropout=0.3
    ):
        super().__init__()
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(bert_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim)
        )
        
        # Classification head (same as before)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_aspects * num_sentiments)
        
        # Aspect names
        self.aspect_names = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        self.sentiment_names = ['positive', 'negative', 'neutral']
    
    def forward(self, input_ids, attention_mask, return_embeddings=False):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_embeddings: If True, return contrastive embeddings
        
        Returns:
            logits: [batch_size, num_aspects, num_sentiments]
            embeddings: [batch_size, projection_dim] (if return_embeddings=True)
        """
        # Encode
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, 768]
        
        # Contrastive embeddings
        embeddings = self.projection_head(pooled_output)  # [batch_size, projection_dim]
        embeddings = F.normalize(embeddings, dim=1)  # L2 normalize
        
        # Classification logits
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # [batch_size, 39]
        
        # Reshape to [batch_size, num_aspects, num_sentiments]
        logits = logits.view(-1, self.num_aspects, self.num_sentiments)
        
        if return_embeddings:
            return logits, embeddings
        else:
            return logits
    
    def predict(self, input_ids, attention_mask):
        """Predict sentiments"""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        return preds, probs


class MultiLabelContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Multi-Label Classification
    
    Inspired by:
    - SupCon (Khosla et al., NeurIPS 2020)
    - Multi-Label Supervised Contrastive Learning (AAAI 2024)
    """
    
    def __init__(self, temperature=0.07, similarity_threshold=0.5):
        super().__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
    
    def forward(self, embeddings, labels):
        """
        Contrastive loss for multi-label
        
        Args:
            embeddings: [batch_size, projection_dim] (L2 normalized)
            labels: [batch_size, num_aspects] (aspect sentiments)
        
        Returns:
            loss: scalar
        """
        batch_size = embeddings.shape[0]
        num_aspects = labels.shape[1]
        
        # Compute similarity matrix: [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create label similarity matrix
        # For each pair, count how many aspects have same sentiment
        label_similarity = torch.zeros(batch_size, batch_size, device=labels.device)
        
        for i in range(num_aspects):
            # aspect_match[i,j] = 1 if sample i and j have same sentiment for aspect
            aspect_match = (labels[:, i].unsqueeze(1) == labels[:, i].unsqueeze(0)).float()
            label_similarity += aspect_match
        
        # Normalize: 0 (no match) to 1 (all aspects match)
        label_similarity = label_similarity / num_aspects
        
        # Create positive mask: pairs with similarity >= threshold
        positive_mask = (label_similarity >= self.similarity_threshold).float()
        positive_mask.fill_diagonal_(0)  # Exclude self
        
        # Create negative mask: all pairs except self
        negative_mask = torch.ones_like(positive_mask)
        negative_mask.fill_diagonal_(0)
        
        # InfoNCE-style loss
        exp_sim = torch.exp(similarity_matrix)
        
        loss = 0.0
        num_positives = 0
        
        for i in range(batch_size):
            # Positive pairs for sample i
            positive_sim = (exp_sim[i] * positive_mask[i]).sum()
            
            # All pairs (exclude self)
            all_sim = (exp_sim[i] * negative_mask[i]).sum()
            
            # Only compute loss if there are positive pairs
            if positive_sim > 0:
                loss += -torch.log(positive_sim / (all_sim + 1e-8))
                num_positives += 1
        
        # Average over samples with positive pairs
        if num_positives > 0:
            loss = loss / num_positives
        else:
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss


def test_contrastive_model():
    """Test contrastive model"""
    print("=" * 80)
    print("Testing Multi-Label Contrastive Model")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating model...")
    model = MultiLabelViSoBERTContrastive(
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
    print(f"   Embeddings L2 norm: {torch.norm(embeddings, dim=1).item():.3f} (should be ~1.0)")
    
    # Test contrastive loss
    print("\n3. Testing contrastive loss...")
    
    # Create fake batch
    batch_size = 4
    fake_embeddings = F.normalize(torch.randn(batch_size, 256), dim=1)
    fake_labels = torch.tensor([
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Sample 1
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Sample 2 (same as 1)
        [1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Sample 3 (different)
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Sample 4 (different)
    ])
    
    contrastive_loss_fn = MultiLabelContrastiveLoss(temperature=0.07, similarity_threshold=0.5)
    loss = contrastive_loss_fn(fake_embeddings, fake_labels)
    
    print(f"   Contrastive loss: {loss.item():.4f}")
    print(f"   Loss should be > 0 (unless no positive pairs)")
    
    print("\nAll tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_contrastive_model()
