"""
Multi-Label Aspect Detection Model for Vietnamese ABSA
Binary classification: Detects which aspects are mentioned in the text
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class AspectDetectionModel(nn.Module):
    """
    Multi-Label Aspect Detection Model
    
    Task: Predict which aspects are mentioned in the review
    Input:  "Pin trâu camera xấu"
    Output: Battery=1 (mentioned), Camera=1 (mentioned), Performance=0 (not mentioned), ...
    """
    
    def __init__(
        self, 
        model_name="5CD-AI/visobert-14gb-corpus",
        num_aspects=11,
        hidden_size=512,
        dropout=0.3
    ):
        super().__init__()
        
        self.num_aspects = num_aspects
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for ViSoBERT
        
        # Binary classifier head for multi-label detection
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        
        # Output: 11 binary predictions (0 = not mentioned, 1 = mentioned)
        self.classifier = nn.Linear(hidden_size, num_aspects)
        
        # Aspect names (for reference, including "Others" for AD stage)
        self.aspect_names = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_aspects] (binary logits for each aspect)
        """
        # Encode text
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Classify
        x = self.dropout(cls_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # [batch_size, num_aspects]
        
        return logits
    
    def predict(self, input_ids, attention_mask, threshold=0.5):
        """
        Predict which aspects are mentioned
        
        Args:
            threshold: Threshold for binary prediction (default: 0.5)
        
        Returns:
            predictions: [batch_size, num_aspects] (binary: 0 or 1)
            probabilities: [batch_size, num_aspects] (sigmoid probabilities)
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
        
        return preds, probs
