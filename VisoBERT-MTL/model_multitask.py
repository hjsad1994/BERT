"""
Dual Task Learning Model for Vietnamese ABSA
Task 1: Aspect Detection (AD) - Binary classification (present/absent)
Task 2: Sentiment Classification (SC) - 3-class classification (positive/negative/neutral)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DualTaskViSoBERT(nn.Module):
    """
    Dual Task Learning Model for ABSA
    
    Input:  "Pin trâu camera xấu"
    Output:
        - Aspect Detection: [1, 0, 0, ...] (binary, aspect present/absent)
        - Sentiment Classification: [positive, negative, neutral, ...] (for detected aspects)
    """
    
    def __init__(
        self, 
        model_name="5CD-AI/visobert-14gb-corpus",
        num_aspects=11, 
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3
    ):
        super().__init__()
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        
        # BERT encoder (shared backbone)
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for ViSoBERT
        
        # Shared feature extraction
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        
        # Task 1: Aspect Detection Head (Binary: present/absent)
        self.aspect_detection_head = nn.Linear(hidden_size, num_aspects)
        
        # Task 2: Sentiment Classification Head (3-class: positive/negative/neutral)
        self.sentiment_classification_head = nn.Linear(hidden_size, num_aspects * num_sentiments)
        
        # Aspect names (for reference)
        self.aspect_names = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        # Sentiment names
        self.sentiment_names = ['positive', 'negative', 'neutral']
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for dual task learning
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            aspect_detection_logits: [batch_size, num_aspects] (binary logits)
            sentiment_logits: [batch_size, num_aspects, num_sentiments] (3-class logits)
        """
        # Encode with shared BERT backbone
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Shared feature extraction
        x = self.dropout(cls_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Task 1: Aspect Detection (binary classification for each aspect)
        aspect_detection_logits = self.aspect_detection_head(x)  # [batch_size, num_aspects]
        
        # Task 2: Sentiment Classification (3-class for each aspect)
        sentiment_logits = self.sentiment_classification_head(x)  # [batch_size, num_aspects * num_sentiments]
        sentiment_logits = sentiment_logits.view(-1, self.num_aspects, self.num_sentiments)
        
        return aspect_detection_logits, sentiment_logits
    
    def predict(self, input_ids, attention_mask):
        """
        Predict aspect detection and sentiment classification
        
        Returns:
            aspect_predictions: [batch_size, num_aspects] (binary: 1=present, 0=absent)
            sentiment_predictions: [batch_size, num_aspects] (sentiment class indices)
            aspect_probs: [batch_size, num_aspects] (probability of aspect being present)
            sentiment_probs: [batch_size, num_aspects, num_sentiments]
        """
        with torch.no_grad():
            aspect_logits, sentiment_logits = self.forward(input_ids, attention_mask)
            
            # Aspect detection: sigmoid for binary classification
            aspect_probs = torch.sigmoid(aspect_logits)  # [batch_size, num_aspects]
            aspect_predictions = (aspect_probs > 0.5).long()  # [batch_size, num_aspects]
            
            # Sentiment classification: softmax for 3-class classification
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)  # [batch_size, num_aspects, num_sentiments]
            sentiment_predictions = torch.argmax(sentiment_logits, dim=-1)  # [batch_size, num_aspects]
        
        return aspect_predictions, sentiment_predictions, aspect_probs, sentiment_probs
    
    def predict_with_names(self, input_ids, attention_mask):
        """
        Predict with aspect and sentiment names
        
        Returns:
            dict: {aspect_name: {'present': bool, 'sentiment': str, 'confidence': float}}
        """
        aspect_preds, sentiment_preds, aspect_probs, sentiment_probs = self.predict(input_ids, attention_mask)
        
        # Convert to dict (assume batch_size = 1)
        aspect_preds = aspect_preds[0].cpu().numpy()  # [num_aspects]
        sentiment_preds = sentiment_preds[0].cpu().numpy()  # [num_aspects]
        aspect_probs = aspect_probs[0].cpu().numpy()  # [num_aspects]
        sentiment_probs = sentiment_probs[0].cpu().numpy()  # [num_aspects, num_sentiments]
        
        results = {}
        for i, aspect in enumerate(self.aspect_names):
            is_present = bool(aspect_preds[i])
            sentiment_idx = sentiment_preds[i]
            sentiment = self.sentiment_names[sentiment_idx]
            sentiment_confidence = float(sentiment_probs[i, sentiment_idx])
            
            results[aspect] = {
                'present': is_present,
                'present_confidence': float(aspect_probs[i]),
                'sentiment': sentiment,
                'sentiment_confidence': sentiment_confidence,
                'probabilities': {
                    'positive': float(sentiment_probs[i, 0]),
                    'negative': float(sentiment_probs[i, 1]),
                    'neutral': float(sentiment_probs[i, 2])
                }
            }
        
        return results


def test_model():
    """Test model forward pass"""
    print("=" * 80)
    print("Testing Dual Task ViSoBERT Model")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating model...")
    model = DualTaskViSoBERT(
        model_name="5CD-AI/visobert-14gb-corpus",
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/visobert-14gb-corpus")
    
    # Test text
    test_text = "Pin tot cam era xau"
    
    # Tokenize
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
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        aspect_logits, sentiment_logits = model(input_ids, attention_mask)
    
    print(f"   Aspect Detection logits shape: {aspect_logits.shape}")
    print(f"   Sentiment Classification logits shape: {sentiment_logits.shape}")
    print(f"   Expected: [1, 11] and [1, 11, 3]")
    
    # Test prediction
    print("\n3. Testing prediction...")
    results = model.predict_with_names(input_ids, attention_mask)
    
    print(f"\n   Predictions:")
    for aspect, data in results.items():
        present = "Present" if data['present'] else "Absent"
        sentiment = data['sentiment'] if data['sentiment'] else "N/A"
        present_conf = data['present_confidence']
        sent_conf = data['sentiment_confidence']
        print(f"   {aspect:<15} {present:<8} ({present_conf:.3f}) | Sentiment: {sentiment:<10} ({sent_conf:.3f})")
    
    print("\nAll tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_model()
