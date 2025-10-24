"""
Multi-Label ABSA Model for Vietnamese Sentiment Analysis
Predicts all 13 aspects × 3 sentiments in one forward pass
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MultiLabelViSoBERT(nn.Module):
    """
    Multi-Label ABSA Model
    
    Input:  "Pin trâu camera xấu"
    Output: Battery=positive, Camera=negative, Performance=neutral, ... (13 aspects)
    """
    
    def __init__(
        self, 
        model_name="5CD-AI/Vietnamese-Sentiment-visobert",  # ViSoBERT - efficient and effective
        num_aspects=11, 
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3
    ):
        super().__init__()
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        
        # BERT encoder (without pooler to avoid warning)
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for ViSoBERT
        
        # Multi-label classifier head
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        
        # Output: 13 aspects × 3 sentiments = 39 classes
        self.classifier = nn.Linear(hidden_size, num_aspects * num_sentiments)
        
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
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, num_aspects, num_sentiments]
        """
        # Encode
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token) as sentence representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Classify
        x = self.dropout(cls_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # [batch_size, 39]
        
        # Reshape to [batch_size, num_aspects, num_sentiments]
        logits = logits.view(-1, self.num_aspects, self.num_sentiments)
        
        return logits
    
    def predict(self, input_ids, attention_mask):
        """
        Predict sentiments for all aspects
        
        Returns:
            predictions: [batch_size, num_aspects] (class indices)
            probabilities: [batch_size, num_aspects, num_sentiments]
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        return preds, probs
    
    def predict_with_names(self, input_ids, attention_mask):
        """
        Predict with aspect and sentiment names
        
        Returns:
            dict: {aspect_name: sentiment_name}
        """
        preds, probs = self.predict(input_ids, attention_mask)
        
        # Convert to dict (assume batch_size = 1)
        preds = preds[0].cpu().numpy()  # [num_aspects]
        probs = probs[0].cpu().numpy()  # [num_aspects, num_sentiments]
        
        results = {}
        for i, aspect in enumerate(self.aspect_names):
            sentiment_idx = preds[i]
            sentiment = self.sentiment_names[sentiment_idx]
            confidence = probs[i, sentiment_idx]
            
            results[aspect] = {
                'sentiment': sentiment,
                'confidence': float(confidence),
                'probabilities': {
                    'positive': float(probs[i, 0]),
                    'negative': float(probs[i, 1]),
                    'neutral': float(probs[i, 2])
                }
            }
        
        return results


def test_model():
    """Test model forward pass"""
    print("=" * 80)
    print("Testing Multi-Label ViSoBERT Model")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating model...")
    model = MultiLabelViSoBERT(
        model_name="5CD-AI/Vietnamese-Sentiment-visobert",
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
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vietnamese-Sentiment-visobert")
    
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
        logits = model(input_ids, attention_mask)
    
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Expected shape: [1, 11, 3]")
    
    # Test prediction
    print("\n3. Testing prediction...")
    results = model.predict_with_names(input_ids, attention_mask)
    
    print(f"\n   Predictions:")
    for aspect, data in results.items():
        sentiment = data['sentiment']
        confidence = data['confidence']
        print(f"   {aspect:<15} {sentiment:<10} (confidence: {confidence:.3f})")
    
    print("\nAll tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_model()
