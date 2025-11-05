"""
Multi-Label Aspect Detection Model for Vietnamese
Detects which aspects are mentioned in the text (binary classification)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class AspectDetectionModel(nn.Module):
    """
    Multi-Label Aspect Detection Model
    
    Input:  "Pin trâu camera xấu"
    Output: Battery=1 (mentioned), Camera=1 (mentioned), Performance=0 (not mentioned), ...
    """
    
    def __init__(
        self, 
        model_name="5CD-AI/Vietnamese-Sentiment-visobert",
        num_aspects=11,
        hidden_size=512,
        dropout=0.3
    ):
        super().__init__()
        
        self.num_aspects = num_aspects
        
        # BERT encoder (without pooler to avoid warning)
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for ViSoBERT
        
        # Multi-label binary classifier head
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        
        # Output: 11 aspects (binary: 0 = not mentioned, 1 = mentioned)
        self.classifier = nn.Linear(hidden_size, num_aspects)
        
        # Aspect names (for reference)
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
        # Encode
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token) as sentence representation
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
            probs = torch.sigmoid(logits)  # Binary classification: sigmoid
            preds = (probs >= threshold).long()
        
        return preds, probs
    
    def predict_with_names(self, input_ids, attention_mask, threshold=0.5):
        """
        Predict with aspect names
        
        Returns:
            dict: {aspect_name: {'mentioned': bool, 'confidence': float}}
        """
        preds, probs = self.predict(input_ids, attention_mask, threshold)
        
        # Convert to dict (assume batch_size = 1)
        preds = preds[0].cpu().numpy()  # [num_aspects]
        probs = probs[0].cpu().numpy()  # [num_aspects]
        
        results = {}
        for i, aspect in enumerate(self.aspect_names):
            mentioned = bool(preds[i])
            confidence = float(probs[i])
            
            results[aspect] = {
                'mentioned': mentioned,
                'confidence': confidence
            }
        
        return results


def test_model():
    """Test model forward pass"""
    print("=" * 80)
    print("Testing Aspect Detection Model")
    print("=" * 80)
    
    # Create model
    print("\n1. Creating model...")
    model = AspectDetectionModel(
        model_name="5CD-AI/Vietnamese-Sentiment-visobert",
        num_aspects=11,
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
    test_text = "Pin tốt camera xấu"
    
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
    print(f"   Expected shape: [1, 11]")
    
    # Test prediction
    print("\n3. Testing prediction...")
    results = model.predict_with_names(input_ids, attention_mask, threshold=0.5)
    
    print(f"\n   Predictions:")
    for aspect, data in results.items():
        mentioned = "Yes" if data['mentioned'] else "No"
        confidence = data['confidence']
        print(f"   {aspect:<15} Mentioned: {mentioned:<3} (confidence: {confidence:.3f})")
    
    print("\nAll tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_model()


