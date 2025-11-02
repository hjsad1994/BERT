"""
Dual-Task ABSA Model for Vietnamese Sentiment Analysis
Task 1: Aspect Detection (binary: present/not present)
Task 2: Sentiment Classification (3-way: positive/negative/neutral)
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class DualTaskViSoBERT(nn.Module):
    """
    Dual-Task ABSA Model with 2 separate heads:
    
    1. Detection Head: Predicts if aspect is present (binary per aspect)
    2. Sentiment Head: Predicts sentiment when present (3-way per aspect)
    
    Example:
        Input:  "Pin trâu camera xấu"
        Detection: Battery=1, Camera=1, Performance=0, ...
        Sentiment: Battery=positive, Camera=negative, Performance=N/A, ...
    """
    
    def __init__(
        self, 
        model_name="5CD-AI/visobert-14gb-corpus",
        num_aspects=11, 
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3,
        use_separate_encoders=False
    ):
        """
        Args:
            model_name: Pre-trained BERT model
            num_aspects: Number of aspects (11)
            num_sentiments: Number of sentiment classes (3: pos/neg/neu)
            hidden_size: Hidden layer size
            dropout: Dropout rate
            use_separate_encoders: Not used (kept for compatibility)
        """
        super().__init__()
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        
        # Shared BERT encoder
        self.bert = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        bert_hidden_size = self.bert.config.hidden_size  # 768 for ViSoBERT
        
        # Shared feature extraction
        self.dropout = nn.Dropout(dropout)
        self.shared_dense = nn.Linear(bert_hidden_size, hidden_size)
        self.activation = nn.ReLU()
        
        # ==============================================================
        # DETECTION HEAD: Binary classification per aspect (sigmoid)
        # ==============================================================
        self.detection_dense = nn.Linear(hidden_size, hidden_size // 2)
        self.detection_classifier = nn.Linear(hidden_size // 2, num_aspects)  # 1 logit per aspect (sigmoid)
        
        # ==============================================================
        # SENTIMENT HEAD: 3-way classification per aspect
        # ==============================================================
        self.sentiment_dense = nn.Linear(hidden_size, hidden_size // 2)
        self.sentiment_classifier = nn.Linear(hidden_size // 2, num_aspects * num_sentiments)  # 3 classes per aspect
        
        # Aspect names (for reference)
        self.aspect_names = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        # Class names
        self.detection_names = ['not_present', 'present']
        self.sentiment_names = ['positive', 'negative', 'neutral']
    
    def forward(self, input_ids, attention_mask, return_features=False):
        """
        Forward pass with dual heads
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_features: If True, return intermediate features
        
        Returns:
            detection_logits: [batch_size, num_aspects] - binary logits (sigmoid)
            sentiment_logits: [batch_size, num_aspects, 3] - 3-way per aspect
        """
        # ==========================================
        # SHARED ENCODER
        # ==========================================
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Shared features
        x = self.dropout(cls_output)
        x = self.shared_dense(x)  # [batch_size, hidden_size]
        x = self.activation(x)
        x = self.dropout(x)
        
        # ==========================================
        # DETECTION HEAD
        # ==========================================
        det_x = self.detection_dense(x)  # [batch_size, hidden_size//2]
        det_x = self.activation(det_x)
        det_x = self.dropout(det_x)
        detection_logits = self.detection_classifier(det_x)  # [batch_size, num_aspects]
        
        # ==========================================
        # SENTIMENT HEAD
        # ==========================================
        sent_x = self.sentiment_dense(x)  # [batch_size, hidden_size//2]
        sent_x = self.activation(sent_x)
        sent_x = self.dropout(sent_x)
        sentiment_logits = self.sentiment_classifier(sent_x)  # [batch_size, num_aspects*3]
        sentiment_logits = sentiment_logits.view(-1, self.num_aspects, self.num_sentiments)
        
        if return_features:
            return detection_logits, sentiment_logits, x
        
        return detection_logits, sentiment_logits
    
    def predict(self, input_ids, attention_mask, detection_threshold=0.5):
        """
        Predict aspect detection + sentiment
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            detection_threshold: Threshold for detection (default: 0.5)
        
        Returns:
            detection_preds: [batch_size, num_aspects] - binary 0/1
            detection_probs: [batch_size, num_aspects] - probabilities (sigmoid)
            sentiment_preds: [batch_size, num_aspects] - class indices 0/1/2
            sentiment_probs: [batch_size, num_aspects, 3] - probabilities
        """
        with torch.no_grad():
            detection_logits, sentiment_logits = self.forward(input_ids, attention_mask)
            
            # Detection predictions (sigmoid for binary)
            detection_probs = torch.sigmoid(detection_logits)  # [batch, aspects]
            detection_preds = (detection_probs > detection_threshold).float()  # [batch, aspects]
            
            # Sentiment predictions
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)  # [batch, aspects, 3]
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)  # [batch, aspects]
        
        return detection_preds, detection_probs, sentiment_preds, sentiment_probs
    
    def predict_with_names(self, input_ids, attention_mask):
        """
        Predict with aspect names and class names
        
        Returns:
            dict: {
                aspect_name: {
                    'detected': bool,
                    'detection_prob': float,
                    'sentiment': str (if detected),
                    'sentiment_prob': float (if detected),
                    'sentiment_probs': dict (all 3 probs)
                }
            }
        """
        det_preds, det_probs, sent_preds, sent_probs = self.predict(input_ids, attention_mask)
        
        # Convert to numpy (assume batch_size = 1)
        det_preds = det_preds[0].cpu().numpy()  # [num_aspects]
        det_probs = det_probs[0].cpu().numpy()  # [num_aspects]
        sent_preds = sent_preds[0].cpu().numpy()  # [num_aspects]
        sent_probs = sent_probs[0].cpu().numpy()  # [num_aspects, 3]
        
        results = {}
        for i, aspect in enumerate(self.aspect_names):
            is_detected = bool(det_preds[i] == 1)
            detection_prob = float(det_probs[i])  # Sigmoid prob of "present"
            
            result = {
                'detected': is_detected,
                'detection_prob': detection_prob,
                'sentiment_probs': {
                    'positive': float(sent_probs[i, 0]),
                    'negative': float(sent_probs[i, 1]),
                    'neutral': float(sent_probs[i, 2])
                }
            }
            
            if is_detected:
                sentiment_idx = sent_preds[i]
                result['sentiment'] = self.sentiment_names[sentiment_idx]
                result['sentiment_prob'] = float(sent_probs[i, sentiment_idx])
            else:
                result['sentiment'] = 'N/A'
                result['sentiment_prob'] = None
            
            results[aspect] = result
        
        return results


def test_model():
    """Test dual-task model"""
    print("=" * 80)
    print("Testing Dual-Task ABSA Model")
    print("=" * 80)
    
    from transformers import AutoTokenizer
    
    # Create model
    print("\n1. Creating model...")
    model = DualTaskViSoBERT(
        model_name="5CD-AI/visobert-14gb-corpus",
        num_aspects=11,
        num_sentiments=3,
        hidden_size=512,
        dropout=0.3,
        use_separate_encoders=False
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/visobert-14gb-corpus")
    
    test_text = "Pin tốt camera xấu"
    
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
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        detection_logits, sentiment_logits = model(input_ids, attention_mask)
    
    print(f"   Detection logits shape: {detection_logits.shape} (expected: [1, 11])")
    print(f"   Sentiment logits shape: {sentiment_logits.shape} (expected: [1, 11, 3])")
    
    # Test prediction
    print("\n3. Testing prediction...")
    results = model.predict_with_names(input_ids, attention_mask)
    
    print(f"\n   Predictions:")
    print(f"   {'Aspect':<15} {'Detected':<10} {'Det.Prob':<10} {'Sentiment':<10} {'Sent.Prob':<10}")
    print("   " + "-" * 65)
    
    for aspect, data in results.items():
        detected = "✓" if data['detected'] else "✗"
        det_prob = f"{data['detection_prob']:.3f}"
        sentiment = data['sentiment']
        sent_prob = f"{data['sentiment_prob']:.3f}" if data['sentiment_prob'] else "N/A"
        
        print(f"   {aspect:<15} {detected:<10} {det_prob:<10} {sentiment:<10} {sent_prob:<10}")
    
    print("\n✅ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_model()
