"""
BiLSTM + CNN Model for Multi-Label Sentiment Classification

Based on research papers:
- "Aspect Based Sentiment Analysis With Feature Enhanced Attention CNN-BiLSTM" (IEEE 2019)
- "CNN-BiLSTM model based on multi-level and multi-scale feature extraction" (SPIE 2024)
- "Enhancing Text Sentiment Classification with Hybrid CNN-BiLSTM" (JAIT 2024)

Architecture:
    Input → Embedding → SpatialDropout → BiLSTM → Conv1D → 
    [GlobalAvgPool + GlobalMaxPool] → Concatenate → Dense → Reshape to [batch, 11, 3]
    
Multi-Label: Predicts sentiment (Positive/Negative/Neutral) for each of 11 aspects
Output: [batch_size, num_aspects, num_sentiments]
    
Note: This model uses ONLY BiLSTM + CNN without pretrained embeddings (no BERT, no transformers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPooling(nn.Module):
    """
    Global Average Pooling + Global Max Pooling
    Concatenates both pooling strategies for richer representations
    """
    def __init__(self):
        super(GlobalPooling, self).__init__()
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, features]
        
        Returns:
            pooled: [batch_size, features * 2] (concatenated avg and max pool)
        """
        # Global Average Pooling
        avg_pool = torch.mean(x, dim=1)  # [batch, features]
        
        # Global Max Pooling
        max_pool, _ = torch.max(x, dim=1)  # [batch, features]
        
        # Concatenate
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [batch, features*2]
        
        return pooled


class BiLSTM_SentimentClassification(nn.Module):
    """
    BiLSTM + CNN Hybrid Model for Multi-Label Sentiment Classification (No Pretrained Models)
    
    Architecture: Embedding → SpatialDropout → BiLSTM → Conv1D → 
                  [GlobalAvgPool + GlobalMaxPool] → Dense → Reshape
    
    Multi-label sentiment classification for 11 aspects × 3 sentiments
    Output: [batch_size, num_aspects, num_sentiments] 
    """
    
    def __init__(
        self,
        vocab_size=30000,
        embedding_dim=300,
        num_aspects=11,
        num_sentiments=3,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        lstm_dropout=0.3,
        spatial_dropout=0.2,
        conv_filters=128,
        conv_kernel_size=3,
        dense_hidden_size=256,
        dense_dropout=0.3,
        padding_idx=0
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings (default: 300)
            num_aspects: Number of aspects (default: 11)
            num_sentiments: Number of sentiment classes per aspect (default: 3)
            lstm_hidden_size: Hidden size of BiLSTM
            lstm_num_layers: Number of BiLSTM layers
            lstm_dropout: Dropout for BiLSTM
            spatial_dropout: Spatial dropout rate for embeddings
            conv_filters: Number of Conv1D filters
            conv_kernel_size: Kernel size for Conv1D
            dense_hidden_size: Hidden size of dense layer
            dense_dropout: Dropout for dense layer
            padding_idx: Index for padding token
        """
        super(BiLSTM_SentimentClassification, self).__init__()
        
        self.num_aspects = num_aspects
        self.num_sentiments = num_sentiments
        self.embedding_dim = embedding_dim
        
        # Aspect names (for reference)
        self.aspect_names = [
            'Battery', 'Camera', 'Performance', 'Display', 'Design',
            'Packaging', 'Price', 'Shop_Service',
            'Shipping', 'General', 'Others'
        ]
        
        # Sentiment names
        self.sentiment_names = ['Positive', 'Negative', 'Neutral']
        
        # 1. Trainable embedding layer (NO pretrained model)
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # 2. Spatial Dropout (applied after embeddings)
        self.spatial_dropout = nn.Dropout2d(spatial_dropout)
        
        # 3. BiLSTM layer
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. Conv1D layer (applied after BiLSTM)
        # Input: [batch, seq_len, lstm_hidden*2]
        # Conv1d expects: [batch, channels, seq_len]
        self.conv1d = nn.Conv1d(
            in_channels=lstm_hidden_size * 2,  # BiLSTM output
            out_channels=conv_filters,
            kernel_size=conv_kernel_size,
            padding='same'  # Keep same sequence length
        )
        
        # 5. Global Pooling (Average + Max)
        self.global_pooling = GlobalPooling()
        
        # 6. Dense layers
        pooled_size = conv_filters * 2  # *2 because of avg+max pooling
        self.dense1 = nn.Linear(pooled_size, dense_hidden_size)
        self.bn1 = nn.BatchNorm1d(dense_hidden_size)
        self.dropout = nn.Dropout(dense_dropout)
        # Output: num_aspects × num_sentiments (will reshape later)
        self.dense2 = nn.Linear(dense_hidden_size, num_aspects * num_sentiments)
        
        print(f"\n[BiLSTM + CNN Model - Multi-Label Sentiment - NO Pretrained Embeddings]")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim} (trainable)")
        print(f"  Spatial Dropout: {spatial_dropout}")
        print(f"  LSTM hidden: {lstm_hidden_size} x {lstm_num_layers} layers (bidirectional)")
        print(f"  Conv1D: {conv_filters} filters, kernel={conv_kernel_size}")
        print(f"  Global Pooling: Average + Max (concatenated)")
        print(f"  Dense: {dense_hidden_size} -> {num_aspects}x{num_sentiments} = {num_aspects * num_sentiments}")
        print(f"  Output shape: [batch, {num_aspects}, {num_sentiments}]")
        print(f"  Total params: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through BiLSTM + CNN architecture
        
        Args:
            input_ids: [batch_size, seq_len] - token indices
            attention_mask: [batch_size, seq_len] - optional, for masking padding
        
        Returns:
            logits: [batch_size, num_aspects, num_sentiments] - raw logits for CrossEntropyLoss
        """
        # 1. Get embeddings (trainable)
        embeddings = self.embeddings(input_ids)  # [batch, seq_len, embedding_dim]
        
        # 2. Apply Spatial Dropout
        # Reshape for Dropout2d: [batch, embedding_dim, seq_len, 1]
        embeddings_2d = embeddings.transpose(1, 2).unsqueeze(-1)  # [batch, embedding_dim, seq_len, 1]
        embeddings_2d = self.spatial_dropout(embeddings_2d)
        embeddings = embeddings_2d.squeeze(-1).transpose(1, 2)  # Back to [batch, seq_len, embedding_dim]
        
        # 3. BiLSTM
        lstm_output, (h_n, c_n) = self.bilstm(embeddings)  # [batch, seq_len, lstm_hidden*2]
        
        # 4. Conv1D (transpose for Conv1d: [batch, channels, seq_len])
        lstm_output_t = lstm_output.transpose(1, 2)  # [batch, lstm_hidden*2, seq_len]
        conv_output = self.conv1d(lstm_output_t)  # [batch, conv_filters, seq_len]
        conv_output = F.relu(conv_output)
        conv_output = conv_output.transpose(1, 2)  # Back to [batch, seq_len, conv_filters]
        
        # 5. Global Pooling (Average + Max)
        pooled = self.global_pooling(conv_output)  # [batch, conv_filters*2]
        
        # 6. Dense layers
        x = self.dense1(pooled)  # [batch, dense_hidden]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        logits = self.dense2(x)  # [batch, num_aspects * num_sentiments]
        
        # Reshape to [batch, num_aspects, num_sentiments]
        logits = logits.view(-1, self.num_aspects, self.num_sentiments)
        
        return logits
    
    def predict(self, input_ids, attention_mask=None):
        """
        Predict sentiment for each aspect
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] - optional
        
        Returns:
            predictions: [batch_size, num_aspects] - sentiment predictions per aspect
            probabilities: [batch_size, num_aspects, num_sentiments] - probabilities
        """
        logits = self.forward(input_ids, attention_mask)
        probabilities = F.softmax(logits, dim=-1)  # Softmax over sentiments
        predictions = torch.argmax(probabilities, dim=-1)  # [batch, num_aspects]
        
        return predictions, probabilities
    
    def predict_with_names(self, input_ids, attention_mask=None):
        """
        Predict with aspect and sentiment names
        
        Returns:
            dict: {aspect_name: {'sentiment': str, 'confidence': float, ...}}
        """
        with torch.no_grad():
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
                    'Positive': float(probs[i, 0]),
                    'Negative': float(probs[i, 1]),
                    'Neutral': float(probs[i, 2])
                }
            }
        
        return results


def test_model():
    """Test BiLSTM + CNN model (no pretrained embeddings)"""
    print("=" * 80)
    print("Testing BiLSTM + CNN Multi-Label Sentiment Classification Model")
    print("=" * 80)
    
    # Create model
    model = BiLSTM_SentimentClassification(
        vocab_size=30000,
        embedding_dim=300,
        num_aspects=11,
        num_sentiments=3,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        spatial_dropout=0.2,
        conv_filters=128,
        conv_kernel_size=3
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(1, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"\nTest Input:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    print(f"\nOutput:")
    print(f"  logits: {logits.shape}")
    print(f"  logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Predictions
    predictions, probabilities = model.predict(input_ids, attention_mask)
    print(f"\nPredictions:")
    print(f"  predictions: {predictions.shape}  # [batch, num_aspects]")
    print(f"  probabilities: {probabilities.shape}  # [batch, num_aspects, num_sentiments]")
    print(f"  predictions[0]: {predictions[0]}")
    
    # Test predict_with_names
    print(f"\nDetailed predictions (first sample):")
    results = model.predict_with_names(input_ids, attention_mask)
    for aspect, data in list(results.items())[:5]:  # Show first 5
        sentiment = data['sentiment']
        confidence = data['confidence']
        print(f"  {aspect:<15} {sentiment:<10} (conf: {confidence:.3f})")
    
    print("\n[OK] BiLSTM + CNN Multi-Label Sentiment model test passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_model()
