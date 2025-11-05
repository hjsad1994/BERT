"""
Model Service for Loading and Predicting with Multi-Label ABSA Model
"""

import torch
import sys
import io
from transformers import AutoTokenizer
import yaml
import os
import numpy as np

# Handle imports from different directories
try:
    from model_multilabel import MultiLabelViSoBERT
except ImportError:
    # Try relative import if running as module
    from .model_multilabel import MultiLabelViSoBERT


class ModelService:
    """Service để load và predict với multi-label ABSA model"""
    
    def __init__(self, config_path=None, model_dir=None):
        """
        Khởi tạo ModelService
        
        Args:
            config_path: Đường dẫn đến file config (None = auto-detect)
            model_dir: Đường dẫn đến thư mục chứa model (default: từ config)
        """
        # Fix encoding cho Windows
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        
        print("=" * 80)
        print("INITIALIZING MODEL SERVICE")
        print("=" * 80)
        
        # Auto-detect config path if not provided
        if config_path is None:
            config_path = self._find_config_path()
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Determine model directory
        if model_dir is None:
            model_dir = self.config['paths']['output_dir']
        
        # Resolve relative paths
        if not os.path.isabs(model_dir):
            # Try relative to config file location
            config_dir = os.path.dirname(os.path.abspath(config_path))
            
            # If model_dir starts with "multi_label/", try to resolve it
            if model_dir.startswith('multi_label/'):
                # Remove "multi_label/" prefix and try relative to config
                relative_path = model_dir.replace('multi_label/', '')
                possible_model_paths = [
                    os.path.join(config_dir, relative_path),  # multi_label/models/...
                    os.path.join(os.path.dirname(config_dir), 'multi_label', relative_path),  # From root
                    os.path.join(config_dir, model_dir),  # Keep original
                    model_dir  # Try as-is
                ]
            else:
                possible_model_paths = [
                    os.path.join(config_dir, model_dir),
                    os.path.join(os.path.dirname(config_dir), model_dir),
                    model_dir  # Try as-is
                ]
            
            for path in possible_model_paths:
                if os.path.exists(path):
                    model_dir = os.path.abspath(path)  # Convert to absolute path
                    break
        
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Model directory: {self.model_dir}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        # Load model
        print("Loading model...")
        self.model = self._load_model()
        
        # Aspect và sentiment names
        self.aspect_names = self.config['valid_aspects']
        self.sentiment_names = ['positive', 'negative', 'neutral']
        
        print("\nModel service initialized successfully!")
        print("=" * 80)
    
    def _find_config_path(self):
        """Tự động tìm đường dẫn đến config file"""
        # Lấy thư mục của file hiện tại
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Thử các đường dẫn có thể
        possible_paths = [
            os.path.join(current_dir, 'config_multi.yaml'),  # Trong multi_label/
            os.path.join(os.path.dirname(current_dir), 'multi_label', 'config_multi.yaml'),  # Từ root
            'multi_label/config_multi.yaml',  # Relative từ root
            'config_multi.yaml'  # Trong cùng thư mục
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found config at: {path}")
                return path
        
        # Nếu không tìm thấy, raise error
        raise FileNotFoundError(
            f"Config file not found. Tried: {', '.join(possible_paths)}"
        )
    
    def _load_config(self, config_path):
        """Load configuration từ YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_model(self):
        """Load trained model từ checkpoint"""
        # Create model
        model = MultiLabelViSoBERT(
            model_name=self.config['model']['name'],
            num_aspects=len(self.config['valid_aspects']),
            num_sentiments=3,
            hidden_size=self.config['model']['hidden_size'],
            dropout=self.config['model']['dropout']
        )
        
        # Load checkpoint - ALWAYS use best_model.pt
        checkpoint_path = os.path.join(self.model_dir, 'best_model.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Verify file exists and get file info
        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # Size in MB
        checkpoint_mtime = os.path.getmtime(checkpoint_path)
        import datetime
        checkpoint_date = datetime.datetime.fromtimestamp(checkpoint_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"=" * 80)
        print(f"LOADING MODEL CHECKPOINT")
        print(f"=" * 80)
        print(f"Checkpoint path: {os.path.abspath(checkpoint_path)}")
        print(f"File size: {checkpoint_size:.2f} MB")
        print(f"Last modified: {checkpoint_date}")
        print(f"=" * 80)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load state dict (with strict=False for compatibility)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'], 
            strict=False
        )
        
        if unexpected_keys:
            print(f"WARNING: Ignored unexpected keys: {unexpected_keys[:5]}...")
        if missing_keys:
            print(f"WARNING: Missing keys: {missing_keys[:5]}...")
        
        model.to(self.device)
        model.eval()
        
        # Print model info
        epoch = checkpoint.get('epoch', 'unknown')
        metrics = checkpoint.get('metrics', {})
        f1_score = metrics.get('overall_f1', 0) * 100 if 'overall_f1' in metrics else 0
        
        print(f"\nModel loaded successfully!")
        print(f"  Epoch: {epoch}")
        print(f"  F1 Score: {f1_score:.2f}%")
        print(f"  Device: {self.device}")
        print(f"  Model: {self.config['model']['name']}")
        print(f"  Aspects: {len(self.aspect_names)}")
        print(f"=" * 80)
        
        return model
    
    def predict(self, text, filter_neutral=True, min_confidence=0.5, max_entropy=1.0, top_k=None):
        """
        Predict sentiment cho tất cả aspects từ text
        
        Args:
            text: Input text (string)
            filter_neutral: Nếu True, filter các aspects có sentiment=neutral với confidence cao
                           (có thể là aspects không được đề cập)
            min_confidence: Minimum confidence để giữ lại aspect (0.0-1.0)
            max_entropy: Maximum entropy để giữ lại aspect (entropy cao = phân bố đều = không rõ ràng)
            top_k: Chỉ giữ lại top K aspects có confidence cao nhất (None = không giới hạn)
        
        Returns:
            dict: {
                'text': str,
                'predictions': {
                    'aspect_name': {
                        'sentiment': str,
                        'confidence': float,
                        'probabilities': {
                            'positive': float,
                            'negative': float,
                            'neutral': float
                        },
                        'entropy': float
                    }
                }
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config['model']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)  # [1, num_aspects, num_sentiments]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        # Convert to CPU numpy
        preds_np = preds[0].cpu().numpy()  # [num_aspects]
        probs_np = probs[0].cpu().numpy()  # [num_aspects, num_sentiments]
        
        # Format results - collect all aspects first
        all_aspects = []
        
        for i, aspect in enumerate(self.aspect_names):
            sentiment_idx = preds_np[i]
            sentiment = self.sentiment_names[sentiment_idx]
            confidence = float(probs_np[i, sentiment_idx])
            
            probs_dict = {
                'positive': float(probs_np[i, 0]),
                'negative': float(probs_np[i, 1]),
                'neutral': float(probs_np[i, 2])
            }
            
            # Calculate entropy (higher entropy = more uniform distribution = less certain)
            probs_array = np.array([probs_dict['positive'], probs_dict['negative'], probs_dict['neutral']])
            entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))
            
            # Filter logic
            should_include = True
            
            # Filter by minimum confidence
            if confidence < min_confidence:
                should_include = False
            
            # Filter by entropy (high entropy = uncertain/unmentioned)
            if entropy > max_entropy:
                should_include = False
            
            # Filter neutral predictions with high confidence (likely unmentioned aspects)
            if filter_neutral and sentiment == 'neutral' and confidence > 0.7:
                should_include = False
            
            if should_include:
                all_aspects.append({
                    'aspect': aspect,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'probabilities': probs_dict,
                    'entropy': float(entropy)
                })
        
        # Apply top_k filtering if specified
        if top_k is not None and top_k > 0:
            # Sort by confidence descending and take top_k
            all_aspects.sort(key=lambda x: x['confidence'], reverse=True)
            all_aspects = all_aspects[:top_k]
        
        # Format final results
        results = {
            'text': text,
            'predictions': {}
        }
        
        for item in all_aspects:
            results['predictions'][item['aspect']] = {
                'sentiment': item['sentiment'],
                'confidence': item['confidence'],
                'probabilities': item['probabilities'],
                'entropy': item['entropy']
            }
        
        return results
    
    def predict_batch(self, texts, filter_neutral=True, min_confidence=0.5, max_entropy=1.0, top_k=None):
        """
        Predict cho nhiều texts cùng lúc
        
        Args:
            texts: List of strings
            filter_neutral: Nếu True, filter các aspects có sentiment=neutral với confidence cao
            min_confidence: Minimum confidence để giữ lại aspect
            max_entropy: Maximum entropy để giữ lại aspect
            top_k: Chỉ giữ lại top K aspects có confidence cao nhất (None = không giới hạn)
        
        Returns:
            list: List of prediction results
        """
        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            max_length=self.config['model']['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)  # [batch_size, num_aspects, num_sentiments]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
        
        # Convert to CPU numpy
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        # Format results
        results = []
        
        for batch_idx, text in enumerate(texts):
            all_aspects = []
            
            for i, aspect in enumerate(self.aspect_names):
                sentiment_idx = preds_np[batch_idx, i]
                sentiment = self.sentiment_names[sentiment_idx]
                confidence = float(probs_np[batch_idx, i, sentiment_idx])
                
                probs_dict = {
                    'positive': float(probs_np[batch_idx, i, 0]),
                    'negative': float(probs_np[batch_idx, i, 1]),
                    'neutral': float(probs_np[batch_idx, i, 2])
                }
                
                # Calculate entropy
                probs_array = np.array([probs_dict['positive'], probs_dict['negative'], probs_dict['neutral']])
                entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))
                
                # Filter logic
                should_include = True
                
                if confidence < min_confidence:
                    should_include = False
                
                if entropy > max_entropy:
                    should_include = False
                
                if filter_neutral and sentiment == 'neutral' and confidence > 0.7:
                    should_include = False
                
                if should_include:
                    all_aspects.append({
                        'aspect': aspect,
                        'sentiment': sentiment,
                        'confidence': confidence,
                        'probabilities': probs_dict,
                        'entropy': float(entropy)
                    })
            
            # Apply top_k filtering if specified
            if top_k is not None and top_k > 0:
                all_aspects.sort(key=lambda x: x['confidence'], reverse=True)
                all_aspects = all_aspects[:top_k]
            
            # Format final result
            result = {
                'text': text,
                'predictions': {}
            }
            
            for item in all_aspects:
                result['predictions'][item['aspect']] = {
                    'sentiment': item['sentiment'],
                    'confidence': item['confidence'],
                    'probabilities': item['probabilities'],
                    'entropy': item['entropy']
                }
            
            results.append(result)
        
        return results


# Global model service instance
_model_service = None


def get_model_service(config_path=None, model_dir=None):
    """Get or create global model service instance"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService(config_path, model_dir)
    return _model_service

