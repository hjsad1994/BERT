"""
Model Service for Loading and Predicting with Dual-Task ABSA Model
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
    from .model_multitask import DualTaskViSoBERT
except ImportError:
    # Try absolute import if running as module
    from model_multitask import DualTaskViSoBERT


class ModelService:
    """Service để load và predict với dual-task ABSA model"""
    
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
        print("INITIALIZING DUAL-TASK MODEL SERVICE")
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
            
            # If model_dir starts with "models/", try to resolve it
            if model_dir.startswith('models/'):
                # Remove "models/" prefix and try relative to config
                relative_path = model_dir.replace('models/', '')
                possible_model_paths = [
                    os.path.join(config_dir, 'models', relative_path),  # dual-task-learning/models/...
                    os.path.join(os.path.dirname(config_dir), 'models', relative_path),  # From root
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
        
        # Aspect và sentiment names - set BEFORE loading model
        # Support both 'aspect_names' and 'valid_aspects' for compatibility
        if 'aspect_names' in self.config:
            self.aspect_names = self.config['aspect_names']
        elif 'valid_aspects' in self.config:
            self.aspect_names = self.config['valid_aspects']
        else:
            raise KeyError("Config must contain either 'aspect_names' or 'valid_aspects'")
        self.sentiment_names = ['positive', 'negative', 'neutral']
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        # Load model
        print("Loading model...")
        self.model = self._load_model()
        
        print("\nModel service initialized successfully!")
        print("=" * 80)
    
    def _find_config_path(self):
        """Tự động tìm đường dẫn đến config file"""
        # Lấy thư mục của file hiện tại (backend/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Thư mục cha (VisoBERT-MTL/)
        parent_dir = os.path.dirname(current_dir)
        
        # Thử các đường dẫn có thể
        possible_paths = [
            os.path.join(parent_dir, 'config_visobert_mtl.yaml'),  # VisoBERT-MTL/config_visobert_mtl.yaml
            os.path.join(current_dir, 'config_visobert_mtl.yaml'),  # backend/config_visobert_mtl.yaml (fallback)
            'config_visobert_mtl.yaml'  # Trong cùng thư mục
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
        # Get aspect names (support both 'aspect_names' and 'valid_aspects')
        if 'aspect_names' in self.config:
            num_aspects = len(self.config['aspect_names'])
        elif 'valid_aspects' in self.config:
            num_aspects = len(self.config['valid_aspects'])
        else:
            num_aspects = self.config['model'].get('num_aspects', 11)
        
        model = DualTaskViSoBERT(
            model_name=self.config['model']['name'],
            num_aspects=num_aspects,
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
    
    def predict(self, text, min_aspect_confidence=0.5, filter_absent=True, min_sentiment_confidence=0.5, top_k=None):
        """
        Predict aspect detection và sentiment classification cho text
        
        Dual-Task Approach:
        1. Step 1: Aspect Detection - detect which aspects are present (binary classification)
        2. Step 2: Sentiment Classification - predict sentiment ONLY for detected aspects
        
        Args:
            text: Input text (string)
            min_aspect_confidence: Minimum confidence để aspect được coi là "present" (0.0-1.0)
            filter_absent: Nếu True, chỉ trả về các aspects được detect (mặc định: True)
                         Nếu False, trả về tất cả aspects với present flag
            min_sentiment_confidence: Minimum confidence cho sentiment prediction (0.0-1.0)
            top_k: Chỉ giữ lại top K aspects có aspect confidence cao nhất (None = không giới hạn)
        
        Returns:
            dict: {
                'text': str,
                'predictions': {
                    'aspect_name': {
                        'present': bool,
                        'present_confidence': float,
                        'sentiment': str,
                        'sentiment_confidence': float,
                        'probabilities': {
                            'positive': float,
                            'negative': float,
                            'neutral': float
                        }
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
            aspect_logits, sentiment_logits = self.model(input_ids, attention_mask)
            
            # Aspect detection: sigmoid for binary classification
            aspect_probs = torch.sigmoid(aspect_logits)  # [1, num_aspects]
            
            # Sentiment classification: softmax for 3-class classification
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)  # [1, num_aspects, num_sentiments]
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)  # [1, num_aspects]
        
        # Convert to CPU numpy
        aspect_probs_np = aspect_probs[0].cpu().numpy()  # [num_aspects]
        sentiment_preds_np = sentiment_preds[0].cpu().numpy()  # [num_aspects]
        sentiment_probs_np = sentiment_probs[0].cpu().numpy()  # [num_aspects, num_sentiments]
        
        # STEP 1: Aspect Detection - detect which aspects are present
        # Dual-Task: First detect aspects using aspect_detection_head (sigmoid)
        detected_aspect_indices = []
        
        for i, aspect in enumerate(self.aspect_names):
            present_confidence = float(aspect_probs_np[i])
            is_present = present_confidence >= min_aspect_confidence
            
            if is_present:
                detected_aspect_indices.append(i)
        
        # STEP 2: Sentiment Classification - only for detected aspects
        # Dual-Task: Only predict sentiment for aspects detected in Step 1
        # This uses sentiment_classification_head (softmax) for detected aspects only
        all_aspects = []
        
        for idx in detected_aspect_indices:
            aspect = self.aspect_names[idx]
            present_confidence = float(aspect_probs_np[idx])
            
            # Get sentiment for this detected aspect
            sentiment_idx = sentiment_preds_np[idx]
            sentiment = self.sentiment_names[sentiment_idx]
            sentiment_confidence = float(sentiment_probs_np[idx, sentiment_idx])
            
            probs_dict = {
                'positive': float(sentiment_probs_np[idx, 0]),
                'negative': float(sentiment_probs_np[idx, 1]),
                'neutral': float(sentiment_probs_np[idx, 2])
            }
            
            # Filter by sentiment confidence
            if sentiment_confidence >= min_sentiment_confidence:
                all_aspects.append({
                    'aspect': aspect,
                    'present': True,  # Always True since we only process detected aspects
                    'present_confidence': present_confidence,
                    'sentiment': sentiment,
                    'sentiment_confidence': sentiment_confidence,
                    'probabilities': probs_dict
                })
        
        # If filter_absent=False, also include absent aspects (for debugging)
        if not filter_absent:
            for i, aspect in enumerate(self.aspect_names):
                if i not in detected_aspect_indices:
                    present_confidence = float(aspect_probs_np[i])
                    if present_confidence < min_aspect_confidence:
                        # Include absent aspects with low confidence
                        all_aspects.append({
                            'aspect': aspect,
                            'present': False,
                            'present_confidence': present_confidence,
                            'sentiment': None,
                            'sentiment_confidence': None,
                            'probabilities': None
                        })
        
        # Apply top_k filtering if specified
        if top_k is not None and top_k > 0:
            # Sort by aspect confidence descending and take top_k
            all_aspects.sort(key=lambda x: x['present_confidence'], reverse=True)
            all_aspects = all_aspects[:top_k]
        
        # Format final results
        results = {
            'text': text,
            'predictions': {}
        }
        
        for item in all_aspects:
            results['predictions'][item['aspect']] = {
                'present': item['present'],
                'present_confidence': item['present_confidence'],
                'sentiment': item['sentiment'],
                'sentiment_confidence': item['sentiment_confidence'],
                'probabilities': item['probabilities']
            }
        
        return results
    
    def predict_batch(self, texts, min_aspect_confidence=0.5, filter_absent=True, min_sentiment_confidence=0.5, top_k=None):
        """
        Predict cho nhiều texts cùng lúc
        
        Args:
            texts: List of strings
            min_aspect_confidence: Minimum confidence để aspect được coi là "present"
            filter_absent: Nếu True, chỉ trả về các aspects được detect
            min_sentiment_confidence: Minimum confidence cho sentiment prediction
            top_k: Chỉ giữ lại top K aspects có aspect confidence cao nhất
        
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
            aspect_logits, sentiment_logits = self.model(input_ids, attention_mask)
            
            # Aspect detection: sigmoid
            aspect_probs = torch.sigmoid(aspect_logits)  # [batch_size, num_aspects]
            
            # Sentiment classification: softmax
            sentiment_probs = torch.softmax(sentiment_logits, dim=-1)  # [batch_size, num_aspects, num_sentiments]
            sentiment_preds = torch.argmax(sentiment_logits, dim=-1)  # [batch_size, num_aspects]
        
        # Convert to CPU numpy
        aspect_probs_np = aspect_probs.cpu().numpy()
        sentiment_preds_np = sentiment_preds.cpu().numpy()
        sentiment_probs_np = sentiment_probs.cpu().numpy()
        
        # Format results
        results = []
        
        for batch_idx, text in enumerate(texts):
            # STEP 1: Aspect Detection - detect which aspects are present
            detected_aspect_indices = []
            
            for i, aspect in enumerate(self.aspect_names):
                present_confidence = float(aspect_probs_np[batch_idx, i])
                is_present = present_confidence >= min_aspect_confidence
                
                if is_present:
                    detected_aspect_indices.append(i)
            
            # STEP 2: Sentiment Classification - only for detected aspects
            all_aspects = []
            
            for idx in detected_aspect_indices:
                aspect = self.aspect_names[idx]
                present_confidence = float(aspect_probs_np[batch_idx, idx])
                
                # Get sentiment for this detected aspect
                sentiment_idx = sentiment_preds_np[batch_idx, idx]
                sentiment = self.sentiment_names[sentiment_idx]
                sentiment_confidence = float(sentiment_probs_np[batch_idx, idx, sentiment_idx])
                
                probs_dict = {
                    'positive': float(sentiment_probs_np[batch_idx, idx, 0]),
                    'negative': float(sentiment_probs_np[batch_idx, idx, 1]),
                    'neutral': float(sentiment_probs_np[batch_idx, idx, 2])
                }
                
                # Filter by sentiment confidence
                if sentiment_confidence >= min_sentiment_confidence:
                    all_aspects.append({
                        'aspect': aspect,
                        'present': True,  # Always True since we only process detected aspects
                        'present_confidence': present_confidence,
                        'sentiment': sentiment,
                        'sentiment_confidence': sentiment_confidence,
                        'probabilities': probs_dict
                    })
            
            # If filter_absent=False, also include absent aspects (for debugging)
            if not filter_absent:
                for i, aspect in enumerate(self.aspect_names):
                    if i not in detected_aspect_indices:
                        present_confidence = float(aspect_probs_np[batch_idx, i])
                        if present_confidence < min_aspect_confidence:
                            # Include absent aspects with low confidence
                            all_aspects.append({
                                'aspect': aspect,
                                'present': False,
                                'present_confidence': present_confidence,
                                'sentiment': None,
                                'sentiment_confidence': None,
                                'probabilities': None
                            })
            
            # Apply top_k filtering if specified
            if top_k is not None and top_k > 0:
                all_aspects.sort(key=lambda x: x['present_confidence'], reverse=True)
                all_aspects = all_aspects[:top_k]
            
            # Format final result
            result = {
                'text': text,
                'predictions': {}
            }
            
            for item in all_aspects:
                result['predictions'][item['aspect']] = {
                    'present': item['present'],
                    'present_confidence': item['present_confidence'],
                    'sentiment': item['sentiment'],
                    'sentiment_confidence': item['sentiment_confidence'],
                    'probabilities': item['probabilities']
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

