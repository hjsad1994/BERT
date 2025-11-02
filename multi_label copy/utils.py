"""
Module Tiện Ích cho ABSA Fine-tuning
====================================
Chứa các hàm và class tiện ích để hỗ trợ quá trình huấn luyện mô hình ABSA

Bao gồm:
    - load_config: Đọc file cấu hình YAML
    - set_seed: Thiết lập seed cho reproducibility
    - load_and_preprocess_data: Tải và xử lý dữ liệu
    - ABSADataset: Custom PyTorch Dataset cho ABSA
    - compute_metrics: Tính toán các chỉ số đánh giá
    - save_predictions: Lưu kết quả dự đoán
"""

import os
import random
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


class FocalLoss(nn.Module):
    """
    Focal Loss để xử lý class imbalance
    
    Focal Loss = -α(1-pt)^γ * log(pt)
    
    Args:
        alpha: Trọng số cho từng class (list hoặc tensor)
        gamma: Focusing parameter (default=2.0). Tăng gamma tăng focus vào hard examples
        reduction: 'mean' hoặc 'sum'
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size)
        
        Returns:
            loss: Focal loss value
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = probability of true class
        
        # Apply focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, np.ndarray)):
                alpha = torch.tensor(self.alpha, device=inputs.device)
            else:
                alpha = self.alpha
            
            # Get alpha for each sample based on target class
            alpha_t = alpha[targets]
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def load_config(config_path):
    """
    Đọc file cấu hình YAML
    
    Args:
        config_path: Đường dẫn đến file config.yaml
        
    Returns:
        dict: Dictionary chứa cấu hình
        
    Raises:
        FileNotFoundError: Nếu file không tồn tại
        yaml.YAMLError: Nếu file YAML không hợp lệ
    """
    print(f"📖 Đang tải cấu hình từ: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ Đã tải cấu hình thành công")
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Lỗi khi đọc file YAML: {str(e)}")


def set_seed(seed):
    """
    Thiết lập seed cho random, numpy, và torch để đảm bảo reproducibility
    
    Args:
        seed: Giá trị seed (integer)
    """
    print(f"🎲 Đang thiết lập seed = {seed} cho reproducibility")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Đã thiết lập seed thành công")


def load_and_preprocess_data(config):
    """
    Tải và xử lý dữ liệu từ các file CSV
    
    Args:
        config: Dictionary chứa cấu hình
        
    Returns:
        tuple: (train_df, val_df, test_df, label_map, id2label)
        
    Raises:
        FileNotFoundError: Nếu file dữ liệu không tồn tại
        ValueError: Nếu dữ liệu không hợp lệ
    """
    print(f"\n{'='*70}")
    print("📊 Đang tải và xử lý dữ liệu...")
    print(f"{'='*70}")
    
    # Lấy đường dẫn từ config
    train_path = config['paths']['train_file']
    val_path = config['paths']['validation_file']
    test_path = config['paths']['test_file']
    
    # Kiểm tra sự tồn tại của các file
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    
    # Đọc các file CSV
    print(f"\n✓ Đang đọc file train: {train_path}")
    train_df = pd.read_csv(train_path, encoding='utf-8-sig')
    
    print(f"✓ Đang đọc file validation: {val_path}")
    val_df = pd.read_csv(val_path, encoding='utf-8-sig')
    
    print(f"✓ Đang đọc file test: {test_path}")
    test_df = pd.read_csv(test_path, encoding='utf-8-sig')
    
    # Data đã được xử lý bởi prepare_data.py (underscores removed)
    # No additional preprocessing needed
    
    # Kiểm tra các cột bắt buộc
    required_columns = ['sentence', 'aspect', 'sentiment']
    for df_name, df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"File {df_name} thiếu các cột: {', '.join(missing_cols)}")
    
    print(f"\n✓ Kích thước dữ liệu:")
    print(f"   Train:      {len(train_df):>6} mẫu")
    print(f"   Validation: {len(val_df):>6} mẫu")
    print(f"   Test:       {len(test_df):>6} mẫu")
    print(f"   Tổng:       {len(train_df) + len(val_df) + len(test_df):>6} mẫu")
    
    # Kiểm tra các khía cạnh hợp lệ (nếu có trong config)
    if 'valid_aspects' in config:
        valid_aspects = set(config['valid_aspects'])
        
        for df_name, df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            invalid_aspects = set(df['aspect'].unique()) - valid_aspects
            if invalid_aspects:
                print(f"\n⚠️  Cảnh báo: File {df_name} chứa các aspect không hợp lệ: {invalid_aspects}")
    
    # Lấy label mapping từ config
    if 'sentiment_labels' in config:
        label_map = config['sentiment_labels']
    else:
        # Tạo label mapping tự động
        unique_sentiments = sorted(set(train_df['sentiment'].unique()) | 
                                   set(val_df['sentiment'].unique()) | 
                                   set(test_df['sentiment'].unique()))
        label_map = {sentiment: idx for idx, sentiment in enumerate(unique_sentiments)}
    
    # Tạo reverse mapping
    id2label = {idx: label for label, idx in label_map.items()}
    
    print(f"\n✓ Label mapping:")
    for label, idx in label_map.items():
        print(f"   {label:>10} -> {idx}")
    
    # Mã hóa sentiment thành label_id
    for df in [train_df, val_df, test_df]:
        df['label_id'] = df['sentiment'].map(label_map)
        
        # Kiểm tra các sentiment không hợp lệ
        invalid_mask = df['label_id'].isna()
        if invalid_mask.any():
            invalid_sentiments = df[invalid_mask]['sentiment'].unique()
            raise ValueError(f"Phát hiện sentiment không hợp lệ: {invalid_sentiments}")
    
    # Phân tích phân bố nhãn
    print(f"\n✓ Phân bố nhãn:")
    for df_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n   {df_name}:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = count / len(df) * 100
            print(f"      {sentiment:>10}: {count:>5} ({percentage:>5.1f}%)")
    
    print(f"\n✓ Hoàn tất việc tải và xử lý dữ liệu")
    
    return train_df, val_df, test_df, label_map, id2label


class ABSADataset(Dataset):
    """
    Custom PyTorch Dataset cho ABSA (Aspect-Based Sentiment Analysis)
    
    Format input cho BERT: [CLS] sentence [SEP] aspect [SEP]
    """
    
    def __init__(self, dataframe, tokenizer, max_length=256):
        """
        Khởi tạo ABSADataset
        
        Args:
            dataframe: pandas DataFrame chứa columns: sentence, aspect, label_id
            tokenizer: Tokenizer từ transformers
            max_length: Độ dài tối đa của sequence
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate required columns
        required_cols = ['sentence', 'aspect', 'label_id']
        missing = [col for col in required_cols if col not in self.dataframe.columns]
        if missing:
            raise ValueError(f"DataFrame thiếu các cột: {', '.join(missing)}")
    
    def __len__(self):
        """Trả về số lượng mẫu trong dataset"""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Lấy một mẫu từ dataset
        
        Args:
            idx: Index của mẫu
            
        Returns:
            dict: Dictionary chứa input_ids, attention_mask, token_type_ids, labels
        """
        row = self.dataframe.iloc[idx]
        
        sentence = str(row['sentence'])
        aspect = str(row['aspect'])
        label = int(row['label_id'])
        
        # Format: [CLS] sentence [SEP] aspect [SEP]
        # Sử dụng tokenizer với sentence pair để tự động thêm [CLS], [SEP]
        encoding = self.tokenizer(
            sentence,
            aspect,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_preds):
    """
    Tính toán các chỉ số đánh giá cho mô hình
    
    Args:
        eval_preds: Tuple (predictions, labels) từ Trainer
        
    Returns:
        dict: Dictionary chứa accuracy, precision, recall, f1
    """
    predictions, labels = eval_preds
    
    # Lấy class có xác suất cao nhất
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions
    
    # Tính accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Tính precision, recall, f1 với weighted average
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='weighted',
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def get_detailed_metrics(predictions, labels, label_names):
    """
    Tính toán metrics chi tiết cho từng class
    
    Args:
        predictions: Array các predictions
        labels: Array các true labels
        label_names: List tên các label
        
    Returns:
        str: Classification report dạng string
    """
    if len(predictions.shape) > 1:
        preds = np.argmax(predictions, axis=1)
    else:
        preds = predictions
    
    # Tạo classification report
    report = classification_report(
        labels,
        preds,
        target_names=label_names,
        digits=4,
        zero_division=0
    )
    
    return report


def save_predictions(trainer, test_dataset, test_df, config, id2label):
    """
    Dự đoán trên test set và lưu kết quả vào CSV
    
    Args:
        trainer: Hugging Face Trainer object
        test_dataset: Test Dataset object
        test_df: Test DataFrame gốc
        config: Dictionary chứa cấu hình
        id2label: Dictionary mapping từ label_id sang tên sentiment
    """
    print(f"\n{'='*70}")
    print("🔮 Đang dự đoán trên tập test...")
    print(f"{'='*70}")
    
    # Dự đoán
    predictions_output = trainer.predict(test_dataset)
    predictions = predictions_output.predictions
    
    # Lấy predicted class
    if len(predictions.shape) > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions
    
    # Tạo DataFrame với kết quả
    results_df = test_df[['sentence', 'aspect', 'sentiment']].copy()
    results_df['predicted_sentiment'] = [id2label[pred] for pred in pred_labels]
    results_df.rename(columns={'sentiment': 'true_sentiment'}, inplace=True)
    
    # Lưu vào file
    output_path = config['paths']['predictions_file']
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ Đã lưu predictions vào: {output_path}")
    print(f"✓ Số lượng predictions: {len(results_df)}")
    
    # Tính accuracy trên test set
    correct = (results_df['true_sentiment'] == results_df['predicted_sentiment']).sum()
    total = len(results_df)
    accuracy = correct / total * 100
    
    print(f"✓ Accuracy trên test set: {accuracy:.2f}% ({correct}/{total})")
    
    # In một số ví dụ
    print(f"\n✓ Một số ví dụ dự đoán:")
    sample_size = min(5, len(results_df))
    for idx in range(sample_size):
        row = results_df.iloc[idx]
        status = "✓" if row['true_sentiment'] == row['predicted_sentiment'] else "✗"
        print(f"\n   {status} Mẫu {idx + 1}:")
        print(f"      Câu:    {row['sentence'][:60]}...")
        print(f"      Aspect: {row['aspect']}")
        print(f"      Thực tế: {row['true_sentiment']:>10} | Dự đoán: {row['predicted_sentiment']:>10}")
    
    return results_df


def save_predictions_from_output(predictions_output, test_df, config, id2label):
    """
    Lưu predictions vào CSV từ predictions_output đã có
    (Tránh predict 2 lần - tiết kiệm memory và thời gian)
    
    Args:
        predictions_output: Output từ trainer.predict() đã có sẵn
        test_df: Test DataFrame gốc
        config: Dictionary chứa cấu hình
        id2label: Dictionary mapping từ label_id sang tên sentiment
    """
    print(f"\n{'='*70}")
    print("💾 Đang lưu predictions vào file...")
    print(f"{'='*70}")
    
    predictions = predictions_output.predictions
    
    # Lấy predicted class
    if len(predictions.shape) > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions
    
    # Tạo DataFrame với kết quả
    results_df = test_df[['sentence', 'aspect', 'sentiment']].copy()
    results_df['predicted_sentiment'] = [id2label[pred] for pred in pred_labels]
    results_df.rename(columns={'sentiment': 'true_sentiment'}, inplace=True)
    
    # Lưu vào file
    output_path = config['paths']['predictions_file']
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ Đã lưu predictions vào: {output_path}")
    print(f"✓ Số lượng predictions: {len(results_df)}")
    
    # Tính accuracy trên test set
    correct = (results_df['true_sentiment'] == results_df['predicted_sentiment']).sum()
    total = len(results_df)
    accuracy = correct / total * 100
    
    print(f"✓ Accuracy trên test set: {accuracy:.2f}% ({correct}/{total})")
    
    # In một số ví dụ
    print(f"\n✓ Một số ví dụ dự đoán:")
    sample_size = min(5, len(results_df))
    for idx in range(sample_size):
        row = results_df.iloc[idx]
        status = "✓" if row['true_sentiment'] == row['predicted_sentiment'] else "✗"
        print(f"\n   {status} Mẫu {idx + 1}:")
        print(f"      Câu:    {row['sentence'][:60]}...")
        print(f"      Aspect: {row['aspect']}")
        print(f"      Thực tế: {row['true_sentiment']:>10} | Dự đoán: {row['predicted_sentiment']:>10}")
    
    return results_df


def print_system_info():
    """In thông tin về hệ thống và các thư viện"""
    print(f"\n{'='*70}")
    print("💻 THÔNG TIN HỆ THỐNG")
    print(f"{'='*70}")
    
    # Python version
    import sys
    print(f"Python version: {sys.version.split()[0]}")
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: True")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU count: {torch.cuda.device_count()}")
    else:
        print(f"✗ CUDA available: False (sẽ sử dụng CPU)")
    
    # Transformers version
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except:
        pass
    
    print(f"{'='*70}\n")

