"""
Script Huấn Luyện Mô Hình ABSA
==============================
Script chính để fine-tune mô hình Vietnamese-Sentiment-visobert 
cho nhiệm vụ Aspect-Based Sentiment Analysis (ABSA)

Usage:
    python train.py --config config.yaml
"""

import os
import sys
import argparse
import torch
import logging
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed as hf_set_seed,
    EarlyStoppingCallback
)

# Import các hàm tiện ích
from utils import (
    load_config,
    set_seed,
    load_and_preprocess_data,
    ABSADataset,
    compute_metrics,
    save_predictions,
    save_predictions_from_output,
    get_detailed_metrics,
    print_system_info
)


class TeeLogger:
    """Logger ghi đồng thời ra console và file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        # Handle console encoding errors (replace problematic chars)
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            # Fallback: replace emoji/special chars with ASCII
            safe_message = message.encode('ascii', errors='replace').decode('ascii')
            self.terminal.write(safe_message)
        
        # Always write full UTF-8 to file
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_logging():
    """Thiết lập logging ra file với timestamp"""
    # Tạo tên file log với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "single_label/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    # Tạo TeeLogger để ghi cả console và file
    tee = TeeLogger(log_file)
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"📝 Training log sẽ được lưu tại: {log_file}\n")
    
    return tee, log_file


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Fine-tune ViSoBERT cho ABSA trên dữ liệu tiếng Việt'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Đường dẫn đến file cấu hình YAML (default: config.yaml)'
    )
    return parser.parse_args()


def main():
    """Hàm main điều phối toàn bộ workflow"""
    
    # =====================================================================
    # 0. SETUP LOGGING TO FILE
    # =====================================================================
    tee_logger, log_file_path = setup_logging()
    
    # =====================================================================
    # 1. PARSE ARGUMENTS VÀ LOAD CONFIG
    # =====================================================================
    print("\n" + "="*70)
    print("🚀 FINE-TUNING VISOBERT CHO ABSA")
    print("="*70)
    
    args = parse_arguments()
    
    # Load cấu hình
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"\n❌ Lỗi khi load config: {str(e)}")
        return
    
    # =====================================================================
    # 2. THIẾT LẬP SEED VÀ IN THÔNG TIN HỆ THỐNG
    # =====================================================================
    # Get training seed from reproducibility config
    seed = config['reproducibility']['training_seed']
    set_seed(seed)
    hf_set_seed(seed)  # Set seed cho transformers
    
    print_system_info()
    
    # =====================================================================
    # 3. PHÁT HIỆN VÀ THIẾT LẬP DEVICE (GPU/CPU)
    # =====================================================================
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"✓ Sử dụng CPU")
    
    print(f"✓ Device: {device}")
    
    # =====================================================================
    # 4. TẢI TOKENIZER VÀ MÔ HÌNH
    # =====================================================================
    print(f"\n{'='*70}")
    print("🤖 Đang tải tokenizer và mô hình...")
    print(f"{'='*70}")
    
    model_name = config['model']['name']
    num_labels = config['model']['num_labels']
    
    try:
        print(f"\n✓ Đang tải tokenizer từ: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"✓ Đang tải mô hình từ: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        print(f"✓ Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"✓ Model parameters: {model.num_parameters():,}")
        print(f"✓ Số lượng labels: {num_labels}")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi tải mô hình: {str(e)}")
        print(f"\nGợi ý: Kiểm tra kết nối internet hoặc tên mô hình trong config.yaml")
        return
    
    # =====================================================================
    # 5. TẢI VÀ XỬ LÝ DỮ LIỆU
    # =====================================================================
    try:
        train_df, val_df, test_df, label_map, id2label = load_and_preprocess_data(config)
    except Exception as e:
        print(f"\n❌ Lỗi khi load dữ liệu: {str(e)}")
        print(f"\nGợi ý: Chạy 'python prepare_data.py' để tạo dữ liệu trước")
        return
    
    # =====================================================================
    # 6. TẠO DATASETS
    # =====================================================================
    print(f"\n{'='*70}")
    print("📦 Đang tạo PyTorch Datasets...")
    print(f"{'='*70}")
    
    max_length = config['model']['max_length']
    
    try:
        train_dataset = ABSADataset(train_df, tokenizer, max_length)
        val_dataset = ABSADataset(val_df, tokenizer, max_length)
        test_dataset = ABSADataset(test_df, tokenizer, max_length)
        
        print(f"\n✓ Train dataset: {len(train_dataset)} mẫu")
        print(f"✓ Val dataset:   {len(val_dataset)} mẫu")
        print(f"✓ Test dataset:  {len(test_dataset)} mẫu")
        
        # In một mẫu để kiểm tra
        print(f"\n✓ Ví dụ một mẫu đã tokenize:")
        sample = train_dataset[0]
        print(f"   Input IDs shape:      {sample['input_ids'].shape}")
        print(f"   Attention mask shape: {sample['attention_mask'].shape}")
        print(f"   Token type IDs shape: {sample['token_type_ids'].shape}")
        print(f"   Label:                {sample['labels'].item()} ({id2label[sample['labels'].item()]})")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi tạo datasets: {str(e)}")
        return
    
    # =====================================================================
    # 7. THIẾT LẬP TRAINING ARGUMENTS
    # =====================================================================
    print(f"\n{'='*70}")
    print("⚙️  Đang thiết lập Training Arguments...")
    print(f"{'='*70}")
    
    training_config = config['training']
    output_dir = config['paths']['output_dir']
    
    training_args = TrainingArguments(
        # Output directory
        output_dir=output_dir,
        
        # Training parameters
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        
        # Optimizer
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        adam_beta1=training_config['adam_beta1'],
        adam_beta2=training_config['adam_beta2'],
        adam_epsilon=training_config['adam_epsilon'],
        max_grad_norm=training_config['max_grad_norm'],
        
        # Scheduler
        warmup_ratio=training_config['warmup_ratio'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        
        # Evaluation
        eval_strategy=training_config['evaluation_strategy'],
        save_strategy=training_config['save_strategy'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=training_config['greater_is_better'],
        
        # Logging
        logging_steps=training_config['logging_steps'],
        logging_first_step=training_config['logging_first_step'],
        
        # Performance
        fp16=training_config['fp16'],
        fp16_full_eval=training_config.get('fp16_full_eval', False),
        optim=training_config.get('optim', 'adamw_torch'),
        dataloader_num_workers=training_config['dataloader_num_workers'],
        dataloader_pin_memory=training_config['dataloader_pin_memory'],
        dataloader_prefetch_factor=training_config.get('dataloader_prefetch_factor', 2),
        dataloader_persistent_workers=training_config.get('dataloader_persistent_workers', False),
        
        # Other
        seed=config['reproducibility']['training_seed'],
        data_seed=config['reproducibility']['dataloader_seed'],
        disable_tqdm=training_config['disable_tqdm'],
        remove_unused_columns=training_config['remove_unused_columns'],
    )
    
    print(f"\n✓ Các tham số huấn luyện chính:")
    print(f"   Learning rate:        {training_config['learning_rate']}")
    print(f"   LR Scheduler:         {training_config['lr_scheduler_type']}")
    print(f"   Optimizer:            {training_config.get('optim', 'adamw_torch')}")
    print(f"   Epochs:               {training_config['num_train_epochs']}")
    print(f"   Train batch size:     {training_config['per_device_train_batch_size']}")
    print(f"   Eval batch size:      {training_config['per_device_eval_batch_size']}")
    print(f"   Gradient accum:       {training_config['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']}")
    print(f"   Warmup ratio:         {training_config['warmup_ratio']}")
    print(f"   FP16:                 {training_config['fp16']}")
    print(f"   Prefetch factor:      {training_config.get('dataloader_prefetch_factor', 2)}")
    print(f"   Output directory:     {output_dir}")
    
    # =====================================================================
    # 8. OVERSAMPLING - TEMPORARILY DISABLED
    # =====================================================================
    print(f"\n{'='*70}")
    print("⚠️  OVERSAMPLING TEMPORARILY DISABLED - Sử dụng dữ liệu gốc")
    print(f"{'='*70}")
    
    # Lưu class counts gốc để tính Focal Loss alpha weights
    from collections import Counter
    class_counts_original = Counter(train_df['sentiment'])  # Dùng cho Focal Loss
    
    # Lưu train_df gốc để visualization sau này
    # train_df_original = train_df.copy()  # DISABLED - không cần vì không oversample
    
    # from oversampling_utils import aspect_wise_oversample  # DISABLED
    
    # In phân bố class trong training data
    print(f"\n📊 Training Data Distribution (ORIGINAL - NO OVERSAMPLING):")
    total_samples = len(train_df)
    for sentiment, count in sorted(class_counts_original.items()):
        pct = (count / total_samples) * 100
        print(f"   {sentiment:10}: {count:6,} samples ({pct:5.2f}%)")
    
    # ======================================================================
    # OVERSAMPLING CODE - COMMENTED OUT (TEMPORARILY DISABLED)
    # ======================================================================
    # Apply aspect-wise oversampling
    # Với mỗi aspect (Battery, Camera, etc.):
    #   - Tìm sentiment có nhiều mẫu nhất
    #   - Oversample các sentiment khác để bằng với sentiment lớn nhất đó
    # print(f"\n🎯 Chiến lược: Cân bằng sentiment cho từng aspect riêng biệt")
    
    # train_df_oversampled = aspect_wise_oversample(
    #     train_df, 
    #     aspect_column='aspect',
    #     sentiment_column='sentiment',
    #     random_state=config['reproducibility']['oversampling_seed']
    # )
    
    # Use oversampled data
    # train_df = train_df_oversampled
    
    # Recreate train_dataset with oversampled data
    # print(f"\n🔄 Recreating train_dataset with oversampled data...")
    # train_dataset = ABSADataset(train_df, tokenizer, max_length)
    print(f"\n✓ Sử dụng train dataset gốc (không oversampling): {len(train_dataset):,} samples")
    
    # Lưu thông tin oversampling để visualization
    # DISABLED - Không lưu thông tin oversampling vì đang tắt oversampling
    # print(f"\n💾 Lưu thông tin oversampling để visualization...")
    # import json
    # from datetime import datetime
    # 
    # oversampling_info = {
    #     'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #     'strategy': 'aspect_wise_oversampling',
    #     'description': 'Cân bằng sentiment cho từng aspect riêng biệt',
    #     'before': {
    #         'total_samples': len(train_df_original),
    #         'sentiment_distribution': dict(class_counts_original),
    #         'aspects': {}
    #     },
    #     'after': {
    #         'total_samples': len(train_df),
    #         'sentiment_distribution': dict(Counter(train_df['sentiment'])),
    #         'aspects': {}
    #     }
    # }
    # 
    # # Lưu phân bố chi tiết theo aspect (before)
    # for aspect in train_df_original['aspect'].unique():
    #     aspect_data = train_df_original[train_df_original['aspect'] == aspect]
    #     oversampling_info['before']['aspects'][aspect] = dict(Counter(aspect_data['sentiment']))
    # 
    # # Lưu phân bố chi tiết theo aspect (after)
    # for aspect in train_df['aspect'].unique():
    #     aspect_data = train_df[train_df['aspect'] == aspect]
    #     oversampling_info['after']['aspects'][aspect] = dict(Counter(aspect_data['sentiment']))
    # 
    # # Lưu vào file JSON
    # os.makedirs('analysis_results', exist_ok=True)
    # oversampling_info_path = 'analysis_results/oversampling_info.json'
    # with open(oversampling_info_path, 'w', encoding='utf-8') as f:
    #     json.dump(oversampling_info, f, indent=2, ensure_ascii=False)
    # 
    # print(f"✓ Đã lưu thông tin oversampling: {oversampling_info_path}")
    
    # =====================================================================
    # 9. TÍNH CLASS WEIGHTS VÀ KHỞI TẠO FOCAL LOSS
    # =====================================================================
    print(f"\n{'='*70}")
    print("🔥 Đang tính class weights cho Focal Loss...")
    print(f"{'='*70}")
    
    # Tính phân bố classes trong training data
    label_counts = class_counts_original
    total = sum(label_counts.values())
    
    # Class distribution
    print(f"\n📊 Phân bố classes trong training data:")
    for label in ['positive', 'negative', 'neutral']:
        count = label_counts.get(label, 0)
        pct = (count / total) * 100
        print(f"   {label:10}: {count:6,} samples ({pct:5.2f}%)")
    
    # Import Focal Loss
    from utils import FocalLoss
    from focal_loss_trainer import CustomTrainer
    
    label_map = config['sentiment_labels']  # {'positive': 0, 'negative': 1, 'neutral': 2}
    
    # Read focal_alpha from config
    focal_config = config.get('single_label', {})
    focal_alpha_config = focal_config.get('focal_alpha', 'auto')
    gamma = focal_config.get('focal_gamma', 2.0)
    
    # Determine alpha weights based on config
    if focal_alpha_config == 'auto':
        # Auto: Tính từ inverse frequency
        print(f"\n🎯 Alpha weights mode: AUTO (inverse frequency)")
        alpha = [0.0, 0.0, 0.0]
        for label, idx in label_map.items():
            count = label_counts.get(label, 1)
            alpha[idx] = total / (len(label_map) * count)
        
        print(f"\n   Calculated alpha weights:")
        for label, idx in label_map.items():
            print(f"   {label:10} (class {idx}): {alpha[idx]:.4f}")
    
    elif isinstance(focal_alpha_config, list) and len(focal_alpha_config) == 3:
        # User-defined weights
        print(f"\n🎯 Alpha weights mode: USER-DEFINED")
        alpha = focal_alpha_config
        print(f"\n   Using custom alpha weights:")
        for label, idx in label_map.items():
            print(f"   {label:10} (class {idx}): {alpha[idx]:.4f}")
    
    elif focal_alpha_config is None:
        # Equal weights (no class weighting)
        print(f"\n🎯 Alpha weights mode: EQUAL (no class weighting)")
        alpha = [1.0, 1.0, 1.0]
        print(f"\n   Using equal weights: {alpha}")
    
    else:
        # Invalid config, fallback to auto
        print(f"\n⚠️  Invalid focal_alpha config: {focal_alpha_config}")
        print(f"   Falling back to AUTO (inverse frequency)")
        alpha = [0.0, 0.0, 0.0]
        for label, idx in label_map.items():
            count = label_counts.get(label, 1)
            alpha[idx] = total / (len(label_map) * count)
    
    # Create Focal Loss
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    print(f"\n✓ Focal Loss created:")
    print(f"   Gamma: {gamma} (focusing parameter)")
    print(f"   Alpha: {alpha}")
    print(f"✓ Focal Loss sẽ focus vào hard examples và handle class imbalance")
    
    # =====================================================================
    # 10. KHỞI TẠO TRAINER VỚI FOCAL LOSS
    # =====================================================================
    print(f"\n{'='*70}")
    print("🏋️  Đang khởi tạo Custom Trainer với Focal Loss...")
    print(f"{'='*70}")
    
    trainer = CustomTrainer.create_trainer_with_focal_loss(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        focal_loss=focal_loss
    )
    
    print(f"✓ Custom Trainer với Focal Loss đã được khởi tạo thành công")
    print(f"✓ Chiến lược xử lý class imbalance:")
    print(f"   • Focal Loss: Tăng trọng số loss cho minority classes")
    print(f"   • Alpha weights được tính dựa trên inverse frequency của mỗi class")
    
    # =====================================================================
    # 10.5. ADD CHECKPOINT RENAMER CALLBACK
    # =====================================================================
    print(f"\n📁 Đang thiết lập Checkpoint Renamer...")
    
    from checkpoint_renamer import SimpleMetricCheckpointCallback
    
    # Add callback để rename checkpoints theo accuracy
    # Example: checkpoint-1352 → checkpoint-91 (91% accuracy)
    checkpoint_callback = SimpleMetricCheckpointCallback(metric_name='eval_accuracy')
    trainer.add_callback(checkpoint_callback)
    
    # Add Early Stopping callback để tránh overfitting
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=training_config.get('early_stopping_patience', 2),
        early_stopping_threshold=training_config.get('early_stopping_threshold', 0.001)
    )
    trainer.add_callback(early_stopping_callback)
    
    print(f"✓ Checkpoints sẽ được đặt tên theo accuracy (vd: checkpoint-90, checkpoint-92)")
    print(f"✓ Early Stopping: sẽ dừng nếu eval_loss không cải thiện sau {training_config.get('early_stopping_patience', 2)} epoch")
    
    # =====================================================================
    # 11. BẮT ĐẦU HUẤN LUYỆN
    # =====================================================================
    print(f"\n{'='*70}")
    print("🎯 BẮT ĐẦU HUẤN LUYỆN")
    print(f"{'='*70}\n")
    
    try:
        train_result = trainer.train()
        
        print(f"\n{'='*70}")
        print("✅ HOÀN TẤT HUẤN LUYỆN")
        print(f"{'='*70}")
        print(f"✓ Training loss: {train_result.training_loss:.4f}")
        print(f"✓ Training time: {train_result.metrics['train_runtime']:.2f}s")
        print(f"✓ Samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình huấn luyện: {str(e)}")
        return
    
    # =====================================================================
    # 9.5. TẠO TRAINER MỚI CHO EVALUATION (KHÔNG CÓ OPTIMIZER)
    # =====================================================================
    print(f"\n{'='*70}")
    print("🔄 TẠO TRAINER MỚI CHO EVALUATION")
    print(f"{'='*70}")
    
    # Lưu model hiện tại
    current_model = trainer.model
    
    # Xóa trainer cũ (có optimizer/scheduler)
    del trainer
    torch.cuda.empty_cache()
    
    # Tạo trainer mới chỉ để eval (không có optimizer/scheduler)
    eval_trainer = Trainer(
        model=current_model,
        args=training_args,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )
    
    print(f"✓ Đã tạo trainer mới cho evaluation (không có optimizer/scheduler)")
    print(f"✓ VRAM đã giảm, sẵn sàng cho evaluation")
    
    # =====================================================================
    # 10. ĐÁNH GIÁ TRÊN TẬP TEST
    # =====================================================================
    print(f"\n{'='*70}")
    print("📊 ĐÁNH GIÁ TRÊN TẬP TEST")
    print(f"{'='*70}")
    
    try:
        # Evaluate
        print("⏳ Đang evaluate trên test dataset...")
        test_results = eval_trainer.evaluate(test_dataset)
        
        print(f"\n✓ Kết quả đánh giá trên tập test:")
        print(f"   Accuracy:  {test_results['eval_accuracy']:.4f}")
        print(f"   Precision: {test_results['eval_precision']:.4f}")
        print(f"   Recall:    {test_results['eval_recall']:.4f}")
        print(f"   F1 Score:  {test_results['eval_f1']:.4f}")
        
        # Giải phóng cache trước khi predict để tránh OOM
        torch.cuda.empty_cache()
        
        # Lấy detailed metrics
        # CHÚ Ý: Chỉ predict 1 LẦN DUY NHẤT ở đây, sau đó tái sử dụng cho save_predictions
        print("\n⏳ Đang predict để lấy detailed metrics...")
        predictions_output = eval_trainer.predict(test_dataset)
        print("✓ Predict hoàn tất")
        label_names = [id2label[i] for i in sorted(id2label.keys())]
        detailed_report = get_detailed_metrics(
            predictions_output.predictions,
            predictions_output.label_ids,
            label_names
        )
        
        print(f"\n✓ Báo cáo chi tiết theo từng class:")
        print(detailed_report)
        
        # Lưu báo cáo vào file
        report_path = config['paths']['evaluation_report']
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ABSA\n")
            f.write("="*70 + "\n\n")
            
            f.write("Tổng quan:\n")
            f.write(f"  Accuracy:  {test_results['eval_accuracy']:.4f}\n")
            f.write(f"  Precision: {test_results['eval_precision']:.4f}\n")
            f.write(f"  Recall:    {test_results['eval_recall']:.4f}\n")
            f.write(f"  F1 Score:  {test_results['eval_f1']:.4f}\n\n")
            
            f.write("Báo cáo chi tiết theo từng class:\n")
            f.write(detailed_report)
            
            f.write("\n" + "="*70 + "\n")
            f.write("Cấu hình mô hình:\n")
            f.write(f"  Model: {model_name}\n")
            f.write(f"  Epochs: {training_config['num_train_epochs']}\n")
            f.write(f"  Learning rate: {training_config['learning_rate']}\n")
            f.write(f"  Batch size: {training_config['per_device_train_batch_size']}\n")
            f.write(f"  Max length: {max_length}\n")
        
        print(f"\n✓ Đã lưu báo cáo chi tiết vào: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi đánh giá: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # =====================================================================
    # 11. LƯU KẾT QUẢ DỰ ĐOÁN
    # =====================================================================
    try:
        # Tái sử dụng predictions_output đã có từ bước trước (tránh predict 2 lần)
        save_predictions_from_output(predictions_output, test_df, config, id2label)
        
        # ALSO save to standard filename for analysis scripts
        predictions_standard_path = "test_predictions.csv"
        if config['paths']['predictions_file'] != predictions_standard_path:
            import shutil
            shutil.copy(config['paths']['predictions_file'], predictions_standard_path)
            print(f"✓ Đã copy predictions sang: {predictions_standard_path}")
    except Exception as e:
        print(f"\n⚠️  Cảnh báo: Không thể lưu predictions: {str(e)}")
    
    # =====================================================================
    # 12. LƯU MÔ HÌNH VÀ TOKENIZER
    # =====================================================================
    print(f"\n{'='*70}")
    print("💾 Đang lưu mô hình và tokenizer...")
    print(f"{'='*70}")
    
    try:
        # load_best_model_at_end=True chỉ load best model vào memory
        # Phải gọi save_model() để lưu ra disk
        final_model_dir = output_dir
        
        # Save best model (dùng eval_trainer thay vì trainer đã bị xóa)
        eval_trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        print(f"\n✓ Mô hình và tokenizer đã được lưu tại: {final_model_dir}")
        print(f"✓ Bạn có thể load lại bằng:")
        print(f"   tokenizer = AutoTokenizer.from_pretrained('{final_model_dir}')")
        print(f"   model = AutoModelForSequenceClassification.from_pretrained('{final_model_dir}')")
        
    except Exception as e:
        print(f"\n⚠️  Cảnh báo: Không thể lưu mô hình: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # =====================================================================
    # 12.5. GIẢI PHÓNG GPU MEMORY TRƯỚC ANALYSIS
    # =====================================================================
    print(f"\n{'='*70}")
    print("🧹 GIẢI PHÓNG GPU MEMORY")
    print(f"{'='*70}")
    
    # Xóa eval_trainer và model sau khi đã save xong
    del eval_trainer
    del current_model
    torch.cuda.empty_cache()
    
    print(f"✓ Đã giải phóng GPU memory")
    
    # =====================================================================
    # 13. TỰ ĐỘNG PHÂN TÍCH KẾT QUẢ
    # =====================================================================
    print(f"\n{'='*70}")
    print("📊 TỰ ĐỘNG PHÂN TÍCH KẾT QUẢ CHI TIẾT")
    print(f"{'='*70}")
    
    try:
        # Import và chạy analyze_results
        import analyze_results
        
        print("✓ Đang chạy phân tích chi tiết...")
        analyze_results.main()
        
    except Exception as e:
        print(f"\n⚠️  Cảnh báo: Không thể tự động phân tích: {str(e)}")
        print(f"   Bạn có thể chạy thủ công: python analyze_results.py")
    
    # =====================================================================
    # 14. KẾT THÚC
    # =====================================================================
    print(f"\n{'='*70}")
    print("🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")
    print(f"{'='*70}")
    
    print(f"\n✓ Tổng kết:")
    print(f"   • Mô hình đã được fine-tune thành công")
    print(f"   • F1 Score trên test: {test_results['eval_f1']:.4f}")
    print(f"   • Mô hình được lưu tại: {output_dir}")
    print(f"   • Báo cáo đánh giá: {config['paths']['evaluation_report']}")
    print(f"   • Predictions: {config['paths']['predictions_file']}")
    print(f"   • Phân tích chi tiết: analysis_results/")
    
    print(f"\n✓ Cảm ơn bạn đã sử dụng! 🙏\n")
    
    # =====================================================================
    # ĐÓNG LOGGER VÀ RESTORE STDOUT/STDERR
    # =====================================================================
    print(f"\n📝 Training log đã được lưu tại: {log_file_path}")
    
    # Restore stdout/stderr và đóng file log
    sys.stdout = tee_logger.terminal
    sys.stderr = tee_logger.terminal
    tee_logger.close()


if __name__ == '__main__':
    main()
