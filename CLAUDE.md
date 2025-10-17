# CLAUDE.md - Vietnamese ABSA Project Guide

## Project Overview

This is a Vietnamese **Aspect-Based Sentiment Analysis (ABSA)** project that fine-tunes ViSoBERT (Vietnamese Sentiment BERT) to analyze sentiment towards specific aspects in product reviews. The system can analyze a sentence and determine the sentiment (positive/negative/neutral) towards different aspects like Battery, Camera, Performance, etc.

**Model Performance:**
- Accuracy: 89.91%
- F1 Score: 90.77%
- Confidence: 83.01%

**Base Model:** `5CD-AI/Vietnamese-Sentiment-visobert`

## Key Commands

### Core Workflow
```bash
# 1. Prepare data (splits dataset.csv into train/val/test)
python prepare_data.py

# 2. Train model (main training pipeline)
python train.py --config config.yaml

# 3. Generate predictions on test set
python generate_test_predictions.py

# 4. Analyze results with detailed visualizations
python analyze_results.py

# 5. Interactive prediction demo
python predict_example.py
```

### Additional Utilities
```bash
# Generate predictions for any CSV file (for error analysis)
python generate_predictions.py --input data/train.csv --output train_predictions.csv

# Shuffle dataset (if needed)
python shuffle_dataset.py
```

## Architecture Overview

### Data Processing Pipeline
1. **Input Format:** `dataset.csv` (multi-label format - one row per review with multiple aspect columns)
2. **Data Preparation:** `prepare_data.py` converts to single-label format (one row per sentence-aspect pair)
3. **Output Format:** `sentence, aspect, sentiment` CSV files in `data/` directory

### Model Architecture
- **Input Format:** `[CLS] sentence [SEP] aspect [SEP]`
- **Model:** ViSoBERT fine-tuned for 3-class classification
- **Classes:** positive (0), negative (1), neutral (2)
- **Max Length:** 128 tokens (covers 97.7% of data)

### Training Pipeline
- **Custom Loss:** Focal Loss with class weighting to handle imbalance
- **Optimizer:** AdamW with cosine learning rate scheduler
- **Mixed Precision:** FP16 training for RTX 3070 optimization
- **Early Stopping:** Patience of 3 epochs
- **Checkpoint Management:** Automatic renaming based on accuracy (e.g., `checkpoint-9185-e3`)

## Key Files and Their Purposes

### Core Scripts
- **`train.py`**: Main training script with complete pipeline including data loading, training, evaluation, and automatic analysis
- **`prepare_data.py`**: Converts multi-label dataset to single-label ABSA format with stratified splitting
- **`analyze_results.py`**: Generates comprehensive analysis with confusion matrices, performance charts, and detailed reports
- **`generate_test_predictions.py`**: Generates predictions on test set
- **`predict_example.py`**: Interactive prediction demo

### Utility Modules
- **`utils.py`**: Core utilities including FocalLoss, ABSADataset, data loading, and metrics computation
- **`oversampling_utils.py`**: Class imbalance handling strategies (currently disabled in training)
- **`focal_loss_trainer.py`**: Custom Trainer wrapper for Focal Loss
- **`checkpoint_renamer.py`**: Callback to rename checkpoints by performance metrics

### Configuration
- **`config.yaml`**: Complete configuration including model settings, training hyperparameters, and paths

## Data Structure

### Input Data (`dataset.csv`)
- **Format:** Multi-label with one row per review
- **Columns:** `data` (review text) + 13 aspect columns (Battery, Camera, Performance, Display, Design, Software, Packaging, Price, Audio, Shop_Service, Shipping, General, Others)
- **Sentiments:** Positive, Negative, Neutral

### Processed Data (`data/` directory)
- **`train.csv`**: Training set (80% by default)
- **`validation.csv`**: Validation set (10% by default)
- **`test.csv`**: Test set (10% by default)
- **`data_metadata.json`**: Metadata about data preparation

### Model Outputs
- **`finetuned_visobert_absa_model/`**: Trained model directory
- **`test_predictions.csv`**: Predictions with confidence scores
- **`evaluation_report.txt`**: Detailed evaluation metrics
- **`analysis_results/`**: Comprehensive analysis visualizations

## Supported Aspects (13 Categories)
1. Battery
2. Camera
3. Performance
4. Display
5. Design
6. Software
7. Packaging
8. Price
9. Audio
10. Shop_Service (includes Warranty)
11. Shipping
12. General
13. Others

## Model Configuration

### Hardware Optimization (RTX 3070 8GB)
- **Batch Size:** 80 (aggressive VRAM usage)
- **GPU Usage:** 95%+
- **VRAM Usage:** 7.5-7.8GB
- **Mixed Precision:** FP16 enabled
- **Data Loading:** 2 workers with prefetching

### Training Hyperparameters
- **Learning Rate:** 2e-5
- **Epochs:** 5 with early stopping
- **Scheduler:** Cosine with 10% warmup
- **Weight Decay:** 0.01
- **Max Gradient Norm:** 1.0

### Class Imbalance Handling
- **Current Strategy:** Focal Loss (gamma=2.0) with inverse frequency alpha weights
- **Previous Strategy:** Aspect-wise oversampling (currently disabled)
- **Labels:** positive=0, negative=1, neutral=2

## Analysis and Visualization

The `analyze_results.py` script generates comprehensive analysis in `analysis_results/`:

### Generated Files
- **`confusion_matrices_all_aspects.png`**: Grid of confusion matrices for all aspects
- **`confusion_matrix_overall.png`**: Overall confusion matrix
- **`metrics_comparison.png`**: Performance comparison across aspects
- **`accuracy_by_aspect.png`**: Accuracy scores by aspect
- **`f1_score_by_aspect.png`**: F1 scores by aspect
- **`precision_by_aspect.png`**: Precision by aspect
- **`recall_by_aspect.png`**: Recall by aspect
- **`sample_distribution.png`**: Sample distribution across aspects
- **`metrics_heatmap.png`**: Performance heatmap
- **`summary_table.png`**: Performance summary table
- **`detailed_analysis_report.txt`**: Detailed text report

## Important Implementation Details

### Text Preprocessing
- **VNCoreNLP Segmentation:** Underscores are automatically removed (e.g., "Chăm_sóc" → "Chăm sóc")
- **Encoding:** UTF-8-sig for BOM handling
- **Tokenization:** BERT tokenizer with sentence-aspect pair format

### Checkpoint Management
- **Naming Convention:** `checkpoint-{accuracy}{epoch}` (e.g., `checkpoint-9185-e3`)
- **Best Model:** Automatically loaded at end via `load_best_model_at_end=True`
- **Retention:** Top 3 checkpoints saved

### Training Optimization
- **Gradient Accumulation:** Disabled (batch size already optimal)
- **Gradient Checkpointing:** Disabled (sufficient VRAM)
- **Deterministic Training:** Seeds set for reproducibility
- **Logging:** Comprehensive logging to `training_logs/` with timestamps

## Common Workflows

### Training New Model
```bash
# 1. Prepare fresh data split
python prepare_data.py

# 2. Start training (automatic analysis will run after training)
python train.py --config config.yaml

# 3. Check results in analysis_results/
```

### Making Predictions
```bash
# Interactive demo
python predict_example.py

# Batch predictions on test set
python generate_test_predictions.py

# Custom dataset predictions
python generate_predictions.py --input custom_data.csv --output predictions.csv
```

### Analyzing Performance
```bash
# Full analysis with visualizations
python analyze_results.py

# Check detailed report
cat analysis_results/detailed_analysis_report.txt

# View evaluation summary
cat evaluation_report.txt
```

## Memory and Performance Notes

### GPU Memory Management
- Training uses ~7.8GB VRAM on RTX 3070
- Model is unloaded after training to free memory for analysis
- Evaluation uses separate trainer without optimizer to reduce memory

### Performance Optimization
- FP16 mixed precision for faster training
- Optimized DataLoader settings
- Batch prediction in inference scripts
- Efficient checkpoint management

## Troubleshooting

### Common Issues
1. **CUDA OOM:** Reduce batch size in config.yaml
2. **Data not found:** Run `prepare_data.py` first
3. **Model not found:** Run `train.py` to train model
4. **Encoding issues:** All files use UTF-8-sig encoding

### File Dependencies
- **Training:** Requires `dataset.csv` → runs `prepare_data.py` automatically if needed
- **Prediction:** Requires trained model in `finetuned_visobert_absa_model/`
- **Analysis:** Requires `test_predictions.csv` from `generate_test_predictions.py`

## Model Usage Examples

### Loading Model for Inference
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("finetuned_visobert_absa_model")
model = AutoModelForSequenceClassification.from_pretrained("finetuned_visobert_absa_model")

# Predict sentiment
inputs = tokenizer("Pin trâu lắm", "Battery", return_tensors="pt")
outputs = model(**inputs)
predicted_class = outputs.logits.argmax().item()
```

### Label Mapping
```python
id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
sentiment = id2label[predicted_class]
```

This project demonstrates a complete ABSA pipeline optimized for Vietnamese product reviews with comprehensive analysis capabilities and production-ready implementation.