# Vietnamese ABSA Model

     # Built-in loop (update every 1 sec)
     nvidia-smi -l 1

     # Compact view
     nvidia-smi dmon -s u

     # Only GPU % and Memory
     nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader --loop=1

Aspect-Based Sentiment Analysis for Vietnamese using ViSoBERT.

## Quick Start

```bash
# 1. Prepare data
python prepare_data.py

# 2. Train model
python train.py

# 3. Generate predictions
python generate_test_predictions.py

# 4. Analyze results
python analyze_results.py
```

## Project Structure

```
BERT/
├── config.yaml                 # Training configuration
├── dataset.csv                 # Main dataset
├── data/                       # Split data (train/val/test)
├── finetuned_visobert_absa_model/  # Trained model
├── analysis_results/           # Analysis outputs
├── training_logs/              # Training logs
├── docs/                       # Documentation
├── tests/                      # Test & analysis scripts
└── backups/                    # Old backups
```

## Main Scripts

| Script | Purpose |
|--------|---------|
| `prepare_data.py` | Split dataset into train/val/test |
| `train.py` | Train the model |
| `generate_test_predictions.py` | Generate predictions on test set |
| `generate_predictions.py` | Generate predictions on any CSV |
| `analyze_results.py` | Detailed analysis with charts |
| `predict_example.py` | Interactive prediction |

## Model Performance

- **Accuracy**: 89.91%
- **F1 Score**: 90.77%
- **Confidence**: 83.01%

## Configuration

Edit `config.yaml` to adjust:
- Batch size (default: 80)
- Learning rate (default: 2e-5)
- Epochs (default: 5)
- GPU optimization settings

## Requirements

```bash
pip install -r requirements.txt
```

## GPU Settings

Optimized for RTX 3070 (8GB VRAM):
- Batch size: 80
- Expected GPU usage: 95%+
- Expected VRAM: 7.5-7.8GB
